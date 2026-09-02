#!/usr/bin/env python3
"""Reproduce the movement-only grid projections used by the report.

The scorer charges ``ceil(sqrt(address))`` for each source-operand read.
Arithmetic and destination writes are free, while every declared output is
read once at program exit.  Multiplying the integer score by 1 fJ gives the
energy proxy reported here.

The Ciresan calculation models the six logical rectangular matrix
multiplications in a forward pass through
``784-2500-2000-1500-1000-500-10``.  Scratch and bulk regions are jointly
packed by reads per cell.  Tile sizes are the best result from three
deterministic coordinate-descent starts within the generalized rectangular
``sa_cache`` schedule family, not global lower bounds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from math import isqrt
except ImportError:  # Python 3.7 compatibility.
    def isqrt(value: int) -> int:
        if value < 0:
            raise ValueError("isqrt() argument must be nonnegative")
        if value < 2:
            return value
        guess = 1 << ((value.bit_length() + 1) // 2)
        while True:
            update = (guess + value // guess) // 2
            if update >= guess:
                return guess
            guess = update


NETWORK_DIMS = (784, 2500, 2000, 1500, 1000, 500, 10)
TRAINING_EXAMPLES = 60_000
JOULES_PER_SCORE_UNIT = 1e-15
SCORE_UNITS_PER_JOULE = 1_000_000_000_000_000
EIGHTK_N = 8192
EIGHTK_TILE = (256, 128)
EIGHTK_SCORE = 118_953_083_721_334


def shell_prefix(last_address: int) -> int:
    """Sum ceil(sqrt(address)) for addresses 1 through last_address."""
    if last_address <= 0:
        return 0
    root = isqrt(last_address)
    return (
        root * (root + 1) * (4 * root - 1) // 6
        + (last_address - root * root) * (root + 1)
    )


def shell_sum(first_address: int, last_address: int) -> int:
    return shell_prefix(last_address) - shell_prefix(first_address - 1)


def pack_regions(regions: list[tuple[str, int, int]]) -> tuple[int, int]:
    """Pack highest-read regions nearest the processor and return (S, R)."""
    score = 0
    paid_reads = 0
    address = 1
    for _name, cell_count, reads_per_cell in sorted(
        regions, key=lambda region: (-region[2], region[0])
    ):
        if not cell_count or not reads_per_cell:
            continue
        last_address = address + cell_count - 1
        score += reads_per_cell * shell_sum(address, last_address)
        paid_reads += cell_count * reads_per_cell
        address = last_address + 1
    return score, paid_reads


def divisors(value: int) -> list[int]:
    return [candidate for candidate in range(1, value + 1) if value % candidate == 0]


def scratch_tiers(
    name: str, specifications: list[tuple[int, int]]
) -> list[tuple[str, int, int]]:
    """Merge differently sized scratch arrays by overlapping their hot prefix."""
    previous_end = 0
    regions = []
    for end in sorted({size for size, _reads in specifications}):
        reads = sum(
            reads_per_cell
            for size, reads_per_cell in specifications
            if size >= end
        )
        regions.append((f"{name}{end}", end - previous_end, reads))
        previous_end = end
    return regions


def one_matmul_score(
    rows: int, contraction: int, columns: int, tile_i: int, tile_j: int
) -> int:
    """Score one rectangular sa_cache matmul under an independently packed layout."""
    macs = rows * contraction * columns
    regions = [
        ("sa", 1, macs),
        ("tmp", 1, rows * columns * (contraction - 1)),
        ("sb", tile_j, macs // tile_j),
        ("sc", tile_i * tile_j, macs // (tile_i * tile_j)),
        ("A", rows * contraction, columns // tile_j),
        ("B", contraction * columns, rows // tile_i),
        ("C", rows * columns, 1),
    ]
    return pack_regions(regions)[0]


def initial_tiles(batch_size: int) -> list[tuple[int, int]]:
    choices = []
    row_tiles = divisors(batch_size)
    for input_width, output_width in zip(NETWORK_DIMS, NETWORK_DIMS[1:]):
        candidates = [
            (tile_i, tile_j)
            for tile_i in row_tiles
            for tile_j in divisors(output_width)
        ]
        choices.append(
            min(
                candidates,
                key=lambda tile: one_matmul_score(
                    batch_size,
                    input_width,
                    output_width,
                    tile[0],
                    tile[1],
                ),
            )
        )
    return choices


def epoch_decomposition(batch_size: int) -> list[tuple[int, int]]:
    """Return ``(rows, invocation_count)`` pairs for one 60k-image epoch."""
    if batch_size < 1 or batch_size > TRAINING_EXAMPLES:
        raise ValueError("batch size must be between 1 and 60,000")
    full_calls, remainder = divmod(TRAINING_EXAMPLES, batch_size)
    calls = []
    if full_calls:
        calls.append((batch_size, full_calls))
    if remainder:
        calls.append((remainder, 1))
    assert sum(rows * count for rows, count in calls) == TRAINING_EXAMPLES
    return calls


def epoch_persistent_score(
    decomposition: list[tuple[int, int]],
    tiles_by_rows: dict[int, list[tuple[int, int]]],
) -> tuple[int, int]:
    """Score an epoch in one persistent address space.

    Weights, the 60k-image input, and the 60k-image output are allocated once.
    Scratch and intermediate activation buffers are reused across invocations.
    This makes the spatial footprint of all source images and final outputs
    visible while allowing computed transient values to overwrite dead values.
    """
    activation_regions = []
    weight_regions = []
    sb_specs = []
    sc_specs = []
    sa_reads = 0
    tmp_reads = 0

    for layer, (input_width, output_width) in enumerate(
        zip(NETWORK_DIMS, NETWORK_DIMS[1:])
    ):
        local_activation_specs = []
        weight_reads_per_cell = 0
        for rows, invocation_count in decomposition:
            tile_i, tile_j = tiles_by_rows[rows][layer]
            macs_per_call = rows * input_width * output_width
            sa_reads += invocation_count * macs_per_call
            tmp_reads += (
                invocation_count
                * rows
                * output_width
                * (input_width - 1)
            )
            sb_specs.append(
                (tile_j, invocation_count * macs_per_call // tile_j)
            )
            sc_specs.append(
                (
                    tile_i * tile_j,
                    invocation_count * macs_per_call // (tile_i * tile_j),
                )
            )

            if layer == 0:
                # These are distinct source images, not a buffer prefix reused
                # by each call.
                activation_regions.append(
                    (
                        f"x0_r{rows}",
                        invocation_count * rows * input_width,
                        output_width // tile_j,
                    )
                )
            else:
                # Intermediate storage is a batch-local buffer whose active
                # prefix is reused on every call.
                local_activation_specs.append(
                    (
                        rows * input_width,
                        invocation_count * (output_width // tile_j),
                    )
                )
            weight_reads_per_cell += invocation_count * (rows // tile_i)

        if layer:
            activation_regions += scratch_tiers(
                f"x{layer}_", local_activation_specs
            )
        weight_regions.append(
            (
                f"W{layer}",
                input_width * output_width,
                weight_reads_per_cell,
            )
        )

    regions = [
        ("sa", 1, sa_reads),
        ("tmp", 1, tmp_reads),
    ]
    regions += scratch_tiers("sb", sb_specs)
    regions += scratch_tiers("sc", sc_specs)
    regions += activation_regions
    regions += weight_regions
    regions.append(("out", TRAINING_EXAMPLES * NETWORK_DIMS[-1], 1))
    return pack_regions(regions)


def optimize_epoch_persistent(
    batch_size: int,
) -> tuple[int, int, dict[int, list[tuple[int, int]]], int, str]:
    """Three-start coordinate descent for one persistent-address-space epoch."""
    decomposition = epoch_decomposition(batch_size)
    starts = [
        (
            "independent_layer_optima",
            {rows: initial_tiles(rows) for rows, _count in decomposition},
        ),
        (
            "all_minimum_tiles",
            {
                rows: [(1, 1) for _output_width in NETWORK_DIMS[1:]]
                for rows, _count in decomposition
            },
        ),
        (
            "all_maximum_tiles",
            {
                rows: [
                    (rows, output_width)
                    for output_width in NETWORK_DIMS[1:]
                ]
                for rows, _count in decomposition
            },
        ),
    ]
    best_result = None

    for start_name, tiles_by_rows in starts:
        score, paid_reads = epoch_persistent_score(
            decomposition, tiles_by_rows
        )
        for sweep in range(1, 101):
            changed = 0
            for rows, _invocation_count in decomposition:
                row_tiles = divisors(rows)
                for layer, output_width in enumerate(NETWORK_DIMS[1:]):
                    old_tile = tiles_by_rows[rows][layer]
                    best_score = score
                    best_reads = paid_reads
                    best_tile = old_tile
                    for tile_i in row_tiles:
                        for tile_j in divisors(output_width):
                            tiles_by_rows[rows][layer] = (tile_i, tile_j)
                            candidate_score, candidate_reads = (
                                epoch_persistent_score(
                                    decomposition, tiles_by_rows
                                )
                            )
                            if candidate_score < best_score:
                                best_score = candidate_score
                                best_reads = candidate_reads
                                best_tile = (tile_i, tile_j)
                    tiles_by_rows[rows][layer] = best_tile
                    if best_tile != old_tile:
                        score = best_score
                        paid_reads = best_reads
                        changed += 1
            if not changed:
                candidate = (
                    score,
                    paid_reads,
                    {
                        rows: list(tiles)
                        for rows, tiles in tiles_by_rows.items()
                    },
                    sweep,
                    start_name,
                )
                if best_result is None or candidate[0] < best_result[0]:
                    best_result = candidate
                break
        else:
            raise RuntimeError("persistent epoch tile search did not converge")

    assert best_result is not None
    return best_result


def persistent_case(batch_size: int) -> dict:
    decomposition = epoch_decomposition(batch_size)
    score, paid_reads, tiles_by_rows, sweeps, winning_start = (
        optimize_epoch_persistent(batch_size)
    )
    return {
        "batch_size": batch_size,
        "batch_invocations": sum(count for _rows, count in decomposition),
        "decomposition": [
            {"rows": rows, "invocation_count": count}
            for rows, count in decomposition
        ],
        "epoch_movement_energy_fJ": score,
        "epoch_paid_reads": paid_reads,
        "average_grid_steps_per_paid_read": score / paid_reads,
        "epoch_movement_energy_J": (
            score / SCORE_UNITS_PER_JOULE
        ),
        "dally_2023_all_reads_small_RAM_same_schedule_sensitivity_J": (
            score * 1e-15 + paid_reads * 400e-15
        ),
        "coordinate_descent_sweeps": sweeps,
        "coordinate_descent_starts": 3,
        "winning_start": winning_start,
        "tiles_by_invocation_rows": {
            str(rows): [
                {
                    "layer": layer,
                    "input_width": input_width,
                    "output_width": output_width,
                    "tile_i": tiles[layer][0],
                    "tile_j": tiles[layer][1],
                }
                for layer, (input_width, output_width) in enumerate(
                    zip(NETWORK_DIMS, NETWORK_DIMS[1:])
                )
            ]
            for rows, tiles in tiles_by_rows.items()
        },
    }


def audit() -> dict:
    # The common rectangular formula must reproduce the independently audited
    # 8192^3 result before it is used for the network projection.
    assert one_matmul_score(
        EIGHTK_N, EIGHTK_N, EIGHTK_N, EIGHTK_TILE[0], EIGHTK_TILE[1]
    ) == EIGHTK_SCORE

    logical_macs_per_example = sum(
        input_width * output_width
        for input_width, output_width in zip(NETWORK_DIMS, NETWORK_DIMS[1:])
    )
    logical_macs_per_epoch = logical_macs_per_example * TRAINING_EXAMPLES
    reference_energy = EIGHTK_SCORE / SCORE_UNITS_PER_JOULE
    mac_only_energy = logical_macs_per_epoch * reference_energy / EIGHTK_N**3
    batch_sizes = (TRAINING_EXAMPLES, 64, 16, 4, 1)
    persistent_cases = {
        ("full" if batch_size == TRAINING_EXAMPLES else str(batch_size)):
        persistent_case(batch_size)
        for batch_size in batch_sizes
    }
    full_persistent_energy = persistent_cases["full"][
        "epoch_movement_energy_fJ"
    ]
    for result in persistent_cases.values():
        result["movement_energy_ratio_to_full_batch"] = (
            result["epoch_movement_energy_fJ"] / full_persistent_energy
        )
    assert persistent_cases["64"]["decomposition"] == [
        {"rows": 64, "invocation_count": 937},
        {"rows": 32, "invocation_count": 1},
    ]
    for result in persistent_cases.values():
        assert sum(
            call["rows"] * call["invocation_count"]
            for call in result["decomposition"]
        ) == TRAINING_EXAMPLES

    return {
        "report_date": "2026-09-02",
        "joules_per_operand_grid_step": JOULES_PER_SCORE_UNIT,
        "movement_energy_unit": (
            "1 fJ per source operand per abstract address-grid step"
        ),
        "distance": "d(address) = ceil(sqrt(address))",
        "int8_step_interpretation": {
            "dally_2023_on_chip_communication_fJ_per_bit_mm": 100,
            "abstract_step_fJ_per_int8_operand": 1,
            "equivalent_distance_um_per_step": 1.25,
            "qualification": (
                "an energy-equivalent scale comparison, not a claim that an "
                "abstract address step is a physical 1.25 um wire"
            ),
        },
        "network": {
            "dimensions": list(NETWORK_DIMS),
            "examples_per_inference_epoch": TRAINING_EXAMPLES,
            "logical_MAC_per_example": logical_macs_per_example,
            "logical_MAC_per_epoch": logical_macs_per_epoch,
        },
        "schedule": {
            "family": "jointly packed generalized rectangular sa_cache",
            "tile_search": (
                "best of three deterministic coordinate-descent starts: "
                "independent layer optima, all-minimum tiles, and all-maximum tiles"
            ),
            "qualification": (
                "best found within this schedule family; not a global lower bound"
            ),
        },
        "batching_analysis": {
            "persistent_epoch_address_space": {
                "description": (
                    "weights, the 60k-image source, and the 60k-image output "
                    "are allocated once and retained; layer-specific batch-local "
                    "scratch and intermediate activation buffers are reused "
                    "across invocations"
                ),
                "input_storage": {
                    "materialized_before_modeled_run": True,
                    "examples": TRAINING_EXAMPLES,
                    "features_per_example": NETWORK_DIMS[0],
                    "distinct_input_cells": (
                        TRAINING_EXAMPLES * NETWORK_DIMS[0]
                    ),
                    "reusable_64_row_input_buffer": False,
                    "batch64_partition": [
                        {
                            "calls": 937,
                            "rows_per_call": 64,
                            "distinct_input_cells": 937 * 64 * NETWORK_DIMS[0],
                        },
                        {
                            "calls": 1,
                            "rows_per_call": 32,
                            "distinct_input_cells": 32 * NETWORK_DIMS[0],
                        },
                    ],
                    "address_assignment": (
                        "all input cells are jointly frequency-packed in the "
                        "persistent epoch address space"
                    ),
                    "initial_materialization_energy": (
                        "excluded; inputs exist at their assigned addresses "
                        "before modeled execution begins"
                    ),
                },
                "transient_storage": (
                    "computed hidden-layer activations use layer-specific "
                    "batch-row buffers that are overwritten across invocations"
                ),
                "cases": persistent_cases,
            },
            "dally_2023_endpoint_sensitivity": {
                "formula": (
                    "E_fJ = movement_energy_fJ + 400*paid_source_reads "
                    "for INT8"
                ),
                "source": (
                    "https://aha.stanford.edu/sites/g/files/sbiybj20066/"
                    "files/media/file/aha-retreat-2023_dally_keynote_en_eff_ai_hw_0.pdf"
                ),
                "source_values": (
                    "100 fJ/(bit mm) on-chip communication and 50 fJ/bit "
                    "small-RAM access"
                ),
                "qualification": (
                    "illustrative sensitivity only: it treats every abstract "
                    "paid source read, including scratch reads, as an 8-bit "
                    "small-RAM access and evaluates the movement-optimized "
                    "tiles without retuning; it is not part of the movement-only model"
                ),
            },
        },
        "mac_count_only_sensitivity": {
            "reference": "retuned 8192^3 sa_cache movement energy",
            "reference_movement_energy_fJ": EIGHTK_SCORE,
            "reference_MAC": EIGHTK_N**3,
            "reference_energy_J": reference_energy,
            "energy_J_per_MAC": reference_energy / EIGHTK_N**3,
            "projected_MNIST_epoch_energy_J_for_either_batching": mac_only_energy,
        },
        "precision_semantics": (
            "The model has unbounded scalar cells and no bit width; its movement "
            "energy does not change between FP32, FP16, INT8, or one-bit values."
        ),
        "omitted_terms": [
            "destination writes",
            "arithmetic, bias, ReLU, saturation, and quantization",
            "Tensor Core shape padding",
            "energy to initially materialize the resident dataset and any "
            "batch-gather or copy operation",
            "kernel launch, control, clocking, leakage, and elapsed time",
            "physical cache, SRAM, HBM, and host-transfer behavior",
            "GPU occupancy and underfilled-kernel efficiency",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="write the audit JSON to this path instead of standard output",
    )
    args = parser.parse_args()
    document = json.dumps(audit(), indent=2) + "\n"
    if args.output:
        args.output.write_text(document)
    else:
        print(document, end="")


if __name__ == "__main__":
    main()
