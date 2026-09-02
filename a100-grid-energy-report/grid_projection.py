#!/usr/bin/env python3
"""Reproduce the movement-only grid projections used by the report.

The scorer charges ``ceil(sqrt(address))`` for each source-operand read.
Arithmetic and destination writes are free, while every declared output is
read once at program exit.  Multiplying the integer score by 1 fJ gives the
energy proxy reported here.

The Ciresan calculation models the six logical rectangular matrix
multiplications in a forward pass through
``784-2500-2000-1500-1000-500-10``.  Scratch and bulk regions are jointly
packed by reads per cell.  Tile sizes are coordinate-descent optima within the
generalized rectangular ``sa_cache`` schedule family, not global lower bounds.
"""

from __future__ import annotations

import json

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


def forward_score(
    batch_size: int, tiles: list[tuple[int, int]]
) -> tuple[int, int]:
    """Jointly pack and score all six logical forward matmuls."""
    activation_regions = []
    weight_regions = []
    sb_specs = []
    sc_specs = []
    sa_reads = 0
    tmp_reads = 0

    for layer, (input_width, output_width) in enumerate(
        zip(NETWORK_DIMS, NETWORK_DIMS[1:])
    ):
        tile_i, tile_j = tiles[layer]
        macs = batch_size * input_width * output_width
        sa_reads += macs
        tmp_reads += batch_size * output_width * (input_width - 1)
        sb_specs.append((tile_j, macs // tile_j))
        sc_specs.append((tile_i * tile_j, macs // (tile_i * tile_j)))
        activation_regions.append(
            (
                f"x{layer}",
                batch_size * input_width,
                output_width // tile_j,
            )
        )
        weight_regions.append(
            (
                f"W{layer}",
                input_width * output_width,
                batch_size // tile_i,
            )
        )

    regions = [("sa", 1, sa_reads), ("tmp", 1, tmp_reads)]
    regions += scratch_tiers("sb", sb_specs)
    regions += scratch_tiers("sc", sc_specs)
    regions += activation_regions
    regions += weight_regions
    regions.append(("out", batch_size * NETWORK_DIMS[-1], 1))
    return pack_regions(regions)


def optimize_forward(batch_size: int) -> tuple[int, int, list[tuple[int, int]], int]:
    """Coordinate-descent tile search from per-layer independent optima."""
    tiles = initial_tiles(batch_size)
    score, paid_reads = forward_score(batch_size, tiles)
    row_tiles = divisors(batch_size)

    for sweep in range(1, 101):
        changed = 0
        for layer, output_width in enumerate(NETWORK_DIMS[1:]):
            old_tile = tiles[layer]
            best_score = score
            best_reads = paid_reads
            best_tile = old_tile
            for tile_i in row_tiles:
                for tile_j in divisors(output_width):
                    tiles[layer] = (tile_i, tile_j)
                    candidate_score, candidate_reads = forward_score(
                        batch_size, tiles
                    )
                    if candidate_score < best_score:
                        best_score = candidate_score
                        best_reads = candidate_reads
                        best_tile = (tile_i, tile_j)
            tiles[layer] = best_tile
            if best_tile != old_tile:
                score = best_score
                paid_reads = best_reads
                changed += 1
        if not changed:
            return score, paid_reads, tiles, sweep
    raise RuntimeError("tile search did not converge")


def case(batch_size: int) -> dict:
    if TRAINING_EXAMPLES % batch_size:
        raise ValueError("this audit requires an exact number of equal-size batches")
    score, paid_reads, tiles, sweeps = optimize_forward(batch_size)
    invocations = TRAINING_EXAMPLES // batch_size
    epoch_score = score * invocations
    epoch_reads = paid_reads * invocations
    return {
        "batch_size": batch_size,
        "batch_invocations": invocations,
        "logical_MAC_per_invocation": (
            batch_size
            * sum(
                input_width * output_width
                for input_width, output_width in zip(
                    NETWORK_DIMS, NETWORK_DIMS[1:]
                )
            )
        ),
        "score_per_invocation": score,
        "paid_reads_per_invocation": paid_reads,
        "epoch_score": epoch_score,
        "epoch_paid_reads": epoch_reads,
        "epoch_energy_J_at_1_fJ_per_score_unit": (
            epoch_score / SCORE_UNITS_PER_JOULE
        ),
        "coordinate_descent_sweeps": sweeps,
        "tiles": [
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
        ],
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

    return {
        "report_date": "2026-09-01",
        "coefficient_J_per_score_unit": JOULES_PER_SCORE_UNIT,
        "score_unit": "one source operand times one abstract address-grid step",
        "distance": "d(address) = ceil(sqrt(address))",
        "network": {
            "dimensions": list(NETWORK_DIMS),
            "examples_per_inference_epoch": TRAINING_EXAMPLES,
            "logical_MAC_per_example": logical_macs_per_example,
            "logical_MAC_per_epoch": logical_macs_per_epoch,
        },
        "schedule": {
            "family": "jointly packed generalized rectangular sa_cache",
            "tile_search": (
                "coordinate descent from independently optimal per-layer tiles"
            ),
            "qualification": (
                "best found within this schedule family; not a global lower bound"
            ),
        },
        "cases": {
            "batch16_epoch": case(16),
            "single_full_batch_epoch": case(TRAINING_EXAMPLES),
        },
        "mac_count_only_sensitivity": {
            "reference": "retuned 8192^3 sa_cache score",
            "reference_score": EIGHTK_SCORE,
            "reference_MAC": EIGHTK_N**3,
            "reference_energy_J": reference_energy,
            "energy_J_per_MAC": reference_energy / EIGHTK_N**3,
            "projected_MNIST_epoch_energy_J_for_either_batching": mac_only_energy,
        },
        "precision_semantics": (
            "The scorer has unbounded scalar cells and no bit width; these grid "
            "scores do not change between FP32, FP16, INT8, or one-bit values."
        ),
        "omitted_terms": [
            "destination writes",
            "arithmetic, bias, ReLU, saturation, and quantization",
            "Tensor Core shape padding",
            "full-dataset storage and batch gather",
            "kernel launch, control, clocking, leakage, and elapsed time",
            "physical cache, SRAM, HBM, and host-transfer behavior",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2) + "\n", end="")
