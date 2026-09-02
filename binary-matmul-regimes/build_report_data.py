#!/usr/bin/env python3
"""Build the derived tables for the native A100 B1-versus-INT8 report.

The raw benchmark JSON is intentionally preserved.  Modal supplied an
A100-SXM4-80GB even though the request named A100-40GB; this script therefore
uses the 80GB product's published 2,039 GB/s HBM2e bandwidth for the roofline
analysis.  Timing and energy values come directly from the raw JSON.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW_PATH = HERE / "a100_b1_vs_int8_results.json"
CSV_PATH = HERE / "results.csv"
DERIVED_PATH = HERE / "derived.json"

A100_INT8_TOPS = 624.0
A100_B1_TOPS = 4_992.0
A100_80GB_HBM_GB_S = 2_039.0
A100_DIE_MM2 = 826.0
A100_L2_MIB = 40.0

DALLY_WIRE_FJ_PER_BIT_MM = 100.0
DALLY_ADD_FJ_PER_BIT = 1.0
TSMC_N7_MINIMUM_METAL_PITCH_NM = 40.0

N = 8_192
TILE_I = 256
TILE_J = 128
GRID_SCORE = 118_953_083_721_334
GRID_CELLS = 1 + 1 + TILE_J + TILE_I * TILE_J + 3 * N * N


def milliseconds(seconds: float) -> float:
    return seconds * 1_000.0


def energy_lookup(raw: dict, label: str, family: str, m: int) -> dict:
    return next(
        row
        for row in raw["energy_results"]
        if row["label"] == label
        and row["shape"]["family"] == family
        and row["shape"]["m"] == m
    )


def timing_rows(raw: dict, family: str) -> list[dict]:
    rows = []
    for record in raw["timing_results"]:
        if record["family"] != family:
            continue
        timings = record["timings"]
        rows.append(
            {
                "family": family,
                "m": record["m"],
                "n": record["n"],
                "k": record["k"],
                "int8_ms": milliseconds(
                    timings["int8"]["seconds_per_call_median"]
                ),
                "b1_prepacked_ms": milliseconds(
                    timings["b1_prepacked"]["seconds_per_call_median"]
                ),
                "b1_pack_a_ms": milliseconds(
                    timings["b1_including_dynamic_a_pack"][
                        "seconds_per_call_median"
                    ]
                ),
                "b1_pack_both_ms": milliseconds(
                    timings["b1_including_both_packs"][
                        "seconds_per_call_median"
                    ]
                ),
                "speedup_prepacked": record[
                    "speedup_b1_prepacked_over_int8"
                ],
                "speedup_pack_a": record["speedup_b1_pack_a_over_int8"],
                "speedup_pack_both": record[
                    "speedup_b1_pack_both_over_int8"
                ],
                "int8_tops": record["effective_int8_TOPS"],
                "b1_tops": record["effective_b1_TOPS"],
            }
        )
    return rows


def energy_rows(raw: dict) -> list[dict]:
    cases = [
        ("square 2048³", "square", 2_048),
        ("square 8192³", "square", 8_192),
        ("output-heavy 8192×8192×256", "output_bound", 8_192),
    ]
    rows = []
    for case, family, m in cases:
        int8 = energy_lookup(raw, "int8", family, m)
        b1 = energy_lookup(raw, "b1_prepacked", family, m)
        pack_a = energy_lookup(
            raw, "b1_including_dynamic_a_pack", family, m
        )
        rows.append(
            {
                "case": case,
                "int8_ms": milliseconds(int8["seconds_per_call_wall"]),
                "int8_j": int8["idle_adjusted_gpu_energy_J_per_call"],
                "b1_ms": milliseconds(b1["seconds_per_call_wall"]),
                "b1_j": b1["idle_adjusted_gpu_energy_J_per_call"],
                "b1_time_reduction": int8["seconds_per_call_wall"]
                / b1["seconds_per_call_wall"],
                "b1_energy_reduction": int8[
                    "idle_adjusted_gpu_energy_J_per_call"
                ]
                / b1["idle_adjusted_gpu_energy_J_per_call"],
                "pack_a_ms": milliseconds(pack_a["seconds_per_call_wall"]),
                "pack_a_j": pack_a[
                    "idle_adjusted_gpu_energy_J_per_call"
                ],
                "pack_a_time_reduction": int8["seconds_per_call_wall"]
                / pack_a["seconds_per_call_wall"],
                "pack_a_energy_reduction": int8[
                    "idle_adjusted_gpu_energy_J_per_call"
                ]
                / pack_a["idle_adjusted_gpu_energy_J_per_call"],
            }
        )
    return rows


def physical_calibration() -> dict:
    radius_steps = math.ceil(math.sqrt(GRID_CELLS))
    # If the score's 1 fJ is for one whole INT8 value, divide Dally's per-bit
    # coefficient by eight.  If it is per bit, the later slide maps 1 fJ to
    # the full 10 micrometres.
    whole_int8_step_um = 1_000.0 / (
        8 * DALLY_WIRE_FJ_PER_BIT_MM
    )
    per_bit_step_um = 1_000.0 / DALLY_WIRE_FJ_PER_BIT_MM
    a100_side_mm = math.sqrt(A100_DIE_MM2)

    def geometry(step_um: float) -> dict:
        reach_mm = radius_steps * step_um / 1_000.0
        base_mm = 2 * reach_mm
        lattice_area_mm2 = GRID_CELLS * (step_um / 1_000.0) ** 2
        return {
            "step_um": step_um,
            "reach_mm": reach_mm,
            "base_mm": base_mm,
            "lattice_area_mm2": lattice_area_mm2,
            "area_over_a100": lattice_area_mm2 / A100_DIE_MM2,
            "reach_over_a100_equal_area_side": reach_mm / a100_side_mm,
            "base_over_a100_equal_area_side": base_mm / a100_side_mm,
            "minimum_metal_pitches_per_step": step_um
            * 1_000.0
            / TSMC_N7_MINIMUM_METAL_PITCH_NM,
            "nominal_7nm_labels_per_step": step_um * 1_000.0 / 7.0,
        }

    return {
        "wire_coefficient_fJ_per_bit_mm": DALLY_WIRE_FJ_PER_BIT_MM,
        "add_energy_fJ_per_bit": DALLY_ADD_FJ_PER_BIT,
        "add_equivalent_distance_um": 10.0,
        "grid_footprint_cells": GRID_CELLS,
        "grid_radius_steps": radius_steps,
        "a100_die_mm2": A100_DIE_MM2,
        "a100_equal_area_side_mm": a100_side_mm,
        "tsmc_n7_minimum_metal_pitch_nm": TSMC_N7_MINIMUM_METAL_PITCH_NM,
        "one_fJ_per_whole_int8_value_step": geometry(whole_int8_step_um),
        "one_fJ_per_bit_step": geometry(per_bit_step_um),
    }


def build() -> dict:
    raw = json.loads(RAW_PATH.read_text())
    square = timing_rows(raw, "square")
    k_sweep = timing_rows(raw, "k_sweep")
    batch = timing_rows(raw, "batch")
    energy = energy_rows(raw)
    cache = raw["weight_residency_result"]
    cache_energy = {
        row["label"]: row
        for row in raw["energy_results"]
        if row["regime"] == "weight_residency"
    }

    n = float(N)
    int8_square_ridge_n = 3 * A100_INT8_TOPS * 1e12 / (
        A100_80GB_HBM_GB_S * 1e9
    )
    b1_square_ridge_n = 2.125 * A100_B1_TOPS * 1e12 / (
        A100_80GB_HBM_GB_S * 1e9
    )
    result = {
        "source": RAW_PATH.name,
        "hardware": raw["hardware"],
        "semantics": raw["semantics"],
        "squares": square,
        "k_sweep": k_sweep,
        "batch": batch,
        "energy": energy,
        "cache": {
            "shape": [cache["m"], cache["n"], cache["k"]],
            "single_weight_int8_mib": cache["single_weight_bytes"][
                "int8"
            ]
            / 1024**2,
            "single_weight_b1_mib": cache["single_weight_bytes"]["b1"]
            / 1024**2,
            "eight_weights_int8_mib": cache["eight_weight_bank_bytes"][
                "int8"
            ]
            / 1024**2,
            "eight_weights_b1_mib": cache["eight_weight_bank_bytes"][
                "b1"
            ]
            / 1024**2,
            "l2_mib": A100_L2_MIB,
            "hot_speedup": cache["hot_speedup"],
            "rotating_speedup": cache["rotating8_speedup"],
            "hot_energy_reduction": cache_energy[
                "int8_hot_static_weight"
            ]["idle_adjusted_gpu_energy_J_per_call"]
            / cache_energy["b1_hot_static_weight"][
                "idle_adjusted_gpu_energy_J_per_call"
            ],
            "rotating_energy_reduction": cache_energy[
                "int8_rotating8_weights"
            ]["idle_adjusted_gpu_energy_J_per_call"]
            / cache_energy["b1_rotating8_weights"][
                "idle_adjusted_gpu_energy_J_per_call"
            ],
        },
        "roofline": {
            "analysis_bandwidth_GB_per_s": A100_80GB_HBM_GB_S,
            "raw_json_bandwidth_warning": (
                "Raw JSON retained 1555 GB/s for the requested 40GB part; "
                "actual hardware was 80GB, so derived rooflines use 2039."
            ),
            "int8_peak_TOPS": A100_INT8_TOPS,
            "b1_peak_TOPS": A100_B1_TOPS,
            "peak_ratio": A100_B1_TOPS / A100_INT8_TOPS,
            "square_int8_ridge_n": int8_square_ridge_n,
            "square_b1_ridge_n": b1_square_ridge_n,
            "both_memory_bound_speedup_ceiling": 24.0 / 17.0,
            "square_roofline_speedup_at_8192": min(
                A100_B1_TOPS / A100_INT8_TOPS,
                n
                / (
                    A100_INT8_TOPS
                    * 1e12
                    * 4.25
                    / (2 * A100_80GB_HBM_GB_S * 1e9)
                ),
            ),
        },
        "eightk": {
            "n": N,
            "pair_contributions": N**3,
            "two_op_count": 2 * N**3,
            "ideal_bmma_warp_instruction_count": N**3 // (16 * 8 * 256),
            "int8_minimum_bytes": 6 * N**2,
            "b1_minimum_bytes": int(4.25 * N**2),
            "grid_score": GRID_SCORE,
            "grid_energy_J_at_1fJ": GRID_SCORE * 1e-15,
        },
        "physical_calibration": physical_calibration(),
    }
    return result


def write_csv(result: dict) -> None:
    fields = [
        "family",
        "m",
        "n",
        "k",
        "int8_ms",
        "b1_prepacked_ms",
        "b1_pack_a_ms",
        "b1_pack_both_ms",
        "speedup_prepacked",
        "speedup_pack_a",
        "speedup_pack_both",
        "int8_tops",
        "b1_tops",
    ]
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for family in ("squares", "k_sweep", "batch"):
            writer.writerows(result[family])


if __name__ == "__main__":
    derived = build()
    DERIVED_PATH.write_text(json.dumps(derived, indent=2) + "\n")
    write_csv(derived)
    print(f"wrote {DERIVED_PATH.name} and {CSV_PATH.name}")
