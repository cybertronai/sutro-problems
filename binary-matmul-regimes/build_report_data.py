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
A100_DIE_MM2 = 826.0
A100_L2_MIB = 40.0

# Published HBM bandwidth per A100 SXM4 memory size.  Modal honoured the
# A100-40GB request with an 80GB SXM4 (HBM2e), so the roofline constant has to
# follow the card that actually ran rather than the card that was requested.
# The threshold is the 80GB part's capacity; anything below it is the 40GB part.
A100_80GB_MIN_BYTES = 60 * 1024**3
A100_80GB_HBM_GB_S = 2_039.0
A100_40GB_HBM_GB_S = 1_555.0


def canonical_hbm_bandwidth_GB_per_s(gpu_memory_bytes: int) -> float:
    """Published HBM bandwidth for the card that actually ran.

    Derived from the reported device capacity instead of from the requested
    Modal SKU, so the raw JSON's stale 1555 GB/s cannot propagate.
    """
    if gpu_memory_bytes >= A100_80GB_MIN_BYTES:
        return A100_80GB_HBM_GB_S
    return A100_40GB_HBM_GB_S

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


def timing_families(raw: dict) -> list[str]:
    """Every family present in the raw timing sweep, in raw-JSON order.

    Derived tables are driven by this list rather than a hand-written tuple so
    that a family present in the raw artifact can never be silently dropped
    from derived.json or results.csv.
    """
    families: list[str] = []
    for record in raw["timing_results"]:
        if record["family"] not in families:
            families.append(record["family"])
    return families


def bandwidth_audit(raw: dict, bandwidth_GB_per_s: float) -> dict:
    """State which HBM constant was used, and which the raw JSON carried."""
    raw_bandwidth = raw["theory_constants"]["a100_hbm_bandwidth_GB_per_s"]
    return {
        "analysis_bandwidth_GB_per_s": bandwidth_GB_per_s,
        "analysis_bandwidth_source": (
            "selected from hardware.gpu_memory_bytes by "
            "canonical_hbm_bandwidth_GB_per_s"
        ),
        "raw_json_bandwidth_GB_per_s": raw_bandwidth,
        "raw_json_bandwidth_matches_hardware": raw_bandwidth
        == bandwidth_GB_per_s,
        "raw_json_bandwidth_warning": (
            f"Raw JSON carries {raw_bandwidth} GB/s, the A100-40GB part, "
            "while hardware.gpu_memory_bytes identifies the 80GB SXM4; this "
            f"build therefore derives the roofline at {bandwidth_GB_per_s} GB/s "
            "and does not copy the raw constant."
        ),
    }


def per_shape_roofline(
    raw: dict, bandwidth_GB_per_s: float
) -> list[dict]:
    """Recompute every shape's HBM and roofline floors at the derived bandwidth.

    The raw JSON's per-shape ``theory`` block was produced with the stale
    1555 GB/s constant.  Byte counts and operation counts are bandwidth
    independent, so the floors are recomputed here from those same numbers
    rather than edited inside the raw artifact.
    """
    bandwidth = bandwidth_GB_per_s * 1e9
    rows = []
    for record in raw["timing_results"]:
        theory = record["theory"]
        m, n, k = record["m"], record["n"], record["k"]
        int8_bytes = m * k + k * n + 4 * m * n
        b1_bytes = (m * k + k * n) // 8 + 4 * m * n
        operations = 2 * m * n * k
        int8_compute_floor = operations / (A100_INT8_TOPS * 1e12)
        b1_compute_floor = operations / (A100_B1_TOPS * 1e12)
        int8_hbm_floor = int8_bytes / bandwidth
        b1_hbm_floor = b1_bytes / bandwidth
        rows.append(
            {
                "family": record["family"],
                "m": m,
                "n": n,
                "k": k,
                "minimum_bytes_int8_s32": int8_bytes,
                "minimum_bytes_b1_s32": b1_bytes,
                "int8_compute_floor_s": int8_compute_floor,
                "b1_compute_floor_s": b1_compute_floor,
                "int8_hbm_floor_s": int8_hbm_floor,
                "b1_hbm_floor_s": b1_hbm_floor,
                "int8_roofline_floor_s": max(int8_compute_floor, int8_hbm_floor),
                "b1_roofline_floor_s": max(b1_compute_floor, b1_hbm_floor),
                "measured_int8_s": record["timings"]["int8"][
                    "seconds_per_call_median"
                ],
                "measured_b1_prepacked_s": record["timings"][
                    "b1_prepacked"
                ]["seconds_per_call_median"],
                "int8_regime": (
                    "memory"
                    if int8_hbm_floor > int8_compute_floor
                    else "compute"
                ),
                "b1_regime": (
                    "memory" if b1_hbm_floor > b1_compute_floor else "compute"
                ),
            }
        )
    return rows


def cache_hbm_consistency(
    cache: dict, bandwidth_GB_per_s: float
) -> dict:
    """Compare each weight-residency call against its own HBM read floor.

    The rotating-eight case is reported as a bandwidth-bound comparison
    because neither bank fits in L2.  That claim is only supportable if the
    measured call time is at or above the time needed to stream that bank's
    bytes from HBM, so the floor is emitted next to the measurement instead of
    being left to the prose.
    """
    bandwidth = bandwidth_GB_per_s * 1e9
    m, n, k = cache["m"], cache["n"], cache["k"]
    output_bytes = 4 * m * n
    variants = {
        "hot": {
            "int8": (m * k + n * k + output_bytes),
            "b1": (m * k + n * k) // 8 + output_bytes,
        },
        "rotating8": {
            "int8": (m * k + 8 * n * k + output_bytes),
            "b1": ((m * k + 8 * n * k) // 8 + output_bytes),
        },
    }
    cases = []
    for regime, banks in variants.items():
        for label, bank_bytes in banks.items():
            key = f"{'int8' if label == 'int8' else 'b1'}_{regime}_static_weight" if regime == "hot" else (
                f"{'int8' if label == 'int8' else 'b1'}_rotating8_weights"
            )
            measured_s = cache["timings"][key]["seconds_per_call_median"]
            floor_s = bank_bytes / bandwidth
            cases.append(
                {
                    "case": key,
                    "min_bytes_read": bank_bytes,
                    "hbm_floor_s_at_analysis_bandwidth": floor_s,
                    "measured_s": measured_s,
                    "measured_over_hbm_floor": measured_s / floor_s,
                    "consistent_with_streaming_from_hbm": measured_s >= floor_s,
                }
            )
    return {
        "analysis_bandwidth_GB_per_s": bandwidth_GB_per_s,
        "cases": cases,
        "rotating8_int8_consistent_with_hbm_bound": next(
            item["consistent_with_streaming_from_hbm"]
            for item in cases
            if item["case"] == "int8_rotating8_weights"
        ),
        "note": (
            "consistency flag is a floor check, not a roofline attribution: a "
            "measured call faster than its own HBM byte floor cannot be "
            "explained by HBM streaming alone."
        ),
    }


def build() -> dict:
    raw = json.loads(RAW_PATH.read_text())
    bandwidth_GB_per_s = canonical_hbm_bandwidth_GB_per_s(
        raw["hardware"]["gpu_memory_bytes"]
    )
    families = timing_families(raw)
    timing_tables = {family: timing_rows(raw, family) for family in families}
    energy = energy_rows(raw)
    cache = raw["weight_residency_result"]
    cache_energy = {
        row["label"]: row
        for row in raw["energy_results"]
        if row["regime"] == "weight_residency"
    }

    n = float(N)
    int8_square_ridge_n = 3 * A100_INT8_TOPS * 1e12 / (bandwidth_GB_per_s * 1e9)
    b1_square_ridge_n = 2.125 * A100_B1_TOPS * 1e12 / (
        bandwidth_GB_per_s * 1e9
    )
    # Square 8192 has exactly one raw timing record; find it rather than
    # re-deriving the number from constants.
    square_8192 = next(
        row
        for row in timing_tables.get("square", [])
        if row["n"] == N
    )
    measured_8192_speedup = square_8192["speedup_prepacked"]
    peak_ratio = A100_B1_TOPS / A100_INT8_TOPS
    # Theoretical B1-vs-INT8 speedup ceiling at square 8192: the arithmetic
    # intensity ratio, capped by the dense-peak ratio.  This is a ceiling from
    # constants only and is NOT a measurement.
    roofline_speedup_ceiling_at_8192 = min(
        peak_ratio,
        n / (A100_INT8_TOPS * 1e12 * 4.25 / (2 * bandwidth_GB_per_s * 1e9)),
    )
    result = {
        "source": RAW_PATH.name,
        "hardware": raw["hardware"],
        "semantics": raw["semantics"],
        "timing_families": families,
        "squares": timing_tables.get("square", []),
        "k_sweep": timing_tables.get("k_sweep", []),
        "batch": timing_tables.get("batch", []),
        "output_bound": timing_tables.get("output_bound", []),
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
            "hbm_floor_consistency": cache_hbm_consistency(
                cache, bandwidth_GB_per_s
            ),
        },
        "roofline": {
            **bandwidth_audit(raw, bandwidth_GB_per_s),
            "int8_peak_TOPS": A100_INT8_TOPS,
            "b1_peak_TOPS": A100_B1_TOPS,
            "peak_ratio": peak_ratio,
            "square_int8_ridge_n": int8_square_ridge_n,
            "square_b1_ridge_n": b1_square_ridge_n,
            "both_memory_bound_speedup_ceiling": 24.0 / 17.0,
            "square_roofline_speedup_ceiling_at_8192": (
                roofline_speedup_ceiling_at_8192
            ),
            "square_measured_speedup_at_8192": measured_8192_speedup,
            "roofline_vs_measurement_note": (
                "square_roofline_speedup_ceiling_at_8192 is computed from the "
                "dense peak TOPS and the HBM bandwidth constant only, and is "
                "capped by the 8x peak ratio; it is a theoretical ceiling, not "
                "a measurement.  square_measured_speedup_at_8192 is the value "
                "observed on hardware for the same shape."
            ),
        },
        "per_shape_roofline": per_shape_roofline(raw, bandwidth_GB_per_s),
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
        # Write whatever families the raw sweep actually carries.
        for key in ("squares", "k_sweep", "batch", "output_bound"):
            writer.writerows(result.get(key, []))


if __name__ == "__main__":
    derived = build()
    DERIVED_PATH.write_text(json.dumps(derived, indent=2) + "\n")
    write_csv(derived)
    print(f"wrote {DERIVED_PATH.name} and {CSV_PATH.name}")
