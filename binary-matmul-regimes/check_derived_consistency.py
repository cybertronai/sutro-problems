#!/usr/bin/env python3
"""Consistency check for the derived tables and the published report.

Verifies that `derived.json` and `results.csv` are byte-reproducible from
`build_report_data.py`, that no timing family present in the raw JSON is
missing from the derived tables, that every derived number traces back to the
raw artifact, and that `index.html` cites nothing those artifacts do not carry.

Run with no arguments:

    python3 check_derived_consistency.py
"""

from __future__ import annotations

import ast
import copy
import csv
import importlib.util
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW_PATH = HERE / "a100_b1_vs_int8_results.json"
DERIVED_PATH = HERE / "derived.json"
CSV_PATH = HERE / "results.csv"
HTML_PATH = HERE / "index.html"
BUILDER = HERE / "build_report_data.py"
HARNESS = HERE / "modal_b1_int8_sweep.py"

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS  ' if condition else 'FAIL  '}{name}" + (f"  {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def rebuilt_bytes() -> tuple[bytes, bytes]:
    """Run the builder in a scratch directory and return its two outputs."""
    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch)
        shutil.copy(BUILDER, target)
        shutil.copy(RAW_PATH, target)
        subprocess.run(
            [sys.executable, "build_report_data.py"],
            cwd=target,
            check=True,
            capture_output=True,
        )
        return (
            (target / "derived.json").read_bytes(),
            (target / "results.csv").read_bytes(),
        )


def harness_call_semantics() -> dict:
    """Run the real rotating loops and timing normalization with harmless mocks."""
    names = {"int8_rotating8", "b1_rotating8", "calibrated_timing"}
    nodes = [
        node for node in ast.walk(ast.parse(HARNESS.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(nodes) != len(names):
        raise ValueError("Expected rotating runners and timing normalization in harness")
    seen = {"int8": [], "b1": []}

    class MockTorch:
        def _int_mm(self, a, b, out=None):
            seen["int8"].append(b)

    def launch(function, a, b, c, m, n, k):
        seen["b1"].append(b)

    def event_duration(runner, repetitions):
        before = len(seen["int8"])
        runner(repetitions)
        if len(seen["int8"]) - before != repetitions:
            raise ValueError("A timing repetition must launch exactly one GEMM")
        return 2.0 * repetitions

    namespace = {
        "torch": MockTorch(), "launch": launch,
        "a8": None, "c8": None, "a1": None, "c1": None,
        "cache_function": None, "b8_views": list(range(8)),
        "b1_banks": list(range(8)), "m": 64, "n": 8192, "k": 8192,
        "event_duration": event_duration, "timing_target_s": 0.08,
        "math": math, "statistics": statistics,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HARNESS), "exec"), namespace)
    namespace["int8_rotating8"](16)
    namespace["b1_rotating8"](16)
    selected = {label: values.copy() for label, values in seen.items()}
    timing = namespace["calibrated_timing"](namespace["int8_rotating8"])
    return {"selected_weights": selected, "seconds_per_call": timing["seconds_per_call_median"]}


def synthetic_builder_checks(raw: dict) -> dict:
    """Exercise new-family coverage and the supported matching 40GB hardware."""
    spec = importlib.util.spec_from_file_location("report_builder", BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    augmented = copy.deepcopy(raw)
    extra = copy.deepcopy(raw["timing_results"][0])
    extra["family"] = "synthetic_future_family"
    augmented["timing_results"].append(extra)
    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch)
        builder.RAW_PATH = target / "raw.json"
        builder.CSV_PATH = target / "results.csv"
        builder.RAW_PATH.write_text(json.dumps(augmented))
        result = builder.build()
        builder.write_csv(result)
        with builder.CSV_PATH.open() as handle:
            rows = list(csv.DictReader(handle))
    table_key = result["timing_table_keys"][extra["family"]]
    forty_gb = copy.deepcopy(raw)
    forty_gb["hardware"]["gpu_memory_bytes"] = 42_405_855_232
    bandwidth = builder.canonical_hbm_bandwidth_GB_per_s(
        forty_gb["hardware"]["gpu_memory_bytes"]
    )
    audit = builder.bandwidth_audit(forty_gb, bandwidth)
    return {
        "new_family_present": result[table_key][0]["family"] == extra["family"]
        and any(row["family"] == extra["family"] for row in rows),
        "all_csv_rows_present": len(rows) == len(augmented["timing_results"]),
        "matching_40gb_bandwidth": bandwidth == 1555.0
        and audit["raw_json_bandwidth_matches_hardware"] is True
        and audit["raw_json_bandwidth_warning"] is None,
    }


def main() -> int:
    raw = json.loads(RAW_PATH.read_text())
    derived = json.loads(DERIVED_PATH.read_text())
    rows = list(csv.DictReader(CSV_PATH.open()))
    html = HTML_PATH.read_text(encoding="utf-8")

    # 1. The derived artifacts are a pure function of the raw artifact.
    rebuilt_derived, rebuilt_csv = rebuilt_bytes()
    check(
        "derived.json reproducible from build_report_data.py",
        DERIVED_PATH.read_bytes() == rebuilt_derived,
    )
    check(
        "results.csv reproducible from build_report_data.py",
        CSV_PATH.read_bytes() == rebuilt_csv,
    )

    # 2. No timing family is silently dropped.
    raw_shapes = {
        (r["family"], r["m"], r["n"], r["k"]) for r in raw["timing_results"]
    }
    csv_shapes = {
        (r["family"], int(r["m"]), int(r["n"]), int(r["k"])) for r in rows
    }
    check(
        "results.csv covers every raw timing shape",
        raw_shapes == csv_shapes and len(rows) == len(raw["timing_results"]),
        f"{len(raw_shapes)} raw shapes vs {len(csv_shapes)} csv rows",
    )
    check(
        "derived families match raw families",
        set(derived["timing_families"])
        == {r["family"] for r in raw["timing_results"]},
        ", ".join(derived["timing_families"]),
    )
    check(
        "output_bound reaches derived.json and results.csv",
        len(derived.get("output_bound", [])) == 1
        and ("output_bound", 8192, 8192, 256) in csv_shapes,
    )

    # 3. Every derived number traces back to the raw artifact.
    by_key = {(r["family"], r["m"], r["n"], r["k"]): r for r in raw["timing_results"]}
    mismatched: list[str] = []
    for row in rows:
        source = by_key[(row["family"], int(row["m"]), int(row["n"]), int(row["k"]))]
        pairs = (
            ("int8_ms", source["timings"]["int8"]["seconds_per_call_median"] * 1e3),
            ("b1_prepacked_ms", source["timings"]["b1_prepacked"]["seconds_per_call_median"] * 1e3),
            ("b1_pack_a_ms", source["timings"]["b1_including_dynamic_a_pack"]["seconds_per_call_median"] * 1e3),
            ("b1_pack_both_ms", source["timings"]["b1_including_both_packs"]["seconds_per_call_median"] * 1e3),
            ("speedup_prepacked", source["speedup_b1_prepacked_over_int8"]),
            ("speedup_pack_a", source["speedup_b1_pack_a_over_int8"]),
            ("speedup_pack_both", source["speedup_b1_pack_both_over_int8"]),
            ("int8_tops", source["effective_int8_TOPS"]),
            ("b1_tops", source["effective_b1_TOPS"]),
        )
        for column, reference in pairs:
            if abs(float(row[column]) - reference) > 1e-9:
                mismatched.append(f"{row['family']} {row['m']} {column}")
    check("every results.csv value traces to raw JSON", not mismatched, str(mismatched[:3]))

    # 4. The roofline bandwidth follows the hardware that actually ran.
    memory_bytes = raw["hardware"]["gpu_memory_bytes"]
    expected_bandwidth = 2039.0 if memory_bytes >= 60 * 1024**3 else 1555.0
    roofline = derived["roofline"]
    check(
        "analysis bandwidth selected from hardware.gpu_memory_bytes",
        roofline["analysis_bandwidth_GB_per_s"] == expected_bandwidth,
        f"gpu_memory_bytes={memory_bytes} -> {expected_bandwidth} GB/s",
    )
    check(
        "stale raw constant recorded as a mismatch, not copied",
        roofline["raw_json_bandwidth_GB_per_s"] == raw["theory_constants"]["a100_hbm_bandwidth_GB_per_s"]
        and roofline["raw_json_bandwidth_matches_hardware"] is False,
        f"raw={roofline['raw_json_bandwidth_GB_per_s']} GB/s",
    )
    check(
        "every per-shape floor recomputed at the derived bandwidth",
        all(
            abs(row["minimum_bytes_int8_s32"] / row["int8_hbm_floor_s"] - expected_bandwidth * 1e9) < 1e3
            and abs(row["minimum_bytes_b1_s32"] / row["b1_hbm_floor_s"] - expected_bandwidth * 1e9) < 1e3
            for row in derived["per_shape_roofline"]
        ),
        f"{len(derived['per_shape_roofline'])} shapes",
    )

    # 5. Theoretical ceiling is not presented as a measurement.
    square_8192 = next(row for row in derived["squares"] if row["n"] == 8192)
    check(
        "theoretical ceiling at 8192 is the dense-peak ratio",
        roofline["square_roofline_speedup_ceiling_at_8192"] == roofline["peak_ratio"] == 8.0,
    )
    check(
        "measured 8192 speedup is a separate key matching the square row",
        abs(roofline["square_measured_speedup_at_8192"] - square_8192["speedup_prepacked"]) < 1e-12,
        f"{roofline['square_measured_speedup_at_8192']:.4f}x",
    )
    check(
        "ambiguous legacy key no longer present",
        "square_roofline_speedup_at_8192" not in roofline,
    )

    # 6. index.html cites nothing the artifacts do not carry.
    output_bound = derived["output_bound"][0]
    check(
        "index.html output-heavy figure matches derived output_bound",
        f"{output_bound['speedup_prepacked']:.2f}" == "1.25" and "1.25× faster" in html,
    )
    check(
        "index.html separates the 8x ceiling from the 6.74x measurement",
        "theoretical ceiling" in html and "6.74×" in html,
    )
    check(
        "index.html does not reuse the removed key name",
        "square_roofline_speedup_at_8192" not in html,
    )
    check(
        "cross-report figures are quoted, not replaced",
        "2.944 ms" in html and "0.37083 ms" in html and "A100-SXM4-40GB" in html,
    )
    check(
        "rotating-bank capacity distinguished from per-call traffic",
        "one selected weight matrix" in html and "does not establish" in html,
    )

    # 7. Use the actual harness's call semantics, not bank capacity per call.
    semantics = harness_call_semantics()
    check(
        "each rotating harness repetition launches one selected matrix",
        all(values == list(range(8)) * 2 for values in semantics["selected_weights"].values()),
    )
    check(
        "harness timing normalizes by individual GEMM calls",
        semantics["seconds_per_call"] == 2.0,
    )
    consistency = derived["cache"]["hbm_floor_consistency"]
    cache = raw["weight_residency_result"]
    m, n, k = cache["m"], cache["n"], cache["k"]
    for label, value_bits in (("int8", 8), ("b1", 1)):
        rotating = next(c for c in consistency["cases"] if c["case"] == f"{label}_rotating8_weights")
        bytes_per_call = (m * k + n * k) * value_bits // 8 + 4 * m * n
        expected_s = bytes_per_call / (expected_bandwidth * 1e9)
        check(
            f"{label} rotating timing uses one matrix per call and an eight-matrix working set",
            rotating["weight_matrices_selected_per_call"] == 1
            and rotating["weight_bank_working_set_bytes"] == cache["eight_weight_bank_bytes"][label]
            and rotating["one_pass_transfer_bytes_per_call"] == bytes_per_call
            and math.isclose(rotating["hbm_floor_s_at_analysis_bandwidth"], expected_s, rel_tol=1e-12)
            and rotating["consistent_with_streaming_from_hbm"] is True,
            f"{rotating['measured_s'] * 1e3:.4f} ms measured vs {expected_s * 1e3:.4f} ms assumed one-pass transfer",
        )
    hot = next(c for c in consistency["cases"] if c["case"] == "int8_hot_static_weight")
    check(
        "single-weight INT8 timing is compatible with the assumed one-pass transfer",
        hot["consistent_with_streaming_from_hbm"] is True,
        f"{hot['measured_over_hbm_floor']:.4f}x of its floor",
    )
    synthetic = synthetic_builder_checks(raw)
    check("a new raw timing family reaches its derived table and CSV", synthetic["new_family_present"] and synthetic["all_csv_rows_present"])
    check("matching 40GB hardware has its own bandwidth and no mismatch warning", synthetic["matching_40gb_bandwidth"])

    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
