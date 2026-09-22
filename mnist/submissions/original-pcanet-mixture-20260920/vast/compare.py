"""Compare three saved K=100 A100 runs without GPU or cloud access.

Usage: python vast/compare.py --raw /path/to/raw --output evidence/vast-comparison.json
The Modal, UK and Slovenia evidence directories default to evidence/rerun,
evidence/vast-uk and evidence/vast-slovenia. Missing results exit 2; failed
checks exit 1. Output files must be new. Per-host verification reuses the
standalone offline checker; cross-host checks and statistics are computed here.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from verify_results import read_json, verify  # noqa: E402


SOURCE_FILES = ("model.py", "cuda_kernel.py", "data.py", "validate.py")
EXPECTED_CONFIG = {
    "L1": 8, "L2": 5, "blocks": 9, "block_size": 14, "stride": 7,
    "features": 2304, "mixture_components_per_class": 8, "rounds": 4,
    "repeats": 60, "idle_seconds": 10, "tf32_matmul": True, "tf32_cudnn": True,
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def relative_difference(a, b):
    return 100 * (a - b) / b if b else None


def differences(first, second):
    return {key: {"baseline": first.get(key), "compared": second.get(key)}
            for key in sorted(set(first) | set(second)) if first.get(key) != second.get(key)}


def compare(directories, raw):
    missing = [str(path / name) for path in directories.values()
               for name in ("results.json", "execution.json", "predictions-k100.npy",
                            "reference-predictions-k100.npy") if not (path / name).is_file()]
    if missing:
        return {"status": "pending", "missing_evidence": missing}, 2

    errors, notes, docs, predictions, references = [], [], {}, {}, {}
    result = {
        "status": "complete", "configuration": "PCANet nine-block, k=8, K=100",
        "audit_kind": "Offline verification and cross-host comparison; no model execution or cloud calls",
        "independent_accuracy_checked_against_raw_data": raw is not None,
        "hosts": {}, "pairwise": {}, "errors": errors, "notes": notes,
    }
    baseline_protocol = read_json(PACKAGE / "protocol.json")
    vast_protocol = read_json(PACKAGE / "vast/protocol.json")
    package_sources = {name: sha256((PACKAGE / name).read_bytes()) for name in SOURCE_FILES}
    for name, digest in package_sources.items():
        if baseline_protocol["source_sha256"].get(name) != digest:
            errors.append("Package source differs from original frozen protocol: " + name)
        if vast_protocol["source_sha256"].get(name) != digest:
            errors.append("Package source differs from frozen Vast protocol: " + name)

    for label, directory in directories.items():
        doc = read_json(directory / "results.json")
        docs[label] = doc
        verification = verify(directory, raw)
        if not verification["passed"]:
            errors.extend(label + ": " + e for e in verification["errors"])
        if any(doc["config"].get(key) != value for key, value in EXPECTED_CONFIG.items()):
            errors.append(label + ": model or measurement configuration differs from frozen protocol")
        expected_dimensions = [100, 80] if label == "modal" else [100]
        if doc["config"].get("dimensions") != expected_dimensions:
            errors.append(label + ": unexpected measured dimensions")
        for name, digest in package_sources.items():
            if doc["source_sha256"].get(name) != digest:
                errors.append(label + ": executed source differs from package: " + name)

        predictions[label] = np.load(directory / "predictions-k100.npy", allow_pickle=False)
        references[label] = np.load(directory / "reference-predictions-k100.npy", allow_pickle=False)
        run = doc["runs"]["100"]
        medians = {key: statistics.median(row[key] for row in run["rounds"])
                   for key in ("task_ms", "gross_j", "adjusted_j_counter", "adjusted_j_sampled")}
        host = {
            "evidence_directory": str(directory),
            "results_json_sha256": sha256((directory / "results.json").read_bytes()),
            "started_utc": doc["started_utc"], "hardware": doc["hardware"],
            "software": doc["software"], "config": doc["config"], "scope": doc["scope"],
            "source_sha256": doc["source_sha256"],
            "data": doc["data"], "validation": run["validation"],
            "prediction_array_sha256": sha256(predictions[label].tobytes()),
            "reference_prediction_array_sha256": sha256(references[label].tobytes()),
            "recomputed_medians": medians,
            "median_energy_above_idle_mj": medians["adjusted_j_counter"] * 1000,
            "counter_minus_sampled_median_pct_of_counter":
                100 * (medians["adjusted_j_counter"] - medians["adjusted_j_sampled"])
                / medians["adjusted_j_counter"] if medians["adjusted_j_counter"] else None,
            "round_ranges": {key: {"min": min(row[key] for row in run["rounds"]),
                                   "max": max(row[key] for row in run["rounds"])}
                             for key in ("task_ms", "adjusted_j_counter", "adjusted_j_sampled")},
            "round_counter_minus_sampled_pct_of_counter": [
                100 * (row["adjusted_j_counter"] - row["adjusted_j_sampled"]) / row["adjusted_j_counter"]
                if row["adjusted_j_counter"] else None for row in run["rounds"]],
            "sensor_check": doc["sensor_check"], "idle_only_control": doc["idle_only_control"],
            "feature_validation": doc["feature_validation"],
            "peak_allocated_bytes": run["peak_allocated_bytes"],
            "peak_reserved_bytes": run["peak_reserved_bytes"],
            "offline_verification": verification,
        }
        if label != "modal":
            host["provisioning"] = vast_protocol["provisioning"][label]
            execution = read_json(directory / "execution.json")
            if execution.get("source_unchanged") is not True:
                errors.append(label + ": runner did not confirm unchanged frozen sources")
            preflight_path = directory / "preflight.json"
            if not preflight_path.is_file():
                errors.append(label + ": missing preflight identity evidence")
            else:
                preflight = read_json(preflight_path)
                host["preflight_sha256"] = sha256(preflight_path.read_bytes())
                if preflight["gpu"]["uuid"] != doc["hardware"]["uuid"]:
                    errors.append(label + ": preflight and measured GPU UUID differ")
        result["hosts"][label] = host

    modal = docs["modal"]
    uuids = [doc["hardware"]["uuid"] for doc in docs.values()]
    result["all_three_gpu_uuids_distinct"] = len(set(uuids)) == len(uuids)
    if not result["all_three_gpu_uuids_distinct"]:
        errors.append("Three distinct physical GPU UUIDs were not established")
    for label, doc in docs.items():
        host = result["hosts"][label]
        host["data_manifest_matches_modal"] = doc["data"] == modal["data"]
        host["scope_matches_modal"] = doc["scope"] == modal["scope"]
        host["software_differences_vs_modal"] = differences(modal["software"], doc["software"])
        host["all_software_versions_match_modal"] = not host["software_differences_vs_modal"]
        host["hardware_differences_vs_modal"] = differences(modal["hardware"], doc["hardware"])
        if not host["data_manifest_matches_modal"] or not host["scope_matches_modal"]:
            errors.append(label + ": data manifest or measured scope differs from Modal")
        if host["software_differences_vs_modal"]:
            notes.append(label + ": software versions differ; see software_differences_vs_modal")

    for left, right in itertools.combinations(directories, 2):
        mismatch = np.flatnonzero(predictions[left] != predictions[right])
        reference_mismatch = np.flatnonzero(references[left] != references[right])
        a = result["hosts"][left]["recomputed_medians"]
        b = result["hosts"][right]["recomputed_medians"]
        result["pairwise"][left + "__" + right] = {
            "left": left, "right": right,
            "prediction_matches": 10000 - len(mismatch),
            "prediction_differences": len(mismatch),
            "prediction_differences_by_test_index": [
                {"index": int(i), "left_prediction": int(predictions[left][i]),
                 "right_prediction": int(predictions[right][i])} for i in mismatch],
            "reference_prediction_differences": len(reference_mismatch),
            "reference_mismatch_test_indices": reference_mismatch.tolist(),
            "right_minus_left_counter_energy_pct_of_left":
                relative_difference(b["adjusted_j_counter"], a["adjusted_j_counter"]),
            "right_minus_left_runtime_pct_of_left": relative_difference(b["task_ms"], a["task_ms"]),
        }
    result["all_k100_predictions_identical"] = all(
        pair["prediction_differences"] == 0 for pair in result["pairwise"].values())
    if not result["all_k100_predictions_identical"]:
        notes.append("K100 predictions differ across hosts; per-index differences are retained.")
    notes.extend([
        "Each host estimate is the median of four windows, each fitting and predicting afresh 60 times; no energy estimates are pooled across hosts.",
        "Reported energy is GPU-board energy above the mean paired-idle baseline, not whole-system or wall-plug energy. Inputs are already normalized and device-resident; loading, transfers, compilation, warmup and test-label scoring are excluded.",
        "Power limits and drivers are recorded per board and may differ; energy differences are observations of these environments, not a controlled causal estimate of provider or power-limit effects.",
        "Counter and sampled-power measurements share NVML sensors. Round ranges are descriptive and are not confidence intervals.",
        "Source/data agreement and prediction comparison are independently checkable from the archive. Feature, repeat-run and ownership observations retained only as scalars cannot be replayed by this offline comparison.",
    ])
    result["passed"] = not errors
    return result, 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modal", type=Path, default=PACKAGE / "evidence/rerun")
    parser.add_argument("--uk", type=Path, default=PACKAGE / "evidence/vast-uk")
    parser.add_argument("--slovenia", type=Path, default=PACKAGE / "evidence/vast-slovenia")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Refusing to overwrite existing output: " + str(args.output))
    result, status = compare({"modal": args.modal, "uk": args.uk, "slovenia": args.slovenia}, args.raw)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
