"""CPU-only independent audit of workspace equivalence and retained counters.

No training and no test labels: predictions are compared byte-for-byte with
the already verified frozen qualification. Missing/failed profiles remain
explicitly unmeasured; allocation size never establishes cache residency.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
HASH_KEYS = ("input_hashes", "initial_parameter_sha256", "epoch_permutation_sha256",
             "final_parameter_sha256", "final_velocity_sha256", "state_vector_sha256",
             "predictions_sha256", "scores_sha256")
METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_sectors_op_read.sum",
           "lts__t_sectors_op_write.sum", "lts__t_sector_hit_rate.pct")
L2_REFERENCE = 40 * 1024**2


def read(path):
    path = Path(path)
    compressed = path.with_name(path.name+".gz")
    if compressed.exists():
        with gzip.open(compressed,"rt") as source:
            return json.load(source)
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ah(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def close(actual, expected, message):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
            math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
            f"{message}: retained {actual}, recomputed {expected}")


def pci(value):
    return tuple(int(part, 16) for part in re.split(r"[:.]", value.lower()))


def report_artifact(path):
    """Verify compressed report contents; retain both transport/content hashes."""
    compressed = path.with_name(path.name+".gz")
    require(path.exists() or compressed.exists(), "Evidence artifact not retained")
    uncompressed_hash = sha(path) if path.exists() else None
    uncompressed_bytes = path.stat().st_size if path.exists() else None
    if compressed.exists():
        digest, size = hashlib.sha256(), 0
        with gzip.open(compressed,"rb") as source:
            for block in iter(lambda: source.read(1024*1024), b""):
                digest.update(block)
                size += len(block)
        require(size > 0, "Empty compressed artifact")
        if uncompressed_hash is not None:
            require(digest.hexdigest() == uncompressed_hash and size == uncompressed_bytes,
                    "Compressed artifact differs from the original")
        uncompressed_hash, uncompressed_bytes = digest.hexdigest(), size
    retained = compressed if compressed.exists() else path
    return {"ncu_report_path": retained.name, "ncu_report_sha256": sha(retained),
            "ncu_report_uncompressed_sha256": uncompressed_hash,
            "ncu_report_uncompressed_bytes": uncompressed_bytes,
            "ncu_report_compressed_sha256": sha(compressed) if compressed.exists() else None,
            "ncu_report_compressed_bytes": compressed.stat().st_size if compressed.exists() else None,
            "ncu_report_gzip_contents_verified": compressed.exists()}


def memory_audit(result):
    memory, inventory = result["memory"], result["buffer_inventory"]
    require(memory["allocated_before_prepare_bytes"] == memory["reserved_before_prepare_bytes"] == 0,
            "Memory measurement did not start in a fresh allocator")
    peak = memory["prepare_and_invoke_peak_allocated_bytes"]
    live = memory["allocated_after_invoke_bytes"]
    require(0 < live <= peak and memory["allocated_after_prepare_bytes"] <= peak,
            "Impossible CUDA allocator accounting")
    require(memory["reserved_after_invoke_bytes"] >= live and
            memory["prepare_and_invoke_peak_reserved_bytes"] >= peak, "Reserved memory accounting differs")
    require(memory["added_peak_over_entry_allocated_bytes"] == peak, "Added peak calculation differs")
    storages = inventory["unique_storages"]
    require(len({s["storage_id"] for s in storages}) == len(storages), "Duplicate storage IDs")
    require(len({(s["device"], s["address"]) for s in storages}) == len(storages), "Aliased storage was counted twice")
    by_id = {s["storage_id"]: s for s in storages}
    aliases = {s["storage_id"]: [] for s in storages}
    widths = {"torch.float32": 4, "torch.float64": 8, "torch.int64": 8,
              "torch.int32": 4, "torch.uint8": 1, "torch.bool": 1}
    for tensor in inventory["named_tensors"]:
        storage = by_id[tensor["storage_id"]]
        aliases[tensor["storage_id"]].append(tensor["name"])
        expected = math.prod(tensor["shape"]) * widths[tensor["dtype"]]
        require(tensor["logical_bytes"] == expected and expected <= storage["bytes"], "Tensor inventory size differs")
    for storage in storages:
        require(storage["aliases"] == aliases[storage["storage_id"]], "Storage alias inventory differs")
    named = sum(s["bytes"] for s in storages)
    require(named == inventory["unique_named_storage_bytes"] and named <= live,
            "Unique storage byte sum differs")
    require(inventory["allocator_allocated_bytes"] == live and
            inventory["allocated_bytes_without_named_storage_reference"] == live-named,
            "Unnamed allocation byte sum differs")
    require(not inventory["is_per_kernel_working_set"] and not inventory["cache_residency_established"],
            "Inventory makes unsupported residency claims")
    require(result["cache_reference"]["bytes"] == L2_REFERENCE, "Capacity reference changed")
    return {"peak_allocated_bytes": peak, "live_allocated_bytes": live,
            "peak_reserved_bytes": memory["prepare_and_invoke_peak_reserved_bytes"],
            "named_storage_bytes": named, "unnamed_live_allocated_bytes": live-named,
            "peak_allocated_mib": peak/1024**2, "nominal_l2_reference_bytes": L2_REFERENCE,
            "peak_allocated_below_nominal_l2_capacity": peak < L2_REFERENCE,
            "cache_residency_established": False, "scope": memory["scope"]}


def energy_audit(energy, metadata):
    require(energy is not None and len(energy["trials"]) == 3, "Missing three-trial energy evidence")
    hardware = energy["hardware"]
    require("A100" in hardware["gpu"] and hardware["nvml_handle_selected_by_cuda_pci_bus_id"] and
            pci(hardware["cuda_pci_bus_id"]) == pci(hardware["nvml_pci_bus_id"]), "CUDA and NVML device identity differs")
    require(energy["repeat_predictions_equal"] and energy["repeat_state_hashes_equal"], "Energy repeats differ")
    for side in ("before_metadata", "after_metadata"):
        current = energy[side]
        require(current["config"] == metadata["config"] and current["fresh_initialization_per_run"] and
                not current["checkpoint_loaded"], "Energy configuration/reset differs")
        for key in HASH_KEYS:
            require(current[key] == metadata[key], f"Energy state differs: {key}")
    rows = []
    for index, trial in enumerate(energy["trials"]):
        repeats = trial["invocations"]
        require(trial["trial"] == index and repeats == energy["repeats_per_trial"] > 0, "Energy trial/repetition mismatch")
        intervals = {}
        for name in ("idle_before", "active", "idle_after"):
            interval = trial[name]
            seconds = interval["end"]["time_s"] - interval["start"]["time_s"]
            joules = (interval["end"]["energy_mj"] - interval["start"]["energy_mj"]) / 1000
            require(seconds > 0 and joules >= 0, "Invalid cumulative energy counters")
            close(interval["duration_s"], seconds, "Counter duration")
            close(interval["energy_j"], joules, "Counter joules")
            close(interval["average_power_w"], joules/seconds, "Counter power")
            if name.startswith("idle"):
                require(seconds >= 2.9 and interval["settle_seconds"] >= 3, "Insufficient idle baseline")
            intervals[name] = (seconds, joules)
        require(trial["idle_before"]["end"]["time_s"] <= trial["active"]["start"]["time_s"] and
                trial["active"]["end"]["time_s"] <= trial["idle_after"]["start"]["time_s"], "Idle intervals do not bracket active work")
        idle = sum(intervals[k][1]/intervals[k][0] for k in ("idle_before", "idle_after"))/2
        seconds, joules = intervals["active"]
        derived = {"idle_adjusted_mj_per_task": (joules-idle*seconds)*1000/repeats,
                   "unadjusted_mj_per_task": joules*1000/repeats,
                   "cuda_ms_per_task": trial["cuda_ms_per_task"],
                   "wall_ms_per_task": seconds*1000/repeats}
        close(trial["paired_idle_w"], idle, "Paired idle power")
        for key, value in derived.items():
            close(trial[key], value, key)
        require(0.7 < derived["cuda_ms_per_task"]/derived["wall_ms_per_task"] < 1.3,
                "CUDA and wall timing disagree")
        rows.append(derived)
    summaries = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        expected = energy["summary"][key]
        require(len(expected["values"]) == 3, "Energy summary omitted a trial")
        for a,b in zip(expected["values"], values):
            close(a,b,"Energy trial value")
        close(expected["mean"], float(np.mean(values)), "Energy mean")
        close(expected["sample_sd"], float(np.std(values, ddof=1)), "Energy sample SD")
        summaries[key] = {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)), "values": values}
    return {"passed": True, "trials": 3, "invocations_per_trial": energy["repeats_per_trial"],
            "raw_counter_arithmetic_recomputed": True, "summary": summaries, "hardware": hardware,
            "scope": "Three paired-idle NVML trials for fresh complete tasks on draw zero"}


def parse_csv_metrics(output):
    """Accept the retained ncu raw CSV, rejecting missing or duplicate metrics."""
    lines = output.splitlines()
    all_rows = list(csv.reader(io.StringIO("\n".join(line for line in lines if line.startswith('"')))))
    metrics, attributes = {}, {}
    scales = {"byte": 1, "Kbyte": 1000, "Mbyte": 1000**2, "Gbyte": 1000**3,
              "sector": 1, "%": 1}
    # --page raw in ncu2025.1 produces one header, one units row, then ranges.
    wide_header = next((i for i,row in enumerate(all_rows) if METRICS[0] in row), None)
    if wide_header is not None:
        header = all_rows[wide_header]
        units = dict(zip(header, all_rows[wide_header+1]))
        ranges = [dict(zip(header,row)) for row in all_rows[wide_header+2:] if len(row) == len(header)]
        require(len(ranges) == 1 and ranges[0]["Kernel Name"] == "range", "Expected one complete profiler range")
        values = ranges[0]
        for name in METRICS:
            require(name in values, f"Missing metric {name}")
            unit = units[name]
            require(unit in scales, f"Unrecognized unit for {name}: {unit}")
            value = float(values[name].replace(",", "")) * scales[unit]
            require(math.isfinite(value) and value >= 0, f"Invalid metric {name}")
            metrics[name] = {"value": value, "raw_value": values[name], "raw_unit": unit,
                             "range_id": values["ID"]}
        for name in ("device__attribute_display_name", "device__attribute_l2_cache_size",
                     "device__attribute_total_memory", "device__attribute_memory_clock_rate",
                     "device__attribute_multiprocessor_count", "profiler__replayer_passes",
                     "profiler__replayer_passes_type_warmup", "gpu__time_duration.sum"):
            if name in values:
                attributes[name] = values[name]
        require(metrics[METRICS[-1]]["value"] <= 100, "L2 hit rate exceeds 100%")
        return metrics, attributes
    header = None
    for row in all_rows:
        if "Metric Name" in row and "Metric Value" in row:
            header = row
            continue
        if header is None or len(row) != len(header):
            continue
        item = dict(zip(header, row))
        name = item.get("Metric Name")
        if name not in METRICS:
            continue
        require(name not in metrics, f"Multiple range records for {name}; aggregate explicitly")
        value = float(item["Metric Value"].replace(",", ""))
        require(math.isfinite(value) and value >= 0, f"Unavailable or negative metric {name}")
        unit = item.get("Metric Unit", "")
        require(unit in scales, f"Unrecognized unit for {name}: {unit}")
        metrics[name] = {"value": value*scales[unit], "raw_value": item["Metric Value"],
                         "raw_unit": unit, "raw_row": item}
    require(set(metrics) == set(METRICS), f"Missing retained ncu metrics: {set(METRICS)-set(metrics)}")
    require(metrics[METRICS[-1]]["value"] <= 100, "L2 hit rate exceeds 100%")
    return metrics, attributes


def audit_record(stem, draw, workspace, action, scope, protocol, plan, entries):
    path = HERE/(stem+".json")
    if not path.exists() and not path.with_name(path.name+".gz").exists():
        return {"status": "pending", "workspace": workspace, "scope": scope}
    record = read(path)
    record_artifact = {key.replace("ncu_report", "result"): value
                       for key,value in report_artifact(path).items()}
    require(record["workspace"] == workspace and record["action"] == action and record["scope"] == scope,
            f"{stem}: requested run differs")
    capabilities = record.get("capabilities", {})
    compact_capabilities = {"ncu_version": capabilities.get("ncu_version")}
    if "metric_query" in capabilities:
        query = capabilities["metric_query"]
        compact_capabilities["metric_query"] = {"returncode": query["returncode"],
            "raw_query_sha256": hashlib.sha256((query["stdout"]+query["stderr"]).encode()).hexdigest(),
            "raw_query_retained_in": record_artifact["result_path"]}
    evidence = {**record_artifact,
                "command": record["command"], "capabilities": compact_capabilities,
                "returncode": record["returncode"], "workspace": workspace, "scope": scope}
    if record["returncode"] != 0:
        return {**evidence, "status": "failed", "blocker": (record["stdout"]+record["stderr"]).strip(),
                "dram_traffic_measured": False}
    result, entry = record["result"], entries[draw]
    base_result = read(BASE/entry["result_path"])
    require(sha(BASE/entry["result_path"]) == entry["result_sha256"], "Original qualification metadata changed")
    require(sha(BASE/entry["prediction_path"]) == entry["prediction_sha256"], "Original qualification predictions changed")
    require(result["source_sha256"]["fixture.py"] == plan["fixture_sha256"], f"{stem}: fixture source differs")
    for name in ("learner.py", "energy.py"):
        require(result["source_sha256"][name] == protocol["source_sha256"][name], f"{stem}: frozen source differs")
    require(result["config"] == result["metadata"]["config"] == protocol["configuration"], f"{stem}: recipe differs")
    require(record["input_sha256"] == result["metadata"]["input_hashes"] == base_result["input_sha256"], f"{stem}: inputs differ")
    require(result["cublas_workspace_config"] == workspace and result["environment_was_set_externally"], "Workspace env differs")
    require(result["test_labels_read"] is False and result["metadata"]["fresh_initialization_per_run"] and
            not result["metadata"]["checkpoint_loaded"], "Fresh-task isolation differs")
    require(set(result["same_process_reference_hashes_matched"]) == set(HASH_KEYS), "Repeated-state check omitted fields")
    mismatched_state_fields = [key for key in HASH_KEYS if result["metadata"][key] != base_result["metadata"][key]]
    fixed_fields = ("input_hashes", "initial_parameter_sha256", "epoch_permutation_sha256", "predictions_sha256")
    require(not set(mismatched_state_fields).intersection(fixed_fields),
            f"{stem}: inputs, seed initialization, epoch order, or predictions differ")
    if workspace == ":4096:8":
        require(not mismatched_state_fields, f"{stem}: unchanged-workspace baseline state differs")
    pred_path = HERE/(stem+"-result.predictions.npy")
    require(sha(pred_path) == result["predictions_file_sha256"], f"{stem}: prediction archive changed")
    predictions = np.load(pred_path, allow_pickle=False)
    original = np.load(BASE/entry["prediction_path"], allow_pickle=False)
    require(predictions.dtype == original.dtype == np.int64 and predictions.shape == original.shape == (10000,), "Invalid prediction arrays")
    require(predictions.tobytes() == original.tobytes() and ah(predictions) == result["metadata"]["predictions_sha256"],
            f"{stem}: predictions are not bit-identical")
    audited = {**evidence, "status": "passed", "draw": draw, "predictions_bit_equal": True,
               "all_state_hashes_bit_equal": not mismatched_state_fields,
               "state_fields_differing_from_original": mismatched_state_fields,
               "state_fields_checked": list(HASH_KEYS),
               "prediction_path": pred_path.name, "prediction_sha256": sha(pred_path),
               "memory": memory_audit(result), "software": result["software"]}
    if result["energy"] is not None:
        audited["energy"] = energy_audit(result["energy"], result["metadata"])
    if action != "profile":
        return audited
    command = record["command"]
    def option(name):
        return command[command.index(name)+1] if name in command else None
    require(option("--replay-mode") == "app-range" and option("--cache-control") == "none" and
            option("--clock-control") == "none", "Profiler alters intended replay/cache/clock policy")
    require("--graph-profiling" not in command and "--profile-from-start" not in command,
            "Incompatible range profiling flags")
    profile = result["profile"]
    require(profile["scope"] == scope and profile["warmup_complete_tasks"] == 2 and
            profile["epochs_per_sequence"] == 2 and profile["synchronized_before_stop"] and
            profile["epoch_order_copies_inside_range"] and not profile["output_serialization_inside_range"] and
            not profile["cpu_validation_inside_range"], "Profiler range differs from declared work")
    sequences = profile["fresh_training_sequences_in_range"]
    require(sequences == (16 if scope == "task" else 1), "Wrong number of profiled sequences")
    require(profile["fresh_tasks_in_range"] == (16 if scope == "task" else 0) and
            profile["reset_inside_range"] == (scope == "task") and
            profile["prediction_inside_range"] == (scope == "task"), "Profiler includes wrong reset/prediction scope")
    output = record["stdout"]+"\n"+record["stderr"]
    try:
        metrics, attributes = parse_csv_metrics(output)
    except (AssertionError, ValueError) as error:
        return {**audited, "status": "unmeasured", "dram_traffic_measured": False, "blocker": str(error)}
    report_path = HERE/(stem+"-profile.ncu-rep")
    artifact = report_artifact(report_path)
    pass_counts = [int(v) for v in re.findall(r"\b(\d+)\s+pass(?:es)?\b", output)]
    if "profiler__replayer_passes" in attributes:
        pass_counts = [int(attributes["profiler__replayer_passes"].replace(",", ""))]
    require(bool(pass_counts) and all(p >= 1 for p in pass_counts), "Profiler pass count was not retained")
    require(attributes.get("device__attribute_display_name") == result["gpu"], "Ncu and fixture GPU identity disagree")
    require(int(attributes["device__attribute_l2_cache_size"].replace(",", "")) == L2_REFERENCE, "Measured GPU L2 capacity differs")
    query = record.get("capabilities", {}).get("metric_query", {})
    query_text = query.get("stdout", "")
    require(query.get("returncode") == 0 and all(name in query_text for name in METRICS), "Metric query did not confirm requested counters")
    values = {name: row["value"] for name,row in metrics.items()}
    return {**audited, "dram_traffic_measured": True, "profile": profile,
            **artifact,
            "metric_rows": metrics, "pass_counts": pass_counts,
            "profile_device_attributes": attributes,
            "pass_count_retained": bool(pass_counts), "range_training_sequences": sequences,
            "dram_read_bytes_per_sequence": values[METRICS[0]]/sequences,
            "dram_write_bytes_per_sequence": values[METRICS[1]]/sequences,
            "dram_total_bytes_per_sequence": (values[METRICS[0]]+values[METRICS[1]])/sequences,
            "l2_read_request_bytes_per_sequence": 32*values[METRICS[2]]/sequences,
            "l2_write_request_bytes_per_sequence": 32*values[METRICS[3]]/sequences,
            "l2_sector_hit_rate_percent": values[METRICS[4]],
            "counter_unit": "per complete fresh reset/train/predict task" if scope == "task" else
                            "per original two-epoch training sequence, reset and prediction outside range"}


def analyze():
    protocol, frozen, plan = read(BASE/"protocol.json"), read(BASE/"prediction-manifest.json"), read(HERE/"plan.json")
    require(plan["base_protocol_sha256"] == frozen["protocol_sha256"] == sha(BASE/"protocol.json"), "Base protocol changed")
    for name,digest in protocol["source_sha256"].items():
        require(sha(BASE/name) == digest, f"Frozen base source changed: {name}")
    require(sha(HERE/"fixture.py") == plan["fixture_sha256"], "Fixture changed after plan")
    runner_candidates = [HERE/"run.py", HERE/"development/run-initial.py"]
    require(any(p.exists() and sha(p) == plan["runner_sha256"] for p in runner_candidates), "Original runner source not retained")
    if (HERE/"profile-plan.json").exists():
        profile_plan = read(HERE/"profile-plan.json")
        require(profile_plan["runner_sha256"] == sha(HERE/"run.py"), "Profile runner changed after plan")
    entries = frozen["entries"]
    require(len(entries) == 11 and [e["draw"] for e in entries] == list(range(11)), "Incomplete original qualification")
    qualification = [audit_record(f"qualify-{i:02}", i, ":16:8", "measure", "task", protocol, plan, entries) for i in range(11)]
    comparisons = {name: audit_record(name, 0, workspace, "measure", "task", protocol, plan, entries)
                   for name,workspace in (("baseline", ":4096:8"), ("small-workspace", ":16:8"), ("zero-workspace", ":0:0"))}
    profiles = {f"profile-{name}-{scope}": audit_record(f"profile-{name}-{scope}",0,workspace,"profile",scope,protocol,plan,entries)
                for name,workspace in (("baseline", ":4096:8"),("small", ":16:8")) for scope in ("task","training")}
    completed = sum(row["status"] == "passed" for row in qualification)
    state_equal = sum(row.get("all_state_hashes_bit_equal", False) for row in qualification)
    base_accuracy = read(BASE/"accuracy.json")
    reductions = {}
    baseline = comparisons["baseline"]
    if baseline["status"] == "passed":
        for name in ("small-workspace", "zero-workspace"):
            row = comparisons[name]
            if row["status"] != "passed":
                continue
            old, new = baseline["memory"]["peak_allocated_bytes"], row["memory"]["peak_allocated_bytes"]
            reductions[name] = {"peak_allocated_bytes_saved": old-new, "peak_allocated_reduction_percent": 100*(old-new)/old,
                                "peak_allocated_ratio_to_baseline": new/old}
            if "energy" in row and "energy" in baseline:
                old_e = baseline["energy"]["summary"]["idle_adjusted_mj_per_task"]["mean"]
                new_e = row["energy"]["summary"]["idle_adjusted_mj_per_task"]["mean"]
                reductions[name]["idle_adjusted_energy_ratio_to_baseline"] = new_e/old_e
    records = qualification + list(comparisons.values()) + list(profiles.values())
    failures = [r for r in records if r["status"] in ("failed", "unmeasured")]
    pending = sum(r["status"] == "pending" for r in records)
    all_passed = all(r["status"] == "passed" for r in records)
    result = {"schema_version": 1, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "analyzer_sha256": sha(__file__), "base_protocol_sha256": sha(BASE/"protocol.json"),
              "plan_sha256": sha(HERE/"plan.json"), "frozen_base_sources_unchanged": True,
              "status": "complete" if all_passed else "blocked" if failures and not pending else "partial",
              "all_available_evidence_verified": True, "all_jobs_passed": all_passed,
              "pending_jobs": pending, "failed_or_unmeasured_jobs": len(failures),
              "qualification": {"complete": completed == 11, "draws_bit_equal": completed,
                                "predictions_bit_equal": completed*10000, "draws": qualification,
                                "inherited_accuracy_valid": completed == 11,
                                "draws_with_all_state_hashes_bit_equal": state_equal,
                                "full_state_bit_equality_established": completed == state_equal == 11,
                                "equivalence_scope": "All class predictions, inputs, initial weights and epoch shuffles match. Different final-state or score hashes are explicitly retained; no numerical error bound is inferred from hashes."},
              "comparisons": comparisons, "reductions": reductions, "profiles": profiles,
              "scope": "Independent CPU comparison against frozen predictions and state, storage accounting, and retained NVML/Ncu counters. No test labels or GPU retraining used.",
              "cache_claim": "Capacity and measured traffic are separate. No claim that every access hits L2 or that profiling proves universal cache residency."}
    if completed == 11:
        result["qualification"]["inherited_accuracy"] = {key: base_accuracy[key] for key in
            ("correct","total","accuracy_percent","sample_sd_pp","error_percent","meets_12_percent_error_target")}
        result["qualification"]["base_accuracy_sha256"] = sha(BASE/"accuracy.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"summary.json")
    args = parser.parse_args()
    result = analyze()
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "all_jobs_passed": result["all_jobs_passed"],
                      "draws_bit_equal": result["qualification"]["draws_bit_equal"],
                      "pending_jobs": result["pending_jobs"], "failed_or_unmeasured_jobs": result["failed_or_unmeasured_jobs"],
                      "reductions": result["reductions"]}, indent=2))
