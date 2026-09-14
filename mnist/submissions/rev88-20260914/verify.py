"""Independently verify the reversible-MNIST qualification and NVML evidence.

CPU-only: regenerate inputs from official IDX sources, recompute predictions'
accuracies, audit the frozen selection and recipe, and rederive energy from
retained cumulative counters. This script neither trains nor changes evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import data_reference as data

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ah(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def close(actual, expected, name):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
            math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-9),
            f"{name}: retained {actual}, independently computed {expected}")


def safe_path(relative):
    path = (HERE / relative).resolve()
    require(path.is_relative_to(HERE.resolve()), f"Evidence path leaves submission: {relative}")
    return path


def source_arrays(raw_dir, expected):
    paths = {key: data.download_source(raw_dir, *data.SOURCES[key])
             for key in ("train_images", "train_labels")}
    actual = {key: {"file": p.name, "sha256": sha(p), "md5": data.file_hash(p, "md5")}
              for key, p in paths.items()}
    require(actual == expected, "Official source checksums differ from the frozen protocol")
    images = data.read_idx(paths["train_images"], 60000, images=True)
    labels = data.read_idx(paths["train_labels"], 60000, images=False).astype(np.int64)
    images = data.area_resize(images.astype(np.float32) / np.float32(255), 9)
    np.clip(images, 0, 1, out=images)
    return images[:, None], labels


def verify_selection(protocol, images, labels):
    plan, validation = read(HERE / "validation-plan.json"), read(HERE / "validation.json")
    require(sha(HERE / "validation.json") == protocol["validation_sha256"], "Frozen validation evidence changed")
    require(validation["complete"] and len(validation["records"]) == len(plan["configs"]) == 16,
            "The declared sixteen-candidate search is incomplete")
    require([row["config"] for row in validation["records"]] == plan["configs"], "Validation candidates changed or disappeared")
    require(plan["dataset_seed"] == protocol["dataset_seeds"][0], "Validation uses a different draw")
    require(plan["learner_sha256"] == protocol["source_sha256"]["learner.py"], "Validation and qualification learner files differ")
    order = np.random.Generator(np.random.PCG64(plan["dataset_seed"])).permutation(60000)
    train, valid = order[:8000], order[8000:10000]
    require(ah(train) == plan["train_indices_sha256"] and ah(valid) == plan["validation_indices_sha256"], "Validation split differs")
    hashes = {"train_images": ah(images[train]), "train_labels": ah(labels[train]), "test_images": ah(images[valid])}
    for row in validation["records"]:
        require(row["total"] == 2000 and 0 <= row["correct"] <= 2000, "Invalid validation count")
        if row.get("failed"):
            require(row["correct"] == 0 and row["result"]["failed"], "Incorrect failed-validation record")
            continue
        result = row["result"]
        require(result["config"] == row["config"], "Validation config/result mismatch")
        require(result["learner_sha256"] == plan["learner_sha256"], "Validation used a different source than its plan")
        require(result["input_sha256"] == hashes and result["metadata"]["input_hashes"] == hashes, "Validation received different data")
        path = safe_path(row["prediction_path"])
        require(sha(path) == row["prediction_sha256"], "Frozen validation prediction archive changed")
        prediction = np.load(path, allow_pickle=False)
        require(prediction.dtype == np.int64 and prediction.shape == (2000,) and
                prediction.min() >= 0 and prediction.max() <= 9, "Invalid validation predictions")
        require(ah(prediction) == result["metadata"]["predictions_sha256"], "Validation prediction payload differs from learner output")
        require(int(np.count_nonzero(prediction == labels[valid])) == row["correct"], "Independently rescored validation count differs")
        close(row["accuracy_percent"], row["correct"] / 20, "Validation percentage")
    eligible = [row for row in validation["records"] if row["correct"] >= 1780]
    require(bool(eligible), "No candidate passed the training-only 89% selection gate")
    selected = min(eligible, key=lambda row: (row["config"]["epochs"], row["config"]["depth"], row["correct"]))
    require(selected["config"] == protocol["configuration"] == read(HERE / "config.json"), "Final configuration differs from the declared selection rule")
    return {"candidates": len(validation["records"]), "failed_candidates": sum(bool(r.get("failed")) for r in validation["records"]),
            "selected_correct": selected["correct"], "selected_total": 2000,
            "validation_learner_sha256": plan["learner_sha256"],
            "validation_matches_final_learner_file": plan["learner_sha256"] == protocol["source_sha256"]["learner.py"],
            "selection_rule_recomputed": True, "validation_predictions_independently_rescored": True,
            "scope": "All successful frozen validation predictions rescored against the training-only split; configs, input hashes and the final selection independently verified."}, selected["result"]["metadata"]["initial_parameter_sha256"]


def pci_identity(value):
    domain, bus, device_function = value.strip().lower().split(":")
    device, function = device_function.split(".")
    return tuple(int(part, 16) for part in (domain, bus, device, function))


def verify_energy(result, metadata):
    energy = result["energy"]
    hardware = energy["hardware"]
    require("A100" in hardware["gpu"] and hardware["total_memory_bytes"] > 0, "Energy hardware is not an A100")
    require(hardware["nvml_handle_selected_by_cuda_pci_bus_id"] and
            pci_identity(hardware["cuda_pci_bus_id"]) == pci_identity(hardware["nvml_pci_bus_id"]),
            "NVML energy counter is not matched to the CUDA device")
    require(len(energy["trials"]) == 3, "Expected three NVML trials")
    require(energy["repeat_predictions_equal"] and energy["repeat_state_hashes_equal"], "Repeated fresh tasks differed")
    before, after = energy["before_metadata"], energy["after_metadata"]
    require(before["config"] == after["config"] == metadata["config"], "Energy used a different learner configuration")
    for key in energy["state_hash_keys_checked"]:
        require(before[key] == after[key] == metadata[key], f"Energy state or input differs from accuracy run: {key}")
    require(before["fresh_initialization_per_run"] and after["fresh_initialization_per_run"], "Energy repetitions reuse learned state")
    require(not before["checkpoint_loaded"] and not after["checkpoint_loaded"], "Energy loaded a checkpoint")
    derived = []
    for index, trial in enumerate(energy["trials"]):
        require(trial["trial"] == index and trial["invocations"] == energy["repeats_per_trial"] > 0, "Wrong energy trial order/repetition count")
        measured = {}
        for name in ("idle_before", "active", "idle_after"):
            interval = trial[name]
            dt = interval["end"]["time_s"] - interval["start"]["time_s"]
            joules = (interval["end"]["energy_mj"] - interval["start"]["energy_mj"]) / 1000
            require(dt > 0 and joules >= 0, f"Invalid cumulative energy interval: {index}/{name}")
            close(interval["duration_s"], dt, f"{name} elapsed seconds")
            close(interval["energy_j"], joules, f"{name} cumulative joules")
            close(interval["average_power_w"], joules / dt, f"{name} average watts")
            if name.startswith("idle"):
                require(dt >= 2.9 and interval["settle_seconds"] >= 3, "Idle sample/settling interval is too short")
            measured[name] = (dt, joules, joules/dt)
        require(trial["idle_before"]["end"]["time_s"] <= trial["active"]["start"]["time_s"] <
                trial["active"]["end"]["time_s"] <= trial["idle_after"]["start"]["time_s"], "Idle measurements do not bracket active work")
        idle_w = (measured["idle_before"][2] + measured["idle_after"][2]) / 2
        dt, joules, _ = measured["active"]
        repeats = trial["invocations"]
        adjusted = (joules - idle_w * dt) * 1000 / repeats
        gross = joules * 1000 / repeats
        close(trial["paired_idle_w"], idle_w, "Paired idle watts")
        close(trial["idle_adjusted_mj_per_task"], adjusted, "Idle-adjusted mJ/task")
        close(trial["unadjusted_mj_per_task"], gross, "Gross mJ/task")
        close(trial["wall_ms_per_task"], dt * 1000 / repeats, "Wall ms/task")
        require(trial["cuda_ms_per_task"] > 0 and 0.7 < trial["cuda_ms_per_task"] / trial["wall_ms_per_task"] < 1.3,
                "CUDA-event and wall timing disagree materially")
        derived.append({"idle_adjusted_mj_per_task": adjusted, "unadjusted_mj_per_task": gross,
                        "cuda_ms_per_task": trial["cuda_ms_per_task"], "wall_ms_per_task": dt*1000/repeats})
    for key, summary in energy["summary"].items():
        values = [row[key] for row in derived]
        require(len(summary["values"]) == 3, "Energy summary omitted trials")
        for actual, expected in zip(summary["values"], values):
            close(actual, expected, f"Energy summary values: {key}")
        close(summary["mean"], np.mean(values), f"Energy summary mean: {key}")
        close(summary["sample_sd"], np.std(values, ddof=1), f"Energy summary SD: {key}")
    require(energy["summary"]["idle_adjusted_mj_per_task"]["mean"] > 0, "Mean adjusted energy is nonpositive")
    cuda_validation = result["cuda_validation"]
    require(cuda_validation["passed"] and cuda_validation["checks"] > 0 and
            math.isfinite(cuda_validation["maximum_abs_error"]), "Independent CUDA replay validation failed")
    require(cuda_validation["tolerance"] == {"atol": 2e-5, "rtol": 3e-3}, "CUDA validation tolerance changed")
    require([case["depth"] for case in cuda_validation["cases"]] == [1,2], "CUDA validation did not cover both depths")
    for case in cuda_validation["cases"]:
        expected = {**metadata["config"], "depth": case["depth"], "epochs": 2, "batch_size": 8, "seed": 781}
        require(case["passed"] and case["config"] == expected and case["epochs"] == 2 and
                case["batch_sizes"] == [8,8,3], "CUDA replay did not verify a final partial batch")
        require(case["complete_replay_byte_repeatable"] and case["predictions_match_independent_eager"],
                "CUDA graph replay or eager-reference equivalence failed")
        require(case["saved_core_activation_storages"] == 1 and case["saved_core_activation_bytes"] == 984 and
                case["saved_parameter_references"] == 2*case["depth"], "CUDA reconstruction retained unexpected states")
    return {"passed": True, "draw_index": 0, "trials": 3,
            "invocations_per_trial": energy["repeats_per_trial"],
            "counter_and_idle_arithmetic_recomputed": True, "repeat_state_hashes_match_accuracy_run": True,
            "cuda_nvml_device_identity_verified": True, "cuda_reconstruction_and_replay_checks_passed": True,
            "mean_idle_adjusted_mj_per_task": energy["summary"]["idle_adjusted_mj_per_task"]["mean"],
            "mean_cuda_ms_per_task": energy["summary"]["cuda_ms_per_task"]["mean"],
            "hardware": energy["hardware"], "scope": "Three repeated-task trials on qualification draw0, not energy averaged across eleven draws"}


def verify(raw_dir):
    protocol, frozen = read(HERE / "protocol.json"), read(HERE / "prediction-manifest.json")
    require(protocol["draw_count"] == 11 and protocol["train_count"] == protocol["test_count"] == 10000, "Wrong medium dataset sizes")
    require(protocol["dataset_seeds"] == list(range(2026091400, 2026091411)), "Unexpected dataset seeds")
    require(protocol["required_correct"] == 96800 and protocol["total"] == 110000 and protocol["error_target_percent"] == 12, "Incorrect accuracy gate")
    for name, digest in protocol["source_sha256"].items():
        require(sha(safe_path(name)) == digest, f"Frozen source changed: {name}")
    require(frozen["protocol_sha256"] == sha(HERE / "protocol.json") and frozen["test_labels_scored"] is False, "Prediction freeze has wrong scope/protocol")
    require(frozen["draw_manifest_sha256"] == sha(HERE / "draw-manifest.json"), "Draw manifest changed after freeze")
    draw_manifest = read(HERE / "draw-manifest.json")
    require(draw_manifest["protocol_sha256"] == sha(HERE / "protocol.json"), "Draws use a different protocol")
    require(len(frozen["entries"]) == len(draw_manifest["draws"]) == 11, "Eleven complete draws are required")
    require([e["draw"] for e in frozen["entries"]] == list(range(11)), "Duplicate/missing frozen draw indices")
    require([d["draw"] for d in draw_manifest["draws"]] == list(range(11)), "Duplicate/missing data draw indices")
    # Validate every frozen prediction/result hash before opening label arrays.
    for entry in frozen["entries"]:
        require(sha(safe_path(entry["prediction_path"])) == entry["prediction_sha256"], "Frozen prediction archive changed")
        require(sha(safe_path(entry["result_path"])) == entry["result_sha256"], "Frozen learner evidence changed")
    images, labels = source_arrays(raw_dir, protocol["raw_sources"])
    selection, initial_hashes = verify_selection(protocol, images, labels)
    config = protocol["configuration"]
    shuffle_rng = np.random.Generator(np.random.PCG64(config["seed"]))
    expected_orders = [ah(shuffle_rng.permutation(10000).astype(np.int64)) for _ in range(config["epochs"])]
    rows, energy_audit = [], None
    confusion_total = np.zeros((10,10), np.int64)
    for index, (entry, draw) in enumerate(zip(frozen["entries"], draw_manifest["draws"])):
        seed = protocol["dataset_seeds"][index]
        require(entry["dataset_seed"] == draw["dataset_seed"] == seed, "Result/data seed mismatch")
        order = np.random.Generator(np.random.PCG64(seed)).permutation(60000)
        train, test = order[:10000], order[10000:20000]
        require(len(np.unique(np.r_[train, test])) == 20000 and draw["train_test_disjoint"], "Train/test overlap")
        require(ah(train) == draw["train_indices_sha256"] and ah(test) == draw["test_indices_sha256"], "Recorded source indices differ")
        inputs = {"train_images": ah(images[train]), "train_labels": ah(labels[train]), "test_images": ah(images[test])}
        result = read(safe_path(entry["result_path"]))
        metadata = result["metadata"]
        require(result["config"] == metadata["config"] == config, "Qualification configuration changed")
        require(result["learner_sha256"] == protocol["source_sha256"]["learner.py"], "Qualification learner source differs")
        require(result["input_sha256"] == metadata["input_hashes"] == draw["input_sha256"] == inputs, "Learner input hashes differ")
        require(metadata["train_count"] == metadata["test_count"] == 10000, "Learner sample counts differ")
        require(metadata["fresh_initialization_per_run"] and not metadata["checkpoint_loaded"], "Learner reused a checkpoint/state")
        require(metadata["initial_parameter_sha256"] == initial_hashes, "Initialization differs between validation and qualification")
        require(metadata["final_parameter_sha256"] != initial_hashes, "No learned parameters changed")
        require(metadata["epoch_permutation_sha256"] == expected_orders, "Epoch shuffles differ from the declared seed")
        require(metadata["parameter_count"] == 2*config["depth"]*41*41 + 82*10 + 10, "Wrong model size")
        require(metadata["reversible_core"] and metadata["input_lift_injective"] and not metadata["head_reversible"], "Incorrect reversibility claims")
        require(metadata["retained_core_states"] == 1 and metadata["core_saved_bytes_per_example"] == 328, "Incorrect retained-state claim")
        require(metadata["validation"]["passed"] and len(metadata["validation"]["checks"]) == 4, "Reconstruction validation missing")
        for check in metadata["validation"]["checks"]:
            require(check["passed"] and check["forward_bitwise_equal"] and check["saved_activation_storages"] == 1,
                    "Reconstruction validation did not pass")
        pred = np.load(safe_path(entry["prediction_path"]), allow_pickle=False)
        require(pred.dtype == np.int64 and pred.shape == (10000,) and pred.min() >= 0 and pred.max() <= 9, "Invalid output labels")
        require(ah(pred) == metadata["predictions_sha256"], "Prediction payload differs from learner output")
        truth = labels[test]
        correct = int(np.count_nonzero(pred == truth))
        confusion = np.bincount(10*truth+pred, minlength=100).reshape(10,10)
        require(int(np.trace(confusion)) == correct and int(confusion.sum()) == 10000, "Independent confusion/count cross-check failed")
        confusion_total += confusion
        rows.append({"draw": index, "dataset_seed": seed, "correct": correct, "total": 10000,
                     "accuracy_percent": correct/100, "test_labels_sha256": ah(truth)})
        if index == 0:
            energy_audit = verify_energy(result, metadata)
        else:
            require("energy" not in result, "Energy scope differs from the declared draw0-only plan")
    accuracy = read(HERE / "accuracy.json")
    require(accuracy["prediction_manifest_sha256"] == sha(HERE / "prediction-manifest.json"), "Accuracy uses different predictions")
    require(accuracy["rows"] == rows, "Independent per-draw accuracy recomputation differs")
    correct = sum(row["correct"] for row in rows)
    require(accuracy["correct"] == correct and accuracy["total"] == 110000, "Incorrect aggregate accuracy counts")
    close(accuracy["accuracy_percent"], correct/1100, "Mean accuracy")
    close(accuracy["error_percent"], (110000-correct)/1100, "Mean error")
    close(accuracy["sample_sd_pp"], np.std([r["accuracy_percent"] for r in rows], ddof=1), "Sample SD across draws")
    require(accuracy["meets_12_percent_error_target"] == (correct >= 96800), "Incorrect qualification decision")
    require(datetime.fromisoformat(accuracy["evaluated_at_utc"]) >= datetime.fromisoformat(frozen["frozen_at_utc"]), "Evaluation predates prediction freeze")
    return {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
            "verifier_sha256": sha(Path(__file__)), "protocol_sha256": sha(HERE / "protocol.json"),
            "prediction_manifest_sha256": sha(HERE / "prediction-manifest.json"),
            "draws_verified": 11, "predictions_verified": 110000,
            "correct": correct, "accuracy_percent": correct/1100,
            "sample_sd_pp": accuracy["sample_sd_pp"], "meets_12_percent_error_target": correct >= 96800,
            "independent_confusion_matrix": confusion_total.tolist(),
            "all_draw_indices_and_inputs_recomputed": True, "all_frozen_source_and_prediction_hashes_match": True,
            "fresh_initializations_and_epoch_shuffles_match": True,
            "selection": selection, "energy": energy_audit,
            "scope": "Independent CPU source/data/count/energy audit. GPU reconstruction and replay checks are retained run evidence; this script does not retrain."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=HERE.parents[1] / "data/raw")
    parser.add_argument("--output", type=Path, default=HERE / "verification.json")
    args = parser.parse_args()
    result = verify(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "draws_verified", "predictions_verified", "correct", "accuracy_percent", "sample_sd_pp", "meets_12_percent_error_target")}, indent=2))
    print("Verified NVML mean:", result["energy"]["mean_idle_adjusted_mj_per_task"], "mJ/task")
