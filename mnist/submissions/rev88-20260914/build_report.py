"""Build the rev88 submission only after all qualification evidence is frozen.

CPU-only: read retained results and source data, verify arithmetic and hashes,
then write README.md and submission.json. Never launches training or energy work.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import numpy as np

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ah(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def close(a, b, message):
    require(np.isclose(a, b, rtol=1e-11, atol=1e-9), message)


def write(path, value):
    text = value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False)+"\n"
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def cost(value):
    """Display exactly two significant figures; raw values remain in JSON."""
    return format(value, "#.2g").rstrip(".")


def freeze_gate():
    names = ["protocol.json", "config.json", "validation-plan.json", "validation.json", "draw-manifest.json",
             "prediction-manifest.json", "accuracy.json"]
    missing = [n for n in names if not (HERE/n).is_file()]
    if missing:
        raise RuntimeError("Submission pending; no partial report written. Missing: "+", ".join(missing))
    protocol, config, selection, validation, draws, predictions, accuracy = [read(HERE/n) for n in names]
    require(protocol["configuration"] == config and protocol["draw_count"] == 11
            and protocol["train_count"] == protocol["test_count"] == 10000, "Protocol/config dimensions differ")
    require(protocol["error_target_percent"] == 12 and protocol["required_correct"] == 96800 and protocol["total"] == 110000,
            "Current MNIST-medium 12% threshold differs")
    require(protocol["dataset_seeds"] == list(range(2026091400, 2026091411)) and protocol["learner_seed"] == config["seed"] == 11,
            "Frozen dataset/learner seeds differ")
    for name, digest in protocol["source_sha256"].items():
        require(sha(HERE/name) == digest, f"Frozen source changed: {name}")
    require(protocol["validation_sha256"] == sha(HERE/"validation.json") and validation["complete"] is True,
            "Training-only selection is incomplete or changed")
    require(selection["learner_sha256"] == protocol["source_sha256"]["learner.py"]
            and len(validation["records"]) == len(selection["configs"]), "Selection plan/model differs")
    require([r["config"] for r in validation["records"]] == selection["configs"], "Selection candidate order differs")
    eligible = [r for r in validation["records"] if not r.get("failed", False) and r["correct"] >= 1780]
    require(eligible, "No training-only validation candidate reaches 89%")
    selected = min(eligible, key=lambda r: (r["config"]["epochs"], r["config"]["depth"], r["correct"]))
    require(selected["config"] == config, "Frozen procedure differs from the planned selection rule")
    require(predictions["protocol_sha256"] == draws["protocol_sha256"] == sha(HERE/"protocol.json")
            and predictions["draw_manifest_sha256"] == sha(HERE/"draw-manifest.json")
            and predictions["test_labels_scored"] is False, "Prediction freeze provenance differs")
    require(accuracy["prediction_manifest_sha256"] == sha(HERE/"prediction-manifest.json"), "Scored freeze changed")
    require(len(draws["draws"]) == len(predictions["entries"]) == len(accuracy["rows"]) == 11, "All eleven draws are required")
    results = []
    for i, entry in enumerate(predictions["entries"]):
        require(entry["draw"] == i and entry["dataset_seed"] == protocol["dataset_seeds"][i], "Frozen draw identity differs")
        require(entry["prediction_path"] == f"predictions/draw-{i:02d}.npy" and entry["result_path"] == f"results/draw-{i:02d}.json",
                "Unexpected result path")
        for path_key, hash_key in (("prediction_path", "prediction_sha256"), ("result_path", "result_sha256")):
            path = HERE/entry[path_key]
            require(path.is_file() and sha(path) == entry[hash_key], f"Frozen output missing or changed: {path.name}")
        results.append(read(HERE/entry["result_path"]))
    require("energy" in results[0] and len(results[0]["energy"]["trials"]) == 3, "Three draw-zero energy trials are required")
    require("cuda_validation" in results[0] and results[0]["cuda_validation"]["passed"] is True, "CUDA reconstruction/replay validation is required")
    return protocol, config, selection, selected, draws, predictions, accuracy, results


def data_audit(raw_dir, protocol, config, draws, predictions, accuracy, results):
    """First evaluation-label read occurs after every output passes freeze_gate."""
    spec = importlib.util.spec_from_file_location("submission_data_reference", HERE/"data_reference.py")
    data = importlib.util.module_from_spec(spec); spec.loader.exec_module(data)
    paths = {key: raw_dir/record["file"] for key, record in protocol["raw_sources"].items()}
    for key, path in paths.items():
        require(path.is_file() and sha(path) == protocol["raw_sources"][key]["sha256"], "MNIST source file missing or changed")
        require(data.file_hash(path, "md5") == protocol["raw_sources"][key]["md5"], "MNIST source MD5 differs")
    native = data.read_idx(paths["train_images"], 60000, images=True)
    labels = data.read_idx(paths["train_labels"], 60000, images=False).astype(np.int64)
    images = data.area_resize(native.astype(np.float32)/np.float32(255), 9)
    np.clip(images, 0, 1, out=images); images = images[:, None, :, :]
    plan, validation = read(HERE/"validation-plan.json"), read(HERE/"validation.json")
    selection_order = np.random.Generator(np.random.PCG64(protocol["dataset_seeds"][0])).permutation(60000)
    vi = selection_order[8000:10000]
    require(ah(selection_order[:8000]) == plan["train_indices_sha256"] and ah(vi) == plan["validation_indices_sha256"],
            "Training-only selection index hashes differ")
    for index, candidate in enumerate(validation["records"]):
        if candidate.get("failed"):
            require(candidate["correct"] == 0 and candidate["result"]["failed"], "Invalid validation failure evidence")
            continue
        require(candidate["prediction_path"] == f"validation-predictions/candidate-{index:02d}.npy", "Unexpected validation prediction path")
        vp = HERE/candidate["prediction_path"]
        require(sha(vp) == candidate["prediction_sha256"], "Validation predictions changed")
        pred = np.load(vp, allow_pickle=False)
        require(pred.shape == (2000,) and pred.dtype == np.int64 and int(np.count_nonzero(pred == labels[vi])) == candidate["correct"],
                "Training-only validation counts differ on independent rescore")
    parameters = 2*config["depth"]*41*41 + 82*10 + 10
    names = {f"weights.{i}" for i in range(2*config["depth"])} | {"head.weight", "head.bias"}
    rng = np.random.Generator(np.random.PCG64(config["seed"]))
    permutations = [ah(rng.permutation(10000).astype(np.int64)) for _ in range(config["epochs"])]
    expected_initial = results[0]["metadata"]["initial_parameter_sha256"]
    rows = []
    for i, (draw, entry, recorded, result) in enumerate(zip(draws["draws"], predictions["entries"], accuracy["rows"], results)):
        seed = protocol["dataset_seeds"][i]
        order = np.random.Generator(np.random.PCG64(seed)).permutation(60000)
        train, test = order[:10000], order[10000:20000]
        require(draw["draw"] == i and draw["dataset_seed"] == seed and draw["train_test_disjoint"] is True
                and len(np.unique(np.r_[train, test])) == 20000, "Draw identity/disjointness differs")
        require(ah(train) == draw["train_indices_sha256"] and ah(test) == draw["test_indices_sha256"], "Sampling replay hash differs")
        inputs = {"train_images": ah(images[train]), "train_labels": ah(labels[train]), "test_images": ah(images[test])}
        metadata = result["metadata"]
        require(inputs == draw["input_sha256"] == result["input_sha256"] == metadata["input_hashes"], "Learner input hash differs")
        require(result["config"] == metadata["config"] == config and result["learner_sha256"] == protocol["source_sha256"]["learner.py"],
                "Executed learner/config differs")
        require(metadata["train_count"] == metadata["test_count"] == 10000 and metadata["parameter_count"] == parameters,
                "Executed input/parameter count differs")
        require(metadata["fresh_initialization_per_run"] is True and metadata["checkpoint_loaded"] is False
                and metadata["reversible_core"] is True and metadata["head_reversible"] is False and metadata["input_lift_injective"] is True,
                "Fresh-state/reversibility policy differs")
        require(metadata["retained_core_states"] == 1 and metadata["core_saved_bytes_per_example"] == 328, "Endpoint retention differs")
        require(metadata["epoch_permutation_sha256"] == permutations and metadata["initial_parameter_sha256"] == expected_initial,
                "Fresh initialization or independent epoch permutation replay differs")
        for field in ("initial_parameter_sha256", "final_parameter_sha256", "final_velocity_sha256"):
            require(set(metadata[field]) == names and all(isinstance(v, str) and len(v) == 64 for v in metadata[field].values()), "State hash schema differs")
        require(metadata["final_parameter_sha256"] != expected_initial, "No parameter updates occurred")
        require(metadata["validation"]["passed"] is True and all(r["passed"] for r in metadata["validation"]["checks"]),
                "CPU ordinary-versus-reconstructed gradient/inverse validation failed")
        memory = result["memory"]
        require(0 <= memory["allocated_before_prepare_bytes"] <= memory["prepare_and_invoke_peak_allocated_bytes"]
                and 0 <= memory["allocated_after_invoke_bytes"] <= memory["prepare_and_invoke_peak_allocated_bytes"]
                and memory["allocated_after_invoke_bytes"] <= memory["reserved_after_invoke_bytes"], "Invalid preparation/invocation memory measurements")
        pred = np.load(HERE/entry["prediction_path"], allow_pickle=False)
        require(pred.shape == (10000,) and pred.dtype == np.int64 and pred.min() >= 0 and pred.max() <= 9,
                "Invalid prediction labels")
        require(ah(pred) == metadata["predictions_sha256"], "Prediction array hash differs")
        correct = int(np.count_nonzero(pred == labels[test]))
        require(recorded == {"draw": i, "dataset_seed": seed, "correct": correct, "total": 10000,
                             "accuracy_percent": correct/100, "test_labels_sha256": ah(labels[test])}, "Independent score replay differs")
        rows.append({**recorded, "learner_seed": config["seed"], "cuda_ms_single_run": result["cuda_ms"]})
    correct = sum(r["correct"] for r in rows)
    values = [r["accuracy_percent"] for r in rows]
    require(correct == accuracy["correct"] and accuracy["total"] == 110000, "Accuracy total differs")
    close(accuracy["accuracy_percent"], correct/1100, "Unrounded mean accuracy differs")
    close(accuracy["sample_sd_pp"], statistics.stdev(values), "Sample SD differs")
    close(accuracy["error_percent"], 100-correct/1100, "Error percentage differs")
    require(accuracy["meets_12_percent_error_target"] == (correct >= 96800), "Qualification decision differs")
    return rows, parameters


def energy_audit(energy, reference):
    hardware = energy["hardware"]
    def pci_key(value):
        return tuple(int(v, 16) for v in value.replace(".", ":").split(":"))
    require(hardware["nvml_handle_selected_by_cuda_pci_bus_id"] is True
            and pci_key(hardware["cuda_pci_bus_id"]) == pci_key(hardware["nvml_pci_bus_id"])
            and "A100" in hardware["gpu"], "CUDA/NVML physical device identity differs")
    require(energy["repeat_predictions_equal"] and energy["repeat_state_hashes_equal"], "Energy replay changed outputs/state")
    require(energy["before_metadata"]["config"] == energy["after_metadata"]["config"] == reference["config"], "Energy measured another procedure")
    for key in energy["state_hash_keys_checked"]:
        require(energy["before_metadata"][key] == energy["after_metadata"][key] == reference[key], "Energy replay hash differs from qualification draw0")
    require(energy["repeats_per_trial"] >= 1 and len(energy["trials"]) == 3, "Energy repeat/trial counts differ")
    def interval(v):
        elapsed = v["end"]["time_s"]-v["start"]["time_s"]
        joules = (v["end"]["energy_mj"]-v["start"]["energy_mj"])/1000
        require(elapsed > 0 and joules >= 0, "Invalid cumulative energy interval")
        close(v["duration_s"], elapsed, "Energy interval duration differs")
        close(v["energy_j"], joules, "Cumulative energy conversion differs")
        close(v["average_power_w"], joules/elapsed, "Average power differs")
    for i, trial in enumerate(energy["trials"]):
        require(trial["trial"] == i and trial["invocations"] == energy["repeats_per_trial"], "Energy trial identity differs")
        for name in ("active", "idle_before", "idle_after"):
            interval(trial[name])
        require(trial["idle_before"]["settle_seconds"] == trial["idle_after"]["settle_seconds"] == 3, "Idle settling differs")
        idle_w = (trial["idle_before"]["average_power_w"]+trial["idle_after"]["average_power_w"])/2
        close(trial["paired_idle_w"], idle_w, "Paired idle power differs")
        close(trial["idle_adjusted_mj_per_task"], (trial["active"]["energy_j"]-idle_w*trial["active"]["duration_s"])*1000/trial["invocations"], "Idle-adjusted energy differs")
        close(trial["unadjusted_mj_per_task"], trial["active"]["energy_j"]*1000/trial["invocations"], "Unadjusted energy differs")
        close(trial["wall_ms_per_task"], trial["active"]["duration_s"]*1000/trial["invocations"], "Wall runtime differs")
        require(trial["cuda_ms_per_task"] > 0, "Invalid CUDA runtime")
    for name, summary in energy["summary"].items():
        values = [trial[name] for trial in energy["trials"]]
        require(summary["values"] == values, "Energy summary values differ")
        close(summary["mean"], statistics.mean(values), "Energy mean differs")
        close(summary["sample_sd"], statistics.stdev(values), "Energy sample SD differs")


def build(raw_dir, rules_file):
    protocol, config, selection, selected, draws, manifest, accuracy, results = freeze_gate()
    rows, parameters = data_audit(raw_dir, protocol, config, draws, manifest, accuracy, results)
    energy = results[0]["energy"]
    energy_audit(energy, results[0]["metadata"])
    e = energy["summary"]["idle_adjusted_mj_per_task"]
    runtime = energy["summary"]["cuda_ms_per_task"]
    endstate_bytes = config["batch_size"]*328
    memory_peaks = [r["memory"]["prepare_and_invoke_peak_allocated_bytes"] for r in results]
    memory_baselines = [r["memory"]["allocated_before_prepare_bytes"] for r in results]
    memory_increments = [peak-baseline for peak, baseline in zip(memory_peaks, memory_baselines)]
    qualified = accuracy["meets_12_percent_error_target"]
    submission = {
        "schema_version": 1, "built_at_utc": datetime.now(timezone.utc).isoformat(), "submission_date": "2026-09-14", "name": "rev88-20260914",
        "tier": "MNIST-medium", "target_error_percent": 12, "minimum_accuracy_percent": 88,
        "required_correct": 96800, "qualifies": qualified, "status": "qualified" if qualified else "does_not_qualify",
        "configuration": config, "parameter_count": parameters, "parameter_payload_bytes": 4*parameters,
        "four_parameter_state_payload_bytes": 16*parameters,
        "memory": {"measured_peak_allocated_bytes": max(memory_peaks), "measured_prepare_and_invoke_peak_bytes_by_draw": memory_peaks,
                   "entry_allocated_baseline_bytes_by_draw": memory_baselines,
                   "added_peak_above_entry_baseline_bytes_by_draw": memory_increments,
                   "added_peak_above_entry_baseline_bytes_max": max(memory_increments),
                   "measured_prepare_and_invoke_peak_bytes_min": min(memory_peaks), "measured_prepare_and_invoke_peak_bytes_median": statistics.median(memory_peaks),
                   "grid_peak_scratch_bytes": None,
                   "core_forward_retained_states": 1, "core_endpoint_bytes_per_example": 328,
                   "core_endpoint_bytes_at_training_batch": endstate_bytes,
                   "scope": "Peak allocator measurement spans preparation and first full invocation before serialization/CUDA diagnostics; includes graph pools/library workspace, excludes driver/context. Report raw peaks, preexisting worker baselines and peak-minus-entry separately. Baselines differ on reused workers; this is not model-dependent growth. Core endpoint/parameter payload are separate exact tensor-size calculations."},
        "accuracy": {"correct": accuracy["correct"], "total": 110000, "mean_percent": accuracy["accuracy_percent"],
                     "sample_sd_pp": accuracy["sample_sd_pp"], "error_percent": accuracy["error_percent"], "draws": rows},
        "dataset_seeds": protocol["dataset_seeds"], "learner_seed": config["seed"], "learner_seed_policy": "same seed, fresh weights and zero momentum in every draw/replay",
        "selection": {"validation_only": True, "training_examples": 8000, "validation_examples": 2000,
                      "candidate_count": len(selection["configs"]), "selected_correct": selected["correct"], "selected_total": selected["total"],
                      "rule": selection["selection"]},
        "a100": {"energy_mj": e["mean"], "energy_sample_sd_mj": e["sample_sd"], "runtime_ms": runtime["mean"],
                 "runtime_sample_sd_ms": runtime["sample_sd"], "draw_measured": 0, "dataset_seed_measured": protocol["dataset_seeds"][0],
                 "trials": 3, "fresh_invocations_per_trial": energy["repeats_per_trial"], "energy_summaries": energy["summary"],
                 "hardware": energy["hardware"], "software": energy["software"], "scope": energy["scope"],
                 "cold_start_or_end_to_end_application_energy_measured": False},
        "grid": {"energy_mj": None, "runtime_ms": None, "peak_scratch_bytes": None, "active_processors": None,
                 "word_node_hops": None, "time_to_score_seconds": None, "model_revision": None, "status": "not measured"},
        "wandb_runs": [], "sources": {"training_source_sha256": protocol["source_sha256"], "raw_mnist_sources": protocol["raw_sources"],
                                      "protocol_sha256": sha(HERE/"protocol.json"), "prediction_manifest_sha256": sha(HERE/"prediction-manifest.json"),
                                      "accuracy_sha256": sha(HERE/"accuracy.json"), "report_builder_sha256": sha(__file__),
                                      "rules_sha256_at_report": sha(rules_file) if rules_file.exists() else None},
        "report_checks": {"passed": True, "all_eleven_outputs_frozen_before_label_read": True,
                          "training_only_selection_rule_and_all_retained_validation_predictions_recomputed": True,
                          "sampling_input_and_epoch_order_hashes_recomputed": True,
                          "all_eleven_accuracies_independently_rescored": True, "aggregate_mean_sample_sd_threshold_recomputed": True,
                          "energy_counter_arithmetic_and_three_trial_statistics_recomputed": True,
                          "energy_replay_state_matches_qualification_draw0": True, "cuda_nvml_pci_identity_verified": True,
                          "cpu_reconstruction_validation_passed": True, "cuda_eager_replay_validation_passed": True}}
    hardware, software = energy["hardware"], energy["software"]
    lines = ["# MNIST-medium: reversible MLP, 12% error target", "", "Submission date: September 14, 2026.", "",
             f"**{accuracy['accuracy_percent']:.4f}% ± {accuracy['sample_sd_pp']:.4f} pp** mean accuracy ± sample standard deviation over all eleven draws, with **{accuracy['correct']:,} / 110,000 correct**. "
             f"The unrounded mean error is {accuracy['error_percent']:.12g}%. This **{'meets' if qualified else 'does not meet'}** the current inclusive 12% error band, which requires at least 88% mean accuracy or 96,800 correct. Qualification uses exact counts, not the rounded display.", "",
             f"Draw-zero A100 measurements give **{cost(e['mean'])} mJ** idle-adjusted energy and **{cost(runtime['mean'])} ms** per complete fresh GPU-resident training-and-prediction task. "
             "These are means of three repeated-measurement trials on one draw, not energy averaged over the eleven accuracy draws.", "",
             "| A100 energy (mJ/task) | A100 CUDA time (ms/task) | Grid energy (mJ) | Grid time (ms) | Grid scratch (bytes) | Time to score (s) |",
             "|---:|---:|---:|---:|---:|---:|",
             f"| {cost(e['mean'])} | {cost(runtime['mean'])} | — | — | — | — |", "",
             "## Accuracy evidence", "", "| Draw | Dataset seed | Learner seed | Correct / total | Accuracy |",
             "|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['draw']:02d} | {row['dataset_seed']} | {config['seed']} | {row['correct']:,} / {row['total']:,} | {row['accuracy_percent']:.2f}% |")
    lines += ["", "Each draw is a separate PCG64 permutation of the official 60,000 MNIST training rows. The first 10,000 rows train the learner and the next 10,000 provide evaluation inputs; the two sets are disjoint within that draw. Independently sampled draws can overlap one another. The official test split is unused. Mean and sample SD (ddof=1) describe these eleven datasets; no confidence interval or independent-image inference is claimed.", "",
              f"The procedure was chosen using only an 8,000/2,000 split inside draw 0's training portion. All {len(selection['configs'])} candidates were evaluated before the qualification predictions were scored. Candidates reaching 89% validation accuracy were ordered by fewest epochs, then shallowest depth, then smallest accuracy surplus over 89%; the selected candidate scored {selected['correct']:,}/{selected['total']:,}. The frozen procedure was then trained freshly on all eleven complete 10,000-example training draws. Selection examples can occur in another draw's evaluation set because the draws share the same source population; selection is not additional qualification evidence.", "",
              "## Reversible learner and memory", "",
              "The only model inputs are the 81 supplied image pixels. Pixels are converted to float32, divided by 255, resized from 28×28 to 9×9 with exact fractional box-area averaging, clipped to [0,1], then transformed inside each measured invocation to `4*x - 0.5`. Appending one zero gives an injective 82-coordinate lift, split into two 41-coordinate halves.", "",
              f"The core has {config['depth']} additive coupling blocks, each with two bias-free 41×41 linear branches applied after ReLU, with residual scale {config['alpha']}. A learned 82→10 linear readout with bias produces class scores. The core is reversible; the classifier is not. The model has **{parameters:,} FP32 parameters** ({4*parameters:,} payload bytes).", "",
              f"Training uses **{config['epochs']} epochs**, batch size **{config['batch_size']}**, mean cross-entropy loss, SGD learning rate **{config['learning_rate']}**, momentum **{config['momentum']}**, and weight decay **{config['weight_decay']}**. There is no augmentation or ensemble. Each epoch uses a seed-only PCG64 permutation; the final 16-example minibatch is included. Branch weights start at normal standard deviation `0.05/sqrt(41)`; the head uses PyTorch's standard Linear initialization. Learner seed {config['seed']} is fixed, but weights, gradients and momentum are reset before every draw and timed invocation.", "",
              "Backward uses a custom autograd function that saves the core endpoint and parameter references, reconstructs each previous state by subtraction, and creates at most one branch's ordinary autograd graph at a time. Thus the number of saved core activation states is one. The referenced parameter tensors are not copied activation snapshots. First-order differentiation is supported; this is not a claim of bit-identical stored/reconstructed gradients or constant total GPU memory.", "",
              f"The retained core endpoint contains 82 FP32 values per example: **328 bytes/example**, or **{endstate_bytes:,} bytes ({endstate_bytes/1024:.2f} KiB)** for batch {config['batch_size']}. Four resident parameter groups—weights, gradients, momentum and the seed-initialization copy—contain {16*parameters:,} payload bytes. These are exact tensor-size calculations. Raw/normalized training and query arrays, resident permutations, classifier tensors, temporary branch gradients and CUDA graph memory are additional costs.", "",
              f"Measured CUDA tensor-allocation peaks over preparation plus the first full invocation span **{min(memory_peaks):,}–{max(memory_peaks):,} bytes** across eleven draws. Entry baselines on reused workers range from {min(memory_baselines):,} to {max(memory_baselines):,} bytes. Subtracting each draw's own entry baseline gives **{min(memory_increments):,}–{max(memory_increments):,} added-peak bytes** ({min(memory_increments)/2**20:.2f}–{max(memory_increments)/2**20:.2f} MiB); the raw variation comes entirely from preexisting resident allocations, not model/data-dependent growth. The cause of those preexisting allocations was not isolated in this submission. This scope includes CUDA graph pools and library workspaces, and is sampled before output serialization and synthetic CUDA diagnostics. It excludes driver/context allocations; it is not a pure activation measurement, a cache-working-set measurement, or energy's preparation-excluded scope. Grid scratch remains unmeasured.", "",
              "CPU FP64/FP32 checks compare reconstructed outputs, all parameter gradients, real-input gradients and inverse recovery against ordinary autograd at both depths. A separate synthetic CUDA check compares eager gradients/SGD state with the captured implementation, verifies exact replay resets and prediction equality, and checks endpoint/parameter-reference retention. These diagnostics run outside energy timing. TF32 is disabled; this PyTorch implementation does not claim the older ordered-kernel arithmetic contract.", "",
              "## A100 measurement method", "",
              f"Hardware: **{hardware['gpu']}**, driver {hardware['driver']}, power limit {hardware['power_limit_w']} W. Runtime: PyTorch {software['torch']}, CUDA {software['cuda']}, NumPy {software['numpy']}, Python {software['python']}, nvidia-ml-py {software['nvidia_ml_py']}. The CUDA runtime's PCI bus ID selects the matching NVML handle; the report verifies both IDs identify the same physical device. GPU identity, clocks, temperature, power samples and raw cumulative NVML counter readings are retained in draw 0's result.", "",
              f"A CUDA-event calibration selected **{energy['repeats_per_trial']} complete invocations per trial**, targeting approximately five seconds of active measurement (capped at 1,000 repetitions). Each of three trials has a three-second idle measurement before and after its active interval; each idle sample follows a three-second settling interval. Paired idle power is the mean of the two idle samples. Idle-adjusted energy is `active joules - paired_idle_watts * active_seconds`, divided by complete invocations and converted to mJ. CUDA events measure execution time; wall time and unadjusted energy are separately retained.", "",
              "| Trial | Fresh invocations | Idle-adjusted mJ/task | Unadjusted mJ/task | CUDA ms/task | Active seconds |",
              "|---:|---:|---:|---:|---:|---:|"]
    for trial in energy["trials"]:
        lines.append(f"| {trial['trial']+1} | {trial['invocations']} | {cost(trial['idle_adjusted_mj_per_task'])} | {cost(trial['unadjusted_mj_per_task'])} | {cost(trial['cuda_ms_per_task'])} | {cost(trial['active']['duration_s'])} |")
    lines += ["", f"Exact three-trial means are {e['mean']:.15g} mJ/task (sample SD {e['sample_sd']:.15g} mJ) and {runtime['mean']:.15g} ms/task (sample SD {runtime['sample_sd']:.15g} ms). Raw precision is retained for reproduction, not a claim of corresponding measurement precision.", "",
              "Every invocation resets initial weights, zeroes gradients and momentum, normalizes all raw training/query pixels, copies the resident epoch permutations device-to-device, performs every training update, and predicts all 10,000 query labels. Repeated runs are not inference from cached trained weights. Predictions and final state hashes are checked before/after all measurements and match qualification draw 0.", "",
              "The GPU-resident scope excludes source download/resizing, CPU permutation generation, host/device transfer, allocation, JIT compilation, graph capture, cold start and verification. Energy measures the GPU board through NVML, excluding host CPU energy. Full application energy and end-to-end wall time are unavailable. This submission does not include a spatial-grid placement, word-node-hop model, theoretical grid energy/runtime or host time-to-score; those columns remain unmeasured.", "",
              "## Reproduction and provenance", "",
              "Install `modal` and `numpy` locally and authenticate Modal. The runner pins its container by digest and installs NumPy 2.2.6 and nvidia-ml-py 12.560.30. Use a fresh output directory so the retained qualification files are never overwritten. From the repository root:", "", "```bash",
              "mkdir -p /tmp/rev88-reproduction",
              "cp mnist/submissions/rev88-20260914/{learner.py,energy.py,data_reference.py,run.py,build_report.py} /tmp/rev88-reproduction/",
              'modal run /tmp/rev88-reproduction/run.py --phase validate --raw-dir "$PWD/mnist/data/raw"',
              'modal run /tmp/rev88-reproduction/run.py --phase fit --raw-dir "$PWD/mnist/data/raw"',
              'modal run /tmp/rev88-reproduction/run.py --phase evaluate --raw-dir "$PWD/mnist/data/raw"',
              'python /tmp/rev88-reproduction/build_report.py --raw-dir "$PWD/mnist/data/raw" --rules-file "$PWD/mnist/README.md"',
              "```", "",
              "The `validate` stage uses no qualification-test labels and freezes the selected procedure before `fit`. Source/version hashes, raw MNIST checksums, all sampling indices' hashes, input hashes, seed-initialization hashes, epoch permutation hashes, final parameter/momentum hashes and prediction hashes are retained. The report builder verifies every frozen output before loading evaluation labels, independently replays sampling and preprocessing, rescores all eleven retained prediction arrays and retained training-only validation predictions, and recomputes energy arithmetic. Earlier development validation archives are not qualification evidence. No learned checkpoint or raw dataset is required in the submission directory.", "",
              "Contributors: Yaroslav Bulatov (requirements), Codex (implementation, measurements and report). No W&B run was created. Reproduction uses at most four ephemeral A100 workers as configured in the runner; execution and shutdown are recorded by the submission coordinator.", "",
              "## Evidence", "",
              "[Current challenge rules](../../README.md) · [Machine-readable submission](submission.json) · [Accuracy counts](accuracy.json) · [Frozen protocol](protocol.json) · [Configuration](config.json)", "",
              "[Training-only selection plan](validation-plan.json) · [Selection results](validation.json) · [Draw manifest](draw-manifest.json) · [Prediction freeze](prediction-manifest.json) · [Draw-zero NVML and CUDA validation](results/draw-00.json)", "",
              "[Learner](learner.py) · [NVML measurement](energy.py) · [Runner](run.py) · [Canonical data routines](data_reference.py) · [Report/check generator](build_report.py)", ""]
    extra = [("Independent verification", "verification.json"), ("Independent verifier", "verify.py"), ("Execution and shutdown record", "execution.json")]
    available = [f"[{label}]({name})" for label, name in extra if (HERE/name).exists()]
    if available:
        lines += [" · ".join(available), ""]
    write(HERE/"submission.json", submission)
    write(HERE/"README.md", "\n".join(lines))
    print(f"Submission {'QUALIFIED' if qualified else 'NOT QUALIFIED'}: {accuracy['correct']}/110000; {accuracy['accuracy_percent']:.6f}% ± {accuracy['sample_sd_pp']:.6f} pp")
    print(f"Draw0 measured mean: {e['mean']:.6f} mJ; {runtime['mean']:.6f} CUDA ms; 3 trials")
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=HERE.parents[1]/"data/raw")
    parser.add_argument("--rules-file", type=Path, default=HERE.parents[1]/"README.md")
    args = parser.parse_args()
    build(args.raw_dir, args.rules_file)
