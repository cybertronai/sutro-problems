"""Independently rescore saved predictions and recompute A100 measurements."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from mnist.code import data


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, context):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError(f"{context}: {actual} != {expected}")


def check_window(window):
    """Derive duration and energy from raw stamps and sampled power."""
    start, end = window["start"], window["end"]
    for stamp in (start, end):
        if stamp["query_finished_s"] < stamp["query_started_s"]:
            raise ValueError("NVML query clock moved backwards")
        close(stamp["time_s"], (stamp["query_started_s"] + stamp["query_finished_s"]) / 2,
              "NVML timestamp midpoint")
    t0, t1 = start["time_s"], end["time_s"]
    seconds = t1 - t0
    joules = (end["energy_mj"] - start["energy_mj"]) / 1000
    if seconds <= 0 or joules < 0:
        raise ValueError("Counter or clock moved backwards")
    close(window["duration_s"], seconds, "Window duration")
    close(window["counter_energy_j"], joules, "Counter energy")
    close(window["counter_power_w"], joules / seconds, "Counter power")
    samples = np.asarray(window["raw_power_samples_time_s_w"], dtype=np.float64)
    if (samples.ndim != 2 or samples.shape[1] != 2 or len(samples) < 2
            or not np.isfinite(samples).all() or np.any(samples[:, 1] < 0)
            or np.any(np.diff(samples[:, 0]) <= 0)
            or samples[0, 0] > t0 or samples[-1, 0] < t1):
        raise ValueError("Invalid power samples or missing interval boundaries")
    times = np.concatenate(([t0], samples[(samples[:, 0] > t0) & (samples[:, 0] < t1), 0], [t1]))
    watts = np.interp(times, samples[:, 0], samples[:, 1])
    sampled_joules = float(np.sum(np.diff(times) * (watts[:-1] + watts[1:]) / 2))
    close(window["sampled_energy_j"], sampled_joules, "Integrated power")
    close(window["sampled_power_w"], sampled_joules / seconds, "Sampled power")


def check_a100(raw_dir, results_dir):
    """Verify two physical hosts, eleven-draw predictions, and energy arithmetic."""
    paths = sorted([*results_dir.glob("*.json"), *results_dir.glob("*.json.gz")])
    if len(paths) != 2:
        raise ValueError("Expected exactly two A100 host JSON records")
    raw_paths = {kind: raw_dir / data.SOURCES[kind][0]
                 for kind in ("train_images", "train_labels")}
    for kind, path in raw_paths.items():
        if data.file_hash(path, "md5") != data.SOURCES[kind][1]:
            raise ValueError(f"Raw dataset hash mismatch: {path.name}")
    pixels = data.read_idx(raw_paths["train_images"], 60000, True)
    labels = data.read_idx(raw_paths["train_labels"], 60000, False)
    config = json.loads((HERE / "config.json").read_text())
    expected_sources = {name: file_hash(HERE / name) for name in
                        ("gpu_benchmark.py", "learner.py", "config.json", "protocol.json", "requirements-gpu.txt")}
    dataset_cache = {}

    def check_predictions(dataset, accuracy, context):
        if (dataset["data_helper_sha256"] != file_hash(ROOT / "mnist/code/data.py")
                or dataset["source_sha256"] != {key: file_hash(value) for key, value in raw_paths.items()}
                or dataset["train_examples"] != 10000 or dataset["test_examples"] != 10000):
            raise ValueError(f"A100 raw dataset identity differs: {context}")
        seed = dataset["dataset_seed"]
        if seed not in dataset_cache:
            order = np.random.Generator(np.random.PCG64(seed)).permutation(60000)
            train, query = order[:10000], order[10000:20000]
            resize = lambda rows: data.area_resize(
                pixels[rows].astype(np.float32) / np.float32(255), 9
            ).reshape(10000, 81)
            arrays = {"train_indices": train, "query_indices": query,
                      "train_pixels": resize(train), "train_labels": labels[train],
                      "query_pixels": resize(query), "test_labels": labels[query]}
            dataset_cache[seed] = ({key: array_hash(value) for key, value in arrays.items()}, labels[query])
        hashes, truth = dataset_cache[seed]
        if dataset["array_sha256"] != hashes:
            raise ValueError(f"A100 reconstructed input differs: {context}")
        predictions_path = results_dir / accuracy["prediction_file"]
        if file_hash(predictions_path) != accuracy["prediction_file_sha256"]:
            raise ValueError(f"A100 prediction file hash differs: {context}")
        predictions = np.load(predictions_path, allow_pickle=False)
        if (predictions.shape != (10000,) or predictions.dtype != np.int64
                or np.any((predictions < 0) | (predictions > 9))
                or array_hash(predictions) != accuracy["prediction_sha256"]
                or accuracy["all_scores_finite"] is not True):
            raise ValueError(f"Invalid A100 predictions: {context}")
        correct = int(np.count_nonzero(predictions == truth))
        if accuracy["correct"] != correct or accuracy["total"] != 10000:
            raise ValueError(f"A100 accuracy count differs: {context}")
        return correct

    def array_hash(value):
        return hashlib.sha256(value.tobytes(order="C")).hexdigest()

    hosts, uuids, machine_ids = [], set(), set()
    for path in paths:
        with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
            doc = json.load(stream)
        if doc["status"] != "complete" or doc["format_version"] != 1:
            raise ValueError(f"Incomplete or unsupported benchmark: {path.name}")
        if doc["source_sha256"] != expected_sources:
            raise ValueError(f"Frozen A100 source changed: {path.name}")
        if doc["configuration"] != {**config, "device": "cuda", "conv_batch_size": 10000}:
            raise ValueError(f"A100 learner configuration differs: {path.name}")
        hardware = doc["hardware"]
        normalize_pci = lambda text: tuple(int(part, 16) for part in text.replace(".", ":").split(":"))
        if ("A100" not in hardware["gpu"] or hardware["nvml_selected_by_cuda_pci_id"] is not True
                or normalize_pci(hardware["cuda_pci_bus_id"]) != normalize_pci(hardware["nvml_pci_bus_id"])):
            raise ValueError(f"A100 CUDA/NVML device identity differs: {path.name}")
        machine_id = doc["vast"]["machine_id"]
        if hardware["uuid"] in uuids or machine_id in machine_ids:
            raise ValueError("Measurements must use two distinct physical hosts and GPUs")
        uuids.add(hardware["uuid"])
        machine_ids.add(machine_id)
        software = doc["software"]
        if any(software[key] is not False for key in ("matmul_tf32", "cudnn_tf32", "cudnn_benchmark")):
            raise ValueError(f"A100 math settings differ: {path.name}")
        dataset, accuracy = doc["dataset"], doc["accuracy"]
        if accuracy["warm_and_captured_predictions_equal"] is not True:
            raise ValueError(f"A100 eager and graph predictions differ: {path.name}")
        correct = check_predictions(dataset, accuracy, path.name)
        qualification = doc["qualification"]
        seeds = list(range(2026091600, 2026091611))
        if (qualification["complete"] is not True or doc["measurement_arguments"]["qualify_all"] is not True
                or qualification["dataset_seeds"] != seeds or len(qualification["draws"]) != 11):
            raise ValueError(f"Expected all eleven frozen GPU qualification draws: {path.name}")
        counts = []
        for index, row in enumerate(qualification["draws"]):
            if row["draw"] != index or row["seed"] != seeds[index] or row["dataset"]["dataset_seed"] != seeds[index]:
                raise ValueError(f"GPU qualification draw identity differs: {path.name}, draw {index}")
            if row["warm_and_captured_predictions_equal"] is not True:
                raise ValueError(f"GPU eager and graph predictions differ: {path.name}, draw {index}")
            if row["seed"] == dataset["dataset_seed"] and (
                    row["matches_measured_draw_predictions"] is not True
                    or row["prediction_sha256"] != accuracy["prediction_sha256"]):
                raise ValueError(f"GPU qualification and measured predictions differ: {path.name}")
            counts.append(check_predictions(row["dataset"], row, f"{path.name}, draw {index}"))
        total = sum(counts)
        if (qualification["total_correct"] != total or qualification["total_predictions"] != 110000
                or qualification["minimum_correct_for_2_percent"] != 107800
                or qualification["meets_2_percent_target"] != (total >= 107800)):
            raise ValueError(f"GPU qualification aggregate differs: {path.name}")
        close(qualification["mean_accuracy"], total / 110000, "GPU qualification accuracy")
        close(qualification["sample_sd_percentage_points"], statistics.stdev(count / 100 for count in counts),
              "GPU qualification sample standard deviation")
        rounds = doc["rounds"]
        if len(rounds) != doc["measurement_arguments"]["rounds"] or not rounds:
            raise ValueError(f"Missing A100 rounds: {path.name}")
        for index, row in enumerate(rounds):
            if (row["round"] != index or row["tasks"] != doc["calibration"]["repeats_per_round"]
                    or row["tasks"] <= 0 or row["predictions_and_score_hashes_equal"] is not True):
                raise ValueError(f"Invalid A100 round identity: {path.name}")
            for key in ("idle_before", "active", "idle_after"):
                check_window(row[key])
            if (row["idle_before"]["end"]["time_s"] > row["active"]["start"]["time_s"]
                    or row["active"]["end"]["time_s"] > row["idle_after"]["start"]["time_s"]):
                raise ValueError("Idle windows do not bracket the active measurement")
            seconds = row["active"]["duration_s"]
            close(row["task_ms"], seconds * 1000 / row["tasks"], "Task time")
            for kind in ("counter", "sampled"):
                idle_w = (row["idle_before"][f"{kind}_power_w"] + row["idle_after"][f"{kind}_power_w"]) / 2
                gross = row["active"][f"{kind}_energy_j"] * 1000 / row["tasks"]
                adjusted = gross - idle_w * seconds * 1000 / row["tasks"]
                close(row[f"paired_idle_w_{kind}"], idle_w, "Paired idle power")
                close(row[f"gross_mj_{kind}"], gross, "Gross task energy")
                close(row[f"adjusted_mj_{kind}"], adjusted, "Adjusted task energy")
        for key in ("task_ms", "gross_mj_counter", "gross_mj_sampled", "adjusted_mj_counter", "adjusted_mj_sampled"):
            close(doc["summary"][key], statistics.median(row[key] for row in rounds), f"Host median {key}")
        expected_spread = [min(row["adjusted_mj_counter"] for row in rounds),
                           max(row["adjusted_mj_counter"] for row in rounds)]
        if doc["adjusted_spread_mj_counter"] != expected_spread:
            raise ValueError(f"A100 energy spread differs: {path.name}")
        sensor = doc.get("sensor_check")
        if sensor:
            for key in ("idle_before", "active", "idle_after"):
                check_window(sensor[key])
            active = sensor["active"]
            difference = abs(active["counter_power_w"] - active["sampled_power_w"]) / active["sampled_power_w"]
            close(sensor["counter_sampled_power_relative_difference"], difference, "Sensor power agreement")
            if sensor["original_diagnostic_pass"] != (active["counter_power_w"] > 150 and difference < 0.1):
                raise ValueError(f"A100 sensor diagnostic result differs: {path.name}")
        hosts.append({"evidence": path.name, "machine_id": machine_id, "gpu_uuid": hardware["uuid"],
                      "measurement_seed": dataset["dataset_seed"], "measurement_correct": correct,
                      "qualification_total_correct": total, "qualification_total_predictions": 110000,
                      "qualification_mean_accuracy_percent": total / 1100,
                      "qualification_sample_sd_percentage_points": qualification["sample_sd_percentage_points"],
                      "meets_2_percent_target": total >= 107800,
                      "rounds": len(rounds), "tasks_per_round": doc["calibration"]["repeats_per_round"],
                      "adjusted_energy_mj": doc["summary"]["adjusted_mj_counter"],
                      "runtime_ms": doc["summary"]["task_ms"],
                      "sensor_diagnostic_pass": sensor["original_diagnostic_pass"] if sensor else None})
    return {"hosts": hosts, "distinct_physical_hosts_verified": True,
            "source_data_prediction_hashes_verified": True,
            "counter_and_sampled_energy_recomputed": True,
            "cross_host_mean_of_medians": {
                "adjusted_energy_mj": statistics.mean(host["adjusted_energy_mj"] for host in hosts),
                "runtime_ms": statistics.mean(host["runtime_ms"] for host in hosts)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="Directory containing the canonical MNIST training IDX gzip files")
    parser.add_argument("--a100-dir", type=Path, default=HERE / "evidence/a100",
                        help="Directory containing two A100 .json or .json.gz records and their predictions")
    args = parser.parse_args()
    if np.__version__ != "2.1.2":
        parser.error("Use NumPy 2.1.2 to reproduce dataset hashes")
    result = {"a100": check_a100(args.raw_dir, args.a100_dir)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
