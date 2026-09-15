#!/usr/bin/env python3
"""Remeasure the frozen F512/T300 CG pair on an A100 using full-batch CUDA graphs.

Run from the repository checkout, with canonical MNIST IDX gzip files available::

    python gpu_benchmark.py --raw-dir /path/to/raw --output /tmp/cg-a100.json

The default timing seed is the first frozen qualification draw, 2026091600.
Use --qualify-all to evaluate all eleven frozen draws with fresh CUDA graphs.
Each graph replay encodes training labels, transforms all pixels, forms both
systems, solves each from zero for 300 CG iterations, and predicts all queries.
Filter generation, host/device transfers, allocation, graph capture, source
resizing, and verification are outside timing.
The data helper is the repository's mnist/code/data.py. No imported benchmark
module is executed. --help needs only the Python standard library.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import socket
import statistics
import sys
import threading
import time


HERE = Path(__file__).resolve().parent
SCOPE = {
    "included": [
        "+/-1 target encoding from resident integer training labels",
        "clamp and asin(sqrt) transform of resident training/query pixels",
        "full-batch convolution, ReLU, pooling and feature centering",
        "feature Gram/RHS and training/query RBF matrix construction",
        "two fresh zero-initialized FP32 Jacobi CG solves, 300 iterations each",
        "score standardization, equal sum, argmax and final device prediction copy",
        "CUDA graph replay submission and synchronization in wall timing",
    ],
    "excluded": [
        "MNIST download and 28x28-to-9x9 dataset preprocessing",
        "input host/device transfers",
        "seed-only random filter generation and filter upload",
        "allocation, compilation, graph capture, warmup and cold start",
        "output device/host transfers and validation",
        "host CPU energy and sensor diagnostic workload",
    ],
    "energy": "GPU board energy from NVML, adjusted by paired settled idle power",
    "time": "synchronized wall time per complete GPU-resident task",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def normalize_pci(value):
    domain, bus, device_function = decode(value).lower().split(":")
    device, function = device_function.split(".")
    return tuple(int(x, 16) for x in (domain, bus, device, function))


def cuda_pci_bus_id(torch):
    """Ask the CUDA runtime, so CUDA_VISIBLE_DEVICES remapping is respected."""
    site = Path(torch.__file__).resolve().parent.parent
    candidates = [str(p) for p in (site / "nvidia/cuda_runtime/lib").glob("libcudart.so*")]
    candidates += [str(p) for p in Path("/usr/local/cuda/lib64").glob("libcudart.so*")]
    located = ctypes.util.find_library("cudart")
    candidates += ([located] if located else []) + ["libcudart.so.12"]
    errors = []
    for candidate in dict.fromkeys(candidates):
        try:
            runtime = ctypes.CDLL(candidate)
            function = runtime.cudaDeviceGetPCIBusId
            function.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
            function.restype = ctypes.c_int
            result = ctypes.create_string_buffer(32)
            status = function(result, len(result), torch.cuda.current_device())
            if status != 0:
                raise RuntimeError(f"CUDA returned status {status}")
            return result.value.decode()
        except (OSError, AttributeError, RuntimeError) as error:
            errors.append(f"{candidate}: {error}")
    raise RuntimeError("Cannot verify CUDA/NVML PCI identity: " + "; ".join(errors))


class PowerSampler(threading.Thread):
    """Retain timestamped NVML samples, including both interval boundary points."""

    def __init__(self, nv, handle, hz):
        super().__init__(daemon=True)
        self.nv, self.handle, self.period = nv, handle, 1.0 / hz
        self.samples = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.error = None

    def run(self):
        try:
            while not self.stop_event.is_set():
                before = time.perf_counter()
                watts = self.nv.nvmlDeviceGetPowerUsage(self.handle) / 1000
                after = time.perf_counter()
                with self.lock:
                    self.samples.append(((before + after) / 2, watts))
                self.stop_event.wait(self.period)
        except Exception as error:
            self.error = error

    def wait_through(self, target):
        deadline = time.perf_counter() + 5
        while True:
            if self.error is not None:
                raise RuntimeError("NVML power sampling failed") from self.error
            with self.lock:
                if self.samples and self.samples[-1][0] >= target:
                    return
            if time.perf_counter() >= deadline:
                raise RuntimeError("Timed out waiting for an NVML power sample")
            time.sleep(min(self.period / 2, 0.01))

    def interval(self, t0, t1):
        # Waiting for the trailing sample occurs outside the measured window.
        self.wait_through(t1)
        with self.lock:
            points = list(self.samples)
        first = next((i for i, p in enumerate(points) if p[0] >= t0), None)
        last = next((i for i, p in enumerate(points) if p[0] >= t1), None)
        if first is None or last is None or first == 0:
            raise RuntimeError("Power samples do not bracket the measured interval")
        raw = points[first - 1:last + 1]

        def at(t, left, right):
            return left[1] + (right[1] - left[1]) * (t - left[0]) / (right[0] - left[0])

        clipped = [(t0, at(t0, raw[0], raw[1]))]
        clipped += [p for p in raw if t0 < p[0] < t1]
        clipped.append((t1, at(t1, raw[-2], raw[-1])))
        joules = sum(
            (b[0] - a[0]) * (a[1] + b[1]) / 2
            for a, b in zip(clipped, clipped[1:])
        )
        return {"raw_power_samples_time_s_w": raw, "sampled_energy_j": joules,
                "sampled_power_w": joules / (t1 - t0),
                "integration": "trapezoids with linearly interpolated boundary samples"}


class Meter:
    def __init__(self, nv, torch, handle, sampler):
        self.nv, self.torch, self.handle, self.sampler = nv, torch, handle, sampler

    def stamp(self):
        before = time.perf_counter()
        energy = int(self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle))
        after = time.perf_counter()
        return {"energy_mj": energy, "time_s": (before + after) / 2,
                "query_started_s": before, "query_finished_s": after}

    def window(self, fn=None, *, seconds=None):
        self.torch.cuda.synchronize()
        first = self.stamp()
        if fn is None:
            time.sleep(seconds)
        else:
            fn()
        self.torch.cuda.synchronize()
        last = self.stamp()
        duration = last["time_s"] - first["time_s"]
        energy = (last["energy_mj"] - first["energy_mj"]) / 1000
        if duration <= 0 or energy < 0:
            raise RuntimeError("NVML counter or clock moved backwards")
        return {"start": first, "end": last, "duration_s": duration,
                "counter_energy_j": energy, "counter_power_w": energy / duration,
                **self.sampler.interval(first["time_s"], last["time_s"])}

    def idle(self, settle_seconds, seconds):
        self.torch.cuda.synchronize()
        time.sleep(settle_seconds)
        return {"settle_seconds": settle_seconds, **self.window(seconds=seconds)}


def load_draw(raw_dir, seed):
    import numpy as np
    sys.path.insert(0, str(HERE.parents[2]))
    from mnist.code import data as ds

    paths = {}
    for key in ("train_images", "train_labels"):
        filename, md5 = ds.SOURCES[key]
        path = raw_dir / filename
        if not path.is_file() or ds.file_hash(path, "md5") != md5:
            raise ValueError(f"Missing or invalid canonical MNIST source: {path}")
        paths[key] = path
    pixels = ds.read_idx(paths["train_images"], 60000, True)
    labels = ds.read_idx(paths["train_labels"], 60000, False)
    order = np.random.Generator(np.random.PCG64(seed)).permutation(60000)
    train, query = order[:10000], order[10000:20000]
    resize = lambda rows: ds.area_resize(
        pixels[rows].astype(np.float32) / np.float32(255), 9
    ).reshape(10000, 81)
    x, y, q, truth = resize(train), labels[train], resize(query), labels[query]
    evidence = {
        "dataset_seed": seed, "train_examples": 10000, "test_examples": 10000,
        "sampling": "PCG64(seed).permutation(60000): train[:10000], query[10000:20000]",
        "preprocessing": "FP32 divide by 255, exact separable box-area resize to 9x9",
        "data_helper_sha256": sha256(ds.__file__),
        "source_sha256": {k: sha256(v) for k, v in paths.items()},
        "array_sha256": {k: array_sha256(v) for k, v in
                         {"train_indices": train, "query_indices": query,
                          "train_pixels": x, "train_labels": y,
                          "query_pixels": q, "test_labels": truth}.items()},
    }
    return x, y, q, truth, evidence


def snapshot(torch, predictions, scores):
    score_hashes = []
    for tensor in scores:
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("CG pair produced nonfinite scores")
        score_hashes.append(array_sha256(tensor.cpu().numpy()))
    values = predictions.cpu().numpy().copy()
    if values.shape != (10000,) or values.min() < 0 or values.max() > 9:
        raise RuntimeError("Invalid prediction labels or shape")
    return values, {"prediction_sha256": array_sha256(values),
                    "score_sha256": score_hashes, "all_scores_finite": True}


def capture_task(torch, learner, x, y, q):
    """Keep graph inputs alive and capture the same fresh fit for every draw."""
    import numpy as np

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    model = learner.CGPair(x, y, q, device="cuda", conv_batch_size=10000)
    labels = torch.tensor(y, dtype=torch.int64, device="cuda")

    def fresh_task():
        # Encoding depends on this task's training labels, so include its kernels.
        model.Y.fill_(-1)
        model.Y.scatter_(1, labels[:, None], 1.0)
        return model.run(return_scores=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            warm_predictions, warm_scores = fresh_task()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    warm_values, _ = snapshot(torch, warm_predictions, warm_scores)
    del warm_predictions, warm_scores
    graph = torch.cuda.CUDAGraph()
    predictions = torch.empty(10000, dtype=torch.int64, device="cuda")
    with torch.cuda.graph(graph, stream=stream):
        captured_predictions, captured_scores = fresh_task()
        predictions.copy_(captured_predictions)
    graph.replay()
    torch.cuda.synchronize()
    values, state = snapshot(torch, predictions, captured_scores)
    np.testing.assert_array_equal(values, warm_values)
    memory = {
        "scope": "fresh model preparation, three eager warmups, graph capture and first replay; excludes sensor diagnostic",
        "allocated_before_bytes": allocated_before,
        "reserved_before_bytes": reserved_before,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "resident_allocated_bytes": torch.cuda.memory_allocated(),
        "resident_reserved_bytes": torch.cuda.memory_reserved(),
        "measurement": "PyTorch CUDA allocator; driver/context allocations are not included",
    }
    return {"model": model, "labels": labels,
            "graph": graph, "predictions": predictions, "scores": captured_scores,
            "values": values, "state": state, "memory": memory}


def qualification_draw(torch, learner, raw_dir, output, draw, seed):
    import numpy as np

    x, y, q, truth, dataset = load_draw(raw_dir, seed)
    task = capture_task(torch, learner, x, y, q)
    path = output.with_name(f"{output.stem}.draw_{draw:02d}.predictions.npy")
    with path.open("xb") as stream:
        np.save(stream, task["values"], allow_pickle=False)
    correct = int((task["values"] == truth).sum())
    return {"draw": draw, "seed": seed, "dataset": dataset,
            "correct": correct, "total": 10000,
            "prediction_file": path.name, "prediction_file_sha256": sha256(path),
            "warm_and_captured_predictions_equal": True, **task["state"]}


def gpu_details(nv, torch, handle, pci):
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    nvml_pci = decode(nv.nvmlDeviceGetPciInfo(handle).busId)
    if normalize_pci(pci) != normalize_pci(nvml_pci):
        raise RuntimeError("CUDA and NVML PCI bus IDs disagree")
    name = decode(nv.nvmlDeviceGetName(handle))
    if "A100" not in name or "A100" not in properties.name:
        raise RuntimeError(f"This benchmark requires an A100; found {name}")
    return {
        "hostname": socket.gethostname(), "gpu": name,
        "uuid": decode(nv.nvmlDeviceGetUUID(handle)),
        "driver": decode(nv.nvmlSystemGetDriverVersion()),
        "cuda_device_index": torch.cuda.current_device(),
        "cuda_pci_bus_id": pci, "nvml_pci_bus_id": nvml_pci,
        "nvml_selected_by_cuda_pci_id": True,
        "total_memory_bytes": properties.total_memory,
        "power_limit_w": nv.nvmlDeviceGetPowerManagementLimit(handle) / 1000,
        "compute_process_ids": [p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-seed", type=int, default=2026091600)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--target-seconds", type=float, default=15)
    parser.add_argument("--idle-seconds", type=float, default=10)
    parser.add_argument("--settle-seconds", type=float, default=3)
    parser.add_argument("--sample-hz", type=float, default=50)
    parser.add_argument("--sensor-seconds", type=float, default=20,
                        help="dense-matmul sensor diagnostic duration; zero skips it")
    parser.add_argument("--qualify-all", action="store_true",
                        help="evaluate all eleven frozen protocol draws with fresh full-batch CUDA graphs")
    args = parser.parse_args(argv)
    if args.rounds < 1 or min(args.target_seconds, args.idle_seconds, args.sample_hz) <= 0:
        parser.error("rounds, target/idle seconds and sample Hz must be positive")
    if args.settle_seconds < 0 or args.sensor_seconds < 0:
        parser.error("settle/sensor seconds must be nonnegative")
    partial = args.output.with_name(args.output.stem + ".partial.json")
    prediction_path = args.output.with_name(args.output.stem + ".predictions.npy")
    qualification_paths = [args.output.with_name(f"{args.output.stem}.draw_{draw:02d}.predictions.npy")
                           for draw in range(11)] if args.qualify_all else []
    if any(p.exists() for p in (args.output, partial, prediction_path, *qualification_paths)):
        parser.error("choose a fresh output prefix; existing evidence is never overwritten")
    protocol = json.loads((HERE / "protocol.json").read_text())
    seeds = list(range(2026091600, 2026091611))
    if protocol["dataset_seeds"] != seeds or protocol["planned_draws"] != 11:
        raise ValueError("Expected the eleven prespecified qualification seeds")

    import numpy as np
    import torch
    import pynvml as nv
    import learner

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script on an A100 host")
    expected = {"filters": 512, "iterations": 300, "filter_seed": 0,
                "ridge_lambda_mean_diagonal": 1e-3, "rbf_gamma": 0.3, "rbf_lambda": 1e-2}
    for key, value in expected.items():
        if learner.CONFIG[key] != value:
            raise RuntimeError(f"Frozen learner configuration changed: {key}")
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    nv.nvmlInit()
    sampler = None
    try:
        pci = cuda_pci_bus_id(torch)
        handle = nv.nvmlDeviceGetHandleByPciBusId(pci.encode())
        hardware = gpu_details(nv, torch, handle, pci)
        nv.nvmlDeviceGetTotalEnergyConsumption(handle)  # Fail before expensive work if unsupported.
        x, y, q, truth, dataset = load_draw(args.raw_dir, args.dataset_seed)
        doc = {
            "format_version": 1, "started_utc": utc_now(), "status": "preparing",
            "hardware": hardware, "dataset": dataset, "scope": SCOPE,
            "configuration": {**learner.CONFIG, "device": "cuda", "conv_batch_size": 10000},
            "measurement_arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "software": {"python": platform.python_version(), "platform": platform.platform(),
                         "torch": str(torch.__version__), "cuda": torch.version.cuda,
                         "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__,
                         "nvidia_ml_py": importlib.metadata.version("nvidia-ml-py"),
                         "matmul_tf32": False, "cudnn_tf32": False, "cudnn_benchmark": False},
            "source_sha256": {p.name: sha256(p) for p in
                              (Path(__file__), HERE / "learner.py", HERE / "config.json",
                               HERE / "protocol.json", HERE / "requirements-gpu.txt")},
            "rounds": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(partial, doc, exclusive=True)
        with torch.no_grad():
            print("Warming and capturing full-batch fresh train-and-predict task", flush=True)
            task = capture_task(torch, learner, x, y, q)
            graph, predictions, captured_scores = task["graph"], task["predictions"], task["scores"]
            baseline_predictions, baseline = task["values"], task["state"]
            doc["gpu_memory"] = task["memory"]
            with prediction_path.open("xb") as output_stream:
                np.save(output_stream, baseline_predictions, allow_pickle=False)
            doc["accuracy"] = {"correct": int((baseline_predictions == truth).sum()),
                               "total": 10000, "prediction_file": prediction_path.name,
                               "prediction_file_sha256": sha256(prediction_path),
                               "warm_and_captured_predictions_equal": True, **baseline}
            start = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            calibration = time.perf_counter() - start
            repeats = max(1, math.ceil(args.target_seconds / calibration))
            doc["calibration"] = {"wall_seconds": calibration, "repeats_per_round": repeats}
            sampler = PowerSampler(nv, handle, args.sample_hz)
            sampler.start()
            sampler.wait_through(time.perf_counter())
            meter = Meter(nv, torch, handle, sampler)
            if args.sensor_seconds:
                print("Checking NVML counter against sampled power under dense matmul", flush=True)
                dense = torch.randn(8192, 8192, device="cuda")
                dense_output = torch.empty_like(dense)

                def burn():
                    deadline = time.perf_counter() + args.sensor_seconds
                    while time.perf_counter() < deadline:
                        for _ in range(10):
                            torch.mm(dense, dense, out=dense_output)
                        torch.cuda.synchronize()

                idle_before = meter.idle(args.settle_seconds, args.idle_seconds)
                load = meter.window(burn)
                idle_after = meter.idle(args.settle_seconds, args.idle_seconds)
                discrepancy = abs(load["counter_power_w"] - load["sampled_power_w"]) / load["sampled_power_w"]
                doc["sensor_check"] = {
                    "idle_before": idle_before, "active": load, "idle_after": idle_after,
                    "counter_sampled_power_relative_difference": discrepancy,
                    "original_diagnostic_pass": load["counter_power_w"] > 150 and discrepancy < 0.1,
                    "criterion": "original diagnostic: loaded counter power >150 W, power agreement <10%",
                }
                del dense, dense_output

            def repeated_tasks():
                for _ in range(repeats):
                    graph.replay()

            for round_id in range(args.rounds):
                idle_before = meter.idle(args.settle_seconds, args.idle_seconds)
                active = meter.window(repeated_tasks)
                idle_after = meter.idle(args.settle_seconds, args.idle_seconds)
                values, state = snapshot(torch, predictions, captured_scores)
                np.testing.assert_array_equal(values, baseline_predictions)
                if state != baseline:
                    raise RuntimeError("Repeated graph task changed final scores or predictions")
                idle_counter = (idle_before["counter_power_w"] + idle_after["counter_power_w"]) / 2
                idle_sampled = (idle_before["sampled_power_w"] + idle_after["sampled_power_w"]) / 2
                seconds = active["duration_s"]
                row = {
                    "round": round_id, "tasks": repeats,
                    "idle_before": idle_before, "active": active, "idle_after": idle_after,
                    "paired_idle_w_counter": idle_counter, "paired_idle_w_sampled": idle_sampled,
                    "task_ms": seconds * 1000 / repeats,
                    "gross_mj_counter": active["counter_energy_j"] * 1000 / repeats,
                    "gross_mj_sampled": active["sampled_energy_j"] * 1000 / repeats,
                    "adjusted_mj_counter": (active["counter_energy_j"] - idle_counter * seconds) * 1000 / repeats,
                    "adjusted_mj_sampled": (active["sampled_energy_j"] - idle_sampled * seconds) * 1000 / repeats,
                    "predictions_and_score_hashes_equal": True,
                }
                doc["rounds"].append(row)
                doc["status"] = "measuring"
                write_json(partial, doc)
                print(f"Round {round_id + 1}/{args.rounds}: {row['adjusted_mj_counter']:.2f} mJ/task, "
                      f"{row['task_ms']:.3f} ms/task", flush=True)
            keys = ("task_ms", "gross_mj_counter", "gross_mj_sampled",
                    "adjusted_mj_counter", "adjusted_mj_sampled")
            doc["summary"] = {key: statistics.median(r[key] for r in doc["rounds"]) for key in keys}
            doc["summary_statistic"] = "median across measurement rounds on this host and dataset"
            doc["adjusted_spread_mj_counter"] = [
                min(r["adjusted_mj_counter"] for r in doc["rounds"]),
                max(r["adjusted_mj_counter"] for r in doc["rounds"]),
            ]
            if args.qualify_all:
                sampler.stop_event.set()
                sampler.join(timeout=5)
                sampler = None
                del task, graph, predictions, captured_scores
                torch.cuda.empty_cache()
                qualification = {
                    "execution": "fresh full-batch CUDA graph per dataset; same capture_task as timed task",
                    "dataset_seeds": seeds, "draws": [], "complete": False,
                }
                doc["qualification"] = qualification
                for draw, seed in enumerate(seeds):
                    print(f"Qualifying draw {draw:02d}, seed {seed}", flush=True)
                    row = qualification_draw(torch, learner, args.raw_dir, args.output, draw, seed)
                    if seed == args.dataset_seed:
                        if row["prediction_sha256"] != baseline["prediction_sha256"]:
                            raise RuntimeError("Fresh qualification predictions differ from the measured draw")
                        row["matches_measured_draw_predictions"] = True
                    qualification["draws"].append(row)
                    doc["status"] = "qualifying"
                    write_json(partial, doc)
                    print(f"Draw {draw:02d}: {row['correct']}/10000", flush=True)
                    torch.cuda.empty_cache()
                correct = sum(row["correct"] for row in qualification["draws"])
                qualification.update({
                    "complete": True, "total_correct": correct, "total_predictions": 110000,
                    "mean_accuracy": correct / 110000,
                    "sample_sd_percentage_points": statistics.stdev(
                        row["correct"] / 100 for row in qualification["draws"]),
                    "minimum_correct_for_2_percent": 107800,
                    "meets_2_percent_target": correct >= 107800,
                })
            doc["finished_utc"] = utc_now()
            doc["status"] = "complete"
            write_json(args.output, doc, exclusive=True)
            partial.unlink()
            print(json.dumps(doc["summary"], indent=2), flush=True)
    finally:
        if sampler is not None:
            sampler.stop_event.set()
            sampler.join(timeout=5)
        nv.nvmlShutdown()


if __name__ == "__main__":
    main()
