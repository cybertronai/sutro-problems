"""Bounded, exact MNIST-small 1NN benchmark on one Modal A100-40GB.

Run from the repository root (Modal credentials must already be configured):
  uvx --with numpy==2.2.6 modal run \
    mnist/submissions/1nn-v4-20260911/gpu_benchmark.py \
    --data mnist/data/small.npz

Only train_images, train_labels, and test_images are loaded/transmitted. Each
timed invocation copies training data into preallocated model memory and then
predicts every test label. The repeated invocation is captured in a CUDA graph.
Host/device transfer, allocation, JIT compilation, validation, graph capture,
and cold start are outside the reported steady-state timing and energy.
"""

from pathlib import Path
import hashlib
import json

import modal


HERE = Path(__file__).resolve().parent
IMAGE_REF = (
    "ghcr.io/ab-10/wikitext-bench@"
    "sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f"
)
image = modal.Image.from_registry(IMAGE_REF).apt_install("gcc").pip_install(
    "numpy==2.2.6", "nvidia-ml-py==12.560.30"
)
app = modal.App("sutro-mnist-small-1nn-v4")


@app.function(
    image=image, gpu="A100-40GB", cpu=4, memory=8192, timeout=300,
    startup_timeout=300, min_containers=0, max_containers=1,
    buffer_containers=0, scaledown_window=2, retries=0,
)
def benchmark(payload: dict, source_sha256: str):
    # Triton resolves annotation names in the JIT function's module globals.
    global tl
    import importlib.metadata
    import os
    import platform
    import statistics
    import subprocess
    import time

    import numpy as np
    import pynvml as nv
    import torch
    import triton
    import triton.language as tl

    torch.set_num_threads(4)

    @triton.jit
    def memorize_kernel(X, Y, MODEL_X, MODEL_Y, N: tl.constexpr,
                        D: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        pixels = tl.load(X + offsets, offsets < N * D, other=0.0)
        tl.store(MODEL_X + offsets, pixels, offsets < N * D)
        labels = tl.load(Y + offsets, offsets < N, other=0)
        tl.store(MODEL_Y + offsets, labels, offsets < N)

    @triton.jit
    def predict_kernel(MODEL_X, MODEL_Y, Q, OUT, INDEX, DISTANCE,
                       N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
        query = tl.program_id(0)
        candidates = tl.arange(0, BLOCK)
        distance = tl.full((BLOCK,), 0.0, tl.float32)
        for feature in tl.static_range(D):
            q = tl.load(Q + query * D + feature)
            x = tl.load(MODEL_X + candidates * D + feature,
                        candidates < N, other=0.0)
            difference = q - x
            square = difference * difference
            distance = distance + square
        distance = tl.where(candidates < N, distance, float("inf"))
        best_distance = tl.min(distance, axis=0)
        best_index = tl.min(tl.where(distance == best_distance, candidates, N), axis=0)
        label = tl.load(MODEL_Y + best_index)
        tl.store(OUT + query, label)
        tl.store(INDEX + query, best_index)
        tl.store(DISTANCE + query, best_distance)

    def unpack(name):
        value = payload[name]
        return np.frombuffer(value["bytes"], dtype=value["dtype"]).reshape(value["shape"]).copy()

    train = unpack("train_images").reshape(600, 9)
    labels = unpack("train_labels").astype(np.int32)
    queries = unpack("test_images").reshape(600, 9)
    assert train.dtype == queries.dtype == np.float32
    assert np.isfinite(train).all() and np.isfinite(queries).all()

    def reference(query_array, label_array):
        distances = np.zeros((len(query_array), len(train)), dtype=np.float32)
        for feature in range(9):
            difference = query_array[:, feature, None] - train[None, :, feature]
            distances += difference * difference
        indices = np.argmin(distances, axis=1)
        return label_array[indices], indices, distances[np.arange(len(query_array)), indices]

    x = torch.from_numpy(train).cuda()
    y = torch.from_numpy(labels).cuda()
    q = torch.from_numpy(queries).cuda()
    model_x = torch.empty_like(x)
    model_y = torch.empty_like(y)
    output = torch.empty((600,), dtype=torch.int32, device="cuda")
    index = torch.empty_like(output)
    distance = torch.empty((600,), dtype=torch.float32, device="cuda")

    def invocation():
        memorize_kernel[(triton.cdiv(600 * 9, 1024),)](
            x, y, model_x, model_y, 600, 9, 1024, num_warps=4,
            enable_fp_fusion=False,
        )
        return predict_kernel[(600,)](
            model_x, model_y, q, output, index, distance, 600, 9, 1024,
            num_warps=4, enable_fp_fusion=False,
        )

    compiled_predict = invocation()
    torch.cuda.synchronize()
    ptx = compiled_predict.asm["ptx"]
    # The source arithmetic is sequential FP32; reject a fused implementation.
    assert "fma.rn.f32" not in ptx and "fma.rn.ftz.f32" not in ptx

    def validate(query_array, label_array):
        expected, expected_index, expected_distance = reference(query_array, label_array)
        invocation()
        torch.cuda.synchronize()
        actual = output.cpu().numpy()
        actual_index = index.cpu().numpy()
        actual_distance = distance.cpu().numpy()
        assert np.array_equal(actual, expected)
        assert np.array_equal(actual_index, expected_index)
        assert np.array_equal(actual_distance.view(np.uint32), expected_distance.view(np.uint32))
        return actual.copy(), {
            "prediction_matches": int(np.count_nonzero(actual == expected)),
            "nearest_index_matches": int(np.count_nonzero(actual_index == expected_index)),
            "nearest_distance_bitwise_matches": int(np.count_nonzero(
                actual_distance.view(np.uint32) == expected_distance.view(np.uint32))),
            "total": len(actual),
        }

    predictions, canonical_validation = validate(queries, labels)
    new_queries = np.random.default_rng(20260911).uniform(0, 1, (600, 9)).astype(np.float32)
    q.copy_(torch.from_numpy(new_queries))
    new_predictions, mutation_validation = validate(new_queries, labels)
    mutation_validation["predictions_changed_from_canonical"] = int(np.count_nonzero(
        new_predictions != predictions))
    assert mutation_validation["predictions_changed_from_canonical"] > 0
    changed_labels = ((labels + 1) % 10).astype(np.int32)
    y.copy_(torch.from_numpy(changed_labels))
    changed_predictions, label_mutation_validation = validate(new_queries, changed_labels)
    label_mutation_validation["predictions_changed_from_original_labels"] = int(np.count_nonzero(
        changed_predictions != new_predictions))
    assert label_mutation_validation["predictions_changed_from_original_labels"] == 600
    y.copy_(torch.from_numpy(labels))
    q.copy_(torch.from_numpy(queries))

    # Capture repeated whole invocations, each with an explicit training copy.
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(32):
            invocation()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    invocations_per_graph = 128
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        for _ in range(invocations_per_graph):
            invocation()
    torch.cuda.synchronize()
    start_event, stop_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(100):
        graph.replay()
    stop_event.record()
    stop_event.synchronize()
    calibration_graph_ms = start_event.elapsed_time(stop_event) / 100
    graph_replays = max(1, int(np.ceil(3000 / calibration_graph_ms)))

    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(0)

    def decode(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    def optional(call):
        try:
            return call()
        except nv.NVMLError as error:
            return {"unavailable": str(error)}

    def energy_stamp():
        before = time.perf_counter()
        energy = int(nv.nvmlDeviceGetTotalEnergyConsumption(handle))
        after = time.perf_counter()
        return {"energy_mj": energy, "time_s": (before + after) / 2,
                "query_latency_s": after - before}

    def interval(first, last):
        duration = last["time_s"] - first["time_s"]
        energy = (last["energy_mj"] - first["energy_mj"]) / 1000
        return {"start": first, "end": last, "duration_s": duration,
                "energy_j": energy, "average_power_w": energy / duration}

    def idle_measurement(seconds=2.0):
        torch.cuda.synchronize()
        first = energy_stamp()
        time.sleep(seconds)
        return interval(first, energy_stamp())

    def telemetry():
        return {
            "temperature_c": nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU),
            "power_w": nv.nvmlDeviceGetPowerUsage(handle) / 1000,
            "graphics_clock_mhz": nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_GRAPHICS),
            "memory_clock_mhz": nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_MEM),
            "throttle_reasons_mask": optional(lambda: int(nv.nvmlDeviceGetCurrentClocksThrottleReasons(handle))),
        }

    props = torch.cuda.get_device_properties(0)
    hardware = {
        "gpu_name": decode(nv.nvmlDeviceGetName(handle)),
        "gpu_uuid": decode(nv.nvmlDeviceGetUUID(handle)),
        "gpu_pci_bus_id": decode(nv.nvmlDeviceGetPciInfo(handle).busId),
        "gpu_total_memory_bytes": int(props.total_memory),
        "multiprocessors": int(props.multi_processor_count),
        "compute_capability": f"{props.major}.{props.minor}",
        "power_limit_w": nv.nvmlDeviceGetPowerManagementLimit(handle) / 1000,
        "mig_mode": optional(lambda: list(nv.nvmlDeviceGetMigMode(handle))),
        "compute_processes": optional(lambda: [
            {"pid": p.pid, "used_gpu_memory_bytes": p.usedGpuMemory}
            for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)]),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_model": next((line.split(":", 1)[1].strip()
                           for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), "unknown"),
    }
    versions = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "triton": triton.__version__,
        "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
        "nvidia_driver": decode(nv.nvmlSystemGetDriverVersion()),
        "nvml": decode(nv.nvmlSystemGetNVMLVersion()),
        "nvidia_ml_py": importlib.metadata.version("nvidia-ml-py"),
        "modal": getattr(modal, "__version__", "runtime-injected; package metadata unavailable"),
        "container_image": IMAGE_REF,
    }
    trials = []
    for trial_id in range(3):
        idle_before = idle_measurement()
        telemetry_before = telemetry()
        start = energy_stamp()
        wall_start = time.perf_counter()
        start_event.record()
        for _ in range(graph_replays):
            graph.replay()
        stop_event.record()
        stop_event.synchronize()
        wall_end = time.perf_counter()
        finish = energy_stamp()
        telemetry_after = telemetry()
        active = interval(start, finish)
        idle_after = idle_measurement()
        repeats = graph_replays * invocations_per_graph
        idle_power = (idle_before["average_power_w"] + idle_after["average_power_w"]) / 2
        adjusted = active["energy_j"] - idle_power * active["duration_s"]
        gpu_duration_s = start_event.elapsed_time(stop_event) / 1000
        trials.append({
            "trial": trial_id + 1, "graph_replays": graph_replays,
            "invocations_per_graph": invocations_per_graph, "invocations": repeats,
            "idle_before": idle_before, "idle_after": idle_after, "active": active,
            "telemetry_before": telemetry_before, "telemetry_after": telemetry_after,
            "paired_idle_power_w": idle_power,
            "idle_adjusted_energy_j": adjusted,
            "idle_adjusted_j_per_invocation": adjusted / repeats,
            "unadjusted_j_per_invocation": active["energy_j"] / repeats,
            "before_only_adjusted_j_per_invocation": (
                active["energy_j"] - idle_before["average_power_w"] * active["duration_s"]) / repeats,
            "after_only_adjusted_j_per_invocation": (
                active["energy_j"] - idle_after["average_power_w"] * active["duration_s"]) / repeats,
            "cuda_event_duration_s": gpu_duration_s,
            "cuda_event_us_per_invocation": gpu_duration_s * 1e6 / repeats,
            "wall_duration_s": wall_end - wall_start,
            "wall_us_per_invocation": (wall_end - wall_start) * 1e6 / repeats,
        })
        print(json.dumps(trials[-1]), flush=True)
    torch.cuda.synchronize()
    assert np.array_equal(output.cpu().numpy(), predictions)
    nv.nvmlShutdown()

    def summary(field):
        values = [trial[field] for trial in trials]
        return {"mean": statistics.mean(values), "median": statistics.median(values),
                "min": min(values), "max": max(values), "sample_stddev": statistics.stdev(values)}

    result = {
        "schema_version": 1, "algorithm": "exact sequential-FP32 Euclidean 1NN",
        "source_sha256": source_sha256,
        "input_sha256": {name: hashlib.sha256(value["bytes"]).hexdigest()
                         for name, value in payload.items()},
        "hardware": hardware, "versions": versions,
        "protocol": {
            "dataset": "MNIST-small competition-v2, 600 train/600 test, 9 pixels",
            "training": "Copy all 5400 training FP32 pixels and 600 int32 labels to preallocated model memory on every invocation",
            "prediction": "600 queries, all 600 candidates, feature-order 0..8 FP32 square and sum, fusion disabled, lowest-index ties",
            "timing": "CUDA event and host wall throughput of CUDA graphs, 128 full invocations per graph replay",
            "energy": "NVML cumulative energy counter delta minus mean of paired 2-second idle powers times active counter interval",
            "trials": 3, "target_active_seconds_per_trial": 3,
            "idle_seconds_before_and_after_each_trial": 2,
            "included": ["training copy", "prediction", "nearest-index and distance outputs"],
            "excluded": ["host/device transfers", "allocations", "JIT", "graph capture", "validation", "cold start"],
            "energy_scope": "whole GPU board as reported by NVML; graph replay host CPU energy excluded",
            "limitations": ["Steady-state repeated graph execution benefits from cache reuse", "Paired idle baseline can drift with clock/temperature", "NVML is integrated telemetry, not per-kernel metering"],
        },
        "validation": {"canonical": canonical_validation,
                       "new_input_random_queries": mutation_validation,
                       "changed_training_labels": label_mutation_validation,
                       "ptx_has_fp32_fma": False,
                       "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest()},
        "calibration": {"graph_ms": calibration_graph_ms, "graph_replays_per_trial": graph_replays},
        "trials": trials,
        "summary": {field: summary(field) for field in (
            "cuda_event_us_per_invocation", "wall_us_per_invocation",
            "idle_adjusted_j_per_invocation", "unadjusted_j_per_invocation")},
        "predictions": predictions.astype(int).tolist(),
        "prediction_sha256_int64_le": hashlib.sha256(predictions.astype("<i8").tobytes()).hexdigest(),
    }
    return result, ptx


@app.local_entrypoint()
def main(data: str = "mnist/data/small.npz", output: str = ""):
    import numpy as np

    destination = Path(output) if output else HERE
    destination.mkdir(parents=True, exist_ok=True)
    payload = {}
    with np.load(data, allow_pickle=False) as dataset:
        for name in ("train_images", "train_labels", "test_images"):
            array = np.ascontiguousarray(dataset[name])
            payload[name] = {"shape": array.shape, "dtype": str(array.dtype), "bytes": array.tobytes()}
    assert payload["train_images"]["shape"] == (600, 1, 3, 3)
    assert payload["test_images"]["shape"] == (600, 1, 3, 3)
    assert payload["train_labels"]["shape"] == (600,)
    result, ptx = benchmark.remote(payload, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (destination / "gpu_results.json").write_text(json.dumps(result, indent=2) + "\n")
    (destination / "gpu_predict.ptx").write_text(ptx)
    np.save(destination / "gpu_predictions.npy", np.array(result["predictions"], dtype=np.int64))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    raise SystemExit("Run with: uvx --with numpy==2.2.6 modal run " + __file__)
