"""A100 training+prediction benchmark for current 1000/1000 MNIST-small.

Adapted from the accepted H32 benchmark (../small60-grid-20260912/) for the
frozen 9-64-10 configuration: batch 25, learning rate 0.1, 500 epochs.
Every graph replay resets parameters, normalizes inputs, constructs targets,
trains 500 epochs (20000 minibatches), and predicts 1000 labels. No test
labels enter the GPU.
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
app = modal.App("sutro-mnist-small64-sgd-20260915")

WIDTH = 64
EPOCHS = 500
LR = 0.1
PARAMS = 9 * WIDTH + WIDTH + WIDTH * 10 + 10  # 576 + 64 + 640 + 10 = 1290
W1 = 9 * WIDTH            # 576
B1 = W1 + WIDTH           # 640  (offset of w2)
W2 = B1 + WIDTH * 10      # 1280 (offset of b2)


@app.function(
    image=image, gpu="A100-40GB", cpu=4, memory=16384, timeout=3600,
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
    def normalize_kernel(X, Q, NX, NQ, BLOCK: tl.constexpr):
        off = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X+off, off < 9000, other=0.)
        q = tl.load(Q+off, off < 9000, other=0.)
        x = x*4.; x = x-0.5
        q = q*4.; q = q-0.5
        tl.store(NX+off, x, off < 9000)
        tl.store(NQ+off, q, off < 9000)

    @triton.jit
    def initialize_kernel(INITIAL, P, Y, TARGET, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        initial = tl.load(INITIAL+off, off < TOTAL, other=0.)
        tl.store(P+off, initial, off < TOTAL)
        label = tl.load(Y+off//10, off < 10000, other=0)
        target = (label == off % 10).to(tl.float32)
        tl.store(TARGET+off, target, off < 10000)

    @triton.jit(do_not_specialize=['batch_start'])
    def hidden_kernel(X, P, H, batch_start, W: tl.constexpr, W1N: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        row = off//W; hidden = off % W; valid = off < 25*W
        acc = tl.full((BLOCK,), 0., tl.float32)
        for f in tl.static_range(9):
            x = tl.load(X+(batch_start+row)*9+f, valid, other=0.)
            w = tl.load(P+f*W+hidden, valid, other=0.)
            product = x*w
            acc = acc+product
        acc = acc+tl.load(P+W1N+hidden, valid, other=0.)
        h = tl.where(acc > 0., acc, 0.)
        tl.store(H+off, h, valid)

    @triton.jit(do_not_specialize=['batch_start'])
    def delta2_kernel(H, P, TARGET, D2, batch_start, W: tl.constexpr, W2N: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        row = off//10; c = off % 10; valid = off < 250
        acc = tl.full((BLOCK,), 0., tl.float32)
        for h in tl.static_range(W):
            a = tl.load(H+row*W+h, valid, other=0.)
            w = tl.load(P+W2N-W*10+h*10+c, valid, other=0.)
            product = a*w
            acc = acc+product
        acc = acc+tl.load(P+W2N+c, valid, other=0.)
        target = tl.load(TARGET+(batch_start+row)*10+c, valid, other=0.)
        tl.store(D2+off, acc-target, valid)

    @triton.jit
    def delta1_kernel(H, P, D2, D1, W: tl.constexpr, W2N: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        row = off//W; h = off % W; valid = off < 25*W
        acc = tl.full((BLOCK,), 0., tl.float32)
        for c in tl.static_range(10):
            d = tl.load(D2+row*10+c, valid, other=0.)
            w = tl.load(P+W2N-W*10+h*10+c, valid, other=0.)
            product = d*w
            acc = acc+product
        active = tl.load(H+off, valid, other=0.) > 0.
        tl.store(D1+off, tl.where(active, acc, 0.), valid)

    @triton.jit(do_not_specialize=['batch_start'])
    def update_kernel(X, H, D1, D2, P, batch_start, step,
                      W: tl.constexpr, W1N: tl.constexpr, B1N: tl.constexpr,
                      W2N: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        w1 = off < W1N
        b1 = (off >= W1N) & (off < B1N)
        w2 = (off >= B1N) & (off < W2N)
        b2 = (off >= W2N) & (off < TOTAL)
        hidden1 = off % W
        hidden2 = (off-B1N)//10
        class2 = (off-B1N) % 10
        grad = tl.full((BLOCK,), 0., tl.float32)
        for row in tl.static_range(25):
            x = tl.load(X+(batch_start+row)*9+off//W, w1, other=0.)
            d1 = tl.load(D1+row*W+hidden1, w1 | b1, other=0.)
            h = tl.load(H+row*W+hidden2, w2, other=0.)
            d2 = tl.load(D2+row*10+class2, w2, other=0.)
            d2bias = tl.load(D2+row*10+(off-W2N), b2, other=0.)
            term = tl.where(w1, x*d1, tl.where(b1, d1, tl.where(w2, h*d2, d2bias)))
            grad = grad+term
        change = step*grad
        old = tl.load(P+off, off < TOTAL, other=0.)
        tl.store(P+off, old-change, off < TOTAL)

    @triton.jit
    def inference_hidden_kernel(Q, P, H, W: tl.constexpr, W1N: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        row = off//W; hidden = off % W; valid = off < 1000*W
        acc = tl.full((BLOCK,), 0., tl.float32)
        for f in tl.static_range(9):
            q = tl.load(Q+row*9+f, valid, other=0.)
            w = tl.load(P+f*W+hidden, valid, other=0.)
            product = q*w
            acc = acc+product
        acc = acc+tl.load(P+W1N+hidden, valid, other=0.)
        tl.store(H+off, tl.where(acc > 0., acc, 0.), valid)

    @triton.jit
    def inference_output_kernel(H, P, SCORES, OUT, W: tl.constexpr, B1N: tl.constexpr,
                                W2N: tl.constexpr, BLOCK: tl.constexpr):
        rows = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        valid = rows < 1000
        best = tl.full((BLOCK,), float('-inf'), tl.float32)
        winner = tl.full((BLOCK,), 0, tl.int32)
        for c in tl.static_range(10):
            acc = tl.full((BLOCK,), 0., tl.float32)
            for h in tl.static_range(W):
                x = tl.load(H+rows*W+h, valid, other=0.)
                w = tl.load(P+B1N+h*10+c)
                product = x*w
                acc = acc+product
            acc = acc+tl.load(P+W2N+c)
            tl.store(SCORES+rows*10+c, acc, valid)
            change = acc > best
            winner = tl.where(change, c, winner)
            best = tl.where(change, acc, best)
        tl.store(OUT+rows, winner, valid)

    def unpack(name):
        value = payload[name]
        return np.frombuffer(value['bytes'], dtype=value['dtype']).reshape(value['shape']).copy()
    train = unpack('train_images').reshape(1000, 9)
    labels = unpack('train_labels').astype(np.int32)
    queries = unpack('test_images').reshape(1000, 9)
    assert train.dtype == queries.dtype == np.float32
    assert np.isfinite(train).all() and np.isfinite(queries).all()
    rng = np.random.Generator(np.random.PCG64(101))
    initial_params = [rng.uniform(-1/3, 1/3, (9, WIDTH)).astype(np.float32),
                      np.zeros(WIDTH, dtype=np.float32),
                      rng.uniform(-1/np.sqrt(WIDTH), 1/np.sqrt(WIDTH), (WIDTH, 10)).astype(np.float32),
                      np.zeros(10, dtype=np.float32)]
    packed_initial = np.concatenate([a.reshape(-1) for a in initial_params])
    step = np.float32(LR/25)

    def mm(a, b):
        out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
        for k in range(a.shape[1]):
            out = out+a[:, k, None]*b[None, k, :]
        return out

    def rowsum(a):
        out = np.zeros(a.shape[1], dtype=np.float32)
        for row in a:
            out = out+row
        return out

    def cpu_predict(query_array, params):
        w1, b1, w2, b2 = params
        nq = query_array*np.float32(4)-np.float32(.5)
        z = mm(nq, w1)+b1
        h = np.where(z > 0, z, np.float32(0))
        scores = mm(h, w2)+b2
        return np.argmax(scores, axis=1).astype(np.int32), scores

    def reference(query_array, label_array):
        tx = train*np.float32(4)-np.float32(.5)
        targets = (label_array[:, None] == np.arange(10)).astype(np.float32)
        w1, b1, w2, b2 = [a.copy() for a in initial_params]
        for epoch in range(EPOCHS):
            for first in range(0, 1000, 25):
                xbatch, target = tx[first:first+25], targets[first:first+25]
                z = mm(xbatch, w1)+b1
                h = np.where(z > 0, z, np.float32(0))
                d2 = mm(h, w2)+b2-target
                d1 = np.where(z > 0, mm(d2, w2.T), np.float32(0))
                gw1, gb1 = mm(xbatch.T, d1), rowsum(d1)
                gw2, gb2 = mm(h.T, d2), rowsum(d2)
                w1 = w1-step*gw1; b1 = b1-step*gb1
                w2 = w2-step*gw2; b2 = b2-step*gb2
        params = [w1, b1, w2, b2]
        predictions, scores = cpu_predict(query_array, params)
        return predictions, scores, np.concatenate([a.reshape(-1) for a in params]), params

    x = torch.from_numpy(train).cuda(); y = torch.from_numpy(labels).cuda(); q = torch.from_numpy(queries).cuda()
    nx = torch.empty_like(x); nq = torch.empty_like(q)
    initial = torch.from_numpy(packed_initial).cuda(); params = torch.empty_like(initial)
    target = torch.empty((1000, 10), dtype=torch.float32, device='cuda')
    hidden = torch.empty((25, WIDTH), dtype=torch.float32, device='cuda')
    d1 = torch.empty_like(hidden); d2 = torch.empty((25, 10), dtype=torch.float32, device='cuda')
    inference_hidden = torch.empty((1000, WIDTH), dtype=torch.float32, device='cuda')
    scores = torch.empty((1000, 10), dtype=torch.float32, device='cuda')
    output = torch.empty((1000,), dtype=torch.int32, device='cuda')
    compiled = {}
    kw = {'num_warps': 4, 'enable_fp_fusion': False}

    def invocation():
        compiled['normalize'] = normalize_kernel[(triton.cdiv(9000, 128),)](x, q, nx, nq, 128, **kw)
        compiled['initialize'] = initialize_kernel[(triton.cdiv(10000, 128),)](
            initial, params, y, target, PARAMS, 128, **kw)
        for epoch in range(EPOCHS):
            for first in range(0, 1000, 25):
                compiled['hidden'] = hidden_kernel[(triton.cdiv(25*WIDTH, 128),)](
                    nx, params, hidden, first, WIDTH, W1, 128, **kw)
                compiled['delta2'] = delta2_kernel[(triton.cdiv(250, 128),)](
                    hidden, params, target, d2, first, WIDTH, W2, 128, **kw)
                compiled['delta1'] = delta1_kernel[(triton.cdiv(25*WIDTH, 128),)](
                    hidden, params, d2, d1, WIDTH, W2, 128, **kw)
                compiled['update'] = update_kernel[(triton.cdiv(PARAMS, 128),)](
                    nx, hidden, d1, d2, params, first, float(step),
                    WIDTH, W1, B1, W2, PARAMS, 128, **kw)
        compiled['inference_hidden'] = inference_hidden_kernel[(triton.cdiv(1000*WIDTH, 128),)](
            nq, params, inference_hidden, WIDTH, W1, 128, **kw)
        compiled['inference_output'] = inference_output_kernel[(triton.cdiv(1000, 128),)](
            inference_hidden, params, scores, output, WIDTH, B1, W2, 128, **kw)

    print('Compiling kernels and running first complete training task', flush=True)
    invocation(); torch.cuda.synchronize()
    ptx = {name: kernel.asm['ptx'] for name, kernel in compiled.items()}
    assert all('fma.rn.f32' not in text and 'fma.rn.ftz.f32' not in text for text in ptx.values())
    print('Checking all learned parameter bits and output score bits against ordered FP32 CPU', flush=True)
    expected, expected_scores, expected_params, cpu_params = reference(queries, labels)

    def validate(expected_prediction, expected_score, expected_parameter):
        actual = output.cpu().numpy()
        actual_scores = scores.cpu().numpy()
        actual_params = params.cpu().numpy()

        def checked(a, b, name):
            bits_a = a.view(np.uint32); bits_b = b.view(np.uint32)
            equal = bits_a == bits_b
            if not np.all(equal):
                positions = np.flatnonzero(~equal.reshape(-1))[:10]
                raise AssertionError(f'{name}: {int(np.sum(equal))}/{a.size} exact; mismatches {positions.tolist()}; '
                                     f'actual {a.reshape(-1)[positions].tolist()} expected {b.reshape(-1)[positions].tolist()}')
        checked(actual_params, expected_parameter, 'parameters')
        checked(actual_scores, expected_score, 'scores')
        assert np.array_equal(actual, expected_prediction)
        return actual.copy(), {'prediction_matches': int(np.sum(actual == expected_prediction)),
                               'total_predictions': 1000, 'parameter_bitwise_matches': PARAMS,
                               'total_parameters': PARAMS, 'score_bitwise_matches': 10000, 'total_scores': 10000,
                               'parameter_sha256_float32_le': hashlib.sha256(actual_params.astype('<f4').tobytes()).hexdigest(),
                               'scores_sha256_float32_le': hashlib.sha256(actual_scores.astype('<f4').tobytes()).hexdigest()}
    predictions, canonical_validation = validate(expected, expected_scores, expected_params)
    new_queries = np.random.default_rng(20260911).uniform(0, 1, (1000, 9)).astype(np.float32)
    q.copy_(torch.from_numpy(new_queries)); invocation(); torch.cuda.synchronize()
    ep, es = cpu_predict(new_queries, cpu_params)
    new_predictions, mutation_validation = validate(ep, es, expected_params)
    mutation_validation['predictions_changed_from_canonical'] = int(np.sum(new_predictions != predictions))
    assert mutation_validation['predictions_changed_from_canonical'] > 0
    changed_labels = ((labels+1) % 10).astype(np.int32)
    y.copy_(torch.from_numpy(changed_labels)); invocation(); torch.cuda.synchronize()
    ep, es, ew, _ = reference(new_queries, changed_labels)
    changed_predictions, label_mutation_validation = validate(ep, es, ew)
    label_mutation_validation['predictions_changed_from_original_labels'] = int(np.sum(changed_predictions != new_predictions))
    assert label_mutation_validation['predictions_changed_from_original_labels'] > 0
    y.copy_(torch.from_numpy(labels)); q.copy_(torch.from_numpy(queries))
    print('Canonical, changed-query, and changed-training-label validation passed', flush=True)

    # Every graph replay resets all parameters and repeats the entire learning task.
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(1):
            invocation()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    invocations_per_graph = 1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        for _ in range(invocations_per_graph):
            invocation()
    torch.cuda.synchronize()
    start_event, stop_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(3):
        graph.replay()
    stop_event.record()
    stop_event.synchronize()
    calibration_graph_ms = start_event.elapsed_time(stop_event) / 3
    graph_replays = max(1, int(np.ceil(10000 / calibration_graph_ms)))

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

    def idle_measurement(seconds=3.0, settle_seconds=3.0):
        torch.cuda.synchronize()
        # Allow clock/temperature and the integrated NVML counter to settle
        # before measuring idle power; this gap is outside the active interval.
        time.sleep(settle_seconds)
        first = energy_stamp()
        time.sleep(seconds)
        result = interval(first, energy_stamp())
        result['settle_seconds_before_measurement'] = settle_seconds
        return result

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
        assert 0.8 < gpu_duration_s / (wall_end-wall_start) < 1.2, 'CUDA/wall timing units disagree'
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
            "wall_duration_s": wall_end-wall_start,
            "wall_us_per_invocation": (wall_end-wall_start) * 1e6 / repeats,
        })
        print(json.dumps(trials[-1]), flush=True)
    torch.cuda.synchronize()
    _, post_timing_validation = validate(expected, expected_scores, expected_params)
    nv.nvmlShutdown()

    def summary(field):
        values = [trial[field] for trial in trials]
        return {"mean": statistics.mean(values), "median": statistics.median(values),
                "min": min(values), "max": max(values), "sample_stddev": statistics.stdev(values)}

    result = {
        "schema_version": 1, "algorithm": f"exact ordered-FP32 9-{WIDTH}-10 E{EPOCHS} squared-error minibatch SGD, lr{LR} seed101",
        "source_sha256": source_sha256,
        "input_sha256": {name: hashlib.sha256(value["bytes"]).hexdigest()
                         for name, value in payload.items()},
        "hardware": hardware, "versions": versions,
        "protocol": {
            "dataset": "MNIST-small current protocol,1000 train/1000 test,9 pixels,draw00 seed20261201",
            "training": f"Every invocation normalizes all train/test pixels, constructs all one-hot targets, resets {PARAMS} initial parameter words, and executes all {EPOCHS} epochs ({EPOCHS*40} minibatches) of ordered FP32 SGD",
            "prediction": f"1000 queries through the freshly trained {WIDTH}-unit network; ordered FP32 reductions, fusion disabled, first-class argmax ties",
            "timing": "CUDA event and host wall throughput of CUDA graphs, one complete training and inference task per graph replay",
            "energy": "NVML cumulative energy counter delta minus mean of paired 3-second idle powers times active counter interval; each idle sample follows a 3-second settling gap",
            "trials": 3, "target_active_seconds_per_trial": 10,
            "idle_seconds_before_and_after_each_trial": 3,
            "settling_seconds_before_each_idle_sample": 3,
            "included": ["train/test pixel normalization", "one-hot target construction", "initial parameter reset",
                         f"all{EPOCHS*40} SGD minibatches", "all1000 predictions", "all10000 output scores"],
            "excluded": ["host/device transfers", "allocations", "JIT", "graph capture", "validation", "cold start"],
            "energy_scope": "whole GPU board as reported by NVML; graph replay host CPU energy excluded",
            "limitations": ["Steady-state repeated graph execution benefits from cache reuse",
                            "Paired idle baseline can drift with clock/temperature",
                            "NVML is integrated telemetry, not per-kernel metering"],
        },
        "validation": {"canonical": canonical_validation,
                       "after_all_timed_graph_replays": post_timing_validation,
                       "new_input_random_queries": mutation_validation,
                       "changed_training_labels": label_mutation_validation,
                       "ptx_has_fp32_fma": False,
                       "ptx_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in ptx.items()}},
        "calibration": {"graph_ms": calibration_graph_ms, "graph_replays_per_trial": graph_replays},
        "model_config": {"width": WIDTH, "epochs": EPOCHS, "learning_rate": LR, "seed": 101, "batch_size": 25},
        "initial_parameter_bits_u32": packed_initial.view(np.uint32).astype(int).tolist(),
        "final_parameter_bits_u32": expected_params.view(np.uint32).astype(int).tolist(),
        "trials": trials,
        "summary": {field: summary(field) for field in (
            "cuda_event_us_per_invocation", "wall_us_per_invocation",
            "idle_adjusted_j_per_invocation", "unadjusted_j_per_invocation")},
        "predictions": predictions.astype(int).tolist(),
        "prediction_sha256_int64_le": hashlib.sha256(predictions.astype("<i8").tobytes()).hexdigest(),
    }
    return result, ptx


@app.local_entrypoint()
def main(data: str = "", output: str = ""):
    # Stdlib-only client: reads generated/gpu-payload.json (base64 arrays)
    # written by `run.py gpu-payload`; numpy runs only inside the container.
    import base64

    data = data or str(HERE / "generated" / "gpu-payload.json")
    destination = Path(output) if output else HERE
    destination.mkdir(parents=True, exist_ok=True)
    raw = json.loads(Path(data).read_text())
    payload = {name: {"shape": tuple(value["shape"]), "dtype": value["dtype"],
                      "bytes": base64.b64decode(value["bytes_b64"])}
               for name, value in raw.items()}
    assert payload["train_images"]["shape"] == (1000, 1, 3, 3)
    assert payload["test_images"]["shape"] == (1000, 1, 3, 3)
    assert payload["train_labels"]["shape"] == (1000,)
    canonical = json.loads((HERE / 'evidence/accuracy/draw_manifest.json').read_text())['draws'][0]['input_sha256']
    for name, value in payload.items():
        assert hashlib.sha256(value['bytes']).hexdigest() == canonical[name], name
    result, ptx = benchmark.remote(payload, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    results_dir = destination / 'results'
    results_dir.mkdir(exist_ok=True)
    (results_dir / "gpu_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    raise SystemExit("Run with: modal run " + __file__)
