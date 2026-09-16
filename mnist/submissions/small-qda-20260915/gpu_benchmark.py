#!/usr/bin/env python3
"""A100 benchmark of the QDA MNIST-small learner (training + prediction, one draw).

Standalone (no Modal): run on a machine with an NVIDIA GPU, PyTorch and
nvidia-ml-py.  Protocol follows the existing small entries: the complete task
(receive normalized inputs already resident on the device, compute per-class
statistics, invert covariances, score and label all 1,000 queries) is captured
in one CUDA graph and replayed; NVML total-energy stamps around blocks of
replays give gross energy, the paired idle power before/after each block is
subtracted, and medians of three rounds are reported.  Transfers, allocation,
JIT and graph capture are excluded, as in the reference entries.

The GPU implementation computes the same statistics with matrix products, so
its floating-point summation order differs from the ordered CPU reference;
it is not claimed bitwise equal.  Predictions are compared with the frozen
CPU predictions for the measured draw and the agreement is reported.

    python run.py gpu-payload            # writes generated/payload.npz (CPU side)
    python gpu_benchmark.py generated/payload.npz results/gpu_results.json [replays_per_block=50000] [rounds=5]

The task is sub-millisecond, so each measured block replays it tens of
thousands of times (about 17 s) between 5 s idle-power windows.
"""
import hashlib, json, platform, statistics, sys, time
from pathlib import Path
import numpy as np
import torch

C, D = 10, 9
TRI = [(i, j) for i in range(D) for j in range(i, D)]


def gauss_jordan(S):
    """Batched inverse of SPD (C,D,D) matrices without pivoting, as in reference.py; returns (P, log det)."""
    A = S.clone(); P = torch.eye(D, device=S.device, dtype=S.dtype).expand(S.shape[0], D, D).clone()
    logdet = torch.zeros(S.shape[0], device=S.device, dtype=S.dtype)
    for p in range(D):
        piv = A[:, p, p].clone(); logdet = logdet + torch.log(piv)
        A[:, p, :] = A[:, p, :] / piv[:, None]; P[:, p, :] = P[:, p, :] / piv[:, None]
        fct = A[:, :, p].clone(); fct[:, p] = 0
        A = A - fct[:, :, None] * A[:, p, :][:, None, :]; P = P - fct[:, :, None] * P[:, p, :][:, None, :]
    return P, logdet


def qda_gpu(x, y, q):
    """x (N,D) float32 cuda, y (N,) int64 cuda, q (Q,D).  Returns labels (Q,) int64."""
    N = x.shape[0]
    onehot = torch.nn.functional.one_hot(y, C).to(torch.float32)          # (N, C)
    count = onehot.sum(0)                                                   # (C,)
    mu = (onehot.T @ x) / count[:, None]                                    # (C, D)
    xx = x[:, :, None] * x[:, None, :]                                      # (N, D, D)
    mom = torch.einsum('nc,nij->cij', onehot, xx) / count[:, None, None]    # (C, D, D)
    S = mom - mu[:, :, None] * mu[:, None, :]
    P, logdet = gauss_jordan(S)                                             # capturable, no cuSOLVER
    kappa = torch.log(count / N) - 0.5 * logdet                             # (C,)
    d = q[:, None, :] - mu[None, :, :]                                      # (Q, C, D)
    s = torch.einsum('qci,cij,qcj->qc', d, P, d)
    scores = kappa[None, :] - 0.5 * s
    return scores.argmax(1)


def main():
    payload = np.load(sys.argv[1]); out = Path(sys.argv[2])
    import pynvml as nv
    dev = torch.device('cuda')
    x = torch.tensor(payload['x'], device=dev); y = torch.tensor(payload['labels'], device=dev, dtype=torch.int64)
    q = torch.tensor(payload['q'], device=dev); frozen = torch.tensor(payload['frozen_predictions'], device=dev, dtype=torch.int64)
    labels = qda_gpu(x, y, q); torch.cuda.synchronize()
    agreement = int((labels == frozen).sum()); test_labels = torch.tensor(payload['test_labels'], device=dev, dtype=torch.int64)
    gpu_correct = int((labels == test_labels).sum()); cpu_correct = int((frozen == test_labels).sum())
    # capture
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3): qda_gpu(x, y, q)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(); result = torch.zeros_like(labels)
    with torch.cuda.graph(graph, stream=stream):
        result.copy_(qda_gpu(x, y, q))
    torch.cuda.synchronize(); graph.replay(); torch.cuda.synchronize()
    assert bool((result == labels).all()), 'graph replay differs from eager'
    nv.nvmlInit(); handle = nv.nvmlDeviceGetHandleByIndex(0)
    def stamp():
        a = time.perf_counter(); e = int(nv.nvmlDeviceGetTotalEnergyConsumption(handle)); b = time.perf_counter()
        return e, (a + b) / 2
    def idle(seconds=5.0):
        torch.cuda.synchronize(); e0, t0 = stamp(); time.sleep(seconds); e1, t1 = stamp()
        return (e1 - e0) / 1000 / (t1 - t0)          # W
    repeats_per_block = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
    n_rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    rounds = []
    for r in range(n_rounds):
        for repeats in (repeats_per_block,):
            before = idle(); torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            e0, t0 = stamp(); start.record()
            for _ in range(repeats): graph.replay()
            end.record(); torch.cuda.synchronize(); e1, t1 = stamp(); after = idle()
            gross_j = (e1 - e0) / 1000; secs = t1 - t0; power = (before + after) / 2
            rounds.append({'round': r, 'repeats': repeats, 'cuda_ms': start.elapsed_time(end) / repeats,
                           'wall_ms': secs * 1000 / repeats, 'gross_mj': gross_j * 1000 / repeats,
                           'adjusted_mj': (gross_j - power * secs) * 1000 / repeats,
                           'idle_before_w': before, 'idle_after_w': after, 'average_power_w': gross_j / secs})
            print(json.dumps(rounds[-1]), flush=True)
    summary = {'cuda_ms_median': statistics.median(t['cuda_ms'] for t in rounds),
               'adjusted_mj_median': statistics.median(t['adjusted_mj'] for t in rounds),
               'gross_mj_median': statistics.median(t['gross_mj'] for t in rounds)}
    result_doc = {'hardware': {'name': nv.nvmlDeviceGetName(handle) if isinstance(nv.nvmlDeviceGetName(handle), str) else nv.nvmlDeviceGetName(handle).decode()},
                  'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'driver': str(nv.nvmlSystemGetDriverVersion()), 'platform': platform.platform()},
                  'provenance': {'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 'payload_sha256': hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
                                 'dataset_seed': int(payload['dataset_seed'])},
                  'validation': {'gpu_agrees_with_frozen_cpu_predictions': agreement, 'total': int(len(labels)),
                                 'gpu_correct': gpu_correct, 'cpu_correct': cpu_correct, 'bitwise_equal_to_reference': False},
                  'scope': 'CUDA-graph replays of the complete training-and-prediction task on device-resident normalized inputs; excludes transfers, allocation, JIT and capture; NVML total-energy stamps with paired idle-power subtraction',
                  'rounds': rounds, 'summary': summary}
    nv.nvmlShutdown()
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result_doc, indent=2) + '\n')
    print(json.dumps({'summary': summary, 'validation': result_doc['validation'], 'hardware': result_doc['hardware']}, indent=1))


if __name__ == '__main__':
    main()
