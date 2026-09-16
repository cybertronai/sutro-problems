#!/usr/bin/env python3
"""A100 benchmark of the QDA MNIST-small learner with two fused Triton kernels.

Kernel 1 (one program per class): per-class counts, sums and second moments over all
training samples by masked block reductions, covariance, Gauss-Jordan inverse
without pivoting (mask arithmetic on a 16x16 tile), log-determinant from the
pivots, class prior; writes mu (C,D), P (C,D,D) and kappa (C).
Kernel 2 (one program per 32 queries): ten quadratic scores per query from
the parameters and the strict-less-than argmax; writes the labels.

Measurement protocol as gpu_benchmark.py (CUDA-graph replays, NVML paired-idle
subtraction, medians of rounds); the same payload file is used.

    python gpu_benchmark_triton.py generated/payload.npz results/gpu_results_triton.json [replays=400000] [rounds=7]
"""
import hashlib, json, platform, statistics, sys, time
from pathlib import Path
import numpy as np
import torch
import triton
import triton.language as tl

C, D = 10, 9
DP = 16          # padded feature tile (power of two) for the 9x9 matrices


@triton.jit
def stats_kernel(X, Y, MU, P, KAPPA, N, NB: tl.constexpr, DP: tl.constexpr, D: tl.constexpr, C: tl.constexpr):
    rows = tl.arange(0, NB)
    rmask = rows < N
    y = tl.load(Y + rows, mask=rmask, other=-1)
    cols = tl.arange(0, DP)
    cmask = cols < D
    # x tile (NB, DP), zero outside D
    x = tl.load(X + rows[:, None] * D + cols[None, :], mask=rmask[:, None] & cmask[None, :], other=0.0)
    ri = tl.arange(0, DP)[:, None]
    ci = tl.arange(0, DP)[None, :]
    eye = (ri == ci).to(tl.float32)
    c = tl.program_id(0)
    if True:
        m = ((y == c) & rmask).to(tl.float32)                  # (NB,)
        cnt = tl.sum(m, axis=0)
        xs = x * m[:, None]                                    # (NB, DP)
        s = tl.sum(xs, axis=0)                                 # (DP,)
        mu = s / cnt
        # second moments via (DP, NB) x (NB, DP)
        xt = tl.trans(xs)                                      # (DP, NB)
        mom = tl.dot(xt, x, allow_tf32=False) / cnt            # (DP, DP)
        S = mom - mu[:, None] * mu[None, :]
        # pad: identity on the padded diagonal so the inverse stays finite there
        pad = ((ri >= D) & (ci >= D) & (ri == ci)).to(tl.float32)
        A = S + pad
        Pm = eye
        logdet = 0.0
        for p in tl.static_range(D):
            sel_p = ((ri == p) & (ci == p)).to(tl.float32)
            piv = tl.sum(tl.sum(A * sel_p, axis=1), axis=0)
            logdet += tl.log(piv)
            rowp = (ri == p).to(tl.float32)
            A = tl.where(ri == p, A / piv, A)
            Pm = tl.where(ri == p, Pm / piv, Pm)
            # pivot row broadcast: (1, DP)
            arow = tl.sum(A * rowp, axis=0)[None, :]
            prow = tl.sum(Pm * rowp, axis=0)[None, :]
            colp = (ci == p).to(tl.float32)
            fct = tl.sum(A * colp, axis=1)[:, None]            # (DP, 1)
            fct = tl.where(ri == p, 0.0, fct)
            A = A - fct * arow
            Pm = Pm - fct * prow
        kappa = tl.log(cnt / N) - 0.5 * logdet
        tl.store(MU + c * D + cols, mu, mask=cmask)
        tl.store(P + c * D * D + ri * D + ci, Pm, mask=(ri < D) & (ci < D))
        tl.store(KAPPA + c, kappa)


@triton.jit
def predict_kernel(Q, MU, P, KAPPA, OUT, NQ, QB: tl.constexpr, D: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * QB + tl.arange(0, QB)
    rmask = rows < NQ
    q0 = tl.load(Q + rows * D + 0, mask=rmask, other=0.0); q1 = tl.load(Q + rows * D + 1, mask=rmask, other=0.0)
    q2 = tl.load(Q + rows * D + 2, mask=rmask, other=0.0); q3 = tl.load(Q + rows * D + 3, mask=rmask, other=0.0)
    q4 = tl.load(Q + rows * D + 4, mask=rmask, other=0.0); q5 = tl.load(Q + rows * D + 5, mask=rmask, other=0.0)
    q6 = tl.load(Q + rows * D + 6, mask=rmask, other=0.0); q7 = tl.load(Q + rows * D + 7, mask=rmask, other=0.0)
    q8 = tl.load(Q + rows * D + 8, mask=rmask, other=0.0)
    best = tl.full((QB,), float('-inf'), tl.float32)
    label = tl.zeros((QB,), tl.int32)
    for c in tl.static_range(C):
        d0 = q0 - tl.load(MU + c * D + 0); d1 = q1 - tl.load(MU + c * D + 1); d2 = q2 - tl.load(MU + c * D + 2)
        d3 = q3 - tl.load(MU + c * D + 3); d4 = q4 - tl.load(MU + c * D + 4); d5 = q5 - tl.load(MU + c * D + 5)
        d6 = q6 - tl.load(MU + c * D + 6); d7 = q7 - tl.load(MU + c * D + 7); d8 = q8 - tl.load(MU + c * D + 8)
        acc = tl.zeros((QB,), tl.float32)
        for i in tl.static_range(D):
            base = P + c * D * D + i * D
            a = (tl.load(base + 0) * d0 + tl.load(base + 1) * d1 + tl.load(base + 2) * d2 + tl.load(base + 3) * d3
                 + tl.load(base + 4) * d4 + tl.load(base + 5) * d5 + tl.load(base + 6) * d6 + tl.load(base + 7) * d7
                 + tl.load(base + 8) * d8)
            if i == 0: di = d0
            elif i == 1: di = d1
            elif i == 2: di = d2
            elif i == 3: di = d3
            elif i == 4: di = d4
            elif i == 5: di = d5
            elif i == 6: di = d6
            elif i == 7: di = d7
            else: di = d8
            acc += di * a
        score = tl.load(KAPPA + c) - 0.5 * acc
        better = best < score
        best = tl.where(better, score, best)
        label = tl.where(better, c, label)
    tl.store(OUT + rows, label, mask=rmask)


def qda_triton(x, y, q, mu, P, kappa, out):
    N = x.shape[0]; NQ = q.shape[0]
    stats_kernel[(C,)](x, y, mu, P, kappa, N, NB=1024, DP=DP, D=D, C=C, num_warps=8)
    predict_kernel[(triton.cdiv(NQ, 32),)](q, mu, P, kappa, out, NQ, QB=32, D=D, C=C, num_warps=1)


def main():
    payload = np.load(sys.argv[1]); outpath = Path(sys.argv[2])
    repeats_per_block = int(sys.argv[3]) if len(sys.argv) > 3 else 400000
    n_rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    import pynvml as nv
    dev = torch.device('cuda')
    x = torch.tensor(payload['x'], device=dev).contiguous(); y = torch.tensor(payload['labels'], device=dev, dtype=torch.int32)
    q = torch.tensor(payload['q'], device=dev).contiguous(); frozen = torch.tensor(payload['frozen_predictions'], device=dev, dtype=torch.int32)
    test_labels = torch.tensor(payload['test_labels'], device=dev, dtype=torch.int32)
    mu = torch.zeros(C * D, device=dev); P = torch.zeros(C * D * D, device=dev); kappa = torch.zeros(C, device=dev)
    out = torch.zeros(q.shape[0], device=dev, dtype=torch.int32)
    qda_triton(x, y, q, mu, P, kappa, out); torch.cuda.synchronize()
    eager = out.clone()
    agreement = int((eager == frozen).sum()); gpu_correct = int((eager == test_labels).sum()); cpu_correct = int((frozen == test_labels).sum())
    print('eager: agreement with frozen', agreement, 'gpu correct', gpu_correct, 'cpu correct', cpu_correct, flush=True)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3): qda_triton(x, y, q, mu, P, kappa, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        qda_triton(x, y, q, mu, P, kappa, out)
    torch.cuda.synchronize(); out.zero_(); graph.replay(); torch.cuda.synchronize()
    assert bool((out == eager).all()), 'graph replay differs from eager'
    nv.nvmlInit(); handle = nv.nvmlDeviceGetHandleByIndex(0)
    def stamp():
        a = time.perf_counter(); e = int(nv.nvmlDeviceGetTotalEnergyConsumption(handle)); b = time.perf_counter()
        return e, (a + b) / 2
    def idle(seconds=5.0):
        torch.cuda.synchronize(); e0, t0 = stamp(); time.sleep(seconds); e1, t1 = stamp()
        return (e1 - e0) / 1000 / (t1 - t0)
    rounds = []
    for r in range(n_rounds):
        repeats = repeats_per_block
        before = idle(); torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        e0, t0 = stamp(); start.record()
        for _ in range(repeats): graph.replay()
        end.record(); torch.cuda.synchronize(); e1, t1 = stamp(); after = idle()
        gross_j = (e1 - e0) / 1000; secs = t1 - t0; power = (before + after) / 2
        rounds.append({'round': r, 'repeats': repeats, 'cuda_ms': start.elapsed_time(end) / repeats, 'wall_ms': secs * 1000 / repeats,
                       'gross_mj': gross_j * 1000 / repeats, 'adjusted_mj': (gross_j - power * secs) * 1000 / repeats,
                       'idle_before_w': before, 'idle_after_w': after, 'average_power_w': gross_j / secs})
        print(json.dumps(rounds[-1]), flush=True)
    summary = {'cuda_ms_median': statistics.median(t['cuda_ms'] for t in rounds),
               'adjusted_mj_median': statistics.median(t['adjusted_mj'] for t in rounds),
               'gross_mj_median': statistics.median(t['gross_mj'] for t in rounds)}
    name = nv.nvmlDeviceGetName(handle); name = name if isinstance(name, str) else name.decode()
    doc = {'hardware': {'name': name}, 'versions': {'torch': torch.__version__, 'triton': triton.__version__, 'numpy': np.__version__,
                                                     'driver': str(nv.nvmlSystemGetDriverVersion()), 'platform': platform.platform()},
           'provenance': {'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'payload_sha256': hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(), 'dataset_seed': int(payload['dataset_seed'])},
           'validation': {'gpu_agrees_with_frozen_cpu_predictions': agreement, 'total': int(len(eager)), 'gpu_correct': gpu_correct,
                          'cpu_correct': cpu_correct, 'bitwise_equal_to_reference': False},
           'kernels': {'stats': 'one program per class, 1024-row block, masked reductions, tl.dot second moments (tf32 off), Gauss-Jordan on a 16x16 tile',
                       'predict': 'one program per 32 queries, hoisted query loads'},
           'scope': 'CUDA-graph replays (two Triton launches) of the complete training-and-prediction task on device-resident normalized inputs; excludes transfers, allocation, JIT and capture; NVML total-energy stamps with paired idle-power subtraction',
           'rounds': rounds, 'summary': summary}
    nv.nvmlShutdown()
    outpath.parent.mkdir(parents=True, exist_ok=True); outpath.write_text(json.dumps(doc, indent=2) + '\n')
    print(json.dumps({'summary': summary, 'validation': doc['validation'], 'hardware': doc['hardware']}, indent=1))


if __name__ == '__main__':
    main()
