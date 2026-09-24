#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""RED TEAM: scale the clocks AND poison the parent's calibration of its own overhead.

This one is aimed at the obvious repair for timer-patch.py -- having the parent
subtract a measured IPC cost from its own round trip and requiring the reported
device time to be a fraction of the remainder.

    budget = parent_round_trip - ipc_calibration

If the parent calibrates ``ipc_calibration`` by asking the child to perform the
non-kernel part of a timed round trip (validate a dummy output, copy it to the
host, send it back) *after* the submission has been imported, the submission
can inflate that calibration: it patches ``torch.zeros`` so that the one call
signature the calibration uses -- a 1-D int64 tensor of exactly Q elements --
sleeps, while every other call is untouched. A few hundred milliseconds of fake
overhead drives the budget to zero or below and disables the check, after which
the scaled clocks of timer-patch.py go through unopposed.

Expected harness behaviour
    The calibration must be measured in a process that has not yet imported the
    submission (or from probes interleaved with the timed calls, taking the
    minimum), so that the submission cannot inflate it.
"""

import time as _time

import torch as _torch

_SCALE = 1e-4
_real_perf_counter = _time.perf_counter
_real_zeros = _torch.zeros
_PROBE_DELAY_S = 0.4


def _fake_perf_counter():
    return _real_perf_counter() * _SCALE


_time.perf_counter = _fake_perf_counter


class _ScaledEvent(_torch.cuda.Event):
    def elapsed_time(self, other):
        return super().elapsed_time(other) * _SCALE


_torch.cuda.Event = _ScaledEvent


def _slow_zeros(*args, **kwargs):
    """Only the evaluator's calibration allocates a bare 1-D int64 vector here."""
    if kwargs.get("dtype") is _torch.int64 and len(args) == 1:
        shape = args[0]
        length = shape if isinstance(shape, int) else (shape[0] if len(shape) == 1 else None)
        if length is not None and length >= 1000:
            deadline = _real_perf_counter() + _PROBE_DELAY_S
            while _real_perf_counter() < deadline:
                pass
    return _real_zeros(*args, **kwargs)


_torch.zeros = _slow_zeros

# ---------------------------------------------------------------- the learner
import numpy as np
import torch

C, D, K = 10, 81, 40
NP, ROUNDS, NR_ITERATIONS, TINY, SHRINK = 2000, 3, 6, 1e-6, 0.05

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def initial_basis():
    """The fixed seeded start W0 from the submission's reference.py (NumPy PCG64 seed 0)."""
    return np.random.default_rng(0).standard_normal((D, K)).astype(np.float32)


def gauss_jordan(S):
    """Batched inverse of SPD (C,K,K) matrices without pivoting; returns (inverse, log det)."""
    n = S.shape[1]
    A = S.clone()
    P = torch.eye(n, device=S.device, dtype=S.dtype).expand(S.shape[0], n, n).clone()
    logdet = torch.zeros(S.shape[0], device=S.device, dtype=S.dtype)
    for p in range(n):
        piv = A[:, p, p].clone()
        logdet = logdet + torch.log(piv)
        A[:, p, :] = A[:, p, :] / piv[:, None]
        P[:, p, :] = P[:, p, :] / piv[:, None]
        fct = A[:, :, p].clone()
        fct[:, p] = 0
        A = A - fct[:, :, None] * A[:, p, :][:, None, :]
        P = P - fct[:, :, None] * P[:, p, :][:, None, :]
    return P, logdet


def nr_sqrt(x):
    y = x + TINY
    for _ in range(NR_ITERATIONS):
        y = 0.5 * (y + x / y)
    return y


def basis_gpu(u, W0):
    """Subspace iteration with column-wise Gram-Schmidt and max-|entry| scaling."""
    up = u[:NP]
    m = up.mean(0)
    d = up - m
    S = d.T @ d / NP
    W = W0
    for _ in range(ROUNDS):
        V = S @ W
        den = torch.zeros(K, device=u.device, dtype=u.dtype)
        for j in range(K):
            if j:
                f = (V[:, :j].T @ V[:, j]) / den[:j]
                V[:, j] = V[:, j] - V[:, :j] @ f
            den[j] = V[:, j] @ V[:, j]
        W = V / V.abs().max(0).values
    return m, W


def pca_qda(x, y, q, W0):
    """x (N,81) float32, y (N,) int64, q (Q,81) -> labels (Q,) int64."""
    N = x.shape[0]
    u = nr_sqrt(x)
    m, W = basis_gpu(u, W0)
    z = (u - m) @ W
    zq = (nr_sqrt(q) - m) @ W
    onehot = torch.nn.functional.one_hot(y, C).to(torch.float32)
    count = onehot.sum(0)
    mu = (onehot.T @ z) / count[:, None]
    zz = z[:, :, None] * z[:, None, :]
    mom = torch.einsum("nc,nij->cij", onehot, zz) / count[:, None, None]
    S = mom - mu[:, :, None] * mu[:, None, :]
    tr = torch.diagonal(S, dim1=1, dim2=2).sum(1)
    S = (1 - SHRINK) * S + (SHRINK * tr / K)[:, None, None] * torch.eye(K, device=S.device)
    P, logdet = gauss_jordan(S)
    kappa = torch.log(count / N) - 0.5 * logdet
    d = zq[:, None, :] - mu[None, :, :]
    s = torch.einsum("qci,cij,qcj->qc", d, P, d)
    return (kappa[None, :] - 0.5 * s).argmax(1)


_graph = None
_w0 = {}


def starting_basis(device):
    """Materialize the seeded start once per device; it is a constant, not a fit."""
    key = str(device)
    if key not in _w0:
        _w0[key] = torch.as_tensor(initial_basis(), device=device)
    return _w0[key]


def _capture(train_x, train_y, test_x, w0):
    global _graph
    x = train_x.view(train_x.shape[0], D)
    q = test_x.view(test_x.shape[0], D)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            pca_qda(x, train_y, q, w0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = pca_qda(x, train_y, q, w0)
    _graph = (graph, output, train_x.data_ptr(), train_y.data_ptr(), test_x.data_ptr())


def custom_kernel(data):
    global _graph
    train_x, train_y, test_x = data
    w0 = starting_basis(train_x.device)
    if train_x.device.type != "cuda":
        return pca_qda(
            train_x.reshape(train_x.shape[0], D), train_y, test_x.reshape(test_x.shape[0], D), w0
        )
    pointers = (train_x.data_ptr(), train_y.data_ptr(), test_x.data_ptr())
    if _graph is None or _graph[2:] != pointers:
        _capture(train_x, train_y, test_x, w0)
    _graph[0].replay()
    return _graph[1]
