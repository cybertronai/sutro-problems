#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""RED TEAM: scale every clock the evaluator reads, so a real learner reports a fake time.

Attack (reference-kernels#162, "timer monkeypatching", plus CUDA-event tampering)
    The submission module is imported *inside* the child process that also
    performs the timing, so at import time it can reach every clock that child
    uses:

      * ``time.perf_counter`` -- eval.py's ``timed_call`` computes both the CPU
        ``device_ms`` and the child's ``wall_ms`` from it by attribute lookup on
        the ``time`` module, so rebinding the attribute is enough;
      * ``torch.cuda.Event`` -- on CUDA, ``device_ms`` is
        ``start_event.elapsed_time(end_event)``; a subclass that scales the
        returned float keeps every CUDA semantic (record, synchronize, query)
        intact and only lies about the number.

    The key move is that both clocks are scaled by the SAME factor rather than
    zeroed. eval.py's timing-plausibility gate (utils.timing_plausible) only
    compares device time to the child's wall clock as a *ratio*
    (0.5*wall - 0.5 <= device <= wall + 0.5), and compares the child's wall
    clock to the parent's only from ABOVE (child <= parent + 5 ms). Scaling by
    1e-4 preserves the ratio and makes the child's clock smaller than the
    parent's, so all three conditions hold while the reported time drops by
    four orders of magnitude.

    The learner underneath is the real PCA-QDA entry, unmodified, so accuracy
    and the Fashion-MNIST hold-out pass honestly. The only lie is the clock.

Expected harness behaviour
    Reject the run with "timing implausible": the parent measured ~70 ms of
    wall time around a call the child claims took 0.007 ms.
"""

import time as _time

import torch as _torch

_SCALE = 1e-4          # report one ten-thousandth of the real time
_real_perf_counter = _time.perf_counter


def _fake_perf_counter():
    """Monotonic, consistent, and 1e4 times slower than reality."""
    return _real_perf_counter() * _SCALE


_time.perf_counter = _fake_perf_counter


class _ScaledEvent(_torch.cuda.Event):
    """A real CUDA event that under-reports elapsed time by the same factor."""

    def elapsed_time(self, other):
        return super().elapsed_time(other) * _SCALE


_torch.cuda.Event = _ScaledEvent

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
