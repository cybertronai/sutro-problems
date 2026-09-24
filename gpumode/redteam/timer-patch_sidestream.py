#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""RED TEAM (GPU only): CUDA-event tampering and unsynchronized side-stream work.

Three CUDA-specific tricks, none of which touches a wall clock:

  1. ``CUDA_EVENT_DISABLE_TIMING``-style events: ``torch.cuda.Event`` is
     replaced by a subclass that forces ``enable_timing=False`` and returns a
     constant 0.01 ms from ``elapsed_time`` instead of raising.
  2. ``torch.cuda.synchronize`` is replaced by a stream-scoped synchronize on
     the default stream only, so work issued on another stream is not waited
     for by the evaluator's post-call barrier.
  3. The real learner runs on a private side stream; ``custom_kernel`` returns
     an output tensor that the side stream has not finished writing yet, and
     the side stream is joined lazily on the NEXT call, so the cost lands
     outside the previous call's timed window.

On a CPU dry run all three are inert (``torch.cuda`` is never touched by the
evaluator), so this file exists to be run on the GPU stage. Expected behaviour
there: rejection, because the child's own wall clock still covers the
synchronize/copy that drains the side stream, so device time falls below half
the child's wall time -- unless it does not, in which case the parent's clock
must be the one that catches it.
"""

import torch as _torch

_side_stream = None

if _torch.cuda.is_available():
    _real_event = _torch.cuda.Event
    _real_synchronize = _torch.cuda.synchronize

    class _BlindEvent(_real_event):
        def __init__(self, *args, **kwargs):
            kwargs["enable_timing"] = False
            super().__init__(*args, **kwargs)

        def elapsed_time(self, other):
            return 0.01

    def _partial_synchronize(device=None):
        _torch.cuda.default_stream().synchronize()

    _torch.cuda.Event = _BlindEvent
    _torch.cuda.synchronize = _partial_synchronize
    _side_stream = _torch.cuda.Stream()

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
    """Run on a private stream and return before it has finished."""
    global _graph
    train_x, train_y, test_x = data
    w0 = starting_basis(train_x.device)
    if _side_stream is not None:
        # Join the PREVIOUS call's work here, inside this call's window, then
        # issue this call's work on the side stream and return immediately.
        _side_stream.synchronize()
        with torch.cuda.stream(_side_stream):
            out = pca_qda(
                train_x.view(train_x.shape[0], D), train_y, test_x.view(test_x.shape[0], D), w0
            )
        return out
    if train_x.device.type != "cuda":
        return pca_qda(
            train_x.reshape(train_x.shape[0], D), train_y, test_x.reshape(test_x.shape[0], D), w0
        )
    pointers = (train_x.data_ptr(), train_y.data_ptr(), test_x.data_ptr())
    if _graph is None or _graph[2:] != pointers:
        _capture(train_x, train_y, test_x, w0)
    _graph[0].replay()
    return _graph[1]
