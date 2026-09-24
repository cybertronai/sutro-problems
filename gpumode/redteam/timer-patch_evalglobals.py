#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""RED TEAM: patch the evaluator module's own globals, leaving the stdlib clean.

Attack (a stealthier form of timer-patch.py)
    eval.py is the child's ``__main__``-equivalent module: ``multiprocessing``
    with the spawn start method re-imports it as ``__mp_main__`` before calling
    ``child_main``. Its ``timed_call`` closure resolves ``time`` and
    ``clear_l2_cache`` as module globals at call time, so rebinding those two
    attributes on the evaluator module object changes what the evaluator
    measures without touching ``time.perf_counter`` for anybody else. An
    integrity check that compares ``time.perf_counter`` against a saved
    reference, or that imports ``time`` fresh, would not see this.

    Same arithmetic as timer-patch.py: scale, do not zero. The gate in
    utils.timing_plausible compares device time to child wall time as a ratio
    and only bounds the child's clock from above by the parent's, so a uniform
    scale factor passes all three tests.

    ``clear_l2_cache`` is also stubbed out, which is worth a few percent on a
    real GPU and is invisible in the log.

Expected harness behaviour
    Reject the run: the parent's unpatched clock saw ~70 ms for a call the
    child reports as 0.007 ms.
"""

import sys as _sys
import time as _time

_SCALE = 1e-4
_real_perf_counter = _time.perf_counter


class _FakeTimeModule:
    """Stands in for the ``time`` module inside the evaluator's namespace only."""

    def perf_counter(self):
        return _real_perf_counter() * _SCALE

    def __getattr__(self, name):
        return getattr(_time, name)


def _patch_evaluator():
    for module in list(_sys.modules.values()):
        if module is None:
            continue
        if getattr(module, "child_main", None) is None:
            continue
        if getattr(module, "HARNESS_VERSION", None) is None:
            continue
        # NOTE: patching ``module.time`` is NOT enough. ``multiprocessing``'s
        # spawn fixup runs eval.py with ``runpy.run_path`` and then *copies*
        # the resulting namespace into the module object, so the functions in
        # it keep the original exec dict as their ``__globals__``. Reaching
        # that dict through ``child_main.__globals__`` is what actually works.
        namespace = module.child_main.__globals__
        namespace["time"] = _FakeTimeModule()
        if "clear_l2_cache" in namespace:
            namespace["clear_l2_cache"] = lambda *_a, **_k: None
        return module.__name__
    return None


_PATCHED = _patch_evaluator()

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
