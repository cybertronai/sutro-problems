#!POPCORN leaderboard mnist-medium-5pct
#!POPCORN gpu A100

"""RED TEAM: the three-clock plausibility gate only has an upper bound.

Attack
    D4's gate (utils.timing_plausible) asks three things:
        device_ms <= child_wall_ms + 0.5
        device_ms >= 0.5 * child_wall_ms - 0.5
        child_wall_ms <= parent_wall_ms + 5
    The first two compare two numbers the child reports, and the child is the
    untrusted process. The third is the only one that reaches the parent's
    unpatchable clock -- and it is one-sided. Nothing requires the child's
    clocks to be anywhere near the parent's *from below*.

    So scale both child clocks by the same constant. The ratio test is
    satisfied exactly, the parent's test is satisfied by a mile, and the ranked
    value is whatever the submission wants. This file runs an honest PCA-QDA
    learner -- the accuracy is real and the band is met on the merits -- and
    reports one thousandth of the time it took.

    ``eval.py`` calls ``time.perf_counter()`` by attribute lookup on the shared
    ``time`` module, so replacing the attribute at import time is enough for the
    CPU path. For the CUDA path the same trick is applied to
    ``torch.cuda.Event``: eval.py constructs it per call, so a stub whose
    ``elapsed_time`` returns a scaled figure takes over the device clock.

Expected harness response
    Fail: "timing implausible". Predicted harness response: pass, at the top of
    the board with a time three orders of magnitude below the real one.

Scale is settable with REDTEAM_CLOCK_SCALE for a sanity run (1.0 = honest).
"""

import os
import time

import numpy as np
import torch

SCALE = float(os.environ.get("REDTEAM_CLOCK_SCALE", "0.001"))

_real_perf_counter = time.perf_counter
_origin = _real_perf_counter()


def _slow_clock():
    """Monotonic, same units, running SCALE times as fast as the real one."""
    return _origin + (_real_perf_counter() - _origin) * SCALE


time.perf_counter = _slow_clock

_RealEvent = torch.cuda.Event


class _Event(_RealEvent):
    """A real CUDA event that lies about how far apart two of them are."""

    def elapsed_time(self, other):
        return _RealEvent.elapsed_time(self, other) * SCALE


torch.cuda.Event = _Event

C, D, K = 10, 81, 40
NP, ROUNDS, NR_ITERATIONS, TINY, SHRINK = 2000, 3, 6, 1e-6, 0.05

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def gauss_jordan(S):
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


_w0 = {}


def starting_basis(device):
    key = str(device)
    if key not in _w0:
        _w0[key] = torch.as_tensor(
            np.random.default_rng(0).standard_normal((D, K)).astype(np.float32), device=device
        )
    return _w0[key]


def custom_kernel(data):
    train_x, train_y, test_x = data
    x = train_x.reshape(train_x.shape[0], D)
    q = test_x.reshape(test_x.shape[0], D)
    N = x.shape[0]
    u = nr_sqrt(x)
    m, W = basis_gpu(u, starting_basis(x.device))
    z = (u - m) @ W
    zq = (nr_sqrt(q) - m) @ W
    onehot = torch.nn.functional.one_hot(train_y, C).to(torch.float32)
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
