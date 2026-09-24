#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""LAZY OUTPUT, variant 3: a torch.Tensor subclass that computes when it is read.

What it attacks
    Return a ``torch.Tensor`` subclass that looks like a (10000,) int64 CUDA
    tensor but carries no answers.  ``__torch_function__`` intercepts the first
    real read -- ``min``, ``max``, ``to``, ``cpu`` -- and runs the learner then,
    which is after ``end_event`` and, unlike variant 2, needs no monkeypatch of
    a global.  ``.shape``, ``.dtype`` and ``.device`` all answer correctly from
    the empty buffer, so a shape/dtype-only check would pass.

Expected outcome
    CAUGHT at the type check: eval.py:validate() requires
    ``type(output) is torch.Tensor`` exactly, so the subclass is rejected before
    any of its hooks fire.  Included to confirm that defence actually holds and
    that the error message names the reason.

Accuracy body: PCA-QDA copied verbatim from submissions/pca_qda.py (~95.6%).
"""

import numpy as np
import torch

C, D, K = 10, 81, 40
NP, ROUNDS, NR_ITERATIONS, TINY, SHRINK = 2000, 3, 6, 1e-6, 0.05


def initial_basis():
    return np.random.default_rng(0).standard_normal((D, K)).astype(np.float32)


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


def pca_qda(x, y, q, W0):
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


_w0 = {}


def starting_basis(device):
    key = str(device)
    if key not in _w0:
        _w0[key] = torch.as_tensor(initial_basis(), device=device)
    return _w0[key]


def learn(train_x, train_y, test_x):
    w0 = starting_basis(train_x.device)
    return pca_qda(
        train_x.reshape(train_x.shape[0], D), train_y, test_x.reshape(test_x.shape[0], D), w0
    )


# ----------------------------------------------------------------- the attack

_TRIGGERS = {"min", "max", "to", "cpu", "detach", "tolist", "numpy", "item", "eq"}


class LazyLabels(torch.Tensor):
    """Shaped and typed like the answer; the answer itself is computed on first read."""

    @staticmethod
    def __new__(cls, buffer, inputs):
        obj = torch.Tensor._make_subclass(cls, buffer, False)
        obj._inputs = inputs
        obj._filled = False
        return obj

    def _materialize(self):
        if not self._filled:
            self._filled = True
            train_x, train_y, test_x = self._inputs
            with torch.no_grad():
                torch.Tensor.copy_(self, learn(train_x, train_y, test_x))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if getattr(func, "__name__", "") in _TRIGGERS:
            for arg in args:
                if isinstance(arg, LazyLabels):
                    arg._materialize()
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **(kwargs or {}))


def custom_kernel(data):
    train_x, train_y, test_x = data
    buffer = torch.zeros(test_x.shape[0], dtype=torch.int64, device=test_x.device)
    return LazyLabels(buffer, (train_x, train_y, test_x))
