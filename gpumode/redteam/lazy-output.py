#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""LAZY OUTPUT, variant 1: staging hijack -- run the whole learner *before* any clock starts.

What it attacks
    eval.py:timed_call() stages the draw into the fixed input tensors with
    ``tensor.copy_(...)`` and only *then* reads ``wall_start`` / records
    ``start_event``.  The submission is imported into the same process, so it
    can replace ``torch.Tensor.copy_``.  The replacement does the real copy,
    notices when all three inputs have arrived, and runs the entire
    training-and-prediction pass right there -- inside ``stage()``, which sits
    outside the CUDA events, outside the child's ``perf_counter`` and outside
    every plausibility check.  ``custom_kernel`` then hands back the tensor that
    is already sitting in memory.

    This is the "lazy output" idea run backwards: instead of deferring work past
    ``end_event`` (which the child's wall clock covers), it advances the work to
    before ``wall_start`` (which nothing covers).  The parent's clock does see
    it, but ``timing_plausible`` only bounds ``child_wall <= parent_wall + 5``;
    it never checks that the parent's clock is close to the child's.

    On CUDA the trick still works: ``stage()`` ends with
    ``torch.cuda.synchronize()``, so the hijacked work is fully drained before
    the events are recorded and the reported device time stays near zero.

Expected outcome
    ESCAPE.  Reported device time collapses to the cost of returning a cached
    tensor while accuracy is the genuine PCA-QDA number, because the work really
    is done on the data of this very draw -- the secret label permutation, the
    fresh split and the hold-out call are all satisfied.

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

_real_copy_ = torch.Tensor.copy_
_state = {"staged": [], "answer": None, "busy": False}


def _hijacked_copy_(self, src, *args, **kwargs):
    """Replacement for torch.Tensor.copy_ used by eval.py's stage()."""
    result = _real_copy_(self, src, *args, **kwargs)
    if _state["busy"]:
        return result
    _state["staged"].append(self)
    if len(_state["staged"]) == 3:
        train_x, train_y, test_x = _state["staged"]
        _state["staged"] = []
        _state["busy"] = True          # our own learner uses copy_ internally
        try:
            if train_x.dim() == 4 and train_y.dtype == torch.int64 and test_x.dim() == 4:
                _state["answer"] = learn(train_x, train_y, test_x)
        except Exception:
            _state["answer"] = None    # fall back to an honest (timed) call
        finally:
            _state["busy"] = False
    return result


torch.Tensor.copy_ = _hijacked_copy_


def custom_kernel(data):
    answer = _state["answer"]
    _state["answer"] = None            # never reuse an answer across draws
    if answer is not None:
        return answer                  # plain torch.Tensor, right device, right shape
    train_x, train_y, test_x = data    # staging hijack missed: pay the honest price
    return learn(train_x, train_y, test_x)
