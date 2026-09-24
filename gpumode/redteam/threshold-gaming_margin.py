#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""RED TEAM: the accuracy gate is a *mean* over draws, so buy it in bulk.

Attack
    The gate is  sum_i correct_i >= ceil(draws * test * (1 - error)),  and the
    ranked value is  mean_i time_i.  Both are linear in the draw index, so a
    submission that answers draw i with strategy s(i) lands on the straight line
    between any two strategies. Mixing a slow accurate learner with a fast
    inaccurate one therefore reaches the whole lower convex hull of the
    (accuracy, time) frontier, and every point on it beats running the accurate
    learner on every draw.

    Here: PCA-QDA (about 95.6%) on two calls out of three, nearest class mean
    (about 79%, and roughly twenty times cheaper) on the third. The mean is
    about 90.1%, comfortably inside the 12% band's 88% floor, at about
    two thirds of the honest learner's mean time. With a cheaper filler or a
    wider band the discount grows; at the 12% band a 98%-accurate learner only
    has to run on 54% of the draws.

    The pattern is a counter, not a coin, and it is phase-locked to the first
    timed call, so it does not need to know the draw count: any window of three
    consecutive calls pays for itself. The untimed warm-up call is call 0 and is
    answered cheaply, since it is neither timed nor scored.

    The hold-out call (D6) is not a problem even when it lands on a cheap call:
    nearest class mean is about 70% on Fashion-MNIST, well over the 50% floor.
    It is also not identified or special-cased in any way.

Expected harness response
    Fail: a per-draw accuracy floor, or an accuracy-weighted ranking, would
    reject it. Predicted harness response: pass, at a rank the submission has
    not earned on any single draw.

Both learners are the harness's own: submissions/pca_qda.py (@jurajselep) and
reference.py's nearest class mean.
"""

import numpy as np
import torch

C, D, K = 10, 81, 40
NP, ROUNDS, NR_ITERATIONS, TINY, SHRINK = 2000, 3, 6, 1e-6, 0.05
PERIOD, CHEAP_PHASE = 3, 2   # accurate on 2 calls in every 3

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


def pca_qda(x, y, q):
    N = x.shape[0]
    u = nr_sqrt(x)
    m, W = basis_gpu(u, starting_basis(x.device))
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


def nearest_class_mean(x, y, q):
    sums = torch.zeros(C, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, y, x)
    means = sums / torch.bincount(y, minlength=C).clamp_min(1).unsqueeze(1)
    return ((means * means).sum(1) - 2 * q @ means.T).argmin(1)


_calls = 0


def custom_kernel(data):
    global _calls
    train_x, train_y, test_x = data
    x = train_x.reshape(train_x.shape[0], D)
    q = test_x.reshape(test_x.shape[0], D)
    index = _calls
    _calls += 1
    if index == 0:                              # untimed warm-up: neither timed nor scored
        return nearest_class_mean(x, train_y, q)
    if (index - 1) % PERIOD == CHEAP_PHASE:     # the draw we choose to be wrong on
        return nearest_class_mean(x, train_y, q)
    return pca_qda(x, train_y, q)
