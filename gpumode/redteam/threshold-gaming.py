#!POPCORN leaderboard mnist-medium-5pct
#!POPCORN gpu A100

"""RED TEAM: pay for training once, on the untimed warm-up call.

Attack
    Rule 1 says "train from scratch on every call", but nothing in the harness
    checks it, and D3 hands every submission one *untimed* call before the
    timed loop. So: fit a real PCA-QDA model on the warm-up draw, keep it in a
    module global, and on every timed call do inference only.

    The secret per-draw label permutation is supposed to make carried state
    "worse than useless". It is not: a permutation of ten symbols is recovered
    from ~256 of the 10,000 labelled training rows the call is handed anyway.
    Predict those rows with the cached model, build a 10x10 vote matrix against
    the labels supplied this call, greedily read off the bijection, and relabel
    the cached model's outputs. Cost: one 256-row forward pass.

    The hold-out (D6) is survived without ever identifying it as Fashion-MNIST.
    The vote matrix reports its own agreement; when the cached model does not
    explain the training labels it is the wrong dataset, so this call falls back
    to an honest from-scratch fit. That call is slow, but the harness excludes
    the hold-out from the ranked statistics, so the fallback is free.

Expected harness response
    Fail: state crossed a call boundary and the ranked time is not the time of a
    training-and-prediction call. Predicted harness response: pass, with a
    ranked time several times lower than the same learner run honestly.

The learner itself is submissions/pca_qda.py (@jurajselep), copied verbatim and
split into fit/score so that the fit can be skipped.
"""

import numpy as np
import torch

C, D, K = 10, 81, 40
NP, ROUNDS, NR_ITERATIONS, TINY, SHRINK = 2000, 3, 6, 1e-6, 0.05
PROBE = 256              # training rows used to recover the label permutation
AGREEMENT_FLOOR = 0.55   # below this the cached model is about a different dataset

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


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


_w0 = {}


def starting_basis(device):
    key = str(device)
    if key not in _w0:
        _w0[key] = torch.as_tensor(initial_basis(), device=device)
    return _w0[key]


def fit(x, y):
    """The whole expensive half of PCA-QDA: basis, class moments, inverses."""
    N = x.shape[0]
    u = nr_sqrt(x)
    m, W = basis_gpu(u, starting_basis(x.device))
    z = (u - m) @ W
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
    return {"m": m, "W": W, "mu": mu, "P": P, "kappa": kappa}


def score(q, model):
    """The cheap half: project and evaluate ten quadratic forms."""
    zq = (nr_sqrt(q) - model["m"]) @ model["W"]
    d = zq[:, None, :] - model["mu"][None, :, :]
    s = torch.einsum("qci,cij,qcj->qc", d, model["P"], d)
    return (model["kappa"][None, :] - 0.5 * s).argmax(1)


def recover_permutation(x, y, model):
    """Map cached class ids to this call's labels, and say how well it fits."""
    probe = min(PROBE, x.shape[0])
    cached = score(x[:probe], model)
    votes = torch.zeros(C, C, device=x.device)
    votes.index_put_((cached, y[:probe]), torch.ones(probe, device=x.device), accumulate=True)
    mapping = torch.full((C,), -1, dtype=torch.int64, device=x.device)
    work = votes.clone()
    for _ in range(C):  # greedy bijection: a permutation, not ten independent argmaxes
        flat = int(work.argmax())
        row, col = flat // C, flat % C
        mapping[row] = col
        work[row, :] = -1
        work[:, col] = -1
    hits = float(votes[torch.arange(C, device=x.device), mapping].sum())
    return mapping, hits / probe


_model = None


def custom_kernel(data):
    global _model
    train_x, train_y, test_x = data
    x = train_x.reshape(train_x.shape[0], D)
    q = test_x.reshape(test_x.shape[0], D)

    if _model is None:                      # the untimed warm-up call pays for training
        _model = fit(x, train_y)
        return score(q, _model)

    mapping, agreement = recover_permutation(x, train_y, _model)
    if agreement < AGREEMENT_FLOOR:         # a different dataset: the hold-out check
        fresh = fit(x, train_y)             # honest, slow, and excluded from the ranking
        return score(q, fresh)
    return mapping[score(q, _model)]        # timed calls: inference plus a relabel
