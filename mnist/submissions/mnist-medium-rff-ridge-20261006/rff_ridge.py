"""RFF-ridge learner for MNIST-medium, per-draw hyperparameter selection.

This is the frozen learner used for every accuracy number in
[README.md](README.md). It performs one complete training-and-prediction run
for one dataset draw.

Model (frozen, no learned state crosses draws)
----------------------------------------------
    u    = x ** p                                    81-d, 9x9 pixels in [0,1]
    W    = default_rng(42).standard_normal((81, D)) * sqrt(2 * gamma)
    b    = default_rng(42).uniform(0, 2*pi, D)        same generator, next draw
    Z    = [cos(U W + b), 1]                         D+1 columns
    A    = solve(Z^T Z + lam I, Z^T Y)                ridge, Y one-hot
    pred = argmax(Zq A)                               first maximum wins

Selection procedure (frozen, train-only)
----------------------------------------
Per-draw grid of 18 points:

    p     in {0.25, 0.50}
    gamma in {0.01, 0.03, 0.10}
    lam   in {0.10, 1.00, 10.00}

Selected on an inner split of THIS draw's own training subset only:

    idx = default_rng(42).permutation(n_train)   ->   inner-train idx[:8000]
                                                      inner-val   idx[8000:]

Metric is inner-val correct count. Tie-break: smaller lam, then smaller gamma,
then smaller p. The selected point is refit on ALL n_train rows of this draw,
then test images are projected and scored.

`test_labels` are never read by this module. The caller may load them only for
checking the train/test index intersection.

Backends
--------
`fit_predict_np`  NumPy, float64 or float32. The float64 path is the accuracy
                  reference; float32 Gram breaks Cholesky for this conditioning
                  (see README).
`fit_predict_torch`  PyTorch, CPU or CUDA. `gram_dtype` selects the Gram
                  accumulation dtype independently of the design rows, and
                  `f32_products` keeps the block products in float32 while the
                  running sum stays in `gram_dtype`. This is the configuration
                  whose energy was measured; see [energy/README.md](energy/README.md).

Learner seed is 42 in every path. No state is carried between draws.
"""
from __future__ import annotations

import numpy as np

LEARNER_SEED = 42
P_GRID = (0.25, 0.50)
GAMMA_GRID = (0.01, 0.03, 0.10)
LAM_GRID = (0.10, 1.00, 10.00)
N_INNER_TRAIN = 8000
SEEDS = tuple(range(20261001, 20261012))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_draw(npz_path):
    """Return train images/labels/indices and test images/indices.

    test_labels are deliberately not returned: the learner must not see them.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        n_tr = len(z["train_labels"])
        n_te = len(z["test_labels"])
        Xtr = z["train_images"].reshape(n_tr, -1).astype(np.float64)
        ytr = z["train_labels"].astype(np.int64)
        Xte = z["test_images"].reshape(n_te, -1).astype(np.float64)
        tidx = z["train_indices"].astype(np.int64)
        teidx = z["test_indices"].astype(np.int64)
    return Xtr, ytr, Xte, tidx, teidx


def inner_split(n, seed=LEARNER_SEED):
    idx = np.random.default_rng(seed).permutation(n)
    return idx[:N_INNER_TRAIN], idx[N_INNER_TRAIN:]


def check_no_test_overlap(train_idx, test_idx):
    """Rows of the inner-validation split that appear in the test set.

    Returns (inner_val_overlap_rows, train_overlap_rows). Both must be zero.
    """
    _, val = inner_split(len(train_idx))
    return (int(np.intersect1d(train_idx[val], test_idx).size),
            int(np.intersect1d(train_idx, test_idx).size))


def projection(D, gamma, seed=LEARNER_SEED):
    """Draw the frozen projection W and offset b for one basis size."""
    r = np.random.default_rng(seed)
    W = r.standard_normal((81, D)) * np.sqrt(2.0 * gamma)
    b = r.uniform(0.0, 2.0 * np.pi, D)
    return W, b


def onehot(y, dtype=np.float64):
    Y = np.zeros((len(y), 10), dtype=dtype)
    Y[np.arange(len(y)), y] = 1.0
    return Y


# --------------------------------------------------------------------------- #
# NumPy backend
# --------------------------------------------------------------------------- #
def _design_np(U, W, b, dtype):
    return np.hstack([np.cos(U @ W + b), np.ones((len(U), 1), dtype=dtype)])


def fit_predict_np(Xtr, ytr, Xte, D, dtype=np.float64):
    """One complete run. Returns (pred, best, table)."""
    Xtr = Xtr.astype(dtype, copy=False)
    Xte = Xte.astype(dtype, copy=False)
    tr, va = inner_split(len(ytr))
    Y = onehot(ytr, dtype)
    eye = np.eye(D + 1, dtype=dtype)
    table = []
    for p in P_GRID:
        U = Xtr ** p
        for gamma in GAMMA_GRID:
            W, b = projection(D, gamma)
            Z = _design_np(U, W, b, dtype)
            Ztr = Z[tr]
            G, R = Ztr.T @ Ztr, Ztr.T @ Y[tr]
            Zva = Z[va]
            for lam in LAM_GRID:
                A = np.linalg.solve(G + lam * eye, R)
                hit = int(np.equal((Zva @ A).argmax(1), ytr[va]).sum())
                table.append({"p": p, "gamma": gamma, "lam": lam,
                              "inner_val_correct": hit,
                              "inner_val_total": int(len(va))})
                del A
            del Z, Ztr, G, R, Zva
        del U
    best = _select(table)
    W, b = projection(D, best["gamma"])
    Z = _design_np(Xtr ** best["p"], W, b, dtype)
    A = np.linalg.solve(Z.T @ Z + best["lam"] * eye, Z.T @ Y)
    del Z
    Zq = _design_np(Xte ** best["p"], W, b, dtype)
    pred = (Zq @ A).argmax(1).astype(np.int64)
    return pred, best, table


def _select(table):
    ranked = sorted(table, key=lambda r: (-r["inner_val_correct"], r["lam"],
                                          r["gamma"], r["p"]))
    return ranked[0]


# --------------------------------------------------------------------------- #
# PyTorch backend
# --------------------------------------------------------------------------- #
def _design_t(U, W, b):
    import torch
    Z = torch.cos(U @ W + b)
    return torch.cat([Z, torch.ones((Z.shape[0], 1), device=Z.device,
                                    dtype=Z.dtype)], 1)


def _gram_t(U, W, b, rows, Y, chunk, gram_dtype, f32_products):
    """(ZtZ, ZtY) over `rows`.

    chunk == 0       one materialized design block
    chunk > 0        row blocks of `chunk` rows; no design block exceeds chunk
    gram_dtype       accumulation dtype of ZtZ / ZtY
    f32_products     keep each block product in the design dtype and promote
                     only for accumulation; this is the mixed-precision Gram
    """
    import torch
    d = W.shape[1] + 1
    acc = getattr(torch, gram_dtype) if gram_dtype else U.dtype
    ZtZ = torch.zeros((d, d), device=U.device, dtype=acc)
    ZtY = torch.zeros((d, 10), device=U.device, dtype=acc)
    if not chunk:
        ridx = torch.as_tensor(rows, device=U.device)
        Z = _design_t(U[ridx], W, b)
        if acc != Z.dtype and not f32_products:
            Z = Z.to(acc)
            Y = Y.to(acc)
        return (Z.T @ Z).to(acc), (Z.T @ Y[ridx]).to(acc)
    for s in range(0, len(rows), chunk):
        ridx = torch.as_tensor(rows[s:s + chunk], device=U.device)
        Z = _design_t(U[ridx], W, b)
        if acc != Z.dtype and not f32_products:
            Z = Z.to(acc)
            Y = Y.to(acc)
        ZtZ += (Z.T @ Z).to(acc)
        ZtY += (Z.T @ Y[ridx]).to(acc)
        del Z
    return ZtZ, ZtY


def _score_t(U, W, b, rows, A, y, chunk):
    import torch
    hit = 0
    for s in range(0, len(rows), chunk or len(rows)):
        ridx = torch.as_tensor(rows[s:s + (chunk or len(rows))], device=U.device)
        Z = _design_t(U[ridx], W, b)
        hit += int((Z @ A.to(Z.dtype)).argmax(1).eq(y[ridx]).sum())
        del Z
    return hit


def _solve_t(ZtZ, ZtY, lam):
    import torch
    L = torch.linalg.cholesky(ZtZ + lam * torch.eye(
        ZtZ.shape[0], device=ZtZ.device, dtype=ZtZ.dtype))
    return torch.cholesky_solve(ZtY, L)


def fit_predict_torch(Xtr, ytr, Xte, D, device="cuda", dtype="float64",
                      chunk=0, gram_dtype=None, f32_products=False):
    """One complete run on device. Returns (pred, best, table)."""
    import torch
    td = getattr(torch, dtype)
    dev = torch.device(device)
    Xtr_t = torch.as_tensor(Xtr, dtype=td, device=dev)
    Xte_t = torch.as_tensor(Xte, dtype=td, device=dev)
    y_t = torch.as_tensor(ytr, dtype=torch.long, device=dev)
    Y = torch.nn.functional.one_hot(y_t, 10).to(td)
    tr, va = inner_split(len(ytr))
    table = []
    for p in P_GRID:
        U = Xtr_t ** p
        for gamma in GAMMA_GRID:
            Wn, bn = projection(D, gamma)
            W = torch.as_tensor(Wn, dtype=td, device=dev)
            b = torch.as_tensor(bn, dtype=td, device=dev)
            ZtZ, ZtY = _gram_t(U, W, b, tr, Y, chunk, gram_dtype, f32_products)
            for lam in LAM_GRID:
                A = _solve_t(ZtZ, ZtY, lam)
                table.append({"p": p, "gamma": gamma, "lam": lam,
                              "inner_val_correct": _score_t(U, W, b, va, A, y_t, chunk),
                              "inner_val_total": int(len(va))})
                del A
            del ZtZ, ZtY, W, b
        del U
    best = _select(table)
    Wn, bn = projection(D, best["gamma"])
    W = torch.as_tensor(Wn, dtype=td, device=dev)
    b = torch.as_tensor(bn, dtype=td, device=dev)
    U = Xtr_t ** best["p"]
    ZtZ, ZtY = _gram_t(U, W, b, list(range(len(ytr))), Y, chunk, gram_dtype,
                       f32_products)
    A = _solve_t(ZtZ, ZtY, best["lam"])
    del ZtZ, ZtY
    Zq = _design_t(Xte_t ** best["p"], W, b)
    pred = (Zq @ A.to(Zq.dtype)).argmax(1).cpu().numpy().astype(np.int64)
    return pred, best, table