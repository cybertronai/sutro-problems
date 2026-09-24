"""Data, release maps and shared helpers for the rotation-obfuscation study.

Question: does releasing z = Q W (x - mu) (Q secret Haar rotation, W identity or
a whitening matrix) stop an entrant from reusing precomputed spatial features
on MNIST-medium (9x9, 10k train / 10k test)?

Everything here is deterministic in the seeds.  Attackers never receive Q, W or
mu; they get only what ``release`` returns.  The 60,000-image official MNIST
training split is the pool; each draw uses 10,000 train + 10,000 test rows and
the remaining 40,000 rows are the "public" set the informed attacker holds.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = Path('/Users/yaroslavvb/git/sutro-problems/output/gpumode-cache/raw')
CACHE = HERE / 'pool9.npz'
GRID, NFEAT = 9, 81
N_TRAIN = N_TEST = 10000


def _idx(path, offset):
    return np.frombuffer(gzip.open(path).read(), np.uint8)[offset:]


def _area_weights(n_in, n_out):
    w = np.zeros((n_out, n_in))
    s = n_in / n_out
    for j in range(n_out):
        lo, hi = j * s, (j + 1) * s
        for i in range(n_in):
            w[j, i] = max(0.0, min(hi, i + 1) - max(lo, i))
    return w / s


def pool():
    """(60000, 81) float32 in [0,1] area-resized 9x9 digits, and uint8 labels."""
    if CACHE.exists():
        d = np.load(CACHE)
        return d['x'], d['y']
    x28 = _idx(RAW / 'train-images-idx3-ubyte.gz', 16).reshape(60000, 28, 28).astype(np.float32) / 255.0
    y = _idx(RAW / 'train-labels-idx1-ubyte.gz', 8).astype(np.uint8)
    w = _area_weights(28, GRID).astype(np.float32)
    x = np.einsum('ji,nik,lk->njl', w, x28, w).reshape(60000, NFEAT)
    x = np.clip(x, 0, 1).astype(np.float32)
    np.savez(CACHE, x=x, y=y)
    return x, y


def draw(seed):
    """Disjoint train / test / public row indices for one draw."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(60000)
    return order[:N_TRAIN], order[N_TRAIN:N_TRAIN + N_TEST], order[N_TRAIN + N_TEST:]


def haar(dim, seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))[None, :]


def whitener(x, method, eps=1e-3, k=None):
    """Return (W, mu) with W (k_or_81, 81) so that W (x - mu) is white (up to eps)."""
    x = np.asarray(x, np.float64)
    mu = x.mean(0)
    cov = np.cov(x - mu, rowvar=False)
    lam, u = np.linalg.eigh(cov)
    lam = np.clip(lam, 0, None)
    # deterministic eigenvector signs (matters only for pca)
    sgn = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(u.shape[1])])
    sgn[sgn == 0] = 1
    u = u * sgn[None, :]
    if method == 'zca':
        w = (u / np.sqrt(lam + eps)[None, :]) @ u.T
    elif method == 'pca':
        if k is not None:
            idx = np.arange(NFEAT - k, NFEAT)
            lam, u = lam[idx], u[:, idx]
        w = (u / np.sqrt(lam + eps)[None, :]).T
    else:
        raise ValueError(method)
    return w, mu, lam


VARIANTS = {
    # name: dict(whiten=None|'zca'|'pca', eps, k, rotate=bool, permute=bool)
    'raw':         dict(whiten=None, rotate=False, permute=False),
    'perm':        dict(whiten=None, rotate=False, permute=True),
    'rot':         dict(whiten=None, rotate=True, permute=False),
    'zca1e-3+rot': dict(whiten='zca', eps=1e-3, rotate=True, permute=False),
    'zca1e-2+rot': dict(whiten='zca', eps=1e-2, rotate=True, permute=False),
    'pca60+rot':   dict(whiten='pca', eps=0.0, k=60, rotate=True, permute=False),
    'zca1e-3':     dict(whiten='zca', eps=1e-3, rotate=False, permute=False),
}


def release(variant, x_fit, x_train, x_test, seed, fit_test_separately=False, x_fit_test=None):
    """Build the released arrays for one variant.

    ``x_fit`` are the rows the organiser fits W, mu on (the draw's train rows
    or the whole pool).  With ``fit_test_separately`` the test rows get their own
    W, mu fitted on ``x_fit_test`` (the test rows themselves), same Q.
    Returns dict with z_train, z_test and the secret map (for scoring only).
    """
    v = VARIANTS[variant]
    dim_out = NFEAT
    if v['whiten'] is None:
        w, mu = np.eye(NFEAT), np.zeros(NFEAT)
        w_te, mu_te = w, mu
    else:
        w, mu, _ = whitener(x_fit, v['whiten'], v.get('eps', 0.0), v.get('k'))
        if fit_test_separately:
            w_te, mu_te, _ = whitener(x_fit_test, v['whiten'], v.get('eps', 0.0), v.get('k'))
        else:
            w_te, mu_te = w, mu
        dim_out = w.shape[0]
    q = haar(dim_out, seed * 7919 + 17) if v['rotate'] else np.eye(dim_out)
    if v['permute']:
        p = np.random.default_rng(seed * 104729 + 3).permutation(dim_out)
        q = q[p]
    a, a_te = q @ w, q @ w_te
    zt = ((x_train.astype(np.float64) - mu) @ a.T).astype(np.float32)
    ze = ((x_test.astype(np.float64) - mu_te) @ a_te.T).astype(np.float32)
    return {'z_train': zt, 'z_test': ze,
            'secret': {'A': a, 'A_test': a_te, 'mu': mu, 'mu_test': mu_te, 'Q': q, 'W': w}}


# ------------------------------------------------------------ scoring helpers
def basis_recovery_report(m, active=None):
    """m[i, p]: weight that recovered coordinate i puts on true pixel p (81x81 or kx81).

    Returns Hungarian-matched energy fraction (1.0 = every recovered coordinate is
    exactly one pixel).  With ``active`` (bool mask over pixels) the matching is
    restricted to active pixel columns and reported as ``active_*``.
    """
    from scipy.optimize import linear_sum_assignment
    e = np.asarray(m, np.float64) ** 2
    e = e / np.maximum(e.sum(1, keepdims=True), 1e-300)
    r, c = linear_sum_assignment(-e)
    matched = e[r, c]
    out = {'matched_energy_mean': float(matched.mean()),
           'matched_energy_median': float(np.median(matched)),
           'n_above_0.5': int((matched > 0.5).sum()),
           'n_above_0.9': int((matched > 0.9).sum()),
           'peak_pixel': c[np.argsort(r)].tolist()}
    if active is not None:
        ea = e[:, np.asarray(active)]
        r, c = linear_sum_assignment(-ea)
        ma = ea[r, c]
        out.update({'active_n_pixels': int(np.asarray(active).sum()), 'active_matched_energy_mean': float(ma.mean()),
                    'active_n_above_0.5': int((ma > 0.5).sum()), 'active_n_above_0.9': int((ma > 0.9).sum())})
    return out


def pixel_reconstruction_error(x_hat, x_true, active_mask=None):
    """Relative RMS error of an attacker's pixel estimate after the best per-pixel affine fit
    is NOT allowed: the attacker must produce pixels in the fixed pixel scale.  We report
    raw relative RMS and the fraction of variance explained."""
    d = x_hat - x_true
    if active_mask is not None:
        d = d[:, active_mask]; x_true = x_true[:, active_mask]
    return {'rel_rms': float(np.sqrt((d ** 2).mean() / max((x_true ** 2).mean(), 1e-12)))}


def adjacency_from_layout(layout):
    """layout[i] = lattice cell of recovered coordinate i.  Returns bool 81x81 adjacency in coordinate index."""
    rc = np.stack([layout // GRID, layout % GRID], 1)
    d2 = ((rc[:, None, :] - rc[None, :, :]) ** 2).sum(-1)
    return d2 == 1


def adjacency_precision(layout, peak_pixel):
    """Fraction of recovered lattice edges that are true pixel neighbours,
    where recovered coordinate i is 'really' pixel peak_pixel[i]."""
    rec = adjacency_from_layout(np.asarray(layout))
    tru = adjacency_from_layout(np.asarray(peak_pixel))
    upper = np.triu(np.ones_like(rec), 1)
    pairs = int((rec & upper).sum())
    return float((rec & tru & upper).sum() / max(pairs, 1))


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


class Timer:
    def __init__(self): self.t = time.perf_counter()
    def lap(self):
        now = time.perf_counter(); d = now - self.t; self.t = now; return d
