"""Shared plumbing for the linear-release red team.

Every attack script in this directory gets its data from the real evaluator:
``eval.make_draw(pool, seed, n_train, n_test, universes, release_dims=K)``.
What comes back is exactly what a submission's ``custom_kernel`` receives --
released train features, released (secretly permuted) train labels, released
test features -- plus the test labels, which the harness keeps and which we use
ONLY for scoring.

Scoring also needs the secret map ``z = Q W (x - mu)``.  We never read it out of
``eval``.  We re-derive the draw's row indices with the evaluator's own RNG
recipe, take the true pixels of those rows from the pool we loaded ourselves,
and least-squares fit ``A`` in ``z = (x - mu) A^T + c``.  The fit is exact (the
release IS linear in the pixels), and ``check_oracle`` asserts the residual is
at the float32 noise floor, which doubles as a check that the rows we recovered
are the rows the evaluator drew.  Any attack that touches ``oracle`` is a
scoring step, never an attacker step.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G = HERE.parent.parent                      # .../gpumode
STUDY = Path('/Users/yaroslavvb/git/sutro-problems/mnist/experiments/rotation-obfuscation-20260923')
POOL_CACHE = Path('/tmp/gpumode-pool')

for p in (str(G), str(STUDY)):
    if p not in sys.path:
        sys.path.insert(0, p)

import mnist_data                            # noqa: E402  (from G)
import eval as harness_eval                  # noqa: E402  (from G)
import common                                # noqa: E402  (rotation study)
import attacks                               # noqa: E402  (rotation study)
import models                                # noqa: E402  (rotation study)

GRID, NFEAT = 9, 81
RELEASE_DIMS = 60
N_TRAIN = N_TEST = 10000
# Case seed of every shipped band (bands.json defaults.test_seed) is 101; the
# ranked seed is secret.  We use the public test seed so the draw is one an
# organiser can regenerate.
CASE_SEED = 101


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def acc(pred, y):
    return float((np.asarray(pred) == np.asarray(y)).mean() * 100)


def jdump(path, obj):
    Path(path).write_text(json.dumps(
        obj, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


# --------------------------------------------------------------- waiting for eval
def release_supported():
    import inspect
    return 'release_dims' in inspect.signature(harness_eval.make_draw).parameters


def require_release(max_wait_s=1200, poll_s=60):
    """Poll until the concurrently-edited eval.py grows ``release_dims``."""
    import importlib
    global harness_eval
    waited = 0.0
    while not release_supported():
        if waited >= max_wait_s:
            raise SystemExit(
                'eval.make_draw still has no release_dims parameter after '
                f'{waited:.0f}s of waiting; rerun when the evaluator change has landed.')
        log(f'eval.make_draw has no release_dims yet; waiting {poll_s}s '
            f'({waited:.0f}/{max_wait_s}s)')
        time.sleep(poll_s)
        waited += poll_s
        harness_eval = importlib.reload(harness_eval)
    return harness_eval


def load_pool(dataset='mnist'):
    return mnist_data.load_pool(POOL_CACHE, dataset, GRID)


# --------------------------------------------------------------- the draw
def draw_rows(pool, seed, n_train, n_test, universes):
    """Re-derive the rows eval.make_draw picked (scoring side only)."""
    train_universe, test_universe = universes
    rng = np.random.default_rng([int(seed), harness_eval.DRAW_SALT])
    train_rows = rng.choice(train_universe, n_train, replace=False)
    test_rows = rng.choice(test_universe, n_test, replace=False)
    return train_rows, test_rows


class Draw:
    """One evaluator draw, with a scoring-side oracle bolted on the side."""

    def __init__(self, dataset='mnist', case_seed=CASE_SEED, draw_seed=None,
                 release_dims=RELEASE_DIMS, n_train=N_TRAIN, n_test=N_TEST):
        require_release()
        # eval.run_case splits the universes from the CASE seed and then draws
        # call i with a derived draw seed; we reproduce both so the draw is one
        # a real ranked run would serve.
        if draw_seed is None:
            draw_seed = (case_seed + harness_eval.TIMED_STRIDE if dataset == 'mnist'
                         else case_seed + harness_eval.HOLDOUT_OFFSET + harness_eval.HOLDOUT_STRIDE)
        self.dataset, self.case_seed, self.seed = dataset, case_seed, draw_seed
        self.release_dims = release_dims
        pool = load_pool(dataset)
        self.pool_x = pool[0].reshape(len(pool[0]), -1).astype(np.float32)
        self.pool_y = pool[1]
        salt = harness_eval.UNIVERSE_SALT + (0 if dataset == 'mnist' else 1)
        self.universes = harness_eval.split_universes(len(pool[1]), case_seed, salt)
        t0 = time.perf_counter()
        visible, self.y_test = harness_eval.make_draw(
            pool, draw_seed, n_train, n_test, self.universes, release_dims=release_dims)
        self.make_draw_seconds = time.perf_counter() - t0
        self.z_train, self.y_train, self.z_test = visible
        self.z_train = np.asarray(self.z_train)
        self.z_test = np.asarray(self.z_test)
        self.y_train = np.asarray(self.y_train)
        self.y_test = np.asarray(self.y_test)
        # ---- scoring oracle
        self.train_rows, self.test_rows = draw_rows(
            pool, draw_seed, n_train, n_test, self.universes)
        self.public_rows = np.setdiff1d(
            np.arange(len(self.pool_y)), np.concatenate([self.train_rows, self.test_rows]))
        self.x_train = self.pool_x[self.train_rows]
        self.x_test = self.pool_x[self.test_rows]
        self.true_y_train = self.pool_y[self.train_rows]
        self.true_y_test = self.pool_y[self.test_rows]
        self._fit_oracle()

    # the secret map, recovered for scoring only
    def _fit_oracle(self):
        if self.release_dims == 0:
            self.A = np.eye(NFEAT)
            self.offset = np.zeros(NFEAT)
            self.oracle_rel_rms = 0.0
            return
        xb = np.concatenate([self.x_train.astype(np.float64),
                             np.ones((len(self.x_train), 1))], 1)
        coef, *_ = np.linalg.lstsq(xb, self.z_train.astype(np.float64), rcond=None)
        self.A = coef[:-1].T                              # k x 81
        self.offset = coef[-1]                            # k
        pred = self.x_train.astype(np.float64) @ self.A.T + self.offset
        d = pred - self.z_train
        self.oracle_rel_rms = float(np.sqrt((d ** 2).mean() / max((self.z_train ** 2).mean(), 1e-30)))

    def check_oracle(self, tol=1e-5):
        if self.oracle_rel_rms > tol:
            raise SystemExit(
                'oracle map fit residual %.3g > %.3g: the rows we re-derived are not the rows '
                'eval.make_draw used, or the release is not linear in the pixels.'
                % (self.oracle_rel_rms, tol))
        # the released labels must be a permutation of the true ones
        perm = self.label_permutation()
        if not np.array_equal(perm[self.true_y_train], self.y_train):
            raise SystemExit('released train labels are not a fixed permutation of the pool labels')
        self.construction = self.verify_construction()
        return True

    def verify_construction(self):
        """Independent check that the release really is Q W (x - mu).

        Refit the documented recipe (exact PCA whitening onto the top
        release_dims directions of the draw's own train rows) with the rotation
        study's ``common.whitener``, then read off Q = A W^+ from the map we
        recovered by least squares.  If the construction is what eval.py
        documents, Q is orthogonal and the offset is exactly -A mu.
        """
        if self.release_dims == 0:
            return {'checked': False}
        w, mu, lam = common.whitener(self.x_train, 'pca', eps=0.0, k=self.release_dims)
        q_hat = self.A @ np.linalg.pinv(w)
        eye = q_hat @ q_hat.T
        return {
            'checked': True,
            'refit_whitener_orthogonality_error': float(
                np.abs(eye - np.eye(len(eye))).max()),
            'offset_matches_minus_A_mu': float(
                np.abs(self.offset + self.A @ mu).max()),
            'smallest_kept_eigenvalue': float(lam.min()),
            'rotation_is_not_identity': float(np.abs(q_hat - np.eye(len(q_hat))).max()),
        }

    def label_permutation(self):
        """relabel[true] = released (scoring side; the evaluator's secret permutation)."""
        perm = np.full(10, -1, np.int64)
        for c in range(10):
            m = self.true_y_train == c
            if m.any():
                perm[c] = int(np.bincount(self.y_train[m], minlength=10).argmax())
        return perm

    def summary(self):
        return {
            'dataset': self.dataset, 'case_seed': self.case_seed, 'draw_seed': self.seed,
            'release_dims': self.release_dims,
            'z_train_shape': list(self.z_train.shape), 'z_test_shape': list(self.z_test.shape),
            'z_train_dtype': str(self.z_train.dtype),
            'released_train_cov_offdiag_rms': float(_cov_offdiag_rms(self.z_train)),
            'released_train_cov_diag_mean': float(np.diag(_cov(self.z_train)).mean()),
            'oracle_map_fit_rel_rms': self.oracle_rel_rms,
            'make_draw_seconds': self.make_draw_seconds,
            'secret_label_permutation_recovered_for_scoring': self.label_permutation().tolist(),
            'construction_check': getattr(self, 'construction', None),
        }


def _cov(z):
    z = np.asarray(z, np.float64)
    return np.cov(z - z.mean(0), rowvar=False)


def _cov_offdiag_rms(z):
    c = _cov(z)
    off = c - np.diag(np.diag(c))
    return np.sqrt((off ** 2).mean())


# ------------------------------------------- attacker-side: undo the secret relabel
def invariant_class_features(v, y, k_eig=12):
    """Features of a class that survive an unknown orthogonal map of the features.

    ``v`` must already be whitened with statistics computed on ALL rows (so the
    whitening itself is equivariant).  Under v -> R v with R orthogonal, the
    class mean norm and the class covariance spectrum are unchanged.
    """
    out = []
    for c in range(10):
        vc = v[y == c]
        mu = vc.mean(0)
        lam = np.linalg.eigvalsh(np.cov(vc - mu, rowvar=False))[::-1][:k_eig]
        out.append(np.concatenate([[np.linalg.norm(mu), len(vc) / len(v)], lam]))
    return np.stack(out)


def recover_label_permutation(z_train, y_train, x_pub, y_pub, k=RELEASE_DIMS):
    """Attacker step: match released class ids to public class ids.

    The evaluator applies a fresh secret permutation of the 10 labels to every
    draw, so an attacker holding public MNIST cannot pair "released class 3"
    with "public class 3" by name.  Both sides are PCA-whitened to their top-k
    subspace, which makes the two feature spaces differ by an orthogonal map;
    class-mean norms and class-covariance spectra are invariants of that map, so
    the pairing is a 10x10 assignment problem.  No secrets are used.

    Returns ``released_of_true`` (index by the public/true class, get the
    released class) and a diagnostics dict.
    """
    from scipy.optimize import linear_sum_assignment
    x_pub = np.asarray(x_pub, np.float64)
    z_train = np.asarray(z_train, np.float64)
    w_pub, mu_pub, _ = common.whitener(x_pub, 'pca', eps=0.0, k=k)
    u = (x_pub - mu_pub) @ w_pub.T
    m_rel = z_train.mean(0)
    lam, ev = np.linalg.eigh(np.cov(z_train - m_rel, rowvar=False))
    keep = np.argsort(-lam)[:k]
    v = (z_train - m_rel) @ (ev[:, keep] / np.sqrt(np.clip(lam[keep], 1e-12, None))[None, :])
    fu, fv = invariant_class_features(u, y_pub), invariant_class_features(v, y_train)
    scale = np.maximum(np.abs(fu).mean(0), 1e-9)
    cost = ((fv[:, None, :] - fu[None, :, :]) / scale[None, None, :]) ** 2
    cost = cost.sum(-1)                     # cost[released, true]
    rel_idx, true_idx = linear_sum_assignment(cost)
    released_of_true = np.empty(10, np.int64)
    released_of_true[true_idx] = rel_idx
    return released_of_true, {'assignment_cost': float(cost[rel_idx, true_idx].sum()),
                              'mean_matched_cost': float(cost[rel_idx, true_idx].mean()),
                              'mean_unmatched_cost': float(cost.mean())}
