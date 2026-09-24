"""Whitened + randomly-rotated variant of the permutation-invariant MNIST-medium task.

Motivation
----------
The study's current obfuscation is a fixed permutation of the 81 features.  It is
not an obfuscation at all in practice: ``topology.recover_layout`` reconstructs
the 9x9 lattice exactly from the *second-order* statistics of the permuted pixels
(a ridge-regularised partial correlation matrix is non-zero essentially only on
lattice edges), after which a convolutional learner runs at full spatial accuracy.

This module implements the obvious fix the user asked about: whiten the pixels
(so the covariance carries no information at all) and then apply a random
orthogonal rotation (so no coordinate corresponds to a pixel any more).

Why the two steps together are the right thing
----------------------------------------------
After whitening, ``Cov(z) = I``.  For **any** orthogonal ``Q``, ``Cov(Qz) = I`` as
well.  So the covariance of the released data is *exactly* the same object no
matter which rotation was used: the second-order statistics contain zero bits
about ``Q``, hence zero bits about the pixel grid.  This is not "the attack gets
harder", it is "the attack's entire input is a constant".  Whitening *without*
a rotation would not do this (ZCA is symmetric and keeps each coordinate close to
its own pixel), and a rotation *without* whitening would not do it either (the
covariance would then be ``Q Sigma Q^T``, from which ``Sigma``'s eigenstructure and
therefore the lattice are recoverable up to the same dihedral ambiguity).

What survives is the higher-order structure: ``z`` is not Gaussian, so ICA-style
attacks on 3rd/4th-order statistics can in principle find a rotation that makes
the coordinates maximally independent, and for natural images that rotation tends
to land on localised, edge-like filters.  That is a real residual attack surface;
this module only removes the cheap second-order one.  ``attack_ica.py`` measures
the residual.

Map
---
    z = A (x - mean),      A = P Q U diag(1/sqrt(lambda + eps)) U^T   (method 'zca')
    z = A (x - mean),      A = P Q diag(1/sqrt(lambda + eps)) U^T     (method 'pca')

``U``/``lambda`` are the eigenvectors/eigenvalues of the pool covariance, ``Q`` is a
random orthogonal matrix and ``P`` is the study's fixed 81-feature permutation.
``P`` is applied for consistency with the existing protocol only; a permutation of
a rotated vector is just another rotation, so ``P Q`` is itself Haar-distributed
and ``P`` adds nothing.  It is kept so that the released variant is the "same
protocol plus a preprocessing step" rather than a different protocol.  For the
same reason ``method`` barely matters once ``Q`` is Haar: ``Q W_zca = (QU) W_pca``
and ``QU`` is Haar whenever ``Q`` is, so the two methods release data with the same
distribution.  Both are provided so the choice can be stated explicitly.

The variance floor is the one real design decision
--------------------------------------------------
``epsilon`` is in variance units of [0,1] pixels.  The 9x9 pool has 17 eigenvalues
below 1e-4 (the always-black frame and its near-duplicates); the smallest is
1.5e-8.  Exact whitening would amplify those directions by up to 8,000x and hand
the learner ~17 unit-variance coordinates of pure area-resize quantisation noise.

But the floor is not free, and this is the subtlety worth stating plainly: with a
floor the released covariance is ``Q diag(lambda/(lambda+eps)) Q^T``, which is
*not* the identity.  Second-order statistics then no longer carry zero bits about
``Q`` -- they identify the low-variance subspace (the blank-border directions,
which in pixel space are close to axis-aligned).  The degenerate top of the
spectrum, where ``lambda/(lambda+eps) ~ 1``, stays unidentifiable, so the leak is
confined to the tail; but it is a leak.

Three positions on that trade-off, all supported here:

* ``epsilon > 0``, 81 features (the default; what the measurements below use).
  Usable data, small residual second-order leak in the low-variance tail.
* ``epsilon = 0``, 81 features.  Exactly identity covariance, zero second-order
  leak, and ~17 coordinates of amplified junk that cost the dense learners
  accuracy.
* ``n_components = k``: keep the top ``k`` eigen-directions, whiten them exactly
  and drop the rest.  Covariance is exactly the identity on ``k`` features, there
  is no tail to leak and no noise to amplify -- at the price of releasing ``k``
  features instead of 81 and discarding the (negligible) variance below the cut.
  This is the cleanest construction of the three.

Measured: the rotation is the active ingredient, the whitening is a tax
-----------------------------------------------------------------------
Two ablations on dev seed 2026092301 settle which half of "whiten and rotate"
does the work (numbers in ``research/whitening-dense-results.md``):

* ZCA whitening with **no** rotation: the topology attack still puts 44% of its
  claimed lattice edges on genuinely adjacent pixels against a 4.4% chance rate.
  ZCA filters are localised centre-surround, so each coordinate still *is* its
  pixel.  Whitening alone is not a defence.
* A random rotation with **no** whitening: the attack drops to 3.5% against the
  same 4.4% chance rate, and the arc-cosine and RBF kernel-ridge baselines are
  unchanged to within 0.23 pp and 0.00 pp respectively (RBF exactly so -- it sees
  only pairwise distances, which a rotation preserves).

That is also what the algebra says.  An attacker can always whiten the released
data for himself, so releasing it pre-whitened hands him nothing he did not have;
both ``Q x`` and ``Q W x`` leave him with the same residual problem of recovering an
unknown orthogonal factor from higher-order statistics.  Whitening only changes
what the *learner* sees -- and it costs the kernels roughly 1 pp at N=1,000 and
0.5-1.5 pp at N=10,000.  ``method='rotate_only'`` is therefore the variant these
measurements support.

The whitening statistics are fitted on the whole 60,000-image pool, i.e. on the
same object as the feature permutation.  This is a *protocol* decision made by the
benchmark designer, exactly like ``perm``; it is not information leaked to a
learner (the learner still only ever sees its own train/query rows), but it does
mean the transform is a property of the pool and must be published with the
protocol.  A pool-independent alternative (fit on the training rows only) would
make the map depend on ``n`` and on the dataset seed; that is measured nowhere here
and is flagged in the write-up.

No Modal, no GPU, no network.  Nothing here reads query labels.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import study  # noqa: E402

METHODS = ("zca", "pca", "rotate_only")
ROTATIONS = ("qr_pcg64", "ortho_group", "none")
DEFAULT_EPSILON = 1e-3
DEFAULT_ROTATION_SEED = 20260923


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _hash(array) -> str:
    """sha256 of the C-order little-endian bytes of ``array`` (float64 canonical)."""
    value = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(value.tobytes()).hexdigest()


def random_rotation(dim: int, seed: int, method: str = "qr_pcg64") -> np.ndarray:
    """Haar-distributed orthogonal matrix, deterministic in ``seed``.

    ``qr_pcg64`` (default) is the Mezzadri construction: QR of an i.i.d. standard
    normal matrix drawn from ``PCG64(seed)``, with the columns rescaled by the sign
    of the diagonal of ``R`` so the result is Haar and not merely orthogonal.  It is
    used as the default because it depends only on numpy's PCG64 stream and on
    LAPACK's QR, so it reproduces across SciPy versions (SciPy is absent from the
    study's Modal image).  ``ortho_group`` calls ``scipy.stats.ortho_group``.
    """
    if method == "none":
        return np.eye(dim)
    if method == "qr_pcg64":
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        gaussian = rng.standard_normal((dim, dim))
        q, r = np.linalg.qr(gaussian)
        return q * np.sign(np.diag(r))[None, :]
    if method == "ortho_group":
        from scipy.stats import ortho_group
        return np.asarray(ortho_group.rvs(dim, random_state=int(seed)), dtype=np.float64)
    raise ValueError(f"unknown rotation {method!r}; choose from {ROTATIONS}")


def fit_transform(pool_images, epsilon: float = DEFAULT_EPSILON,
                  rotation_seed: int = DEFAULT_ROTATION_SEED, method: str = "zca",
                  rotation: str = "qr_pcg64", apply_permutation: bool = True,
                  n_components: int = None) -> dict:
    """Fit the whitening + rotation map on the unpermuted pool.

    Returns a dict describing ``z = A (x - mean)``.  Deterministic in
    ``(pool_images, epsilon, rotation_seed, method, rotation, apply_permutation,
    n_components)``.

    ``n_components=k`` keeps only the ``k`` largest-variance eigen-directions; ``A``
    is then ``(k, 81)`` and ``A_inv`` is its pseudo-inverse (a right inverse:
    ``A @ A_inv = I_k``).  The study permutation cannot act on ``k != 81`` features
    and is skipped in that case -- it was redundant with ``Q`` anyway.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    epsilon = float(epsilon)
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    x = np.asarray(pool_images, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("pool_images must be 2-D")
    rows, dim = x.shape
    if rows < dim + 1:
        raise ValueError("need more rows than features to estimate the covariance")

    mean = x.mean(axis=0)
    centered = x - mean
    covariance = centered.T @ centered / float(rows - 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, vectors = np.linalg.eigh(covariance)      # ascending
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    # Fix the eigenvector sign convention so the map is reproducible across
    # LAPACK builds (eigh's signs are arbitrary).  Irrelevant for 'zca', which is
    # sign-invariant, but it makes 'pca' deterministic too.
    signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(dim)])
    signs[signs == 0] = 1.0
    vectors = vectors * signs[None, :]

    kept = dim if n_components is None else int(n_components)
    if not (1 <= kept <= dim):
        raise ValueError(f"n_components must satisfy 1 <= k <= {dim}, got {n_components}")
    if kept < dim:
        # keep the largest-variance directions (eigh returns them ascending)
        keep_index = np.arange(dim - kept, dim)
        used_values, used_vectors = eigenvalues[keep_index], vectors[:, keep_index]
    else:
        used_values, used_vectors = eigenvalues, vectors

    inverse_root = 1.0 / np.sqrt(used_values + epsilon)
    if method == "rotate_only":
        # No whitening at all: z = P Q (x - mean).  Kept as a first-class option
        # because the measurements show the random rotation, not the whitening,
        # is what defeats the pixel-topology attack -- and a rotation alone is
        # free for every rotation-invariant kernel.
        if kept != dim:
            raise ValueError("rotate_only cannot drop components")
        whitener = np.eye(dim)
    elif method == "zca" and kept == dim:
        whitener = (used_vectors * inverse_root[None, :]) @ used_vectors.T
    else:
        # A reduced-rank 'zca' is not defined; with a Haar Q the two coincide in
        # distribution anyway (Q W_zca = (QU) W_pca).
        whitener = (used_vectors * inverse_root[None, :]).T
    q = random_rotation(kept, rotation_seed, rotation)
    forward = q @ whitener
    if apply_permutation and kept == dim:
        permutation = study.feature_permutation()
        if permutation.shape != (dim,):
            raise ValueError("the study feature permutation does not match the features")
        forward = forward[permutation, :]
    else:
        permutation = np.arange(kept, dtype=np.int64)
    inverse = np.linalg.inv(forward) if kept == dim else np.linalg.pinv(forward)

    below = {threshold: int((eigenvalues < threshold).sum())
             for threshold in (1e-3, 1e-4, 1e-5, 1e-6)}
    return {
        "A": np.ascontiguousarray(forward),
        "A_inv": np.ascontiguousarray(inverse),
        "mean": np.ascontiguousarray(mean),
        "eigenvalues": np.ascontiguousarray(eigenvalues),
        "epsilon": epsilon,
        "rotation_seed": int(rotation_seed),
        "rotation": str(rotation),
        "method": str(method),
        "apply_permutation": bool(apply_permutation and kept == dim),
        "n_features": int(dim),
        "n_released": int(kept),
        "n_components": None if n_components is None else int(n_components),
        "n_pool_rows": int(rows),
        "eigenvalue_min": float(eigenvalues.min()),
        "eigenvalue_max": float(eigenvalues.max()),
        "eigenvalue_min_kept": float(used_values.min()),
        "eigenvalues_below": below,
        # variance actually achieved per released direction: lambda/(lambda+eps)
        "achieved_variance_min": float((used_values / (used_values + epsilon)).min())
        if epsilon > 0 else 1.0,
        "achieved_variance_mean": float((used_values / (used_values + epsilon)).mean())
        if epsilon > 0 else 1.0,
        "condition_number": float(np.sqrt((used_values.max() + epsilon)
                                          / (used_values.min() + epsilon))),
        "condition_number_A": float(np.linalg.cond(forward)),
        "pool_images_sha256": _hash(np.asarray(pool_images, dtype=np.float32)),
        "A_sha256": _hash(forward),
        "A_inv_sha256": _hash(inverse),
        "mean_sha256": _hash(mean),
        "eigenvalues_sha256": _hash(eigenvalues),
        "permutation_sha256": hashlib.sha256(
            np.ascontiguousarray(permutation, dtype=np.int64).tobytes()).hexdigest(),
        "description": (
            "z = A (x - mean) with A = P Q W, W = %s whitening with variance floor "
            "epsilon=%g, Q a %s random rotation (seed %d), P the study's fixed "
            "81-feature permutation" % (method, epsilon, rotation, int(rotation_seed))),
    }


def default_transform(epsilon: float = DEFAULT_EPSILON,
                      rotation_seed: int = DEFAULT_ROTATION_SEED,
                      method: str = "zca", rotation: str = "qr_pcg64") -> dict:
    """``fit_transform`` on the study's canonical unpermuted pool."""
    return fit_transform(study.pool_images(), epsilon=epsilon,
                         rotation_seed=rotation_seed, method=method, rotation=rotation)


def apply(images, transform: dict) -> np.ndarray:
    """Map raw (M,81) unpermuted pixels through the fitted transform; float32 out."""
    values = np.asarray(images, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != transform["A"].shape[1]:
        raise ValueError("images must be (M, n_features) unpermuted pixels")
    z = (values - transform["mean"][None, :]) @ transform["A"].T
    return np.ascontiguousarray(z, dtype=np.float32)


def invert(z, transform: dict) -> np.ndarray:
    """Inverse of :func:`apply` (float64 out); used only by diagnostics."""
    values = np.asarray(z, dtype=np.float64)
    return np.ascontiguousarray(values @ transform["A_inv"].T + transform["mean"][None, :])


def job_arrays(seed: int, n: int, transform: dict) -> dict:
    """Mirror of ``study.job_arrays`` in the whitened + rotated space.

    Identical draws (``study.draw_order``; train = order[:n], query =
    order[10000:20000]); the only difference is that the released features are
    ``z = A(x - mean)`` instead of ``x[:, perm]``.  The permutation is already
    folded into ``A``.  Query labels are never returned.
    """
    n = int(n)
    if not (1 <= n <= study.QUERY_START):
        raise ValueError(f"n must satisfy 1 <= n <= {study.QUERY_START}, got {n}")
    images = study.pool_images()
    labels = np.load(study.POOL_LABELS_PATH)
    order = study.draw_order(seed)
    train = order[:n]
    query = order[study.QUERY_START:study.QUERY_START + study.QUERY_COUNT]
    if query.shape[0] != study.QUERY_COUNT:
        raise AssertionError("query slice must contain exactly 10,000 rows")
    if np.intersect1d(train, query).size != 0:
        raise AssertionError("train and query indices overlap")
    return {
        "train_x": apply(images[train], transform),
        "train_y": np.ascontiguousarray(labels[train], dtype=np.uint8),
        "query_x": apply(images[query], transform),
    }


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def neighbour_correlation_report(z, permutation=None) -> dict:
    """Mean |Pearson r| over true grid-neighbour vs non-neighbour feature pairs.

    ``z`` are released features.  When ``permutation`` is given (the study's
    permuted-pixel variant) feature ``j`` is pixel ``permutation[j]``; otherwise
    feature ``j`` is pixel ``j``.  In the whitened+rotated variant no feature is a
    pixel at all, so "neighbour pairs" are the pairs that *would* be neighbours if
    the rotation were the identity -- the point of the diagnostic is that the two
    numbers become indistinguishable.
    """
    values = np.asarray(z, dtype=np.float64)
    dim = values.shape[1]
    grid = int(round(dim ** 0.5))
    if grid * grid != dim:
        raise ValueError("the neighbour diagnostic needs a square feature count")
    corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    pixel_of = np.arange(dim) if permutation is None else np.asarray(permutation)
    rows, cols = pixel_of // grid, pixel_of % grid
    d2 = (rows[:, None] - rows[None, :]) ** 2 + (cols[:, None] - cols[None, :]) ** 2
    neighbour = (d2 == 1)
    other = (d2 > 1)
    magnitude = np.abs(corr)
    return {
        "mean_abs_corr_neighbour": float(magnitude[neighbour].mean()),
        "mean_abs_corr_non_neighbour": float(magnitude[other].mean()),
        "max_abs_corr_offdiagonal": float(magnitude[~np.eye(dim, dtype=bool)].max()),
        "n_neighbour_pairs": int(neighbour.sum() // 2),
        "n_non_neighbour_pairs": int(other.sum() // 2),
    }


def filter_peaks(transform: dict) -> dict:
    """Where each released coordinate lives in pixel space.

    Coordinate ``j`` of ``z`` is the projection of ``x - mean`` onto row ``j`` of ``A``;
    the *synthesis* filter (what ``z_j`` contributes to the reconstructed image) is
    column ``j`` of ``A_inv``.  For the permuted-pixel protocol each synthesis filter
    is a single pixel (energy fraction 1.0); after a random rotation the filters are
    delocalised and the peak-pixel fraction collapses toward 1/81.
    """
    inverse = np.asarray(transform["A_inv"], dtype=np.float64)
    energy = inverse ** 2
    total = energy.sum(axis=0)
    peak_pixel = np.argmax(energy, axis=0)
    peak_fraction = energy[peak_pixel, np.arange(energy.shape[1])] / np.maximum(total, 1e-300)
    return {
        "peak_pixel": peak_pixel.astype(np.int64),
        "peak_energy_fraction": peak_fraction,
        "mean_peak_energy_fraction": float(peak_fraction.mean()),
        "median_peak_energy_fraction": float(np.median(peak_fraction)),
        "distinct_peak_pixels": int(np.unique(peak_pixel).size),
        "uniform_reference": 1.0 / energy.shape[0],
    }
