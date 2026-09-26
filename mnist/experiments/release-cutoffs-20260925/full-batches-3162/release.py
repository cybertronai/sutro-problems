"""Verbatim copy of the gpumode harness release map, so no gpumode file is shipped.

SOURCE
------
gpumode harness 1.2.0 -- ``/Users/yaroslavvb/git/sutro-problems/gpumode/eval.py``
sha256 ``0279a99ae94990f4d2828576440874cee2592ea728f6fb625687e44f86801c36``
extracted 2026-09-24 for mnist/experiments/release-ladder-20260924.

The four salts and the four functions below (``split_universes``,
``haar_rotation``, ``release_map``, ``apply_release``) are byte-for-byte the
harness definitions, pulled out with ``ast.get_source_segment`` rather than
retyped; ``test_study.py`` re-imports ``eval.py`` from the harness directory and
asserts bit-identical output on random data and on one real draw. Nothing else
lives in this module: it is baked into the Modal image, so anything added here
would travel to the container.

``z = Q W (x - mu)``: ``W`` is exact PCA whitening onto the top ``release_dims``
principal directions of the draw's TRAINING rows (no variance floor), ``mu`` is
their mean, and ``Q`` is a per-draw secret Haar rotation. The map is refitted at
every training level; ``Q`` depends only on the dataset seed, so it is shared by
all levels of one seed.
"""
from __future__ import annotations

import numpy as np

HARNESS_VERSION_NOTE = "gpumode harness 1.2.0"
HARNESS_VERSION_STRING = "sutro-mnist-medium-time/1.2.0"  # utils.HARNESS_VERSION
HARNESS_EVAL_SHA256 = "0279a99ae94990f4d2828576440874cee2592ea728f6fb625687e44f86801c36"
HARNESS_EVAL_PATH = "/Users/yaroslavvb/git/sutro-problems/gpumode/eval.py"

# ---- verbatim: eval.py seed salts -----------------------------------------
LABEL_SALT = 0x5EED
UNIVERSE_SALT = 0x5711
DRAW_SALT = 0xD4A7
RELEASE_SALT = 0x4C17

# ---- verbatim: eval.py release map ----------------------------------------


def split_universes(count, seed, salt):
    """Split a pool in half, once per evaluation, from the secret seed.

    Training halves are drawn from one half and test halves from the other, so
    no test image is ever shown with a label earlier in the same run. Without
    this, every draw re-splits the same 60,000 rows and a submission can build a
    hash table of the images it has already been taught.
    """
    order = np.random.default_rng([int(seed), salt]).permutation(count)
    middle = count // 2
    return order[:middle], order[middle:]

def haar_rotation(dim, seed):
    """A Haar-uniform ``dim x dim`` orthogonal matrix from the secret seed.

    QR of a standard-normal matrix, with the sign of each column fixed by the
    sign of the matching diagonal entry of ``R``; without that fix the QR is not
    Haar-uniform. Ported from ``common.haar`` of the rotation-obfuscation study.
    """
    rng = np.random.default_rng([int(seed), RELEASE_SALT])
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))[None, :]

def release_map(train_rows, seed, release_dims):
    """Fit one draw's secret release map ``(mu, W, Q)`` on its training rows.

    ``W`` (``release_dims x D``) is exact PCA whitening onto the top
    ``release_dims`` principal directions of the training rows: rows
    ``u_i.T / sqrt(lambda_i)``, no variance floor, eigenvector signs pinned by
    the largest-magnitude entry so the fit is deterministic. ``mu`` is the
    training-row mean and ``Q`` is Haar-random per draw. The released array is
    ``z = Q W (x - mu)``.

    A floor (ZCA with ``eps``) would leak the dead-border subspace through the
    bottom eigenvectors, which is why this is exact whitening on the top
    directions instead (DESIGN.md D11, and the rotation-obfuscation study of
    2026-09-23).

    The caller must keep the result secret: it is the inverse of the obfuscation
    and never leaves this process. Exposed as a function so the tests can refit
    it; ``make_draw`` does not return it.
    """
    flat = np.asarray(train_rows, dtype=np.float64).reshape(len(train_rows), -1)
    dim = flat.shape[1]
    if not 0 < release_dims <= dim:
        raise ValueError(
            f"release_dims must be between 1 and {dim} for {dim}-pixel images, got {release_dims}"
        )
    if len(flat) <= release_dims:
        raise ValueError(
            f"a draw of {len(flat)} training rows cannot support a {release_dims}-dimensional "
            "whitening fit"
        )
    mu = flat.mean(0)
    covariance = np.cov(flat - mu, rowvar=False)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(dim)])
    signs[signs == 0] = 1.0
    vectors = vectors * signs[None, :]
    top = np.arange(dim - release_dims, dim)  # eigh returns ascending eigenvalues
    eigenvalues, vectors = eigenvalues[top], vectors[:, top]
    if eigenvalues.min() <= 1e-10:
        raise ValueError(
            f"the draw's training rows are rank deficient: principal direction "
            f"{release_dims} has variance {eigenvalues.min():.3e}, so whitening would divide "
            "by zero; lower release_dims or raise train"
        )
    whitener = (vectors / np.sqrt(eigenvalues)[None, :]).T
    return mu, whitener, haar_rotation(release_dims, seed)

def apply_release(images, mu, transform):
    """``z = A (x - mu)`` for a stack of images, in float64, returned as float32."""
    flat = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
    return np.ascontiguousarray((flat - mu) @ transform.T, dtype=np.float32)
