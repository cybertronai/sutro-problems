"""Residual attack on the whitened + randomly-rotated variant: ICA.

Whitening plus a random rotation removes every bit of second-order information
about the pixel grid (``Cov(z) = I`` no matter which rotation was used).  It does
not remove information: ``z = A(x - mu)`` is an invertible linear map of the
pixels, so everything about the images is still there, just in another basis.
What is left to attack is the higher-order structure.

MNIST pixels are an unusually favourable target for ICA-style attacks:

* they are non-negative and heavily zero-inflated -- each pixel's marginal is
  extremely sparse and right-skewed, i.e. maximally non-Gaussian;
* distinct pixels are (conditionally on the digit) far from Gaussian-dependent,
  so "make the coordinates as independent / as non-Gaussian as possible" has an
  answer close to the pixel basis itself.

So the theory says: the rotation *is* identifiable from 3rd/4th-order statistics,
and an attacker who recovers it is handed back the permuted-pixel problem, where
the existing second-order topology attack already works.  This module measures
whether that pipeline actually closes at 81 dimensions with ~20,000 unlabelled
images -- which is the only question that matters for the benchmark.

Pipeline
--------
1. FastICA on the released features (train rows plus, optionally, the query rows
   used unlabelled -- the protocol permits that and records it).
2. Fix each source's sign by its skewness (pixels are right-skewed).
3. Localisation diagnostic: the composite analysis map ``M = W_ica A`` should be a
   scaled signed permutation matrix if the pixels were recovered.  Report the
   per-row peak energy fraction and whether the peak assignment is a bijection.
4. Hand the recovered sources to the existing second-order attack and, if it
   succeeds, to the frozen cnn-09 recipe.  Step 4 is the end-to-end number.

Nothing here reads query labels.  Query *images* are used unlabelled, which the
protocol allows and which an attacker would obviously do.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import study      # noqa: E402
import topology   # noqa: E402
import whiten     # noqa: E402

GRID = 9
NFEAT = 81


def signed_permutation_report(matrix) -> dict:
    """How close is ``matrix`` to a scaled signed permutation matrix?

    ``matrix[i, p]`` is the weight source ``i`` puts on pixel ``p``.  For a perfect
    recovery every row has all of its energy on one pixel and the peak pixels are
    a bijection.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    energy = matrix ** 2
    total = energy.sum(axis=1)
    peak = np.argmax(energy, axis=1)
    fraction = energy[np.arange(len(peak)), peak] / np.maximum(total, 1e-300)
    return {
        "peak_pixel": peak.astype(np.int64),
        "peak_energy_fraction": fraction,
        "mean_peak_energy_fraction": float(fraction.mean()),
        "median_peak_energy_fraction": float(np.median(fraction)),
        "max_peak_energy_fraction": float(fraction.max()),
        "distinct_peak_pixels": int(np.unique(peak).size),
        "is_bijection": bool(np.unique(peak).size == len(peak)),
        "uniform_reference": 1.0 / matrix.shape[1],
    }


def matching_report(matrix, active=None) -> dict:
    """Optimal one-to-one source<->pixel matching, restricted to active pixels.

    The row-wise peak of ``matrix`` can be degenerate (several sources peaking on
    the same pixel).  A Hungarian matching on the row-normalised energy gives the
    honest "how much of the pixel basis did ICA actually recover" number.
    """
    from scipy.optimize import linear_sum_assignment

    matrix = np.asarray(matrix, dtype=np.float64)
    energy = matrix ** 2
    energy = energy / np.maximum(energy.sum(axis=1, keepdims=True), 1e-300)
    if active is not None:
        energy = energy[:, np.asarray(active)]
    rows, cols = linear_sum_assignment(-energy)
    matched = energy[rows, cols]
    return {
        "matched_energy_mean": float(matched.mean()),
        "matched_energy_median": float(np.median(matched)),
        "matched_above_0.5": int((matched > 0.5).sum()),
        "matched_above_0.9": int((matched > 0.9).sum()),
        "n_matched": int(len(matched)),
    }


def composite_controls(transform, seed=12345) -> dict:
    """Reference values for the localisation metric.

    * ``random``: a random orthogonal rotation of the released features -- what an
      attacker gets for free, i.e. the null.
    * ``zca``: the pixel-space ZCA whitening matrix -- the most pixel-aligned white
      basis there is, i.e. the best any ICA-style attack could hope to return.
      (Note that this is the *ceiling* of the metric, not 1.0: no orthogonal
      rotation of whitened data can reproduce the raw pixels, because raw pixels
      are correlated and every ICA output is white.)
    """
    a = np.asarray(transform["A"], dtype=np.float64)
    dim = a.shape[0]
    random = whiten.random_rotation(dim, seed) @ a
    out = {"control_random_rotation": signed_permutation_report(random)
           ["mean_peak_energy_fraction"]}
    eigenvalues = np.asarray(transform["eigenvalues"], dtype=np.float64)
    if a.shape[0] == a.shape[1]:
        pool = np.asarray(study.pool_images(), dtype=np.float64)
        covariance = np.cov(pool, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        zca = (vectors * (1.0 / np.sqrt(np.clip(values, 0, None)
                                        + float(transform["epsilon"])))[None, :]) @ vectors.T
        out["control_zca_whitening"] = signed_permutation_report(zca)[
            "mean_peak_energy_fraction"]
        out["control_zca_matched_energy_mean"] = matching_report(zca)[
            "matched_energy_mean"]
    del eigenvalues
    return out


def positive_part_features(sources) -> np.ndarray:
    """Re-express ICA sources so ``topology``'s ``sqrt(clip(.))`` front end is affine.

    ``recover_layout`` variance-stabilises with ``sqrt(clip(x, 0, None))`` because it
    expects non-negative, zero-inflated pixels.  Feeding it signed sources would
    silently discard half of every coordinate.  Mapping each source through
    ``u = ((s - min) / range) ** 2`` makes the pipeline's square root recover
    ``(s - min) / range`` exactly -- a per-feature affine map, which leaves every
    correlation and partial correlation unchanged.
    """
    s = np.asarray(sources, dtype=np.float64)
    low = s.min(axis=0, keepdims=True)
    span = np.maximum(s.max(axis=0, keepdims=True) - low, 1e-12)
    return np.ascontiguousarray(((s - low) / span) ** 2, dtype=np.float32)


def run_ica(features, n_components=None, fun="logcosh", seed=0, max_iter=2000,
            tol=1e-5):
    """FastICA; returns (sources, unmixing) with ``sources = (x - mean) @ unmixing.T``."""
    from sklearn.decomposition import FastICA
    model = FastICA(n_components=n_components, algorithm="parallel", fun=fun,
                    whiten="unit-variance", max_iter=max_iter, tol=tol,
                    random_state=int(seed))
    sources = model.fit_transform(np.asarray(features, dtype=np.float64))
    # sklearn: S = (X - mean_) @ components_.T
    unmixing = np.asarray(model.components_, dtype=np.float64)
    skew = ((sources - sources.mean(0)) ** 3).mean(0)
    sign = np.where(skew < 0, -1.0, 1.0)
    return sources * sign[None, :], unmixing * sign[:, None], model


def attack(variant_kwargs, n=10000, use_query=True, n_components=None,
           fun="logcosh", seed=0, run_cnn=False, dev_seed=2026092301):
    """Full ICA attack on one whitened variant.  Returns a JSON-ready record."""
    pool = np.asarray(study.pool_images())
    transform = whiten.fit_transform(pool, **variant_kwargs)
    arrays = whiten.job_arrays(dev_seed, n, transform)
    unlabeled = np.concatenate([arrays["train_x"], arrays["query_x"]], axis=0) \
        if use_query else arrays["train_x"]

    started = time.time()
    sources, unmixing, model = run_ica(unlabeled, n_components=n_components,
                                       fun=fun, seed=seed)
    ica_seconds = time.time() - started

    composite = unmixing @ np.asarray(transform["A"], dtype=np.float64)
    report = signed_permutation_report(composite)
    record = {
        "variant": variant_kwargs, "n_train": int(n),
        "n_unlabeled_rows": int(unlabeled.shape[0]),
        "uses_query_images_unlabeled": bool(use_query),
        "ica_fun": fun, "ica_components": int(sources.shape[1]),
        "ica_seconds": ica_seconds,
        "ica_converged_iterations": int(getattr(model, "n_iter_", -1)),
        "mean_peak_energy_fraction": report["mean_peak_energy_fraction"],
        "median_peak_energy_fraction": report["median_peak_energy_fraction"],
        "max_peak_energy_fraction": report["max_peak_energy_fraction"],
        "distinct_peak_pixels": report["distinct_peak_pixels"],
        "peak_assignment_is_bijection": report["is_bijection"],
        "uniform_reference": report["uniform_reference"],
    }

    # reference points for the localisation number
    record["reference_synthesis_random_rotation"] = float(
        whiten.filter_peaks(transform)["mean_peak_energy_fraction"])
    record.update(composite_controls(transform))
    pool_variance = np.asarray(study.pool_images(), dtype=np.float64).var(axis=0)
    active = np.where(pool_variance > 1e-4)[0]
    record["n_active_pixels"] = int(active.size)
    record.update({"matching_" + k: v for k, v in
                   matching_report(composite, active).items()})

    # The attacker's estimate of the pixel values: source j contributes s_j * b_j
    # to the image, with b_j the j-th column of the mixing matrix.  If b_j is
    # (nearly) a single pixel, then s_j * ||b_j|| is (nearly) that pixel's value.
    # Rescaling by ||b_j|| is what undoes ICA's unit-variance convention and turns
    # white sources back into correlated pixel estimates -- without which the
    # second-order topology attack has nothing to work with.
    mixing = np.linalg.pinv(composite)
    scale = np.linalg.norm(mixing, axis=0)
    sources = sources * scale[None, :]
    if sources.shape[1] < NFEAT:
        # recover_layout's contract is 81 features.  Pad the discarded directions
        # with constants: the variance screen parks them on the leftover lattice
        # cells, which is exactly what an attacker who kept k < 81 components
        # would do.
        pad = np.zeros((sources.shape[0], NFEAT - sources.shape[1]))
        sources = np.concatenate([sources, pad], axis=1)
        composite = np.concatenate(
            [composite, np.zeros((NFEAT - composite.shape[0], composite.shape[1]))],
            axis=0)
        report = signed_permutation_report(composite)
        record["padded_to_81"] = True
    train_sources = sources[:n] if use_query else sources
    feed = positive_part_features(train_sources)
    started = time.time()
    try:
        layout = np.asarray(topology.recover_layout(feed, max_seconds=120.0),
                            dtype=np.int64)
        record["recover_layout_failed"] = None
    except Exception as error:
        record["recover_layout_failed"] = f"{type(error).__name__}: {error}"
        layout = None
    record["recover_layout_seconds"] = time.time() - started

    if layout is not None and feed.shape[1] == NFEAT:
        pixel_of = np.asarray(report["peak_pixel"])
        rows, cols = pixel_of // GRID, pixel_of % GRID
        d2 = (rows[:, None] - rows[None, :]) ** 2 + (cols[:, None] - cols[None, :]) ** 2
        true_adjacent = d2 == 1
        recovered_adjacent = topology.ADJACENCY[np.ix_(layout, layout)] > 0
        upper = np.triu(np.ones((NFEAT, NFEAT), dtype=bool), 1)
        pairs = int((recovered_adjacent & upper).sum())
        hits = int((recovered_adjacent & true_adjacent & upper).sum())
        record["adjacency_precision"] = hits / max(pairs, 1)
        record["chance_adjacency_precision"] = float((true_adjacent & upper).sum()
                                                     / upper.sum())
        record["exact_placement_fraction_best_dihedral"] = max(
            float((relabel[layout] == pixel_of).mean())
            for relabel in topology.DIHEDRAL)

        if run_cnn:
            import learners
            labels = np.load(study.POOL_LABELS_PATH)
            truth = labels[study.query_indices(dev_seed)]
            if not use_query:
                raise ValueError("run_cnn needs the query rows through the same ICA")
            query_sources = sources[n:]
            low = train_sources.min(0, keepdims=True)
            span = np.maximum(train_sources.max(0, keepdims=True) - low, 1e-12)
            train_feed = np.ascontiguousarray((train_sources - low) / span, np.float32)
            query_feed = np.ascontiguousarray((query_sources - low) / span, np.float32)
            topology.recover_layout = (                       # noqa: E731
                lambda train_x, orient=False, max_seconds=120.0,
                return_details=False, _layout=layout:
                (_layout, {"substituted": True}) if return_details else _layout)
            out = learners.fit_predict(train_feed, arrays["train_y"], query_feed,
                                       {"family": "topo_cnn", "member_seeds": [101],
                                        "epochs": 25},
                                       seed=11, deadline_unix=time.time() + 3600.0,
                                       device="cpu")
            record["cnn_query_error_pct"] = 100.0 * float((out["labels"] != truth).mean())
    return record


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="ICA attack on the whitened variant")
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--rotation-seed", type=int, default=20260923)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--components", type=int, default=None)
    parser.add_argument("--fun", default="logcosh", choices=["logcosh", "exp", "cube"])
    parser.add_argument("--no-query", action="store_true")
    parser.add_argument("--cnn", action="store_true")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args(argv)

    record = attack({"epsilon": arguments.epsilon,
                     "rotation_seed": arguments.rotation_seed, "method": "zca"},
                    n=arguments.n, use_query=not arguments.no_query,
                    n_components=arguments.components, fun=arguments.fun,
                    run_cnn=arguments.cnn)
    print(json.dumps({k: v for k, v in record.items()
                      if not isinstance(v, np.ndarray)}, indent=2, default=str))
    if arguments.out:
        Path(arguments.out).write_text(json.dumps(record, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
