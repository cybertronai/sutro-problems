"""CPU classical baselines for permutation-invariant MNIST-medium (9x9 -> 81).

Every family here is exactly permutation invariant in the sense that matters for
this study: it consumes the 81 features as an unordered feature vector and never
uses pixel coordinates.  Query labels are never an argument; hyper-parameters are
chosen by stratified 5-fold cross-validation on the training rows alone, so the
selection is bit-identical for any ``query_x``.

``fit_predict(train_x, train_y, query_x, config, seed)`` returns
``{'logits': float32 (Q,10), 'labels': uint8 (Q,), 'metrics': dict}`` with
``labels`` the first-maximum argmax of ``logits`` (ties resolved to the lowest
class), matching the study-wide prediction contract.
"""

from __future__ import annotations

import platform
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

FAMILIES = ("svm_rbf", "knn", "kernel_ridge", "hgb")
KERNELS = ("rbf", "arccos1", "ntk_relu")
NORMALIZATION = "4*x - 0.5"
N_CLASSES = 10
N_FOLDS = 5
TINY = 1e-12


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def normalize(x):
    """Study-wide input scaling for the kernel and SVM families."""
    return 4.0 * np.asarray(x, dtype=np.float64) - 0.5


def argmax_labels(logits):
    """First-maximum argmax (ties resolved to the lowest class), as uint8."""
    return np.argmax(np.asarray(logits), axis=1).astype(np.uint8)


def _check(train_x, train_y, query_x):
    train_x = np.asarray(train_x, dtype=np.float64)
    query_x = np.asarray(query_x, dtype=np.float64)
    train_y = np.asarray(train_y).astype(np.int64)
    if train_x.ndim != 2 or query_x.ndim != 2:
        raise ValueError("train_x and query_x must be 2-D")
    if train_x.shape[1] != query_x.shape[1]:
        raise ValueError("train_x and query_x must share the feature dimension")
    if train_y.shape != (train_x.shape[0],):
        raise ValueError("train_y must have one label per training row")
    if train_y.min() < 0 or train_y.max() >= N_CLASSES:
        raise ValueError("labels must lie in 0..9")
    return train_x, train_y, query_x


def _folds(train_y, seed):
    counts = np.bincount(train_y, minlength=N_CLASSES)
    n_splits = int(min(N_FOLDS, counts[counts > 0].min()))
    n_splits = max(n_splits, 2)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    return list(splitter.split(np.zeros(len(train_y)), train_y)), n_splits


def _cv_subset(train_y, limit, seed):
    """Deterministic stratified subsample of the training rows for the CV stage."""
    n = len(train_y)
    if limit is None or limit >= n:
        return np.arange(n)
    rng = np.random.default_rng(int(seed) + 991)
    keep = []
    for cls in range(N_CLASSES):
        rows = np.where(train_y == cls)[0]
        if rows.size == 0:
            continue
        take = max(int(round(limit * rows.size / n)), min(N_FOLDS, rows.size))
        keep.append(rng.permutation(rows)[:take])
    return np.sort(np.concatenate(keep))


def _one_hot(y):
    out = np.zeros((len(y), N_CLASSES))
    out[np.arange(len(y)), y] = 1.0
    return out


# --------------------------------------------------------------------------- #
# explicit kernels (float64)
# --------------------------------------------------------------------------- #
def rbf_kernel(a, b, gamma):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sq = ((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T))
    return np.exp(-float(gamma) * np.maximum(sq, 0.0))


def _arccos_angles(cross, norm_a, norm_b):
    scale = np.sqrt(np.outer(norm_a, norm_b))
    cosine = np.clip(cross / np.maximum(scale, TINY), -1.0, 1.0)
    return scale, np.arccos(cosine)


def arccos1_kernel(a, b, depth=1):
    """Cho & Saul order-1 arc-cosine kernel, composed ``depth`` times.

    ``k_1(x,y) = (1/pi)|x||y|(sin t + (pi - t) cos t)`` with ``t`` the angle
    between ``x`` and ``y``; the diagonal ``k_1(x,x) = |x|^2`` is preserved, so
    the self-norms used by deeper compositions are the input norms.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    depth = int(depth)
    if depth < 1:
        raise ValueError("depth must be >= 1")
    diag_a = (a * a).sum(1)
    diag_b = (b * b).sum(1)
    cross = a @ b.T
    for _ in range(depth):
        scale, theta = _arccos_angles(cross, diag_a, diag_b)
        cross = scale * (np.sin(theta) + (np.pi - theta) * np.cos(theta)) / np.pi
    return cross


def ntk_relu_kernel(a, b, depth=3):
    """Infinite-width ReLU NTK with He (``c_sigma = 2``) scaling, ``depth`` layers."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    depth = int(depth)
    if depth < 1:
        raise ValueError("depth must be >= 1")
    diag_a = (a * a).sum(1)
    diag_b = (b * b).sum(1)
    sigma = a @ b.T
    theta_ntk = sigma.copy()
    for _ in range(depth):
        scale, theta = _arccos_angles(sigma, diag_a, diag_b)
        sigma = scale * (np.sin(theta) + (np.pi - theta) * np.cos(theta)) / np.pi
        sigma_dot = (np.pi - theta) / np.pi
        theta_ntk = theta_ntk * sigma_dot + sigma
    return theta_ntk


def _kernel_fn(name, config):
    if name == "rbf":
        gamma = config.get("gamma", 0.02)
        return lambda p, q: rbf_kernel(p, q, gamma)
    if name == "arccos1":
        depth = config.get("depth", 1)
        return lambda p, q: arccos1_kernel(p, q, depth)
    if name == "ntk_relu":
        depth = config.get("depth", 3)
        return lambda p, q: ntk_relu_kernel(p, q, depth)
    raise ValueError(f"unknown kernel {name!r}; choose from {KERNELS}")


def build_kernel(name, a, b, config):
    """Gram matrix, filled in row blocks so the float64 temporaries stay small.

    A 10,000 x 10,000 float64 block is 800 MB; computing one arc-cosine or NTK
    layer needs three such temporaries at once, so an unblocked build peaks at
    several gigabytes.  Blocking caps the temporaries at ``kernel_block`` rows.
    """
    kernel = _kernel_fn(name, config)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    block = int(config.get("kernel_block", 2000))
    if block <= 0 or a.shape[0] <= block:
        return kernel(a, b)
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.float64)
    for start in range(0, a.shape[0], block):
        stop = min(start + block, a.shape[0])
        out[start:stop] = kernel(a[start:stop], b)
    return out


def _ridge_solve(gram, targets, lam, inplace=False):
    """Solve ``(K + lam*n I) alpha = Y``; return ``(alpha, used_lstsq_fallback)``.

    ``inplace=True`` lets the Cholesky overwrite ``gram`` (the caller must not
    reuse it) -- worth 800 MB at N=10,000.  The kernels here are PSD and
    ``lam > 0``, so the factorisation only fails on genuine numerical trouble;
    when a copy still exists we fall back to a least-squares solve *and say so*,
    and when it does not we surface the failure rather than solve a
    half-overwritten matrix.  Cross-validation refuses to select a lambda whose
    fold solves needed the fallback, because the full-N refit (``inplace=True``)
    has no fallback available and would abort the job.
    """
    gram = np.ascontiguousarray(gram)
    n = gram.shape[0]
    matrix = gram if inplace else gram.copy()
    matrix.flat[::n + 1] += lam * n
    from scipy.linalg import cho_factor, cho_solve
    try:
        factor = cho_factor(matrix, lower=True, check_finite=False,
                            overwrite_a=True)
        return cho_solve(factor, targets, check_finite=False), False
    except Exception as error:
        if inplace:
            raise RuntimeError(
                f"Cholesky failed for lambda={lam!r} on an n={n} kernel; "
                "increase lambda_grid") from error
        rebuilt = gram.copy()
        rebuilt.flat[::n + 1] += lam * n
        return np.linalg.lstsq(rebuilt, targets, rcond=None)[0], True


# --------------------------------------------------------------------------- #
# families
# --------------------------------------------------------------------------- #
def _fit_svm_rbf(train_x, train_y, query_x, config, seed, metrics):
    c_grid = list(config.get("C_grid", [2, 5, 10, 20]))
    gamma_grid = list(config.get("gamma_grid", ["scale", 0.01, 0.02, 0.05]))
    subset = _cv_subset(train_y, config.get("cv_subsample", 4000), seed)
    sub_x, sub_y = train_x[subset], train_y[subset]
    folds, n_splits = _folds(sub_y, seed)
    best = None
    table = []
    for c_value in c_grid:
        for gamma in gamma_grid:
            correct = 0
            total = 0
            for train_idx, test_idx in folds:
                model = SVC(C=float(c_value), gamma=gamma, kernel="rbf",
                            decision_function_shape="ovr", random_state=int(seed),
                            cache_size=int(config.get("cache_size", 500)))
                model.fit(sub_x[train_idx], sub_y[train_idx])
                predicted = model.predict(sub_x[test_idx])
                correct += int((predicted == sub_y[test_idx]).sum())
                total += len(test_idx)
            score = correct / total
            table.append({"C": c_value, "gamma": gamma, "cv_accuracy": score})
            if best is None or score > best[0] + 1e-12:
                best = (score, c_value, gamma)
    score, c_value, gamma = best
    metrics.update(chosen_C=c_value, chosen_gamma=gamma, cv_accuracy=score,
                   cv_folds=n_splits, cv_rows=int(len(subset)), cv_table=table)
    model = SVC(C=float(c_value), gamma=gamma, kernel="rbf",
                decision_function_shape="ovr", random_state=int(seed),
                cache_size=int(config.get("cache_size", 500)))
    started = time.time()
    model.fit(train_x, train_y)
    metrics["training_seconds"] = time.time() - started
    metrics["n_support_vectors"] = int(model.support_vectors_.shape[0])
    started = time.time()
    scores = np.asarray(model.decision_function(query_x), dtype=np.float64)
    metrics["inference_seconds"] = time.time() - started
    logits = np.zeros((query_x.shape[0], N_CLASSES))
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    logits[:, model.classes_.astype(int)] = scores
    return logits


def _fit_knn(train_x, train_y, query_x, config, seed, metrics):
    k_grid = [int(k) for k in config.get("k_grid", [1, 3, 5, 7])]
    folds, n_splits = _folds(train_y, seed)
    best = None
    table = []
    for k in k_grid:
        correct = 0
        total = 0
        for train_idx, test_idx in folds:
            usable = min(k, len(train_idx))
            model = KNeighborsClassifier(n_neighbors=usable, algorithm="brute")
            model.fit(train_x[train_idx], train_y[train_idx])
            predicted = model.predict(train_x[test_idx])
            correct += int((predicted == train_y[test_idx]).sum())
            total += len(test_idx)
        score = correct / total
        table.append({"k": k, "cv_accuracy": score})
        if best is None or score > best[0] + 1e-12:
            best = (score, k)
    score, k = best
    metrics.update(chosen_k=k, cv_accuracy=score, cv_folds=n_splits,
                   cv_rows=int(len(train_y)), cv_table=table)
    usable = min(k, len(train_y))
    model = KNeighborsClassifier(n_neighbors=usable, algorithm="brute")
    started = time.time()
    model.fit(train_x, train_y)
    metrics["training_seconds"] = time.time() - started
    started = time.time()
    neighbours = model.kneighbors(query_x, return_distance=False)
    votes = np.zeros((query_x.shape[0], N_CLASSES))
    labels = train_y[neighbours]
    for cls in range(N_CLASSES):
        votes[:, cls] = (labels == cls).sum(1)
    logits = votes / float(usable)
    metrics["inference_seconds"] = time.time() - started
    return logits


def _kernel_ridge_cv(gram, train_y, lambda_grid, folds):
    """Cross-validate the ridge strength; never return a numerically unusable one.

    A lambda whose fold solve fell back to ``lstsq`` still scores, but selecting
    it would hand the full-N refit a Cholesky it cannot factor (the refit runs
    ``inplace=True`` and raises by design), aborting the job.  Such rows are kept
    in the reported table with ``unstable: true`` and excluded from the choice.
    """
    targets = _one_hot(train_y)
    table = []
    best = None
    for lam in lambda_grid:
        correct = 0
        total = 0
        unstable = False
        for train_idx, test_idx in folds:
            alpha, fell_back = _ridge_solve(gram[np.ix_(train_idx, train_idx)],
                                            targets[train_idx], float(lam))
            unstable = unstable or fell_back
            predicted = np.argmax(gram[np.ix_(test_idx, train_idx)] @ alpha, axis=1)
            correct += int((predicted == train_y[test_idx]).sum())
            total += len(test_idx)
        score = correct / total
        table.append({"lambda": lam, "cv_accuracy": score, "unstable": unstable})
        if unstable:
            continue
        if best is None or score > best[0] + 1e-12:
            best = (score, lam)
    if best is None:
        # Every lambda was unstable: take the largest (best conditioned) one and
        # let the caller record that the grid needs raising.
        lam = max(float(value) for value in lambda_grid)
        score = max(row["cv_accuracy"] for row in table if float(row["lambda"]) == lam)
        best = (score, lam)
    return best, table


def _fit_kernel_ridge(train_x, train_y, query_x, config, seed, metrics):
    kernel = config.get("kernel", "rbf")
    lambda_grid = list(config.get("lambda_grid", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]))
    gamma_grid = list(config.get("gamma_grid", [0.01, 0.02, 0.05])) if kernel == "rbf" else [None]
    subset = _cv_subset(train_y, config.get("cv_subsample", 4000), seed)
    sub_x, sub_y = train_x[subset], train_y[subset]
    folds, n_splits = _folds(sub_y, seed)
    best = None
    table = []
    for gamma in gamma_grid:
        settings = dict(config)
        if gamma is not None:
            settings["gamma"] = gamma
        gram = build_kernel(kernel, sub_x, sub_x, settings)
        scale = float(np.mean(np.diag(gram)))
        gram = gram / max(scale, TINY)
        (score, lam), rows = _kernel_ridge_cv(gram, sub_y, lambda_grid, folds)
        for row in rows:
            row["gamma"] = gamma
        table.extend(rows)
        if best is None or score > best[0] + 1e-12:
            best = (score, lam, gamma)
    score, lam, gamma = best
    settings = dict(config)
    if gamma is not None:
        settings["gamma"] = gamma
    metrics["cv_unstable_rows"] = int(sum(bool(row.get("unstable")) for row in table))
    metrics["cv_all_unstable"] = bool(table) and metrics["cv_unstable_rows"] == len(table)
    metrics.update(chosen_kernel=kernel, chosen_lambda=lam, chosen_gamma=gamma,
                   chosen_depth=int(config.get("depth", 1 if kernel == "arccos1" else 3)),
                   cv_accuracy=score, cv_folds=n_splits, cv_rows=int(len(subset)),
                   cv_table=table)
    started = time.time()
    gram = build_kernel(kernel, train_x, train_x, settings)
    scale = float(np.mean(np.diag(gram)))
    gram /= max(scale, TINY)
    alpha, _ = _ridge_solve(gram, _one_hot(train_y), float(lam), inplace=True)
    del gram
    metrics["training_seconds"] = time.time() - started
    metrics["kernel_diagonal_scale"] = scale
    started = time.time()
    logits = np.empty((query_x.shape[0], N_CLASSES))
    block = int(config.get("predict_block", 2000))
    for start in range(0, query_x.shape[0], block):
        stop = min(start + block, query_x.shape[0])
        cross = build_kernel(kernel, query_x[start:stop], train_x, settings)
        cross /= max(scale, TINY)
        logits[start:stop] = cross @ alpha
        del cross
    metrics["inference_seconds"] = time.time() - started
    return logits


def _fit_hgb(train_x, train_y, query_x, config, seed, metrics):
    model = HistGradientBoostingClassifier(
        max_iter=int(config.get("max_iter", 500)),
        learning_rate=float(config.get("learning_rate", 0.1)),
        max_leaf_nodes=int(config.get("max_leaf_nodes", 31)),
        early_stopping=False,
        random_state=int(seed),
    )
    started = time.time()
    model.fit(train_x, train_y)
    metrics["training_seconds"] = time.time() - started
    metrics.update(max_iter=int(config.get("max_iter", 500)),
                   learning_rate=float(config.get("learning_rate", 0.1)),
                   max_leaf_nodes=int(config.get("max_leaf_nodes", 31)),
                   cv_accuracy=None)
    started = time.time()
    scores = np.asarray(model.decision_function(query_x), dtype=np.float64)
    metrics["inference_seconds"] = time.time() - started
    logits = np.zeros((query_x.shape[0], N_CLASSES))
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    logits[:, model.classes_.astype(int)] = scores
    return logits


_DISPATCH = {
    "svm_rbf": _fit_svm_rbf,
    "knn": _fit_knn,
    "kernel_ridge": _fit_kernel_ridge,
    "hgb": _fit_hgb,
}


def fit_predict(train_x, train_y, query_x, config, seed):
    """Fit one classical family and predict the query rows.  No query labels."""
    family = config.get("family")
    if family not in _DISPATCH:
        raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")
    train_x, train_y, query_x = _check(train_x, train_y, query_x)
    wall = time.time()
    scaled = family in ("svm_rbf", "knn", "kernel_ridge")
    fit_x = normalize(train_x) if scaled else train_x
    use_x = normalize(query_x) if scaled else query_x
    metrics = {
        "family": family,
        "normalization": NORMALIZATION if scaled else "none",
        "n_train": int(train_x.shape[0]),
        "n_query": int(query_x.shape[0]),
        "n_features": int(train_x.shape[1]),
        "seed": int(seed),
        "uses_query_images_unlabeled": False,
        "query_labels_supplied": False,
        "truncated": False,
        "epochs_completed": None,
        "platform": platform.platform(),
    }
    logits = _DISPATCH[family](fit_x, train_y, use_x, config, seed, metrics)
    logits = np.ascontiguousarray(logits, dtype=np.float32)
    labels = argmax_labels(logits)
    metrics["total_seconds"] = time.time() - wall
    # training_seconds covers the final refit only; make the rest of the
    # train-side cost (the CV grid, kernel builds, normalisation) visible so the
    # three timings add up to the wall time the runner records.
    metrics["cv_seconds"] = max(0.0, metrics["total_seconds"]
                                - float(metrics.get("training_seconds", 0.0))
                                - float(metrics.get("inference_seconds", 0.0)))
    metrics["train_label_histogram"] = np.bincount(train_y, minlength=N_CLASSES).tolist()
    metrics["predicted_label_histogram"] = np.bincount(labels, minlength=N_CLASSES).tolist()
    return {"logits": logits, "labels": labels, "metrics": metrics}
