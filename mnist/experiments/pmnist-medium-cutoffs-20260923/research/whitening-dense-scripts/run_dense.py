"""Measure dense (permutation-invariant, non-spatial) methods on the permuted-pixel
variant vs the whitened+rotated variant.  Dev seed 2026092301 only.

Query labels come from raw/pool_labels.npy indexed by study.query_indices(seed);
this is a DEV seed, and the labels are used only to score, never inside a fit.

usage:  python run_dense.py <task> [...]
tasks:  classical <variant> <kernel> <n>
        mlp <variant> <n>
"""
import json
import os
import sys
import time

import numpy as np

ROOT = "/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923"
sys.path.insert(0, ROOT)

import study      # noqa: E402
import whiten     # noqa: E402

SEED = 2026092301
OUT = "/tmp/whiten/results"
os.makedirs(OUT, exist_ok=True)

# The two whitened variants.  'w2' is the epsilon chosen by the train-only CV
# sweep (best of 1e-2/1e-3/1e-4); 'w4' is a much stronger whitening, kept so the
# accuracy cost and the attack resistance can be read off the same axis.
VARIANTS = {
    "permuted": None,
    "w2": dict(epsilon=1e-2, rotation_seed=20260923, method="zca"),
    "w4": dict(epsilon=1e-4, rotation_seed=20260923, method="zca"),
    "wexact64": dict(epsilon=0.0, rotation_seed=20260923, n_components=64),
    # ablations for the attack table only (not measured for accuracy)
    "zca_norot": dict(epsilon=1e-2, rotation_seed=20260923, method="zca",
                      rotation="none"),
    "rot_only": dict(epsilon=0.0, rotation_seed=20260923, method="rotate_only"),
}

LAMBDA_GRID = [1e-07, 1e-06, 1e-05, 0.0001, 0.001]
STUDY_GAMMA_GRID = [0.005, 0.01, 0.02, 0.05]


def query_labels():
    """Dev-seed query labels, read directly from the pool (scoring only)."""
    labels = np.load(study.POOL_LABELS_PATH)
    return np.ascontiguousarray(labels[study.query_indices(SEED)], dtype=np.uint8)


def arrays_for(variant, n):
    if variant == "permuted":
        return study.job_arrays(SEED, n), None
    transform = whiten.fit_transform(np.asarray(study.pool_images()),
                                     **VARIANTS[variant])
    return whiten.job_arrays(SEED, n, transform), transform


def mean_pairwise_sq(x, rng_seed=0, rows=1000):
    """Mean squared pairwise distance in classical's 4x-0.5 input space."""
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(x.shape[0], size=min(rows, x.shape[0]), replace=False)
    v = 4.0 * np.asarray(x[idx], dtype=np.float64) - 0.5
    sq = ((v * v).sum(1)[:, None] + (v * v).sum(1)[None, :] - 2.0 * (v @ v.T))
    off = ~np.eye(len(idx), dtype=bool)
    return float(np.maximum(sq, 0.0)[off].mean())


def run_classical(variant, kernel, n):
    import classical
    arrays, transform = arrays_for(variant, n)
    config = {"family": "kernel_ridge", "kernel": kernel,
              "lambda_grid": list(LAMBDA_GRID), "cv_subsample": 4000,
              "predict_block": 2000, "kernel_block": 2000}
    extra = {}
    if kernel == "arccos1":
        config["depth"] = 3
    elif kernel == "rbf":
        # The study's gamma grid is calibrated to 4x-0.5 pixel units.  Whitened
        # features live on a much larger scale, so the same numeric gammas would
        # push every kernel entry to ~0.  The grid is therefore the study grid
        # PLUS a scale-matched copy (multiplied by the ratio of mean squared
        # pairwise distances against the permuted-pixel variant); the existing
        # train-only CV picks between them, so no query label is involved.
        reference = mean_pairwise_sq(study.job_arrays(SEED, min(n, 2000))["train_x"])
        here = mean_pairwise_sq(arrays["train_x"])
        ratio = reference / here
        scaled = [float(g * ratio) for g in STUDY_GAMMA_GRID]
        grid = sorted(set([float(g) for g in STUDY_GAMMA_GRID] + scaled))
        config["gamma_grid"] = grid
        extra = {"gamma_scale_ratio": ratio, "mean_pairwise_sq": here,
                 "mean_pairwise_sq_reference": reference, "gamma_grid": grid}
    started = time.time()
    out = classical.fit_predict(arrays["train_x"], arrays["train_y"],
                                arrays["query_x"], config, seed=11)
    wall = time.time() - started
    truth = query_labels()
    error = 100.0 * float((out["labels"] != truth).mean())
    record = {
        "method": "kernel_ridge:" + kernel, "variant": variant, "n": int(n),
        "seed": SEED, "query_error_pct": error, "wall_seconds": wall,
        "cv_accuracy": float(out["metrics"]["cv_accuracy"]),
        "chosen_lambda": float(out["metrics"]["chosen_lambda"]),
        "chosen_gamma": out["metrics"].get("chosen_gamma"),
        "config": {k: v for k, v in config.items() if k != "lambda_grid"},
        **extra,
    }
    if transform is not None:
        record["transform"] = {k: transform[k] for k in
                               ("epsilon", "method", "rotation_seed", "n_released",
                                "condition_number", "A_sha256")}
    return record


def run_mlp(variant, n):
    import learners
    arrays, transform = arrays_for(variant, n)
    config = {"family": "mlp", "normalization": "standardize",
              "widths": [1024, 1024], "dropout": 0.3, "input_noise_std": 0.3,
              "lr": 0.002, "weight_decay": 0.01,
              "epochs": int(os.environ.get("MLP_EPOCHS", 300 if n <= 1000 else 150)),
              "batch_size": 128, "warmup_fraction": 0.1, "members": 1}
    started = time.time()
    out = learners.fit_predict(arrays["train_x"], arrays["train_y"],
                               arrays["query_x"], config, seed=11,
                               deadline_unix=time.time() + 3600.0, device="cpu")
    wall = time.time() - started
    truth = query_labels()
    error = 100.0 * float((out["labels"] != truth).mean())
    record = {
        "method": "mlp-standardize", "variant": variant, "n": int(n), "seed": SEED,
        "query_error_pct": error, "wall_seconds": wall,
        "epochs_completed": int(out["metrics"].get("epochs_completed", -1)),
        "truncated": bool(out["metrics"].get("truncated", False)),
        "config": {k: v for k, v in config.items()},
    }
    if transform is not None:
        record["transform"] = {k: transform[k] for k in
                               ("epsilon", "method", "rotation_seed", "n_released",
                                "condition_number", "A_sha256")}
    return record


def main():
    task = sys.argv[1]
    if task == "classical":
        variant, kernel, n = sys.argv[2], sys.argv[3], int(sys.argv[4])
        record = run_classical(variant, kernel, n)
        name = f"classical-{kernel}-{variant}-n{n}"
    elif task == "mlp":
        variant, n = sys.argv[2], int(sys.argv[3])
        record = run_mlp(variant, n)
        name = f"mlp-{variant}-n{n}"
    else:
        raise SystemExit(f"unknown task {task}")
    path = os.path.join(OUT, name + ".json")
    json.dump(record, open(path, "w"), indent=2, sort_keys=True, default=str)
    print("%-34s error %6.2f%%  (%.0fs)  -> %s"
          % (name, record["query_error_pct"], record["wall_seconds"], path), flush=True)


if __name__ == "__main__":
    main()
