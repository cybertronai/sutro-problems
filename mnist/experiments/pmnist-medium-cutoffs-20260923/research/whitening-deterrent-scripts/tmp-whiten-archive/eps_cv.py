"""Pick epsilon by train-only 5-fold CV of arc-cosine depth-3 kernel ridge.

Dev seed 2026092301, N=1000, training rows only.  No query labels are read; the
query argument is a throwaway copy of ten training rows so classical.fit_predict
has something to predict.  classical._fit_kernel_ridge reports the 5-fold CV
accuracy of the selected lambda in metrics['cv_accuracy'].
"""
import json
import sys
import time

import numpy as np

ROOT = "/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923"
sys.path.insert(0, ROOT)

import classical   # noqa: E402
import study       # noqa: E402
import whiten      # noqa: E402

SEED = 2026092301
N = 1000
CONFIG = {
    "family": "kernel_ridge", "kernel": "arccos1", "depth": 3,
    "lambda_grid": [1e-07, 1e-06, 1e-05, 0.0001, 0.001],
    "cv_subsample": 4000, "predict_block": 2000, "kernel_block": 2000,
}


def cv_accuracy(train_x, train_y, tag):
    started = time.time()
    out = classical.fit_predict(train_x, train_y, train_x[:10], CONFIG, seed=11)
    metrics = out["metrics"]
    row = {
        "variant": tag,
        "cv_accuracy": float(metrics["cv_accuracy"]),
        "cv_error_pct": 100.0 * (1.0 - float(metrics["cv_accuracy"])),
        "chosen_lambda": float(metrics["chosen_lambda"]),
        "cv_folds": int(metrics["cv_folds"]),
        "seconds": time.time() - started,
    }
    print("%-42s cv_err %6.2f%%  lambda %g  (%.1fs)"
          % (tag, row["cv_error_pct"], row["chosen_lambda"], row["seconds"]), flush=True)
    return row


def main():
    pool = np.asarray(study.pool_images())
    rows = []

    base = study.job_arrays(SEED, N)
    rows.append(cv_accuracy(base["train_x"], base["train_y"], "permuted-pixels (baseline)"))

    transforms = {}
    for epsilon in (1e-2, 1e-3, 1e-4, 1e-5, 0.0):
        t = whiten.fit_transform(pool, epsilon=epsilon, rotation_seed=20260923,
                                 method="zca")
        transforms[epsilon] = t
        arrays = whiten.job_arrays(SEED, N, t)
        row = cv_accuracy(arrays["train_x"], arrays["train_y"],
                          "whitened zca eps=%g" % epsilon)
        row["epsilon"] = epsilon
        row["achieved_variance_min"] = t["achieved_variance_min"]
        row["condition_number"] = t["condition_number"]
        rows.append(row)
        # same data, but pre-descaled so classical's 4x-0.5 map reproduces z exactly
        descaled = (arrays["train_x"].astype(np.float64) + 0.5) / 4.0
        row2 = cv_accuracy(descaled, arrays["train_y"],
                           "whitened zca eps=%g (no 4x-0.5)" % epsilon)
        row2["epsilon"] = epsilon
        rows.append(row2)

    for k in (48, 64, 72):
        t = whiten.fit_transform(pool, epsilon=0.0, rotation_seed=20260923,
                                 n_components=k)
        arrays = whiten.job_arrays(SEED, N, t)
        row = cv_accuracy(arrays["train_x"], arrays["train_y"],
                          "whitened exact, top-%d components" % k)
        row["epsilon"] = 0.0
        row["n_components"] = k
        rows.append(row)

    json.dump({"seed": SEED, "n": N, "config": CONFIG, "rows": rows},
              open("/tmp/whiten/eps_cv.json", "w"), indent=2)
    print("wrote /tmp/whiten/eps_cv.json")


if __name__ == "__main__":
    main()
