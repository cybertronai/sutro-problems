"""Reproduce the eleven predeclared MNIST-medium draws for one basis size.

Runs the frozen RFF-ridge learner on each draw, writes predictions, records
every selection decision, then scores with the official evaluator.

Usage
-----
    python run_protocol.py --D 12000 --data-root <dir with data-<seed>> \
        --outdir out --seeds 20261001,...,20261011

For each draw the script expects ``<data-root>/data-<seed>/medium.npz`` produced
by the official generator:

    python -m mnist.code.data --profile medium-error-targets-v1 \
        --output <data-root>/data-<seed> --seed <seed>

Run from the repository root so that the ``mnist.code`` package resolves.
Scoring is done by the official evaluator only; this file performs no accuracy
arithmetic of its own beyond summing the ``correct`` fields it copies from the
evaluator JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import rff_ridge as R


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_official(data_dir, pred_path, bands, repo_root):
    """Return {band: {correct, total, required_correct, meets}} from the evaluator."""
    out = {}
    for band in bands:
        r = subprocess.run(
            [sys.executable, "-m", "mnist.code.evaluate", "--tier", "medium",
             "--error-target", band, "--data-dir", str(data_dir),
             "--predictions", str(pred_path)],
            cwd=repo_root, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"evaluator failed for band {band}: {r.stderr[-500:]}")
        d = json.loads(r.stdout)
        out[band] = {"correct": d["correct"], "total": d["total"],
                     "required_correct": d["required_correct"],
                     "meets": d["meets_accuracy_target"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--seeds", default=",".join(str(s) for s in R.SEEDS))
    ap.add_argument("--backend", default="numpy", choices=("numpy", "torch"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--gram-dtype", default=None)
    ap.add_argument("--f32-products", action="store_true")
    ap.add_argument("--bands", default="2,3,5,8,12")
    ap.add_argument("--repo-root", default=".")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    bands = a.bands.split(",")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for seed in seeds:
        data_dir = Path(a.data_root) / f"data-{seed}"
        npz = data_dir / "medium.npz"
        Xtr, ytr, Xte, tidx, teidx = R.load_draw(npz)
        val_overlap, train_overlap = R.check_no_test_overlap(tidx, teidx)
        assert val_overlap == 0 and train_overlap == 0, (seed, val_overlap,
                                                          train_overlap)
        t0 = time.perf_counter()
        if a.backend == "numpy":
            pred, best, table = R.fit_predict_np(Xtr, ytr, Xte, a.D)
        else:
            pred, best, table = R.fit_predict_torch(
                Xtr, ytr, Xte, a.D, device=a.device, dtype=a.dtype,
                chunk=a.chunk, gram_dtype=a.gram_dtype, f32_products=a.f32_products)
        seconds = round(time.perf_counter() - t0, 2)
        assert pred.shape == (len(Xte),) and pred.dtype == np.int64
        assert pred.min() >= 0 and pred.max() <= 9

        p_path = out / f"pred-D{a.D}-{seed}.npy"
        np.save(p_path, pred)
        bands_doc = score_official(data_dir, p_path, bands, a.repo_root)

        rec = {"seed": seed, "D": a.D, "selected": best, "grid": table,
               "predictions": str(p_path), "sha256": sha256(p_path),
               "seconds": seconds,
               "val_test_index_overlap_rows": val_overlap,
               "train_test_index_overlap_rows": train_overlap,
               "bands": bands_doc}
        records.append(rec)
        print(json.dumps({"seed": seed, "selected": best,
                          "inner_val": f"{best['inner_val_correct']}/2000",
                          "seconds": seconds,
                          "correct_band3": bands_doc.get("3", {}).get("correct")}),
              flush=True)
        del Xtr, Xte

    summary = {}
    for band in bands:
        per = [r["bands"][band]["correct"] for r in records]
        acc = np.array([100.0 * r["bands"][band]["correct"] / r["bands"][band]["total"]
                        for r in records])
        required = records[0]["bands"][band]["required_correct"] * len(records)
        total = int(sum(per))
        summary[band] = {"n_draws": len(records), "per_draw_correct": per,
                         "total_correct": total,
                         "mean_accuracy_pct": float(acc.mean()),
                         "sd_pp_ddof1": float(acc.std(ddof=1)),
                         "min_pct": float(acc.min()), "max_pct": float(acc.max()),
                         "required_total": required,
                         "status": "PASS" if total >= required else "FAIL",
                         "margin": total - required}
        print(json.dumps({"band": band, **summary[band]}), flush=True)

    Path(a.records).write_text(json.dumps(
        {"D": a.D, "learner_seed": R.LEARNER_SEED, "seeds": seeds,
         "backend": a.backend, "device": a.device, "dtype": a.dtype,
         "chunk": a.chunk, "gram_dtype": a.gram_dtype,
         "f32_products": a.f32_products,
         "selection_protocol": {
             "p_grid": list(R.P_GRID), "gamma_grid": list(R.GAMMA_GRID),
             "lam_grid": list(R.LAM_GRID),
             "inner_split": f"default_rng({R.LEARNER_SEED}).permutation(n_train)"
                            f"[:{R.N_INNER_TRAIN}] train / rest validation",
             "metric": "inner-validation correct count",
             "tie_break": "smaller lam, then smaller gamma, then smaller p",
             "refit": "selected point refit on all n_train rows of this draw",
             "test_labels_read_by_learner": False},
         "records": records, "summary": summary}, indent=1) + "\n")
    print(f"wrote {a.records}")


if __name__ == "__main__":
    main()