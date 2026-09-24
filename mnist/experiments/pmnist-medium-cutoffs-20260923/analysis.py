#!/usr/bin/env python3
"""Five-level ladder analysis for the permutation-invariant MNIST-medium study.

This module reads ONLY a scores JSON produced by ``score.py`` (plus, optionally,
the already published per-draw CSV of the spatial-CNN cutoff calibration).  It
never opens an MNIST label file, a prediction ``.npz`` or the pool arrays.

Ladder definition (matches ``study.py``)
---------------------------------------
``N_i = 1000 * 10 ** (i / 4)`` for ``i = 0..4``; the exact floats are
1000, 1778.2794…, 3162.2777…, 5623.4133…, 10000 and the integers actually
trained on are the rounded 1000, 1778, 3162, 5623, 10000.  Every constant
multiplier is ``10 ** 0.25 = 1.7782794…``.  Unlike the 9×9 spatial study
(``mnist/experiments/nine-geometric-cutoffs-20260923``), which measured only the
two endpoints and interpolated the middle three, all five levels here are
measured on eleven dataset seeds (2026092001..2026092011) with 10,000 queries
each, so the power-law fits are checks of the geometric-spacing assumption
rather than the source of the middle cutoffs.

Scores-file contract and assumptions
------------------------------------
The loader is deliberately tolerant, because it was written alongside
``score.py``.  It accepts either a bare list of job records or an object; in the
object case the job list is taken from the first present key among
``jobs / records / results / runs / scores`` and the optional pre-aggregated
table from ``aggregate / aggregates / by_candidate_n / summary``.  Per job the
following field names are accepted (first match wins):

    id                 <- id | job_id
    candidate_id       <- candidate_id | candidate
    stage              <- stage
    seed               <- seed | dataset_seed
    n                  <- n | n_train | train_n | examples
    correct            <- correct | n_correct | correct_count
    total              <- total | n_total | query_count | total_count
    metrics            <- metrics (nested dict) or the same names inline:
                          training_seconds, fit_wall_seconds, epochs_completed,
                          truncated, uses_query_images_unlabeled

Assumptions that the rest of the file relies on:

*   ``correct`` and ``total`` are integers.  **Every mean is computed from those
    integers** -- ``100 * (sum(total) - sum(correct)) / sum(total)`` -- and a
    stored ``error_pct`` is only ever cross-checked against them, never summed.
    A stored value disagreeing by more than 1e-6 pp is reported in
    ``summary.json`` under ``integrity.error_pct_mismatches``.
*   One record per ``(candidate_id, seed, n)``.  Duplicates are an error.
*   Any pre-aggregated block in the scores file is recomputed here and only
    cross-checked; this analysis never trusts it.

Usage
-----
    python analysis.py --scores results/scores_final.json --candidate <id> \
        [--reference-csv .../cutoff-calibration-20260920/analysis_outputs/per_draw.csv] \
        --out analysis/

    python analysis.py --dev --scores results/scores_dev.json --out analysis/

All randomness is seeded (``--bootstrap-seed``, default 20260923).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist
from scipy.stats import t as student_t

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

# ------------------------------------------------------------------ protocol ---
LEVEL_COUNT = 5
LEVELS_EXACT = [1000.0 * 10.0 ** (i / 4.0) for i in range(LEVEL_COUNT)]
LEVELS = [int(round(value)) for value in LEVELS_EXACT]  # 1000 1778 3162 5623 10000
LEVEL_MULTIPLIER = 10.0 ** 0.25
FINAL_SEEDS = list(range(2026092001, 2026092012))  # eleven paired dataset draws
QUERY_COUNT = 10000

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260923
ROUNDING_STEPS = (0.05, 0.1)
TOPO_PREFIX = "topo"

# Reference spatial study: rows keyed by (dataset_seed, n) with these columns.
REFERENCE_CORRECT_COLUMN = "ensemble_final_correct"
REFERENCE_ERROR_COLUMN = "ensemble_final_error_pct"
REFERENCE_INTERPOLATION_BRACKET = (800, 1600)

# dataviz reference palette, light mode, first three categorical slots only
# (all-pairs forms cap at three).  Text/grid wear ink tokens, never series hues.
COLOR_MAIN = "#2a78d6"      # slot 1, blue   -- the selected pmnist candidate
COLOR_REFERENCE = "#eb6834"  # slot 2, orange -- spatial-CNN reference study
COLOR_TOPO = "#1baf7a"       # slot 3, aqua   -- topology-recovered candidate
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID_COLOR = "#e3e2de"
SURFACE = "#fcfcfb"

JOB_KEYS = ("jobs", "records", "results", "runs", "scores")
AGGREGATE_KEYS = ("aggregate", "aggregates", "by_candidate_n", "summary")
METRIC_NAMES = ("training_seconds", "fit_wall_seconds", "epochs_completed",
                "truncated", "uses_query_images_unlabeled", "train_accuracy",
                "device", "device_name")


# ------------------------------------------------------------------- helpers ---
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def pick(mapping, names, default=None):
    """First present, non-None value among ``names``."""
    for name in names:
        if isinstance(mapping, dict) and mapping.get(name) is not None:
            return mapping[name]
    return default


def round_to(value, step):
    """Half-up rounding to a multiple of ``step``.

    Decimal arithmetic on the decimal text of the value, because neither
    ``round`` (banker's rounding: 1.025 -> 1.00 at a 0.05 step) nor a binary
    ``floor(value / step + 0.5)`` (1.025 / 0.05 == 20.499999999999996) gives the
    documented behaviour at an exact half.
    """
    if value is None or not math.isfinite(value):
        return None
    quotient = Decimal(str(float(value))) / Decimal(str(float(step)))
    return float(quotient.quantize(Decimal(1), rounding=ROUND_HALF_UP)
                 * Decimal(str(float(step))))


def error_pct(correct, total):
    """Error percentage from integer counts; never from a stored percentage."""
    correct, total = int(correct), int(total)
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= correct <= total:
        raise ValueError(f"correct={correct} outside [0, {total}]")
    return 100.0 * (total - correct) / total


def level_index(n):
    """Index of ``n`` in the five-level ladder, or None when off-ladder."""
    n = int(n)
    return LEVELS.index(n) if n in LEVELS else None


def pava_decreasing(values):
    """Least-squares nonincreasing projection (pool-adjacent-violators)."""
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("PAVA needs a finite one-dimensional sequence")
    blocks = []
    for value in y:
        blocks.append([1.0, float(value)])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][0] < blocks[-1][1] / blocks[-1][0]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0] + right[0], left[1] + right[1]])
    fitted, position = np.empty_like(y), 0
    for weight, total in blocks:
        count = int(round(weight))
        fitted[position:position + count] = total / weight
        position += count
    return fitted


def t_interval(values):
    """Mean with a Student-t 95% interval; ``df = count - 1``."""
    sample = np.asarray(values, dtype=float)
    count = sample.size
    mean = float(sample.mean()) if count else None
    if count < 2:
        return {"mean": mean, "sd": None, "sem": None, "df": max(count - 1, 0),
                "ci95": [None, None], "count": int(count)}
    sd = float(sample.std(ddof=1))
    sem = sd / math.sqrt(count)
    half = float(student_t.ppf(0.975, count - 1)) * sem
    return {"mean": mean, "sd": sd, "sem": sem, "df": count - 1,
            "ci95": [mean - half, mean + half], "count": int(count)}


def safe_float(value):
    """JSON-safe float: non-finite becomes None (summary.json forbids NaN)."""
    value = float(value)
    return value if math.isfinite(value) else None


def nan_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return safe_float(values.mean()) if values.size else None


def quantile_ci(samples):
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [None, None]
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


# -------------------------------------------------------------------- loading ---
def normalize_job(raw, index, default_stage=None):
    """Coerce one scores record into the canonical shape used below."""
    if not isinstance(raw, dict):
        raise ValueError(f"job #{index} is not an object")
    metrics_block = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
    metrics = {}
    for name in METRIC_NAMES:
        value = metrics_block.get(name, raw.get(name))
        metrics[name] = value
    correct = pick(raw, ("correct", "n_correct", "correct_count"))
    total = pick(raw, ("total", "n_total", "query_count", "total_count"), QUERY_COUNT)
    if correct is None:
        raise ValueError(f"job #{index} has no correct count")
    job = {
        "id": pick(raw, ("id", "job_id"), f"job-{index}"),
        "candidate_id": str(pick(raw, ("candidate_id", "candidate"), "unknown")),
        # score.py records the stage once at the top level, not per job.
        "stage": pick(raw, ("stage",), default_stage),
        "seed": int(pick(raw, ("seed", "dataset_seed"))),
        "n": int(pick(raw, ("n", "n_train", "train_n", "examples"))),
        "correct": int(correct),
        "total": int(total),
        "metrics": metrics,
    }
    job["error_pct"] = error_pct(job["correct"], job["total"])
    stored = pick(raw, ("error_pct", "error_percent", "error"))
    job["stored_error_pct"] = float(stored) if stored is not None else None
    return job


def load_scores(path):
    """Tolerant reader for ``results/scores_<stage>.json``."""
    path = Path(path)
    payload = json.loads(path.read_text())
    detected = {"jobs_key": None, "aggregate_key": None}
    stage = payload.get("stage") if isinstance(payload, dict) else None
    if isinstance(payload, list):
        raw_jobs, raw_aggregate = payload, None
    elif isinstance(payload, dict):
        raw_jobs = raw_aggregate = None
        for key in JOB_KEYS:
            if isinstance(payload.get(key), list):
                raw_jobs, detected["jobs_key"] = payload[key], key
                break
        for key in AGGREGATE_KEYS:
            if isinstance(payload.get(key), list):
                raw_aggregate, detected["aggregate_key"] = payload[key], key
                break
        if raw_jobs is None:
            raise ValueError(f"{path}: no job list under any of {JOB_KEYS}")
    else:
        raise ValueError(f"{path}: unsupported top-level JSON type")

    jobs, mismatches, seen = [], [], {}
    for index, raw in enumerate(raw_jobs):
        job = normalize_job(raw, index, default_stage=stage)
        key = (job["candidate_id"], job["seed"], job["n"])
        if key in seen:
            raise ValueError(f"duplicate scores record for candidate/seed/n {key}")
        seen[key] = job["id"]
        if job["stored_error_pct"] is not None and \
                abs(job["stored_error_pct"] - job["error_pct"]) > 1e-6:
            mismatches.append({"id": job["id"], "stored": job["stored_error_pct"],
                               "recomputed": job["error_pct"]})
        jobs.append(job)
    if not jobs:
        raise ValueError(f"{path}: scores file contains no job records")
    return {"path": str(path), "jobs": jobs, "aggregate_in_file": raw_aggregate,
            "detected_keys": detected, "error_pct_mismatches": mismatches, "stage": stage,
            "top_level_keys": sorted(payload) if isinstance(payload, dict) else None}


def candidates_in(jobs):
    return sorted({job["candidate_id"] for job in jobs})


def select(jobs, candidate_id, n=None):
    rows = [job for job in jobs if job["candidate_id"] == candidate_id]
    if n is not None:
        rows = [job for job in rows if job["n"] == int(n)]
    return sorted(rows, key=lambda job: (job["n"], job["seed"]))


def check_aggregate(loaded, recomputed):
    """Cross-check any aggregate block shipped in the scores file.

    ``score.py`` reports ``mean_error_pct`` as the mean of the per-job
    ``error_pct`` values; here the primary statistic is the pooled error from
    integer counts.  The two coincide when every job has the same ``total``, so
    a row is accepted if it matches either.  This analysis never consumes the
    stored aggregate -- it only records disagreements.
    """
    block = loaded.get("aggregate_in_file")
    if not block:
        return {"present": False, "checked": 0, "disagreements": []}
    index = {(row.get("candidate_id"), int(row["n"])): row
             for row in recomputed if row.get("candidate_id")}
    disagreements, checked = [], 0
    for row in block:
        key = (str(pick(row, ("candidate_id", "candidate"), "")), int(pick(row, ("n",), 0)))
        ours = index.get(key)
        if ours is None:
            continue
        checked += 1
        theirs = pick(row, ("mean_error_pct", "mean_error", "error_pct"))
        pooled = pick(ours, ("expected_error_pct", "mean_error_pct"))
        by_draw = ours.get("mean_of_draw_errors_pct")
        candidates = [value for value in (pooled, by_draw) if value is not None]
        if theirs is not None and candidates and \
                all(abs(float(theirs) - value) > 1e-6 for value in candidates):
            disagreements.append({"candidate_id": key[0], "n": key[1],
                                  "in_file": float(theirs), "pooled": pooled,
                                  "mean_of_draws": by_draw})
    return {"present": True, "checked": checked, "disagreements": disagreements}


# ------------------------------------------------------------- level statistics ---
def metric_values(rows, name):
    return [row["metrics"][name] for row in rows
            if isinstance(row["metrics"].get(name), (int, float))
            and not isinstance(row["metrics"].get(name), bool)]


def flag_summary(rows, name):
    values = [bool(row["metrics"][name]) for row in rows
              if row["metrics"].get(name) is not None]
    if not values:
        return {"reported": 0, "true": 0, "all": None, "any": None}
    return {"reported": len(values), "true": int(sum(values)),
            "all": bool(all(values)), "any": bool(any(values))}


def level_stats(rows, n):
    """Per-level summary for one candidate; all means from integer counts."""
    rows = sorted(rows, key=lambda row: row["seed"])
    correct = sum(row["correct"] for row in rows)
    total = sum(row["total"] for row in rows)
    draw_errors = np.array([row["error_pct"] for row in rows], dtype=float)
    interval = t_interval(draw_errors)
    best = min(rows, key=lambda row: row["error_pct"])
    worst = max(rows, key=lambda row: row["error_pct"])
    training = metric_values(rows, "training_seconds")
    wall = metric_values(rows, "fit_wall_seconds")
    epochs = metric_values(rows, "epochs_completed")
    index = level_index(n)
    return {
        "n": int(n),
        "level": None if index is None else index + 1,
        "n_exact": None if index is None else LEVELS_EXACT[index],
        "draws": len(rows),
        "seeds": [row["seed"] for row in rows],
        "aggregate_correct": correct,
        "aggregate_total": total,
        # Primary statistic: pooled over the integer counts of every draw.
        "expected_error_pct": error_pct(correct, total),
        "expected_accuracy_pct": 100.0 * correct / total,
        "mean_of_draw_errors_pct": interval["mean"],
        "sd_error_pp": interval["sd"],
        "sem_error_pp": interval["sem"],
        "mean_ci95_pct": interval["ci95"],
        "ci_df": interval["df"],
        "min_draw": {"seed": best["seed"], "error_pct": best["error_pct"], "id": best["id"]},
        "max_draw": {"seed": worst["seed"], "error_pct": worst["error_pct"], "id": worst["id"]},
        "per_draw_error_pct": draw_errors.tolist(),
        "mean_training_seconds": float(np.mean(training)) if training else None,
        "mean_fit_wall_seconds": float(np.mean(wall)) if wall else None,
        "mean_epochs_completed": float(np.mean(epochs)) if epochs else None,
        "truncated_fits": flag_summary(rows, "truncated")["true"],
        "truncated_reported": flag_summary(rows, "truncated")["reported"],
        "uses_query_images_unlabeled": flag_summary(rows, "uses_query_images_unlabeled"),
    }


def ladder_matrix(jobs, candidate_id, levels=LEVELS):
    """(draws, levels) integer correct/total matrices over the seeds present at
    EVERY level, so a bootstrap resample keeps each draw's levels together."""
    by_level = {n: {row["seed"]: row for row in select(jobs, candidate_id, n)} for n in levels}
    shared = sorted(set.intersection(*[set(v) for v in by_level.values()])) if by_level else []
    correct = np.array([[by_level[n][seed]["correct"] for n in levels] for seed in shared], dtype=np.int64)
    total = np.array([[by_level[n][seed]["total"] for n in levels] for seed in shared], dtype=np.int64)
    missing = [{"n": n, "seed": seed} for n in levels
               for seed in sorted(set(by_level[n]) - set(shared))]
    return shared, correct, total, missing


# -------------------------------------------------------------------- fitting ---
def two_anchor_power_law(n_low, error_low, n_high, error_high):
    """error(N) = a * (N/1000)^(-b) through the two ladder endpoints."""
    if error_low <= 0 or error_high <= 0:
        raise ValueError("Power-law anchors must be positive errors")
    b = math.log(error_low / error_high) / math.log(n_high / n_low)
    a = error_low * (n_low / 1000.0) ** b
    return {"a": float(a), "b": float(b)}


def power_law_at(fit, n):
    n = np.asarray(n, dtype=float)
    return fit.get("c", 0.0) + fit["a"] * (n / 1000.0) ** (-fit["b"])


def fit_floor_power_law(ns, errors, sigma=None):
    """e(N) = c + a*(N/1000)^(-b) with c >= 0, by scipy least squares."""
    x = np.asarray(ns, dtype=float) / 1000.0
    y = np.asarray(errors, dtype=float)
    weights = np.ones_like(y) if sigma is None else np.maximum(np.asarray(sigma, float), 1e-6)

    def residual(p):
        return (p[0] + p[1] * x ** (-p[2]) - y) / weights

    bounds = ([0.0, 1e-8, 0.01], [float(max(y.max(), 1.0)) * 10.0, 1000.0, 5.0])
    starts = [[0.0, max(float(np.median(y)), 1e-3), 0.5],
              [max(float(y[-1]) * 0.7, 0.0), max(float(np.median(y)), 1e-3), 0.8],
              [0.0, max(float(y[0]), 1e-3), 0.3]]
    solutions = [least_squares(residual, start, bounds=bounds, max_nfev=2000,
                               ftol=1e-12, xtol=1e-12, gtol=1e-12) for start in starts]
    best = min(solutions, key=lambda s: float(np.sum(s.fun ** 2)))
    if not best.success or not np.isfinite(best.x).all():
        raise RuntimeError("three-parameter power-law optimizer did not converge")
    c, a, b = (float(v) for v in best.x)
    predicted = c + a * x ** (-b)
    residuals = predicted - y
    dof = max(len(y) - 3, 0)
    weighted_sse = float(np.sum((residuals / weights) ** 2))
    return {
        "form": "error_pct = c + a * (N/1000) ** (-b), c >= 0",
        "c": c, "a": a, "b": b,
        "predicted_error_pct": predicted.tolist(),
        "residual_pp": residuals.tolist(),
        "rmse_pp": float(np.sqrt(np.mean(residuals ** 2))),
        "max_abs_residual_pp": float(np.max(np.abs(residuals))),
        "residual_dof": dof,
        "weighted_sse": weighted_sse,
        "weighted_sse_p_value": float(chi2_dist.sf(weighted_sse, dof)) if dof > 0 else None,
        "weights": "unit" if sigma is None else "1/SEM of the eleven draws (floored at 1e-6 pp)",
        "floor_interpretation": "Fitted asymptote of this recipe on this pool; not a Bayes-error estimate.",
    }


# ------------------------------------------------------------------ bootstrap ---
def bootstrap_ladder(correct, total, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Resample WHOLE draws (every level of a draw moves together)."""
    draws = correct.shape[0]
    if draws < 2:
        raise ValueError("bootstrap needs at least two draws")
    indices = np.random.default_rng(seed).integers(0, draws, size=(resamples, draws))
    resampled_correct = correct[indices].sum(axis=1)
    resampled_total = total[indices].sum(axis=1)
    errors = 100.0 * (resampled_total - resampled_correct) / resampled_total
    return {"indices": indices, "errors": errors}


def bootstrap_summary(boot, levels=LEVELS):
    errors = boot["errors"]
    with np.errstate(divide="ignore", invalid="ignore"):
        endpoint = np.where(errors[:, -1] > 0, errors[:, 0] / errors[:, -1], np.nan)
        exponent = np.where((errors[:, -1] > 0) & (errors[:, 0] > 0),
                            np.log(np.maximum(errors[:, 0], 1e-12) /
                                   np.maximum(errors[:, -1], 1e-12)) /
                            math.log(levels[-1] / levels[0]), np.nan)
        steps = []
        for i in range(len(levels) - 1):
            ratio = np.where(errors[:, i + 1] > 0, errors[:, i] / errors[:, i + 1], np.nan)
            steps.append({"from_n": int(levels[i]), "to_n": int(levels[i + 1]),
                          "mean_ratio": nan_mean(ratio),
                          "ci95": quantile_ci(ratio)})
    return {
        "resamples": int(errors.shape[0]),
        "seed": BOOTSTRAP_SEED,
        "unit": "one complete dataset draw, all five levels together (pairing preserved)",
        "statistic": "pooled error 100*(sum(total)-sum(correct))/sum(total) over the resampled draws",
        "interval": "percentile 95% (2.5th / 97.5th)",
        "level_error_ci95_pct": [quantile_ci(errors[:, j]) for j in range(errors.shape[1])],
        "level_error_mean_pct": [float(np.mean(errors[:, j])) for j in range(errors.shape[1])],
        "endpoint_ratio": {"definition": f"error({levels[0]}) / error({levels[-1]})",
                           "mean": nan_mean(endpoint), "ci95": quantile_ci(endpoint)},
        "two_anchor_exponent": {"definition": "log(error_low/error_high)/log(N_high/N_low)",
                                "mean": nan_mean(exponent), "ci95": quantile_ci(exponent)},
        "step_ratios": steps,
    }


# ------------------------------------------------------------------- reference ---
def load_reference_csv(path):
    """Per-draw rows of the spatial-CNN cutoff calibration (read-only)."""
    rows = {}
    with Path(path).open(newline="") as handle:
        for raw in csv.DictReader(handle):
            seed, n = int(raw["dataset_seed"]), int(raw["n"])
            correct, total = int(raw[REFERENCE_CORRECT_COLUMN]), int(raw["total"])
            rows[(seed, n)] = {
                "seed": seed, "n": n, "correct": correct, "total": total,
                "error_pct": error_pct(correct, total),
                "stored_error_pct": float(raw[REFERENCE_ERROR_COLUMN]),
                "draw_index": int(raw["draw_index"]),
            }
    if not rows:
        raise ValueError(f"{path}: no reference rows")
    return rows


def reference_curve(reference):
    """Pooled error per n over every reference draw."""
    grid = sorted({n for _, n in reference})
    points = []
    for n in grid:
        chosen = [row for (_, m), row in reference.items() if m == n]
        correct = sum(row["correct"] for row in chosen)
        total = sum(row["total"] for row in chosen)
        points.append({"n": n, "draws": len(chosen), "error_pct": error_pct(correct, total)})
    return points


def reference_interpolated_at(reference, seed, target, bracket=REFERENCE_INTERPOLATION_BRACKET):
    """Per-draw log-log interpolation of the spatial study at an unmeasured N.

    Two variants: a plain log-linear interpolation between the bracketing rows,
    and an isotonic variant that first forces the draw's whole curve to be
    nonincreasing in N.  Neither is a measurement.
    """
    low, high = bracket
    if (seed, low) not in reference or (seed, high) not in reference:
        return None
    plain = float(np.exp(np.interp(
        math.log(target), [math.log(low), math.log(high)],
        [math.log(reference[(seed, low)]["error_pct"]),
         math.log(reference[(seed, high)]["error_pct"])])))
    grid = sorted(n for (s, n) in reference if s == seed)
    curve = pava_decreasing([reference[(seed, n)]["error_pct"] for n in grid])
    isotonic = float(np.exp(np.interp(math.log(target),
                                      np.log(np.asarray(grid, dtype=float)),
                                      np.log(np.maximum(curve, 1e-12)))))
    return {"log_linear": plain, "isotonic_log_linear": isotonic,
            "bracket": [low, high],
            "bracket_error_pct": [reference[(seed, low)]["error_pct"],
                                  reference[(seed, high)]["error_pct"]]}


def paired_difference(pairs, label, bootstrap_seed=BOOTSTRAP_SEED,
                      resamples=BOOTSTRAP_RESAMPLES):
    """Paired ours-minus-reference summary with t and bootstrap intervals."""
    if not pairs:
        return {"available": False, "reason": "no shared dataset seeds", "label": label}
    differences = np.array([pair["difference_pp"] for pair in pairs], dtype=float)
    interval = t_interval(differences)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    boot = differences[indices].mean(axis=1)
    return {
        "available": True,
        "label": label,
        "pairs": pairs,
        "draws": int(differences.size),
        "mean_difference_pp": interval["mean"],
        "sd_difference_pp": interval["sd"],
        "t_ci95_pp": interval["ci95"],
        "t_df": interval["df"],
        "bootstrap_ci95_pp": quantile_ci(boot),
        "bootstrap_resamples": int(resamples),
        "sign": "negative favours the permutation-invariant candidate",
        "wins": int(np.sum(differences < 0)),
        "ties": int(np.sum(differences == 0)),
    }


def compare_with_reference(jobs, candidate_id, reference, bootstrap_seed=BOOTSTRAP_SEED):
    """Paired comparisons at 10,000 (measured) and 1,000 (interpolated)."""
    result = {"candidate_id": candidate_id,
              "reference_column": REFERENCE_ERROR_COLUMN,
              "reference_grid": sorted({n for _, n in reference})}

    ours_high = {row["seed"]: row for row in select(jobs, candidate_id, LEVELS[-1])}
    pairs = []
    for seed in sorted(ours_high):
        other = reference.get((seed, LEVELS[-1]))
        if other is None:
            continue
        pairs.append({"seed": seed, "ours_error_pct": ours_high[seed]["error_pct"],
                      "reference_error_pct": other["error_pct"],
                      "difference_pp": ours_high[seed]["error_pct"] - other["error_pct"]})
    result["at_10000"] = paired_difference(pairs, "measured vs measured at N=10,000",
                                           bootstrap_seed)
    result["at_10000"]["basis"] = (
        "Both sides measured on the same dataset seeds and the same query slice.")

    ours_low = {row["seed"]: row for row in select(jobs, candidate_id, LEVELS[0])}
    for variant in ("log_linear", "isotonic_log_linear"):
        pairs = []
        for seed in sorted(ours_low):
            interpolated = reference_interpolated_at(reference, seed, LEVELS[0])
            if interpolated is None:
                continue
            value = interpolated[variant]
            pairs.append({"seed": seed, "ours_error_pct": ours_low[seed]["error_pct"],
                          "reference_error_pct": value,
                          "reference_bracket": interpolated["bracket"],
                          "difference_pp": ours_low[seed]["error_pct"] - value})
        key = f"at_1000_{variant}"
        result[key] = paired_difference(
            pairs, f"measured vs {variant} interpolation at N=1,000", bootstrap_seed)
        result[key]["basis"] = (
            "INTERPOLATION, NOT A MEASUREMENT: the spatial study has no 1,000-example row; "
            f"its per-draw error at 1,000 is interpolated in log error / log N between "
            f"{REFERENCE_INTERPOLATION_BRACKET[0]} and {REFERENCE_INTERPOLATION_BRACKET[1]}"
            + (" after a nonincreasing (PAVA) projection of the draw's curve."
               if variant.startswith("isotonic") else "."))
    return result


# ---------------------------------------------------------------------- ladder ---
def build_ladder(jobs, candidate_id, resamples=BOOTSTRAP_RESAMPLES,
                 bootstrap_seed=BOOTSTRAP_SEED):
    seeds, correct, total, missing = ladder_matrix(jobs, candidate_id)
    if not seeds:
        raise ValueError(f"candidate {candidate_id!r} has no dataset seed present at all "
                         f"five levels {LEVELS}")
    levels = []
    for j, n in enumerate(LEVELS):
        rows = [row for row in select(jobs, candidate_id, n) if row["seed"] in set(seeds)]
        levels.append(level_stats(rows, n))

    means = np.array([entry["expected_error_pct"] for entry in levels], dtype=float)
    sems = np.array([entry["sem_error_pp"] or 1e-6 for entry in levels], dtype=float)
    two_anchor = two_anchor_power_law(LEVELS[0], means[0], LEVELS[-1], means[-1])
    two_anchor["form"] = "error_pct = a * (N/1000) ** (-b) through the two ladder endpoints"
    two_anchor["anchors"] = {"low": {"n": LEVELS[0], "error_pct": float(means[0])},
                             "high": {"n": LEVELS[-1], "error_pct": float(means[-1])}}
    two_anchor["predicted_error_pct"] = [float(v) for v in power_law_at(two_anchor, LEVELS)]
    two_anchor["predicted_error_pct_exact_levels"] = [
        float(v) for v in power_law_at(two_anchor, LEVELS_EXACT)]
    two_anchor["interior_residual_pp"] = [
        float(two_anchor["predicted_error_pct"][j] - means[j]) for j in (1, 2, 3)]
    two_anchor["max_abs_interior_residual_pp"] = float(
        max(abs(v) for v in two_anchor["interior_residual_pp"]))
    two_anchor["role"] = ("Check of the geometric-spacing assumption only. Every level here "
                          "is measured; nothing in the ladder is taken from this fit.")

    floor_fit = fit_floor_power_law(LEVELS, means)
    floor_fit_weighted = fit_floor_power_law(LEVELS, means, sems)

    boot = bootstrap_ladder(correct, total, resamples=resamples, seed=bootstrap_seed)
    boot_summary = bootstrap_summary(boot)

    rows = []
    for j, entry in enumerate(levels):
        cutoff = entry["expected_error_pct"]
        per_draw = np.asarray(entry["per_draw_error_pct"], dtype=float)
        rounded = {}
        for step in ROUNDING_STEPS:
            value = round_to(cutoff, step)
            rounded[f"{step:g}"] = {
                "error_pct": value,
                "accuracy_pct": None if value is None else 100.0 - value,
                "shift_pp": None if value is None else value - cutoff,
                "pass_fraction": float(np.mean(per_draw <= value)) if value is not None else None,
                "passes": int(np.sum(per_draw <= value)) if value is not None else None,
            }
        rows.append({
            **entry,
            "expected_cutoff_error_pct": cutoff,
            "bootstrap_ci95_pct": boot_summary["level_error_ci95_pct"][j],
            "two_anchor_predicted_error_pct": two_anchor["predicted_error_pct"][j],
            "two_anchor_minus_measured_pp": two_anchor["predicted_error_pct"][j] - cutoff,
            "floor_fit_predicted_error_pct": floor_fit["predicted_error_pct"][j],
            "pass_count": int(np.sum(per_draw <= cutoff)),
            "pass_fraction": float(np.mean(per_draw <= cutoff)),
            "rounded_options": rounded,
        })

    step_ratios = []
    for i in range(LEVEL_COUNT - 1):
        step_ratios.append({
            "from_n": LEVELS[i], "to_n": LEVELS[i + 1],
            "sample_multiplier": LEVELS_EXACT[i + 1] / LEVELS_EXACT[i],
            "measured_error_ratio": means[i] / means[i + 1],
            "bootstrap_ci95": boot_summary["step_ratios"][i]["ci95"],
        })

    return {
        "candidate_id": candidate_id,
        "draw_seeds": seeds,
        "draw_count": len(seeds),
        "levels": rows,
        "missing_jobs": missing,
        "endpoint_ratio": {"measured": float(means[0] / means[-1]),
                           "bootstrap_ci95": boot_summary["endpoint_ratio"]["ci95"]},
        "step_ratios": step_ratios,
        "two_anchor_power_law": two_anchor,
        "floor_power_law": floor_fit,
        "floor_power_law_sem_weighted": floor_fit_weighted,
        "bootstrap": boot_summary,
    }


# ---------------------------------------------------------------------- output ---
def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=False,
                                     allow_nan=False, default=float) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def format_number(value, digits=4, missing="n/a"):
    """Plain fixed-point text; nested f-strings are avoided for Python 3.11."""
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return missing
    if not math.isfinite(float(value)):
        return missing
    return f"{float(value):.{digits}f}"


def format_pct(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}%"


def format_interval(interval, digits=3, unit="%"):
    if not interval or interval[0] is None:
        return "n/a"
    return f"[{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]{unit}"


def ladder_markdown(ladder, comparisons, notes):
    lines = ["# Five levels of sample complexity",
             f"Permutation-invariant MNIST-medium (9x9, 81 permuted features) - "
             f"candidate `{ladder['candidate_id']}`",
             "",
             f"Levels are `N_i = 1000 * 10 ** (i/4)`, i = 0..4, rounded to "
             f"**{', '.join(f'{n:,}' for n in LEVELS)}** training examples; every step "
             f"multiplies the budget by **{LEVEL_MULTIPLIER:.6f}**.",
             f"All five levels are **measured** on {ladder['draw_count']} dataset draws "
             f"({ladder['draw_seeds'][0]}..{ladder['draw_seeds'][-1]}) with "
             f"{ladder['levels'][0]['aggregate_total'] // max(ladder['draw_count'], 1):,} "
             "queries each.",
             "",
             "## The difficulty ladder",
             "",
             "| Level | Examples | Expected error | Accuracy | 95% CI | Pass fraction | "
             "Rounded 0.05 / 0.1 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in ladder["levels"]:
        rounded = row["rounded_options"]
        lines.append(
            f"| {row['level']} | {row['n']:,} | {format_pct(row['expected_cutoff_error_pct'])} | "
            f"{format_pct(row['expected_accuracy_pct'])} | "
            f"{format_interval(row['mean_ci95_pct'])} | "
            f"{row['pass_count']}/{row['draws']} | "
            f"{format_pct(rounded['0.05']['error_pct'], 2)} / "
            f"{format_pct(rounded['0.1']['error_pct'], 2)} |")
    lines += [
        "",
        "Expected error is the pooled mean over the draws, "
        "`100 * (sum(total) - sum(correct)) / sum(total)`, computed from integer counts. "
        "The 95% CI is a Student-t interval for the mean of the per-draw errors "
        f"({ladder['levels'][0]['ci_df']} df). Pass fraction counts draws at or below the "
        "expected-mean cutoff; a mean-achievement cutoff is not a per-run guarantee.",
        "",
        "## Per level: dispersion and cost",
        "",
        "| Level | Examples | Mean of draws | SD (pp) | Min draw | Max draw | "
        "Bootstrap 95% | Mean train s | Mean fit wall s | Truncated |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in ladder["levels"]:
        lines.append(
            f"| {row['level']} | {row['n']:,} | {format_pct(row['mean_of_draw_errors_pct'])} | "
            f"{format_number(row['sd_error_pp'], 4)} | "
            f"{format_pct(row['min_draw']['error_pct'], 2)} | "
            f"{format_pct(row['max_draw']['error_pct'], 2)} | "
            f"{format_interval(row['bootstrap_ci95_pct'])} | "
            f"{format_number(row['mean_training_seconds'], 1)} | "
            f"{format_number(row['mean_fit_wall_seconds'], 1)} | "
            f"{row['truncated_fits']}/{row['draws']} |")

    two = ladder["two_anchor_power_law"]
    floor = ladder["floor_power_law"]
    lines += [
        "",
        "## Is the geometric spacing consistent with a power law?",
        "",
        f"Two-anchor model through the endpoints: **error(N) = {two['a']:.4f} * "
        f"(N / 1,000) ** (-{two['b']:.4f}) percent**. Because the three middle levels are "
        "measured here, this is a check, not a source of cutoffs.",
        "",
        "| Level | Examples | Measured | Two-anchor prediction | Prediction - measured (pp) |",
        "| --- | --- | --- | --- | --- |"]
    for row in ladder["levels"]:
        lines.append(
            f"| {row['level']} | {row['n']:,} | {format_pct(row['expected_cutoff_error_pct'])} | "
            f"{format_pct(row['two_anchor_predicted_error_pct'])} | "
            f"{row['two_anchor_minus_measured_pp']:+.4f} |")
    lines += [
        "",
        f"Largest interior deviation: **{two['max_abs_interior_residual_pp']:.4f} pp**.",
        "",
        f"Three-parameter fit with a nonnegative floor: **e(N) = {floor['c']:.4f} + "
        f"{floor['a']:.4f} * (N / 1,000) ** (-{floor['b']:.4f})**, RMSE "
        f"{floor['rmse_pp']:.4f} pp, max |residual| {floor['max_abs_residual_pp']:.4f} pp "
        f"on {floor['residual_dof']} residual degrees of freedom. "
        f"{floor['floor_interpretation']}",
        "",
        "## Bootstrap over whole draws",
        "",
        f"{ladder['bootstrap']['resamples']:,} resamples, seed "
        f"{ladder['bootstrap']['seed']}, unit = {ladder['bootstrap']['unit']}. "
        "Percentile 95% intervals.",
        "",
        f"Endpoint ratio error(1,000)/error(10,000) = "
        f"**{ladder['endpoint_ratio']['measured']:.4f}** "
        f"{format_interval(ladder['endpoint_ratio']['bootstrap_ci95'], 3, '')}.",
        "",
        "| Step | Sample multiplier | Error ratio | Bootstrap 95% |",
        "| --- | --- | --- | --- |"]
    for step in ladder["step_ratios"]:
        lines.append(
            f"| {step['from_n']:,} -> {step['to_n']:,} | {step['sample_multiplier']:.6f} | "
            f"{step['measured_error_ratio']:.4f} | "
            f"{format_interval(step['bootstrap_ci95'], 3, '')} |")

    for comparison in comparisons:
        lines += ["", f"## Paired comparison: `{comparison['candidate_id']}` vs the spatial "
                      "CNN reference", ""]
        for key, title in (("at_10000", "N = 10,000 (both measured)"),
                           ("at_1000_log_linear", "N = 1,000 (reference interpolated)"),
                           ("at_1000_isotonic_log_linear",
                            "N = 1,000 (reference isotonic-interpolated)")):
            block = comparison.get(key)
            if not block or not block.get("available"):
                continue
            lines += [
                f"**{title}.** Mean paired difference (ours minus spatial) "
                f"**{block['mean_difference_pp']:+.4f} pp**, t 95% "
                f"{format_interval(block['t_ci95_pp'], 4, ' pp')}, bootstrap 95% "
                f"{format_interval(block['bootstrap_ci95_pp'], 4, ' pp')} over "
                f"{block['draws']} paired draws; ours is lower in "
                f"{block['wins']}/{block['draws']}. {block['basis']}",
                ""]

    lines += ["", "## Scope", ""] + [f"- {note}" for note in notes] + [""]
    return "\n".join(lines)


# --------------------------------------------------------------------- figures ---
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.8)
    ax.grid(True, which="minor", color=GRID_COLOR, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def figure_error_vs_n(ladder, path, reference=None, topo_ladder=None):
    """Error vs N on a log sample-count axis: the ladder's headline figure."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    levels = np.array(LEVELS, dtype=float)
    means = np.array([row["expected_cutoff_error_pct"] for row in ladder["levels"]])
    sds = np.array([row["sd_error_pp"] or 0.0 for row in ladder["levels"]])
    low = np.array([row["bootstrap_ci95_pct"][0] for row in ladder["levels"]])
    high = np.array([row["bootstrap_ci95_pct"][1] for row in ladder["levels"]])

    ax.fill_between(levels, low, high, color=COLOR_MAIN, alpha=0.16, linewidth=0,
                    label="95% bootstrap interval (draw resampling)")
    for j, row in enumerate(ladder["levels"]):
        values = row["per_draw_error_pct"]
        ax.plot([levels[j]] * len(values), values, linestyle="none", marker="o",
                markersize=4, color=COLOR_MAIN, alpha=0.35,
                markeredgecolor="none", zorder=2,
                label="per-draw error" if j == 0 else None)
    dense = np.geomspace(levels[0], levels[-1], 200)
    ax.plot(dense, power_law_at(ladder["two_anchor_power_law"], dense), linestyle="--",
            color=COLOR_MAIN, linewidth=1.6, alpha=0.9,
            label="two-anchor power law (endpoints only)")
    ax.errorbar(levels, means, yerr=sds, fmt="o", color=COLOR_MAIN, markersize=8,
                linewidth=2.0, capsize=4, markeredgecolor=SURFACE, markeredgewidth=1.4,
                zorder=5, label=f"{ladder['candidate_id']} mean +/-1 SD")

    if reference:
        points = reference_curve(reference)
        inside = [p for p in points if levels[0] <= p["n"] <= levels[-1]]
        if inside:
            ax.plot([p["n"] for p in inside], [p["error_pct"] for p in inside],
                    marker="s", markersize=6, linewidth=2.0, color=COLOR_REFERENCE,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4,
                    label="spatial CNN reference (true topology)")
    if topo_ladder:
        topo_means = [row["expected_cutoff_error_pct"] for row in topo_ladder["levels"]]
        ax.plot(levels, topo_means, marker="D", markersize=6, linewidth=2.0,
                color=COLOR_TOPO, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4,
                label=f"{topo_ladder['candidate_id']} (recovered topology)")

    for j in (0, len(levels) - 1):  # selective direct labels: endpoints only
        ax.annotate(f"{means[j]:.3f}%", (levels[j], means[j]),
                    textcoords="offset points", xytext=(0, 13), ha="center",
                    fontsize=9, color=INK_PRIMARY, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{n:,}" for n in LEVELS])
    ax.minorticks_off()
    ax.margins(y=0.10)  # headroom so off-ladder reference markers are not clipped
    ax.set_xlabel("Labeled training examples (log scale)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Query error (%)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Permutation-invariant MNIST-medium: error versus training examples",
                 color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    # Lead the legend with the candidate the ladder is about.
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)),
                   key=lambda i: 0 if labels[i].startswith(ladder["candidate_id"]) else 1)
    legend = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                       frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figure_paired_endpoints(ladder, path, comparison=None):
    """Draw-level pairing at the two endpoints, plus the paired difference dots."""
    panels = 2 if comparison and comparison.get("at_10000", {}).get("available") else 1
    fig, axes = plt.subplots(1, panels, figsize=(4.6 * panels + 1.0, 4.6))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    style_axes(ax)
    low_row, high_row = ladder["levels"][0], ladder["levels"][-1]
    xs = [0.0, 1.0]
    for i, seed in enumerate(ladder["draw_seeds"]):
        ys = [low_row["per_draw_error_pct"][i], high_row["per_draw_error_pct"][i]]
        ax.plot(xs, ys, color=COLOR_MAIN, alpha=0.32, linewidth=1.2, marker="o",
                markersize=4, markeredgecolor="none", zorder=2,
                label="one dataset draw" if i == 0 else None)
    for x, row in zip(xs, (low_row, high_row)):
        ax.errorbar([x], [row["expected_cutoff_error_pct"]], yerr=[row["sd_error_pp"] or 0.0],
                    fmt="s", color=COLOR_MAIN, markersize=10, linewidth=2.2, capsize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5,
                    label="mean +/-1 SD" if x == xs[0] else None)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{LEVELS[0]:,} examples", f"{LEVELS[-1]:,} examples"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Query error (%)", color=INK_SECONDARY, fontsize=10)
    ax.set_title(f"Paired endpoints, {ladder['draw_count']} draws",
                 color=INK_PRIMARY, fontsize=11, loc="left", pad=10)
    legend = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    if panels == 2:
        block = comparison["at_10000"]
        ax = axes[1]
        style_axes(ax)
        differences = [pair["difference_pp"] for pair in block["pairs"]]
        ax.axhline(0.0, color=INK_MUTED, linewidth=1.0)
        ax.plot(range(len(differences)), differences, linestyle="none", marker="o",
                markersize=7, color=COLOR_REFERENCE, alpha=0.75,
                markeredgecolor=SURFACE, markeredgewidth=1.0, label="per-draw difference")
        mean = block["mean_difference_pp"]
        ax.axhline(mean, color=COLOR_REFERENCE, linewidth=2.0,
                   label=f"mean {mean:+.3f} pp")
        if block["t_ci95_pp"][0] is not None:
            ax.axhspan(block["t_ci95_pp"][0], block["t_ci95_pp"][1],
                       color=COLOR_REFERENCE, alpha=0.14, linewidth=0,
                       label="t 95% interval")
        ax.set_xticks(range(len(differences)))
        ax.set_xticklabels([str(pair["seed"])[-2:] for pair in block["pairs"]], fontsize=8)
        ax.set_xlabel("Dataset seed (last two digits)", color=INK_SECONDARY, fontsize=10)
        ax.set_ylabel("Ours minus spatial CNN (pp)", color=INK_SECONDARY, fontsize=10)
        ax.set_title("Paired difference at 10,000 examples", color=INK_PRIMARY,
                     fontsize=11, loc="left", pad=10)
        legend = ax.legend(frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figure_dev_time_vs_error(rows, n, path):
    """Development stage: mean fit time against mean error, one point per candidate."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    times = [row["mean_training_seconds"] for row in rows]
    errors = [row["expected_error_pct"] for row in rows]
    usable = [(t, e, row) for t, e, row in zip(times, errors, rows) if t is not None]
    if usable:
        ax.plot([t for t, _, _ in usable], [e for _, e, _ in usable], linestyle="none",
                marker="o", markersize=8, color=COLOR_MAIN, alpha=0.85,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        for t, e, row in usable:
            ax.annotate(row["candidate_id"], (t, e), textcoords="offset points",
                        xytext=(7, 3), fontsize=8, color=INK_SECONDARY)
    else:
        ax.text(0.5, 0.5, "no training_seconds reported", transform=ax.transAxes,
                ha="center", color=INK_SECONDARY, fontsize=10)
    ax.set_xlabel("Mean training seconds per fit", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Mean query error (%)", color=INK_SECONDARY, fontsize=10)
    ax.set_title(f"Development candidates at N = {int(n):,}", color=INK_PRIMARY,
                 fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ dev report ---
def development_table(jobs):
    table = []
    for candidate_id in candidates_in(jobs):
        for n in sorted({row["n"] for row in select(jobs, candidate_id)}):
            rows = select(jobs, candidate_id, n)
            entry = level_stats(rows, n)
            entry["candidate_id"] = candidate_id
            entry["per_seed"] = [{"seed": row["seed"], "error_pct": row["error_pct"],
                                  "correct": row["correct"], "total": row["total"],
                                  "training_seconds": row["metrics"].get("training_seconds"),
                                  "fit_wall_seconds": row["metrics"].get("fit_wall_seconds"),
                                  "epochs_completed": row["metrics"].get("epochs_completed"),
                                  "truncated": row["metrics"].get("truncated"),
                                  "id": row["id"]} for row in rows]
            table.append(entry)
    table.sort(key=lambda entry: (entry["n"], entry["expected_error_pct"]))
    return table


def development_markdown(table):
    lines = ["# Development stage: candidates by training-set size", "",
             "Pooled error is `100 * (sum(total) - sum(correct)) / sum(total)` over the "
             "development seeds, from integer counts. Sorted by error within each N.", ""]
    for n in sorted({entry["n"] for entry in table}):
        lines += [f"## N = {n:,}", "",
                  "| Candidate | Mean error | SD (pp) | Draws | Mean train s | "
                  "Mean fit wall s | Truncated | Unlabeled queries |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for entry in [e for e in table if e["n"] == n]:
            flag = entry["uses_query_images_unlabeled"]
            if flag["reported"] == 0:
                flag_text = "n/r"
            elif flag["all"]:
                flag_text = "yes"
            elif flag["any"]:
                flag_text = f"mixed ({flag['true']}/{flag['reported']})"
            else:
                flag_text = "no"
            sd = format_number(entry["sd_error_pp"], 4)
            train = format_number(entry["mean_training_seconds"], 1)
            wall = format_number(entry["mean_fit_wall_seconds"], 1)
            lines.append(f"| `{entry['candidate_id']}` | "
                         f"{format_pct(entry['expected_error_pct'])} | {sd} | "
                         f"{entry['draws']} | {train} | {wall} | "
                         f"{entry['truncated_fits']}/{entry['draws']} | {flag_text} |")
        lines.append("")
    lines += ["## Per-seed errors", "",
              "| Candidate | N | Seed | Error | Correct / total | Train s | Epochs | Truncated |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for entry in table:
        for row in entry["per_seed"]:
            train = format_number(row["training_seconds"], 1)
            lines.append(f"| `{entry['candidate_id']}` | {entry['n']:,} | {row['seed']} | "
                         f"{format_pct(row['error_pct'], 2)} | "
                         f"{row['correct']} / {row['total']} | {train} | "
                         f"{row['epochs_completed']} | {row['truncated']} |")
    lines.append("")
    return "\n".join(lines)


def run_development(loaded, out_dir):
    jobs = loaded["jobs"]
    table = development_table(jobs)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written = []
    for n in sorted({entry["n"] for entry in table}):
        path = figures / f"dev-time-vs-error-n{n}.png"
        figure_dev_time_vs_error([e for e in table if e["n"] == n], n, path)
        written.append(str(path))

    flat = [{"candidate_id": entry["candidate_id"], "n": entry["n"],
             "draws": entry["draws"], "mean_error_pct": entry["expected_error_pct"],
             "mean_of_draw_errors_pct": entry["mean_of_draw_errors_pct"],
             "sd_error_pp": entry["sd_error_pp"],
             "mean_training_seconds": entry["mean_training_seconds"],
             "mean_fit_wall_seconds": entry["mean_fit_wall_seconds"],
             "mean_epochs_completed": entry["mean_epochs_completed"],
             "truncated_fits": entry["truncated_fits"],
             "uses_query_images_unlabeled": entry["uses_query_images_unlabeled"]["all"]}
            for entry in table]
    per_seed = [{"candidate_id": entry["candidate_id"], "n": entry["n"], **row}
                for entry in table for row in entry["per_seed"]]

    payload = {"study": "pmnist-medium-cutoffs-20260923", "mode": "development",
               "generated_at_utc": utc_now(), "scores_file": loaded["path"],
               "detected_keys": loaded["detected_keys"],
               "candidates": candidates_in(jobs), "job_count": len(jobs),
               "table": table, "figures": written,
               "integrity": {"error_pct_mismatches": loaded["error_pct_mismatches"],
                             "aggregate_cross_check": check_aggregate(loaded, flat)},
               "method": "Every mean is pooled from integer correct/total counts; no rounded "
                         "percentage is ever averaged."}
    write_json(out_dir / "dev_summary.json", payload)
    write_csv(out_dir / "dev_table.csv", flat)
    write_csv(out_dir / "dev_per_seed.csv", per_seed)
    (out_dir / "dev_table.md").write_text(development_markdown(table))
    return payload


# ---------------------------------------------------------------- final report ---
SCOPE_NOTES = [
    "Errors are measured on the 10,000 held-out query images of each dataset draw, drawn "
    "from the official MNIST training split; the official test split is not used.",
    "Every mean comes from integer correct/total counts; no rounded percentage is averaged.",
    "The cutoffs are expected-mean thresholds for this frozen recipe. They are not minimum "
    "necessary sample counts, a convergence certificate, or a per-run pass guarantee.",
    "The bootstrap resamples whole dataset draws with all five levels together, so it "
    "reflects draw-level variability conditional on the fixed pool, the frozen recipe and "
    "the fixed feature permutation. It is not a prediction interval for a new run.",
    "The training prefixes are nested within a dataset seed, so the five levels of one draw "
    "are positively dependent; the paired bootstrap keeps that dependence.",
    f"Rounded cutoff options use half-up rounding to {ROUNDING_STEPS[0]} pp and "
    f"{ROUNDING_STEPS[1]} pp; rounding changes the implied spacing between levels.",
]


def run_final(loaded, candidate_id, out_dir, reference_csv=None,
              resamples=BOOTSTRAP_RESAMPLES, bootstrap_seed=BOOTSTRAP_SEED):
    jobs = loaded["jobs"]
    available = candidates_in(jobs)
    if candidate_id is None:
        primary = [c for c in available if not c.lower().startswith(TOPO_PREFIX)]
        if len(primary) == 1:
            candidate_id = primary[0]
        elif len(available) == 1:
            candidate_id = available[0]
        else:
            raise SystemExit(f"--candidate is required; scores file holds {available}")
    if candidate_id not in available:
        raise SystemExit(f"candidate {candidate_id!r} not in scores file; have {available}")

    ladder = build_ladder(jobs, candidate_id, resamples=resamples,
                          bootstrap_seed=bootstrap_seed)

    topo_ids = [c for c in available
                if c.lower().startswith(TOPO_PREFIX) and c != candidate_id]
    topo_ladder = None
    if topo_ids:
        try:
            topo_ladder = build_ladder(jobs, topo_ids[0], resamples=resamples,
                                       bootstrap_seed=bootstrap_seed)
        except ValueError:
            topo_ladder = None

    reference = load_reference_csv(reference_csv) if reference_csv else None
    comparisons = []
    if reference is not None:
        comparisons.append(compare_with_reference(jobs, candidate_id, reference,
                                                  bootstrap_seed))
        for topo_id in topo_ids:
            block = compare_with_reference(jobs, topo_id, reference, bootstrap_seed)
            block["loophole_check"] = (
                "Recovered-topology CNN versus the true-topology spatial CNN on the same "
                "eleven dataset seeds. A small difference means the permutation cost little; "
                "it does not prove the topology was recovered exactly.")
            comparisons.append(block)

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    curve_path = figures_dir / "error-vs-n.png"
    paired_path = figures_dir / "paired-endpoints.png"
    figure_error_vs_n(ladder, curve_path, reference=reference, topo_ladder=topo_ladder)
    figure_paired_endpoints(ladder, paired_path,
                            comparison=comparisons[0] if comparisons else None)

    ladder_rows = [{
        "level": row["level"], "n": row["n"], "n_exact": row["n_exact"],
        "draws": row["draws"],
        "aggregate_correct": row["aggregate_correct"], "aggregate_total": row["aggregate_total"],
        "expected_error_pct": row["expected_cutoff_error_pct"],
        "expected_accuracy_pct": row["expected_accuracy_pct"],
        "mean_of_draw_errors_pct": row["mean_of_draw_errors_pct"],
        "sd_error_pp": row["sd_error_pp"],
        "ci95_low_pct": row["mean_ci95_pct"][0], "ci95_high_pct": row["mean_ci95_pct"][1],
        "bootstrap_ci95_low_pct": row["bootstrap_ci95_pct"][0],
        "bootstrap_ci95_high_pct": row["bootstrap_ci95_pct"][1],
        "min_draw_error_pct": row["min_draw"]["error_pct"],
        "max_draw_error_pct": row["max_draw"]["error_pct"],
        "mean_training_seconds": row["mean_training_seconds"],
        "mean_fit_wall_seconds": row["mean_fit_wall_seconds"],
        "truncated_fits": row["truncated_fits"],
        "pass_count": row["pass_count"], "pass_fraction": row["pass_fraction"],
        "two_anchor_predicted_error_pct": row["two_anchor_predicted_error_pct"],
        "floor_fit_predicted_error_pct": row["floor_fit_predicted_error_pct"],
        "rounded_0p05_error_pct": row["rounded_options"]["0.05"]["error_pct"],
        "rounded_0p05_pass_fraction": row["rounded_options"]["0.05"]["pass_fraction"],
        "rounded_0p1_error_pct": row["rounded_options"]["0.1"]["error_pct"],
        "rounded_0p1_pass_fraction": row["rounded_options"]["0.1"]["pass_fraction"],
    } for row in ladder["levels"]]

    per_draw_rows = [{
        "candidate_id": job["candidate_id"], "n": job["n"], "seed": job["seed"],
        "correct": job["correct"], "total": job["total"], "error_pct": job["error_pct"],
        "training_seconds": job["metrics"].get("training_seconds"),
        "fit_wall_seconds": job["metrics"].get("fit_wall_seconds"),
        "epochs_completed": job["metrics"].get("epochs_completed"),
        "truncated": job["metrics"].get("truncated"),
        "uses_query_images_unlabeled": job["metrics"].get("uses_query_images_unlabeled"),
        "id": job["id"],
    } for job in sorted(jobs, key=lambda j: (j["candidate_id"], j["n"], j["seed"]))]

    payload = {
        "study": "pmnist-medium-cutoffs-20260923",
        "mode": "final",
        "generated_at_utc": utc_now(),
        "scores_file": loaded["path"],
        "detected_keys": loaded["detected_keys"],
        "candidate_id": candidate_id,
        "candidates_in_file": available,
        "protocol": {
            "levels_rule": "N_i = 1000 * 10 ** (i/4), i = 0..4",
            "levels_exact": LEVELS_EXACT,
            "levels_rounded": LEVELS,
            "level_multiplier": LEVEL_MULTIPLIER,
            "expected_dataset_seeds": FINAL_SEEDS,
            "queries_per_draw": QUERY_COUNT,
        },
        "ladder": ladder,
        "topo_ladder": topo_ladder,
        "reference_comparisons": comparisons,
        "reference_csv": str(reference_csv) if reference_csv else None,
        "reference_curve": reference_curve(reference) if reference else None,
        "figures": {"error_vs_n": str(curve_path), "paired_endpoints": str(paired_path)},
        "integrity": {
            "job_count": len(jobs),
            "error_pct_mismatches": loaded["error_pct_mismatches"],
            "aggregate_cross_check": check_aggregate(loaded, [
                {"candidate_id": candidate_id, "n": row["n"],
                 "expected_error_pct": row["expected_cutoff_error_pct"],
                 "mean_of_draw_errors_pct": row["mean_of_draw_errors_pct"]}
                for row in ladder["levels"]]),
            "missing_expected_seeds": sorted(set(FINAL_SEEDS) - set(ladder["draw_seeds"])),
            "unexpected_seeds": sorted(set(ladder["draw_seeds"]) - set(FINAL_SEEDS)),
            "levels_covered": ladder["draw_count"] == len(ladder["draw_seeds"]),
        },
        "scope_notes": SCOPE_NOTES,
    }
    write_json(out_dir / "summary.json", payload)
    write_csv(out_dir / "ladder.csv", ladder_rows)
    write_csv(out_dir / "per_draw.csv", per_draw_rows)
    (out_dir / "summary.md").write_text(ladder_markdown(ladder, comparisons, SCOPE_NOTES))
    return payload


# ------------------------------------------------------------------------ main ---
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", required=True, type=Path,
                        help="results/scores_<stage>.json written by score.py")
    parser.add_argument("--candidate", default=None,
                        help="candidate id for the final ladder (inferred when unambiguous)")
    parser.add_argument("--reference-csv", default=None, type=Path,
                        help="per_draw.csv of the spatial cutoff-calibration study")
    parser.add_argument("--out", default=Path("analysis"), type=Path,
                        help="output directory")
    parser.add_argument("--dev", action="store_true",
                        help="development-stage table instead of the final ladder")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.bootstrap < 100:
        raise SystemExit("--bootstrap needs at least 100 resamples")
    loaded = load_scores(args.scores)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dev:
        payload = run_development(loaded, out_dir)
        print(f"development table: {len(payload['table'])} (candidate, n) cells from "
              f"{payload['job_count']} jobs -> {out_dir}")
        return 0
    payload = run_final(loaded, args.candidate, out_dir,
                        reference_csv=args.reference_csv,
                        resamples=args.bootstrap, bootstrap_seed=args.bootstrap_seed)
    ladder = payload["ladder"]
    print(f"candidate {payload['candidate_id']} over {ladder['draw_count']} draws")
    for row in ladder["levels"]:
        print(f"  level {row['level']}  N={row['n']:>6,}  "
              f"error {row['expected_cutoff_error_pct']:.4f}%  "
              f"SD {row['sd_error_pp']:.4f} pp  "
              f"pass {row['pass_count']}/{row['draws']}")
    print(f"wrote summary.json, summary.md, ladder.csv, per_draw.csv, figures/ -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
