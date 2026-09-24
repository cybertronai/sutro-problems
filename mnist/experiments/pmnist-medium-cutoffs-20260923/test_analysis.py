#!/usr/bin/env python3
"""Unit tests for analysis.py.

Everything here is fabricated in a temporary directory: no MNIST file, no
prediction, no Modal call.  The synthetic scores follow the same schema that
``score.py`` writes (``{"stage", "jobs": [...], "aggregate": [...]}``).

Construction trick used throughout
----------------------------------
The synthetic truth is ``6.25 * (N/1000) ** (-b)`` with ``b = 4*log10(1.25)``,
so each ladder step divides the error by exactly 1.25 and the five level errors
are **6.25, 5.00, 4.00, 3.20, 2.56 percent** -- all exact multiples of 0.01 pp,
i.e. exact integer wrong-counts out of 10,000.  Draw offsets are multiples of
0.01 pp that sum to exactly zero.  The pooled mean over the eleven draws is
therefore *exactly* the truth, and the two-anchor exponent, the endpoint ratio
and every per-step ratio are recovered to machine precision on noiseless data.

Run:  /tmp/pmnist-env/bin/python -m unittest -v test_analysis.py
"""
from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analysis

# 6.25 * 1.25**-i at level i: 6.25, 5.00, 4.00, 3.20, 2.56 -- all on the
# 0.01 pp integer-count grid, so the fabricated means carry no rounding error.
TRUE_A = 6.25
TRUE_B = 4.0 * math.log10(1.25)
TRUE_STEP_RATIO = 1.25
TOPO_A = 5.0
TOPO_B = 0.32
REFERENCE_A = 7.5  # deliberately worse than TRUE_A so the paired sign is known
REFERENCE_B = 0.30
COUNT_GRID_PP = 0.01  # 10,000 queries -> error is a multiple of 0.01 pp
FINAL_SEEDS = analysis.FINAL_SEEDS
DEV_SEEDS = [2026092301, 2026092302, 2026092303]
REFERENCE_GRID = [100, 200, 400, 800, 1600, 3200, 6400, 10000]


def truth_at(n, a=TRUE_A, b=TRUE_B):
    return a * (n / 1000.0) ** (-b)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def zero_sum_offsets(count, step=0.02):
    """Offsets that are multiples of 0.01 pp and sum to exactly zero."""
    offsets = [round((i - (count - 1) / 2.0) * step, 10) for i in range(count)]
    assert abs(sum(offsets)) < 1e-12
    for offset in offsets:
        assert abs(offset / COUNT_GRID_PP - round(offset / COUNT_GRID_PP)) < 1e-9
    return offsets


def make_job(candidate_id, seed, n, error, total=10000, training=None, truncated=False,
             unlabeled=False, epochs=150, stage="final", wrong_stored_error=False):
    """One scores record with an exact integer correct count."""
    wrong = int(round(error * total / 100.0))
    correct = total - wrong
    exact_error = 100.0 * wrong / total
    return {
        "id": f"{stage}-{candidate_id}-s{seed}-n{n}",
        "candidate_id": candidate_id,
        "seed": seed,
        "n": n,
        "correct": correct,
        "total": total,
        "error_pct": 99.0 if wrong_stored_error else exact_error,
        "predictions_sha256": "0" * 64,
        "result_sha256": "1" * 64,
        "metrics": {
            "training_seconds": training if training is not None else 10.0 + n / 100.0,
            "fit_wall_seconds": (training if training is not None else 10.0 + n / 100.0) + 5.0,
            "epochs_completed": epochs,
            "truncated": truncated,
            "uses_query_images_unlabeled": unlabeled,
        },
    }


def make_scores(path, candidates, seeds, levels=None, noisy=True, stage="final",
                extra_jobs=()):
    """Write a scores JSON in score.py's schema and return the payload."""
    levels = list(levels or analysis.LEVELS)
    offsets = zero_sum_offsets(len(seeds)) if noisy else [0.0] * len(seeds)
    jobs = []
    for candidate_id, (a, b) in candidates.items():
        for index, seed in enumerate(seeds):
            for n in levels:
                error = truth_at(n, a, b) + offsets[index]
                jobs.append(make_job(candidate_id, seed, n, error, stage=stage,
                                     unlabeled=candidate_id.startswith("topo")))
    jobs.extend(extra_jobs)
    aggregate = []
    for candidate_id in sorted({job["candidate_id"] for job in jobs}):
        for n in sorted({job["n"] for job in jobs if job["candidate_id"] == candidate_id}):
            errors = [job["error_pct"] for job in jobs
                      if job["candidate_id"] == candidate_id and job["n"] == n]
            aggregate.append({
                "candidate_id": candidate_id, "n": n, "count": len(errors),
                "mean_error_pct": float(np.mean(errors)),
                "sd_error_pct": float(np.std(errors, ddof=1)) if len(errors) > 1 else None,
                "min_error_pct": min(errors), "max_error_pct": max(errors)})
    payload = {"schema_version": 1, "stage": stage, "scored_at_utc": "2026-09-23T00:00:00Z",
               "job_count": len(jobs), "jobs": jobs, "aggregate": aggregate}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def make_reference_csv(path, seeds=FINAL_SEEDS, a=REFERENCE_A, b=REFERENCE_B, total=10000):
    """Reference per_draw.csv with the calibration study's real column set."""
    columns = ["draw_index", "dataset_seed", "n", "total", "ensemble_final_correct",
               "ensemble_final_error_pct", "single101_final_correct",
               "single101_final_error_pct", "ensemble_midpoint_correct",
               "ensemble_midpoint_error_pct", "training_ms", "final_prediction_ms",
               "time_ms", "job_wall_ms"]
    offsets = zero_sum_offsets(len(seeds), step=0.03)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, seed in enumerate(seeds):
            for n in REFERENCE_GRID:
                error = truth_at(n, a, b) + offsets[index]
                wrong = int(round(error * total / 100.0))
                writer.writerow({
                    "draw_index": index, "dataset_seed": seed, "n": n, "total": total,
                    "ensemble_final_correct": total - wrong,
                    "ensemble_final_error_pct": 100.0 * wrong / total,
                    "single101_final_correct": total - wrong - 3,
                    "single101_final_error_pct": 100.0 * (wrong + 3) / total,
                    "ensemble_midpoint_correct": total - wrong - 1,
                    "ensemble_midpoint_error_pct": 100.0 * (wrong + 1) / total,
                    "training_ms": 1000.0 * n / 100.0, "final_prediction_ms": 50.0,
                    "time_ms": 1000.0 * n / 100.0 + 50.0, "job_wall_ms": 1000.0 * n / 80.0})
    return Path(path)


class HelperTests(unittest.TestCase):
    def test_levels_match_the_geometric_rule(self):
        self.assertEqual(analysis.LEVELS, [1000, 1778, 3162, 5623, 10000])
        for i, exact in enumerate(analysis.LEVELS_EXACT):
            self.assertAlmostEqual(exact, 1000 * 10 ** (i / 4), places=9)
        ratios = [analysis.LEVELS_EXACT[i + 1] / analysis.LEVELS_EXACT[i] for i in range(4)]
        for ratio in ratios:
            self.assertAlmostEqual(ratio, 10 ** 0.25, places=12)

    def test_error_pct_is_exact_integer_arithmetic(self):
        self.assertEqual(analysis.error_pct(9895, 10000), 1.05)
        self.assertEqual(analysis.error_pct(10000, 10000), 0.0)
        with self.assertRaises(ValueError):
            analysis.error_pct(10001, 10000)

    def test_round_to_is_half_up(self):
        self.assertAlmostEqual(analysis.round_to(1.0250, 0.05), 1.05)
        self.assertAlmostEqual(analysis.round_to(1.0249, 0.05), 1.00)
        self.assertAlmostEqual(analysis.round_to(1.1500, 0.1), 1.20)
        self.assertAlmostEqual(analysis.round_to(2.1275, 0.05), 2.15)
        self.assertAlmostEqual(analysis.round_to(2.1275, 0.1), 2.10)

    def test_pava_projects_onto_nonincreasing_sequences(self):
        np.testing.assert_allclose(analysis.pava_decreasing([12, 8, 9, 5]), [12, 8.5, 8.5, 5])
        np.testing.assert_allclose(analysis.pava_decreasing([1, 2, 3]), [2, 2, 2])
        np.testing.assert_allclose(analysis.pava_decreasing([5, 4, 3]), [5, 4, 3])


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_loads_score_py_schema_and_recomputes_errors(self):
        path = self.root / "scores_final.json"
        make_scores(path, {"pmnist-a": (TRUE_A, TRUE_B)}, FINAL_SEEDS)
        loaded = analysis.load_scores(path)
        self.assertEqual(loaded["detected_keys"]["jobs_key"], "jobs")
        self.assertEqual(loaded["detected_keys"]["aggregate_key"], "aggregate")
        self.assertEqual(loaded["stage"], "final")
        self.assertEqual(len(loaded["jobs"]), 5 * len(FINAL_SEEDS))
        self.assertEqual(loaded["error_pct_mismatches"], [])
        for job in loaded["jobs"]:
            self.assertEqual(job["stage"], "final")  # inherited from the top level
            self.assertAlmostEqual(
                job["error_pct"], 100.0 * (job["total"] - job["correct"]) / job["total"], 12)

    def test_a_stored_percentage_is_never_trusted(self):
        path = self.root / "scores_final.json"
        payload = make_scores(path, {"pmnist-a": (TRUE_A, TRUE_B)}, FINAL_SEEDS)
        payload["jobs"][0]["error_pct"] = 99.0  # deliberately wrong
        path.write_text(json.dumps(payload))
        loaded = analysis.load_scores(path)
        self.assertEqual(len(loaded["error_pct_mismatches"]), 1)
        self.assertEqual(loaded["error_pct_mismatches"][0]["stored"], 99.0)
        broken = loaded["jobs"][0]
        # The stored 99% is ignored; the value used comes from the integer counts.
        self.assertNotEqual(broken["error_pct"], 99.0)
        self.assertEqual(broken["error_pct"],
                         100.0 * (broken["total"] - broken["correct"]) / broken["total"])
        self.assertEqual(broken["stored_error_pct"], 99.0)

    def test_bare_list_and_alternate_key_names(self):
        jobs = [{"job_id": "x", "candidate": "c", "dataset_seed": 2026092001,
                 "n_train": 1000, "n_correct": 9800, "n_total": 10000,
                 "training_seconds": 3.0, "truncated": False}]
        path = self.root / "bare.json"
        path.write_text(json.dumps(jobs))
        loaded = analysis.load_scores(path)
        self.assertEqual(loaded["jobs"][0]["candidate_id"], "c")
        self.assertEqual(loaded["jobs"][0]["seed"], 2026092001)
        self.assertEqual(loaded["jobs"][0]["n"], 1000)
        self.assertEqual(loaded["jobs"][0]["error_pct"], 2.0)
        self.assertEqual(loaded["jobs"][0]["metrics"]["training_seconds"], 3.0)

        alt = {"records": jobs, "by_candidate_n": []}
        path2 = self.root / "alt.json"
        path2.write_text(json.dumps(alt))
        self.assertEqual(analysis.load_scores(path2)["detected_keys"]["jobs_key"], "records")

    def test_duplicate_records_are_rejected(self):
        jobs = [make_job("a", 2026092001, 1000, 2.0), make_job("a", 2026092001, 1000, 2.0)]
        path = self.root / "dup.json"
        path.write_text(json.dumps({"stage": "final", "jobs": jobs}))
        with self.assertRaises(ValueError):
            analysis.load_scores(path)


class LadderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.path = self.root / "scores_final.json"
        self.payload = make_scores(
            self.path, {"pmnist-a": (TRUE_A, TRUE_B), "topo-cnn": (TOPO_A, TOPO_B)},
            FINAL_SEEDS)
        self.loaded = analysis.load_scores(self.path)
        self.ladder = analysis.build_ladder(self.loaded["jobs"], "pmnist-a")

    def test_level_means_equal_pooled_integer_counts(self):
        """The headline statistic must be sum(correct)/sum(total), exactly."""
        for row in self.ladder["levels"]:
            jobs = [job for job in self.payload["jobs"]
                    if job["candidate_id"] == "pmnist-a" and job["n"] == row["n"]]
            correct = sum(job["correct"] for job in jobs)
            total = sum(job["total"] for job in jobs)
            self.assertEqual(row["aggregate_correct"], correct)
            self.assertEqual(row["aggregate_total"], total)
            self.assertEqual(row["expected_error_pct"], 100.0 * (total - correct) / total)
            self.assertAlmostEqual(row["expected_accuracy_pct"],
                                   100.0 - row["expected_error_pct"], places=10)
            # Equal totals per draw -> pooled and mean-of-draws coincide.
            self.assertAlmostEqual(row["expected_error_pct"],
                                   row["mean_of_draw_errors_pct"], places=10)

    def test_level_means_are_exactly_the_generating_power_law(self):
        expected = [6.25, 5.00, 4.00, 3.20, 2.56]
        for row, value in zip(self.ladder["levels"], expected):
            self.assertAlmostEqual(row["expected_error_pct"], value, places=12)
            self.assertAlmostEqual(row["expected_error_pct"], truth_at(row["n"]),
                                   delta=COUNT_GRID_PP)

    def test_ladder_shape_and_metadata(self):
        self.assertEqual([row["n"] for row in self.ladder["levels"]], analysis.LEVELS)
        self.assertEqual([row["level"] for row in self.ladder["levels"]], [1, 2, 3, 4, 5])
        self.assertEqual(self.ladder["draw_seeds"], FINAL_SEEDS)
        self.assertEqual(self.ladder["draw_count"], 11)
        for row in self.ladder["levels"]:
            self.assertEqual(row["ci_df"], 10)  # Student-t with 11 draws
            self.assertEqual(len(row["per_draw_error_pct"]), 11)
            self.assertLessEqual(row["min_draw"]["error_pct"], row["max_draw"]["error_pct"])
            self.assertIsNotNone(row["mean_training_seconds"])
            self.assertEqual(row["truncated_fits"], 0)

    def test_t_interval_matches_a_hand_computation(self):
        from scipy.stats import t as student_t
        row = self.ladder["levels"][0]
        values = np.asarray(row["per_draw_error_pct"])
        half = student_t.ppf(0.975, 10) * values.std(ddof=1) / math.sqrt(11)
        self.assertAlmostEqual(row["mean_ci95_pct"][0], values.mean() - half, places=10)
        self.assertAlmostEqual(row["mean_ci95_pct"][1], values.mean() + half, places=10)

    def test_bootstrap_intervals_contain_the_truth(self):
        boot = self.ladder["bootstrap"]
        self.assertEqual(boot["resamples"], analysis.BOOTSTRAP_RESAMPLES)
        self.assertEqual(boot["seed"], 20260923)
        truths = [6.25, 5.00, 4.00, 3.20, 2.56]
        for j, row in enumerate(self.ladder["levels"]):
            low, high = boot["level_error_ci95_pct"][j]
            self.assertLess(low, high)
            self.assertLessEqual(low, truths[j])
            self.assertGreaterEqual(high, truths[j])
            self.assertLessEqual(low, row["expected_error_pct"])
            self.assertGreaterEqual(high, row["expected_error_pct"])

        true_endpoint_ratio = TRUE_STEP_RATIO ** 4  # 6.25 / 2.56
        self.assertAlmostEqual(self.ladder["endpoint_ratio"]["measured"],
                               true_endpoint_ratio, places=12)
        low, high = self.ladder["endpoint_ratio"]["bootstrap_ci95"]
        self.assertLessEqual(low, true_endpoint_ratio)
        self.assertGreaterEqual(high, true_endpoint_ratio)

        low, high = boot["two_anchor_exponent"]["ci95"]
        self.assertLessEqual(low, TRUE_B)
        self.assertGreaterEqual(high, TRUE_B)

        self.assertEqual(len(boot["step_ratios"]), 4)
        for step, measured in zip(boot["step_ratios"], self.ladder["step_ratios"]):
            self.assertAlmostEqual(measured["measured_error_ratio"], TRUE_STEP_RATIO,
                                   places=12)
            self.assertLessEqual(step["ci95"][0], TRUE_STEP_RATIO)
            self.assertGreaterEqual(step["ci95"][1], TRUE_STEP_RATIO)

    def test_bootstrap_is_reproducible_and_pairs_whole_draws(self):
        correct = np.array([[9800, 9900]] * 4, dtype=np.int64)
        total = np.full((4, 2), 10000, dtype=np.int64)
        first = analysis.bootstrap_ladder(correct, total, resamples=50, seed=7)
        second = analysis.bootstrap_ladder(correct, total, resamples=50, seed=7)
        np.testing.assert_array_equal(first["indices"], second["indices"])
        np.testing.assert_allclose(first["errors"], second["errors"])
        # Identical draws -> every resample reproduces the same curve exactly.
        np.testing.assert_allclose(first["errors"], np.tile([2.0, 1.0], (50, 1)))

    def test_two_anchor_fit_recovers_the_exponent_on_noiseless_data(self):
        path = self.root / "noiseless.json"
        make_scores(path, {"pmnist-a": (TRUE_A, TRUE_B)}, FINAL_SEEDS, noisy=False)
        ladder = analysis.build_ladder(analysis.load_scores(path)["jobs"], "pmnist-a")
        fit = ladder["two_anchor_power_law"]
        # The fabricated errors land exactly on the integer-count grid, so the
        # endpoint-anchored exponent is recovered to machine precision.
        self.assertAlmostEqual(fit["b"], TRUE_B, places=12)
        self.assertAlmostEqual(fit["a"], TRUE_A, places=12)
        # Interior levels deviate only because 1778/3162/5623 are the ROUNDED
        # geometric budgets, not 10**0.25 multiples of 1000.
        self.assertLess(fit["max_abs_interior_residual_pp"], 0.01)
        for row in ladder["levels"]:
            self.assertAlmostEqual(row["two_anchor_predicted_error_pct"],
                                   truth_at(row["n"]), delta=0.02)
        # On the exact (unrounded) level floats the fit is exact at every level.
        exact = analysis.power_law_at(fit, analysis.LEVELS_EXACT)
        np.testing.assert_allclose(exact, [6.25, 5.00, 4.00, 3.20, 2.56], rtol=1e-12)

    def test_two_anchor_fit_is_exact_on_analytic_inputs(self):
        fit = analysis.two_anchor_power_law(1000, 4.0, 10000, 1.0)
        self.assertAlmostEqual(fit["b"], math.log10(4.0), places=12)
        self.assertAlmostEqual(fit["a"], 4.0, places=12)
        np.testing.assert_allclose(analysis.power_law_at(fit, [1000, 10000]), [4.0, 1.0])

    def test_floor_fit_recovers_a_known_three_parameter_curve(self):
        ns = analysis.LEVELS
        errors = [0.8 + 3.0 * (n / 1000) ** (-0.6) for n in ns]
        fit = analysis.fit_floor_power_law(ns, errors)
        self.assertAlmostEqual(fit["c"], 0.8, places=4)
        self.assertAlmostEqual(fit["a"], 3.0, places=4)
        self.assertAlmostEqual(fit["b"], 0.6, places=4)
        self.assertLess(fit["rmse_pp"], 1e-6)
        self.assertEqual(fit["residual_dof"], 2)

    def test_floor_fit_on_the_study_ladder_is_reported_with_lack_of_fit(self):
        fit = self.ladder["floor_power_law"]
        self.assertGreaterEqual(fit["c"], 0.0)
        self.assertEqual(len(fit["residual_pp"]), 5)
        self.assertLess(fit["max_abs_residual_pp"], 0.05)
        self.assertIsNotNone(self.ladder["floor_power_law_sem_weighted"]["weighted_sse"])

    def test_rounded_options_and_pass_fractions(self):
        for row in self.ladder["levels"]:
            cutoff = row["expected_cutoff_error_pct"]
            per_draw = np.asarray(row["per_draw_error_pct"])
            self.assertEqual(row["pass_count"], int(np.sum(per_draw <= cutoff)))
            self.assertAlmostEqual(row["pass_fraction"], row["pass_count"] / row["draws"])
            for step, key in ((0.05, "0.05"), (0.1, "0.1")):
                option = row["rounded_options"][key]
                self.assertAlmostEqual(option["error_pct"] / step,
                                       round(option["error_pct"] / step), places=9)
                self.assertLessEqual(abs(option["shift_pp"]), step / 2 + 1e-9)
                self.assertAlmostEqual(
                    option["pass_fraction"],
                    float(np.mean(per_draw <= option["error_pct"])), places=12)
            # Zero-mean symmetric offsets -> the median draw sits at the mean.
            self.assertGreaterEqual(row["pass_count"], 5)

    def test_missing_seed_at_one_level_shrinks_the_paired_draw_set(self):
        payload = json.loads(self.path.read_text())
        payload["jobs"] = [job for job in payload["jobs"]
                           if not (job["candidate_id"] == "pmnist-a"
                                   and job["seed"] == FINAL_SEEDS[0]
                                   and job["n"] == 3162)]
        path = self.root / "gap.json"
        path.write_text(json.dumps(payload))
        ladder = analysis.build_ladder(analysis.load_scores(path)["jobs"], "pmnist-a")
        self.assertEqual(ladder["draw_count"], 10)
        self.assertNotIn(FINAL_SEEDS[0], ladder["draw_seeds"])
        self.assertTrue(any(entry["seed"] == FINAL_SEEDS[0] for entry in ladder["missing_jobs"]))


class ReferenceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.scores = self.root / "scores_final.json"
        make_scores(self.scores, {"pmnist-a": (TRUE_A, TRUE_B), "topo-cnn": (TOPO_A, TOPO_B)},
                    FINAL_SEEDS)
        self.reference_csv = make_reference_csv(self.root / "per_draw.csv")
        self.reference = analysis.load_reference_csv(self.reference_csv)
        self.jobs = analysis.load_scores(self.scores)["jobs"]

    def test_reference_csv_columns_and_recomputed_errors(self):
        self.assertEqual(len(self.reference), len(FINAL_SEEDS) * len(REFERENCE_GRID))
        for row in self.reference.values():
            self.assertAlmostEqual(row["error_pct"], row["stored_error_pct"], places=9)

    def test_interpolation_at_1000_sits_between_the_bracketing_rows(self):
        seed = FINAL_SEEDS[0]
        interpolated = analysis.reference_interpolated_at(self.reference, seed, 1000)
        low, high = interpolated["bracket_error_pct"]
        for key in ("log_linear", "isotonic_log_linear"):
            self.assertLess(interpolated[key], low)
            self.assertGreater(interpolated[key], high)
        self.assertEqual(interpolated["bracket"], [800, 1600])

    def test_paired_comparison_runs_at_both_endpoints(self):
        comparison = analysis.compare_with_reference(self.jobs, "pmnist-a", self.reference)
        block = comparison["at_10000"]
        self.assertTrue(block["available"])
        self.assertEqual(block["draws"], 11)
        self.assertEqual(block["t_df"], 10)
        for pair in block["pairs"]:
            self.assertAlmostEqual(
                pair["difference_pp"],
                pair["ours_error_pct"] - pair["reference_error_pct"], places=12)
        expected = float(np.mean([pair["difference_pp"] for pair in block["pairs"]]))
        self.assertAlmostEqual(block["mean_difference_pp"], expected, places=12)
        self.assertLess(block["t_ci95_pp"][0], block["mean_difference_pp"])
        self.assertGreater(block["t_ci95_pp"][1], block["mean_difference_pp"])
        # Reference truth 7.5*(N/1000)^-0.30 is 3.76% at 10k against our 2.56%.
        self.assertLess(block["mean_difference_pp"], 0.0)
        self.assertEqual(block["wins"], 11)

        for key in ("at_1000_log_linear", "at_1000_isotonic_log_linear"):
            interpolated = comparison[key]
            self.assertTrue(interpolated["available"])
            self.assertEqual(interpolated["draws"], 11)
            self.assertIn("INTERPOLATION, NOT A MEASUREMENT", interpolated["basis"])

    def test_topology_candidate_gets_its_own_paired_comparison(self):
        comparison = analysis.compare_with_reference(self.jobs, "topo-cnn", self.reference)
        self.assertTrue(comparison["at_10000"]["available"])
        self.assertEqual(comparison["candidate_id"], "topo-cnn")


class FinalCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.scores = self.root / "results" / "scores_final.json"
        make_scores(self.scores, {"pmnist-a": (TRUE_A, TRUE_B), "topo-cnn": (TOPO_A, TOPO_B)},
                    FINAL_SEEDS)
        self.reference_csv = make_reference_csv(self.root / "per_draw.csv")
        self.out = self.root / "analysis"

    def test_final_mode_writes_every_artifact(self):
        code = analysis.main(["--scores", str(self.scores), "--candidate", "pmnist-a",
                              "--reference-csv", str(self.reference_csv),
                              "--out", str(self.out), "--bootstrap", "400"])
        self.assertEqual(code, 0)
        for name in ("summary.json", "summary.md", "ladder.csv", "per_draw.csv",
                     "figures/error-vs-n.png", "figures/paired-endpoints.png"):
            path = self.out / name
            self.assertTrue(path.exists(), f"missing {name}")
            self.assertGreater(path.stat().st_size, 0)

        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["mode"], "final")
        self.assertEqual(summary["candidate_id"], "pmnist-a")
        self.assertEqual(summary["protocol"]["levels_rounded"], [1000, 1778, 3162, 5623, 10000])
        self.assertAlmostEqual(summary["protocol"]["levels_exact"][1], 1000 * 10 ** 0.25, 9)
        self.assertEqual(summary["integrity"]["missing_expected_seeds"], [])
        self.assertEqual(summary["integrity"]["error_pct_mismatches"], [])
        self.assertEqual(
            summary["integrity"]["aggregate_cross_check"]["disagreements"], [])
        self.assertEqual(summary["ladder"]["bootstrap"]["resamples"], 400)
        self.assertIsNotNone(summary["topo_ladder"])
        self.assertEqual(summary["topo_ladder"]["candidate_id"], "topo-cnn")
        self.assertEqual(len(summary["reference_comparisons"]), 2)
        self.assertIn("loophole_check", summary["reference_comparisons"][1])
        self.assertEqual(summary["reference_comparisons"][1]["candidate_id"], "topo-cnn")

        rows = read_csv(self.out / "ladder.csv")
        self.assertEqual([int(row["n"]) for row in rows], analysis.LEVELS)
        for row, level in zip(rows, summary["ladder"]["levels"]):
            self.assertAlmostEqual(float(row["expected_error_pct"]),
                                   level["expected_cutoff_error_pct"], places=12)

        text = (self.out / "summary.md").read_text()
        self.assertIn("| Level | Examples | Expected error | Accuracy | 95% CI | "
                      "Pass fraction | Rounded 0.05 / 0.1 |", text)
        for n in analysis.LEVELS:
            self.assertIn(f"{n:,}", text)
        self.assertIn("two-anchor", text.lower())
        self.assertIn("Paired comparison", text)
        self.assertIn("INTERPOLATION, NOT A MEASUREMENT", text)

    def test_final_mode_without_a_reference_csv(self):
        code = analysis.main(["--scores", str(self.scores), "--candidate", "pmnist-a",
                              "--out", str(self.out), "--bootstrap", "200"])
        self.assertEqual(code, 0)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["reference_comparisons"], [])
        self.assertIsNone(summary["reference_curve"])
        self.assertTrue((self.out / "figures" / "paired-endpoints.png").exists())

    def test_unknown_candidate_is_refused(self):
        with self.assertRaises(SystemExit):
            analysis.main(["--scores", str(self.scores), "--candidate", "nope",
                           "--out", str(self.out)])

    def test_candidate_is_inferred_when_only_one_non_topo_candidate_exists(self):
        code = analysis.main(["--scores", str(self.scores), "--out", str(self.out),
                              "--bootstrap", "200"])
        self.assertEqual(code, 0)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["candidate_id"], "pmnist-a")

    def test_summary_json_is_nan_free(self):
        analysis.main(["--scores", str(self.scores), "--candidate", "pmnist-a",
                       "--reference-csv", str(self.reference_csv),
                       "--out", str(self.out), "--bootstrap", "200"])
        text = (self.out / "summary.json").read_text()
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        json.loads(text)


class DevCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.scores = self.root / "results" / "scores_dev.json"
        extra = [make_job("slow-mlp", seed, 1000, 4.0 + 0.1 * index, training=900.0,
                          truncated=True, unlabeled=True, stage="dev")
                 for index, seed in enumerate(DEV_SEEDS)]
        make_scores(self.scores, {"pmnist-a": (TRUE_A, TRUE_B), "topo-cnn": (TOPO_A, TOPO_B)},
                    DEV_SEEDS, levels=[1000, 10000], noisy=True, stage="dev",
                    extra_jobs=extra)
        self.out = self.root / "analysis"

    def test_dev_mode_writes_table_and_figures(self):
        code = analysis.main(["--dev", "--scores", str(self.scores), "--out", str(self.out)])
        self.assertEqual(code, 0)
        for name in ("dev_summary.json", "dev_table.csv", "dev_per_seed.csv", "dev_table.md",
                     "figures/dev-time-vs-error-n1000.png",
                     "figures/dev-time-vs-error-n10000.png"):
            path = self.out / name
            self.assertTrue(path.exists(), f"missing {name}")
            self.assertGreater(path.stat().st_size, 0)

        payload = json.loads((self.out / "dev_summary.json").read_text())
        self.assertEqual(payload["mode"], "development")
        self.assertEqual(sorted(payload["candidates"]), ["pmnist-a", "slow-mlp", "topo-cnn"])

        rows = read_csv(self.out / "dev_table.csv")
        by_n = {}
        for row in rows:
            by_n.setdefault(int(row["n"]), []).append(row)
        for n, group in by_n.items():
            errors = [float(row["mean_error_pct"]) for row in group]
            self.assertEqual(errors, sorted(errors), f"n={n} is not sorted by error")
            for row in group:
                self.assertEqual(int(row["draws"]), len(DEV_SEEDS))
        slow = [row for row in rows if row["candidate_id"] == "slow-mlp"][0]
        self.assertEqual(int(slow["truncated_fits"]), 3)
        self.assertEqual(slow["uses_query_images_unlabeled"], "True")
        self.assertAlmostEqual(float(slow["mean_training_seconds"]), 900.0, places=6)

        per_seed = read_csv(self.out / "dev_per_seed.csv")
        self.assertEqual(len(per_seed), len(rows) * len(DEV_SEEDS))
        for row in per_seed:
            self.assertAlmostEqual(
                float(row["error_pct"]),
                100.0 * (int(row["total"]) - int(row["correct"])) / int(row["total"]),
                places=12)

        text = (self.out / "dev_table.md").read_text()
        self.assertIn("## N = 1,000", text)
        self.assertIn("## Per-seed errors", text)
        self.assertIn("`slow-mlp`", text)

    def test_dev_means_are_pooled_from_integer_counts(self):
        loaded = analysis.load_scores(self.scores)
        table = analysis.development_table(loaded["jobs"])
        raw = json.loads(self.scores.read_text())["jobs"]
        for entry in table:
            jobs = [job for job in raw if job["candidate_id"] == entry["candidate_id"]
                    and job["n"] == entry["n"]]
            correct = sum(job["correct"] for job in jobs)
            total = sum(job["total"] for job in jobs)
            self.assertEqual(entry["expected_error_pct"], 100.0 * (total - correct) / total)


if __name__ == "__main__":
    unittest.main(verbosity=2)
