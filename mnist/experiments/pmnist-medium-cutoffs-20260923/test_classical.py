"""Synthetic unit tests for the CPU classical baselines and their kernels."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import classical  # noqa: E402

PI = math.pi


def scalar(gram):
    """Single entry of a 1x1 kernel matrix."""
    return float(np.asarray(gram)[0, 0])


def toy(n_train=150, n_query=40, seed=0, features=81):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(classical.N_CLASSES, features)).astype(np.float32)
    labels = rng.integers(0, classical.N_CLASSES, n_train).astype(np.uint8)
    train_x = (centres[labels] + 0.6 * rng.normal(size=(n_train, features))).astype(np.float32)
    query_x = rng.normal(size=(n_query, features)).astype(np.float32)
    return train_x, labels, query_x


CONFIGS = {
    "svm_rbf": {"family": "svm_rbf", "C_grid": [1, 5], "gamma_grid": ["scale", 0.02],
                "cv_subsample": None},
    "knn": {"family": "knn", "k_grid": [1, 3, 5]},
    "kernel_ridge_rbf": {"family": "kernel_ridge", "kernel": "rbf",
                         "gamma_grid": [0.01, 0.05], "lambda_grid": [1e-4, 1e-2],
                         "cv_subsample": None},
    "kernel_ridge_arccos1": {"family": "kernel_ridge", "kernel": "arccos1", "depth": 2,
                             "lambda_grid": [1e-4, 1e-2], "cv_subsample": None},
    "kernel_ridge_ntk": {"family": "kernel_ridge", "kernel": "ntk_relu", "depth": 3,
                         "lambda_grid": [1e-4, 1e-2], "cv_subsample": None},
    "hgb": {"family": "hgb", "max_iter": 25, "learning_rate": 0.2, "max_leaf_nodes": 15},
}


class FamilyContract(unittest.TestCase):
    def test_shapes_dtypes_and_argmax_labels(self):
        train_x, train_y, query_x = toy()
        for name, config in CONFIGS.items():
            with self.subTest(family=name):
                out = classical.fit_predict(train_x, train_y, query_x, config, 3)
                self.assertEqual(set(out), {"logits", "labels", "metrics"})
                self.assertEqual(out["logits"].shape, (query_x.shape[0], 10))
                self.assertEqual(out["logits"].dtype, np.float32)
                self.assertEqual(out["labels"].shape, (query_x.shape[0],))
                self.assertEqual(out["labels"].dtype, np.uint8)
                self.assertTrue(np.isfinite(out["logits"]).all())
                expected = np.argmax(out["logits"], axis=1).astype(np.uint8)
                np.testing.assert_array_equal(out["labels"], expected)
                for key in ("training_seconds", "inference_seconds", "family",
                            "uses_query_images_unlabeled"):
                    self.assertIn(key, out["metrics"])
                self.assertFalse(out["metrics"]["uses_query_images_unlabeled"])

    def test_labels_break_ties_to_the_lowest_class(self):
        logits = np.zeros((4, 10), dtype=np.float32)
        logits[1, [2, 7]] = 1.0
        logits[2, :] = 5.0
        logits[3, 9] = 0.5
        np.testing.assert_array_equal(classical.argmax_labels(logits),
                                      np.array([0, 2, 0, 9], dtype=np.uint8))

    def test_unknown_family_and_bad_shapes_raise(self):
        train_x, train_y, query_x = toy(40, 5)
        with self.assertRaises(ValueError):
            classical.fit_predict(train_x, train_y, query_x, {"family": "nope"}, 0)
        with self.assertRaises(ValueError):
            classical.fit_predict(train_x, train_y, query_x[:, :10], CONFIGS["knn"], 0)
        with self.assertRaises(ValueError):
            classical.fit_predict(train_x, train_y[:5], query_x, CONFIGS["knn"], 0)

    def test_knn_logits_are_vote_fractions(self):
        train_x, train_y, query_x = toy(120, 17, seed=5)
        out = classical.fit_predict(train_x, train_y, query_x,
                                    {"family": "knn", "k_grid": [5]}, 1)
        np.testing.assert_allclose(out["logits"].sum(1), 1.0, atol=1e-6)
        scaled = out["logits"] * 5.0
        np.testing.assert_allclose(scaled, np.round(scaled), atol=1e-5)

    def test_normalization_is_recorded(self):
        train_x, train_y, query_x = toy(80, 9)
        out = classical.fit_predict(train_x, train_y, query_x, CONFIGS["knn"], 0)
        self.assertEqual(out["metrics"]["normalization"], classical.NORMALIZATION)
        np.testing.assert_allclose(classical.normalize(np.array([[0.0, 0.5, 1.0]])),
                                   np.array([[-0.5, 1.5, 3.5]]))


class KernelMath(unittest.TestCase):
    def test_all_kernels_are_psd_on_random_data(self):
        rng = np.random.default_rng(11)
        data = rng.normal(size=(60, 12))
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        grams = {
            "rbf": classical.rbf_kernel(data, data, 0.3),
            "arccos1_d1": classical.arccos1_kernel(data, data, 1),
            "arccos1_d2": classical.arccos1_kernel(data, data, 2),
            "arccos1_d3": classical.arccos1_kernel(data, data, 3),
            "ntk_d1": classical.ntk_relu_kernel(data, data, 1),
            "ntk_d3": classical.ntk_relu_kernel(data, data, 3),
        }
        for name, gram in grams.items():
            with self.subTest(kernel=name):
                np.testing.assert_allclose(gram, gram.T, atol=1e-10)
                self.assertGreater(float(np.linalg.eigvalsh(gram).min()), -1e-8)

    def test_arccos1_closed_form_on_hand_computed_vectors(self):
        parallel = np.array([[1.0, 0.0]])
        orthogonal = np.array([[0.0, 1.0]])
        opposite = np.array([[-1.0, 0.0]])
        scaled = np.array([[3.0, 0.0]])
        other = np.array([[0.0, 4.0]])
        diagonal = np.array([[1.0, 1.0]])
        # theta = 0  ->  (1/pi)|x||y|(sin 0 + pi cos 0) = |x||y|
        self.assertAlmostEqual(scalar(classical.arccos1_kernel(parallel, parallel, 1)), 1.0, places=12)
        # theta = pi/2 -> (1/pi)|x||y|
        self.assertAlmostEqual(scalar(classical.arccos1_kernel(parallel, orthogonal, 1)),
                               1.0 / PI, places=12)
        self.assertAlmostEqual(scalar(classical.arccos1_kernel(scaled, other, 1)),
                               12.0 / PI, places=12)
        # theta = pi -> sin pi + 0 * cos pi = 0
        self.assertAlmostEqual(scalar(classical.arccos1_kernel(parallel, opposite, 1)),
                               0.0, places=12)
        # theta = pi/4 with |x| = sqrt(2), |y| = 1
        theta = PI / 4.0
        expected = math.sqrt(2.0) * (math.sin(theta) + (PI - theta) * math.cos(theta)) / PI
        self.assertAlmostEqual(scalar(classical.arccos1_kernel(diagonal, parallel, 1)),
                               expected, places=12)

    def test_arccos1_preserves_norms_and_composes(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(9, 5))
        norms = (data * data).sum(1)
        for depth in (1, 2, 3):
            gram = classical.arccos1_kernel(data, data, depth)
            np.testing.assert_allclose(np.diag(gram), norms, rtol=1e-10)
        once = classical.arccos1_kernel(data, data, 1)
        twice = classical.arccos1_kernel(data, data, 2)
        scale = np.sqrt(np.outer(norms, norms))
        angle = np.arccos(np.clip(once / scale, -1.0, 1.0))
        manual = scale * (np.sin(angle) + (PI - angle) * np.cos(angle)) / PI
        np.testing.assert_allclose(twice, manual, rtol=1e-10, atol=1e-12)

    def test_ntk_relu_is_symmetric_and_matches_depth_one_formula(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(7, 6))
        b = rng.normal(size=(5, 6))
        cross = classical.ntk_relu_kernel(a, b, 3)
        back = classical.ntk_relu_kernel(b, a, 3)
        np.testing.assert_allclose(cross, back.T, rtol=1e-10, atol=1e-12)
        gram = classical.ntk_relu_kernel(a, a, 2)
        np.testing.assert_allclose(gram, gram.T, atol=1e-10)

        depth_one = classical.ntk_relu_kernel(a, b, 1)
        sigma0 = a @ b.T
        scale = np.sqrt(np.outer((a * a).sum(1), (b * b).sum(1)))
        angle = np.arccos(np.clip(sigma0 / scale, -1.0, 1.0))
        sigma1 = scale * (np.sin(angle) + (PI - angle) * np.cos(angle)) / PI
        expected = sigma0 * ((PI - angle) / PI) + sigma1
        np.testing.assert_allclose(depth_one, expected, rtol=1e-10, atol=1e-12)
        # orthogonal unit vectors: Sigma0 = 0, Sigma1 = 1/pi, Theta1 = 1/pi
        e1 = np.array([[1.0, 0.0]])
        e2 = np.array([[0.0, 1.0]])
        self.assertAlmostEqual(scalar(classical.ntk_relu_kernel(e1, e2, 1)), 1.0 / PI, places=12)
        # identical unit vectors: Sigma0 = 1, Sigma1 = 1, Sigmadot = 1, Theta1 = 2
        self.assertAlmostEqual(scalar(classical.ntk_relu_kernel(e1, e1, 1)), 2.0, places=12)

    def test_rbf_kernel_matches_definition(self):
        rng = np.random.default_rng(6)
        a = rng.normal(size=(4, 3))
        b = rng.normal(size=(6, 3))
        gram = classical.rbf_kernel(a, b, 0.7)
        manual = np.exp(-0.7 * ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
        np.testing.assert_allclose(gram, manual, rtol=1e-12, atol=1e-14)

    def test_build_kernel_rejects_unknown_names(self):
        with self.assertRaises(ValueError):
            classical.build_kernel("mystery", np.zeros((2, 2)), np.zeros((2, 2)), {})


class SelectionIgnoresQueries(unittest.TestCase):
    """Hyper-parameter choice must be a function of the training rows alone."""

    SELECTION_KEYS = ("chosen_C", "chosen_gamma", "chosen_k", "chosen_lambda",
                      "chosen_kernel", "chosen_depth", "cv_accuracy", "cv_rows",
                      "cv_folds", "cv_table")

    def _selection(self, metrics):
        return {k: metrics[k] for k in self.SELECTION_KEYS if k in metrics}

    def test_identical_choice_for_two_different_query_sets(self):
        train_x, train_y, query_a = toy(150, 40, seed=21)
        rng = np.random.default_rng(99)
        query_b = (10.0 * rng.normal(size=(17, train_x.shape[1]))).astype(np.float32)
        for name, config in CONFIGS.items():
            if name == "hgb":
                continue  # no cross-validated selection
            with self.subTest(family=name):
                first = classical.fit_predict(train_x, train_y, query_a, config, 5)
                second = classical.fit_predict(train_x, train_y, query_b, config, 5)
                self.assertEqual(self._selection(first["metrics"]),
                                 self._selection(second["metrics"]))
                self.assertTrue(len(self._selection(first["metrics"])) >= 3)

    def test_cv_subset_is_deterministic_and_stratified(self):
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 10, 900)
        first = classical._cv_subset(labels, 300, 7)
        second = classical._cv_subset(labels, 300, 7)
        np.testing.assert_array_equal(first, second)
        self.assertLessEqual(len(first), 400)
        self.assertEqual(len(np.unique(labels[first])), 10)
        np.testing.assert_array_equal(classical._cv_subset(labels, None, 7), np.arange(900))


class RidgeSelectionStability(unittest.TestCase):
    """A lambda the full-N refit cannot factor must never win the CV grid."""

    def test_unstable_lambda_is_flagged_and_never_selected(self):
        rng = np.random.default_rng(2)
        labels = rng.integers(0, 10, 60)
        gram = np.eye(60)
        folds = [(np.arange(0, 40), np.arange(40, 60)),
                 (np.arange(20, 60), np.arange(0, 20))]
        real_solve = classical._ridge_solve

        def flaky(matrix, targets, lam, inplace=False):
            solution, _ = real_solve(matrix, targets, lam, inplace=inplace)
            # pretend the tiny ridge needed the lstsq fallback, and make it look
            # like the best score by handing back the exact one-hot targets
            if float(lam) < 1e-6:
                return targets.copy(), True
            return solution, False

        classical._ridge_solve = flaky
        try:
            (score, lam), table = classical._kernel_ridge_cv(
                gram, labels, [1e-9, 1e-2], folds)
        finally:
            classical._ridge_solve = real_solve
        self.assertEqual(lam, 1e-2)
        flagged = {row["lambda"]: row["unstable"] for row in table}
        self.assertTrue(flagged[1e-9])
        self.assertFalse(flagged[1e-2])
        self.assertGreater(max(row["cv_accuracy"] for row in table), score - 1e-12)

    def test_ridge_solve_reports_whether_it_fell_back(self):
        rng = np.random.default_rng(5)
        data = rng.normal(size=(30, 6))
        gram = data @ data.T
        alpha, fell_back = classical._ridge_solve(gram, np.eye(30)[:, :3], 1e-2)
        self.assertFalse(fell_back)
        self.assertEqual(alpha.shape, (30, 3))


class PlanGuards(unittest.TestCase):
    """make_plan must refuse the same off-protocol requests plan.py refuses."""

    def setUp(self):
        import run_classical
        self.runner = run_classical
        self.candidates = ROOT / "plans" / "classical_candidates.json"
        if not (ROOT / "study.py").exists():
            self.skipTest("study.py not present yet")

    def test_rejects_off_protocol_levels_and_final_seeds_in_dev(self):
        out = Path(tempfile.mkdtemp()) / "plan.json"
        with self.assertRaises(ValueError):
            self.runner.make_plan(self.candidates, [900], [2026092301], out)
        with self.assertRaises(ValueError):
            self.runner.make_plan(self.candidates, [1000], [2026092001], out)
        with self.assertRaises(ValueError):
            self.runner.make_plan(self.candidates, [1000], [2026092301], out, stage="probe")
        self.assertFalse(out.exists())

    def test_final_requires_a_committed_selection(self):
        scratch = Path(tempfile.mkdtemp())
        out = scratch / "plan.json"
        with self.assertRaises(ValueError):
            self.runner.make_plan(self.candidates, [1000], [2026092001], out,
                                  stage="final",
                                  selection_path=scratch / "selection.json")
        (scratch / "selection.json").write_text(json.dumps(
            {"candidates": [{"id": "knn", "config": {}}]}))
        with self.assertRaises(ValueError):
            # svm-rbf is not in the committed selection
            self.runner.make_plan(self.candidates, [1000], [2026092001], out,
                                  stage="final",
                                  selection_path=scratch / "selection.json")
        only_knn = scratch / "knn.json"
        only_knn.write_text(json.dumps([{"candidate_id": "knn", "family": "knn",
                                         "k_grid": [3]}]))
        jobs = self.runner.make_plan(only_knn, [1000], [2026092001], out,
                                     stage="final",
                                     selection_path=scratch / "selection.json")
        self.assertEqual([job["id"] for job in jobs],
                         ["final-knn-s2026092001-n1000"])

    def test_dev_plan_over_protocol_levels_is_accepted(self):
        out = Path(tempfile.mkdtemp()) / "plan.json"
        jobs = self.runner.make_plan(self.candidates, [1000], [2026092301], out)
        self.assertTrue(out.exists())
        self.assertTrue(all(job["stage"] == "dev" and job["n"] == 1000 for job in jobs))


class ResumeIsExact(unittest.TestCase):
    """A stored result counts as done only for the identical job and code."""

    def setUp(self):
        import run_classical
        self.runner = run_classical
        if not (ROOT / "study.py").exists():
            self.skipTest("study.py not present yet")
        self.scratch = Path(tempfile.mkdtemp())
        (self.scratch / "results").mkdir()
        (self.scratch / "predictions").mkdir()
        candidates = self.scratch / "candidates.json"
        candidates.write_text(json.dumps([{"candidate_id": "knn", "family": "knn",
                                           "k_grid": [1, 3]}]))
        self.job = self.runner.make_plan(candidates, [1000], [2026092301],
                                         self.scratch / "plan.json")[0]

    def _run(self):
        return self.runner.run_job(self.job, self.scratch / "results",
                                   self.scratch / "predictions")

    def test_rerun_when_config_seed_budget_or_source_changes(self):
        record = self._run()
        self.assertIn("fit_wall_seconds", record["metrics"])
        self.assertEqual(record["metrics"]["fit_wall_seconds"],
                         record["metrics"]["wall_seconds"])
        self.assertTrue(self.runner._already_done(
            self.job, self.scratch / "results", self.scratch / "predictions"))
        for mutate in ({"config": dict(self.job["config"], k_grid=[11, 25])},
                       {"learner_seed": self.job["learner_seed"] + 1},
                       {"time_budget_seconds": 600}):
            changed = dict(self.job, **mutate)
            self.assertFalse(
                self.runner._already_done(changed, self.scratch / "results",
                                          self.scratch / "predictions", verbose=False),
                f"{mutate} must force a rerun")
        stale = json.loads((self.scratch / "results" / f"{self.job['id']}.json").read_text())
        stale["provenance"]["source_sha256"]["classical.py"] = "0" * 64
        (self.scratch / "results" / f"{self.job['id']}.json").write_text(json.dumps(stale))
        self.assertFalse(self.runner._already_done(
            self.job, self.scratch / "results", self.scratch / "predictions",
            verbose=False))

    def test_predictions_path_points_at_the_file_that_was_written(self):
        record = self._run()
        written = self.scratch / "predictions" / f"{self.job['id']}.npz"
        resolved = ROOT / record["predictions_path"]
        self.assertEqual(resolved.resolve(), written.resolve())
        self.assertEqual(record["predictions_sha256"], self.runner._sha_file(written))

    def test_one_failing_job_does_not_abort_the_batch(self):
        broken = dict(self.job, id="dev-broken-s2026092301-n1000",
                      candidate_id="broken",
                      config={"family": "no-such-family"})
        plan = self.scratch / "mixed.json"
        plan.write_text(json.dumps([broken, self.job]))
        status = self.runner.main(["--plan", str(plan),
                                   "--results-dir", str(self.scratch / "results"),
                                   "--predictions-dir", str(self.scratch / "predictions")])
        self.assertEqual(status, 1)
        log = json.loads((self.scratch / "logs" / "classical_failures.json").read_text())
        self.assertEqual([entry["id"] for entry in log["failures"]], [broken["id"]])
        # the healthy job that followed it still ran
        self.assertTrue((self.scratch / "results" / f"{self.job['id']}.json").exists())


class RunnerModule(unittest.TestCase):
    """run_classical must import (and expose its helpers) before study.py exists."""

    def test_imports_without_study_and_hashes_match_canonical_data(self):
        import run_classical
        self.assertTrue(hasattr(run_classical, "run_job"))
        self.assertTrue(hasattr(run_classical, "make_plan"))
        array = np.arange(12, dtype=np.float32).reshape(3, 4)
        import canonical_data
        self.assertEqual(run_classical._array_hash(array), canonical_data.array_hash(array))


if __name__ == "__main__":
    unittest.main(verbosity=2)
