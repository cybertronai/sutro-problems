"""Unit tests for the permutation-invariant MNIST-medium data/protocol layer.

Run: /tmp/pmnist-env/bin/python -m unittest -v test_study.py
No GPU, no Modal, no network. Nothing here writes into ROOT/results.
"""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np

import canonical_data
import study

ROOT = study.ROOT
PYTHON = sys.executable


def _ensure_pool():
    if not study.POOL_IMAGES_PATH.exists():
        study.prepare()


@contextlib.contextmanager
def _without_learner_modules():
    """Drop any learner/runner module another test file imported into this process.

    ``study.query_labels`` refuses to run while they are loaded (that guard is
    tested separately); this keeps the scoring tests independent of test order.
    """
    saved = {name: sys.modules.pop(name) for name in ("learners", "runner")
             if name in sys.modules}
    try:
        yield
    finally:
        sys.modules.update(saved)


class PermutationTests(unittest.TestCase):
    def setUp(self):
        _ensure_pool()

    def test_permutation_is_a_bijection_and_stable(self):
        first = study.feature_permutation()
        second = study.feature_permutation()
        self.assertEqual(first.dtype, np.int64)
        self.assertEqual(first.shape, (81,))
        np.testing.assert_array_equal(np.sort(first), np.arange(81))
        np.testing.assert_array_equal(first, second)
        expected = np.random.Generator(np.random.PCG64(20260923)).permutation(81)
        np.testing.assert_array_equal(first, expected)

    def test_permutation_is_not_the_identity(self):
        self.assertFalse(np.array_equal(study.feature_permutation(), np.arange(81)))


class DrawTests(unittest.TestCase):
    def setUp(self):
        _ensure_pool()

    def test_orders_are_deterministic_permutations(self):
        for seed in study.DEV_SEEDS[:1] + study.FINAL_SEEDS[:2]:
            order = study.draw_order(seed)
            self.assertEqual(order.shape, (60000,))
            self.assertEqual(order.dtype, np.int64)
            np.testing.assert_array_equal(np.sort(order), np.arange(60000))
            np.testing.assert_array_equal(order, study.draw_order(seed))

    def test_orders_match_the_canonical_generator(self):
        for seed in (study.DEV_SEEDS[0], study.FINAL_SEEDS[0]):
            reference, _ = canonical_data.source_permutations(seed=seed)
            np.testing.assert_array_equal(study.draw_order(seed), reference)

    def test_distinct_seeds_give_distinct_draws(self):
        a = study.train_indices(study.FINAL_SEEDS[0], 1000)
        b = study.train_indices(study.FINAL_SEEDS[1], 1000)
        self.assertFalse(np.array_equal(a, b))

    def test_prefixes_are_nested_and_query_is_disjoint(self):
        for seed in (study.DEV_SEEDS[0], study.FINAL_SEEDS[-1]):
            query = study.query_indices(seed)
            self.assertEqual(query.shape, (10000,))
            self.assertEqual(np.unique(query).size, 10000)
            previous = None
            for n in study.LEVELS:
                train = study.train_indices(seed, n)
                self.assertEqual(train.shape, (n,))
                self.assertEqual(np.unique(train).size, n)
                if previous is not None:
                    np.testing.assert_array_equal(train[: previous.shape[0]], previous)
                previous = train
                self.assertEqual(np.intersect1d(train, query).size, 0)

    def test_levels_are_the_rounded_geometric_budgets(self):
        entries = study.levels()
        self.assertEqual([entry["n"] for entry in entries], study.LEVELS)
        for i, entry in enumerate(entries):
            self.assertAlmostEqual(entry["exact"], 1000.0 * 10.0 ** (i / 4.0), places=9)
            self.assertEqual(entry["n"], int(round(entry["exact"])))

    def test_seed_sets(self):
        self.assertEqual(study.DEV_SEEDS, [2026092301, 2026092302, 2026092303])
        self.assertEqual(study.FINAL_SEEDS, list(range(2026092001, 2026092012)))
        self.assertEqual(len(study.FINAL_SEEDS), 11)
        self.assertEqual(set(study.DEV_SEEDS) & set(study.FINAL_SEEDS), set())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        _ensure_pool()

    def test_twenty_rows_recomputed_independently(self):
        pool = study.pool_images()
        rows = np.random.default_rng(12345).choice(60000, 20, replace=False)
        raw = canonical_data.read_idx(
            study.SOURCE_DIR / "train-images-idx3-ubyte.gz", 60000, True)
        subset = raw[rows].astype(np.float32) / np.float32(255)
        resized = canonical_data.area_resize(subset, 9)
        np.clip(resized, 0.0, 1.0, out=resized)
        expected = resized.reshape(20, 81)
        np.testing.assert_array_equal(pool[rows], expected)
        self.assertEqual(pool.dtype, np.float32)
        self.assertGreaterEqual(float(pool.min()), 0.0)
        self.assertLessEqual(float(pool.max()), 1.0)

    def test_labels_match_the_source_file(self):
        raw = canonical_data.read_idx(
            study.SOURCE_DIR / "train-labels-idx1-ubyte.gz", 60000, False)
        labels = np.load(study.POOL_LABELS_PATH)
        self.assertEqual(labels.dtype, np.uint8)
        np.testing.assert_array_equal(labels, raw.astype(np.uint8))

    def test_ahash_matches_canonical_array_hash(self):
        for array in (study.pool_images()[:5].copy(),
                      np.load(study.POOL_LABELS_PATH)[:7],
                      study.feature_permutation().copy()):
            self.assertEqual(study.ahash(array), canonical_data.array_hash(array))

    def test_manifest_hashes_are_current(self):
        manifest = json.loads(study.DATA_MANIFEST_PATH.read_text())
        self.assertFalse(manifest["test_labels_created"])
        self.assertEqual(manifest["arrays"]["pool_images"]["sha256"],
                         study.ahash(np.load(study.POOL_IMAGES_PATH)))
        self.assertEqual(manifest["arrays"]["pool_labels"]["sha256"],
                         study.ahash(np.load(study.POOL_LABELS_PATH)))
        self.assertEqual(manifest["arrays"]["feature_permutation"]["sha256"],
                         study.ahash(np.load(study.PERMUTATION_PATH)))

    def test_prepare_is_idempotent(self):
        before = json.loads(study.DATA_MANIFEST_PATH.read_text())
        after = study.prepare()
        self.assertEqual(before, after)


class JobArrayTests(unittest.TestCase):
    def setUp(self):
        _ensure_pool()

    def test_job_arrays_content_and_absence_of_query_labels(self):
        seed, n = study.DEV_SEEDS[1], 1778
        arrays = study.job_arrays(seed, n)
        self.assertEqual(set(arrays), {"train_x", "train_y", "query_x"})
        pool = study.pool_images()
        labels = np.load(study.POOL_LABELS_PATH)
        perm = study.feature_permutation()
        train = study.train_indices(seed, n)
        query = study.query_indices(seed)
        np.testing.assert_array_equal(arrays["train_x"], pool[train][:, perm])
        np.testing.assert_array_equal(arrays["query_x"], pool[query][:, perm])
        np.testing.assert_array_equal(arrays["train_y"], labels[train])
        self.assertEqual(arrays["train_x"].dtype, np.float32)
        self.assertEqual(arrays["query_x"].dtype, np.float32)
        self.assertEqual(arrays["train_y"].dtype, np.uint8)
        self.assertEqual(arrays["query_x"].shape, (10000, 81))
        truth = labels[query]
        for name, value in arrays.items():
            self.assertFalse(value.shape == truth.shape and value.dtype == truth.dtype
                             and np.array_equal(value, truth),
                             f"{name} leaks the query labels")

    def test_job_arrays_are_writable_copies(self):
        arrays = study.job_arrays(study.DEV_SEEDS[0], 1000)
        arrays["train_x"][0, 0] = 0.5
        again = study.job_arrays(study.DEV_SEEDS[0], 1000)
        self.assertNotEqual(float(again["train_x"][0, 0]), 0.5)
        self.assertTrue(again["train_x"].flags["C_CONTIGUOUS"])

    def test_job_arrays_rejects_oversized_n(self):
        with self.assertRaises(ValueError):
            study.job_arrays(study.DEV_SEEDS[0], 10001)

    def test_make_job_is_deterministic(self):
        config = {"family": "mlp", "width": 512, "nested": {"dropout": 0.2}}
        first = study.make_job("dev", 2026092301, 1000, "mlp-a", config, 7)
        second = study.make_job("dev", 2026092301, 1000, "mlp-a", dict(config), 7)
        self.assertEqual(first, second)
        self.assertEqual(first["id"], "dev-mlp-a-s2026092301-n1000")
        self.assertEqual(first["time_budget_seconds"], 1200)
        arrays = study.job_arrays(2026092301, 1000)
        self.assertEqual(first["input_sha256"],
                         {k: study.ahash(v) for k, v in arrays.items()})
        self.assertEqual(first["train_indices_sha256"],
                         study.ahash(study.train_indices(2026092301, 1000)))
        self.assertEqual(first["query_indices_sha256"],
                         study.ahash(study.query_indices(2026092301)))
        config["width"] = 999
        self.assertEqual(first["config"]["width"], 512)

    def test_make_job_binds_stage_to_its_seed_set(self):
        config = {"family": "mlp"}
        for stage in ("dev", "smoke"):
            with self.assertRaises(ValueError):
                study.make_job(stage, study.FINAL_SEEDS[0], 1000, "cheat", config, 1)
        with self.assertRaises(ValueError):
            study.make_job("final", study.DEV_SEEDS[0], 1000, "cheat", config, 1)
        with self.assertRaises(ValueError):
            study.make_job("probe", study.FINAL_SEEDS[0], 1000, "cheat", config, 1)
        # The permitted combinations still build.
        self.assertTrue(study.make_job("dev", study.DEV_SEEDS[0], 1000, "ok", config, 1))
        self.assertTrue(study.make_job("final", study.FINAL_SEEDS[0], 1000, "ok", config, 1))


class LeakageGuardTests(unittest.TestCase):
    def setUp(self):
        _ensure_pool()

    def test_query_labels_refuses_inside_a_learner_process(self):
        with _without_learner_modules():
            for name in ("learners", "runner"):
                self.assertNotIn(name, sys.modules)
                sys.modules[name] = types.ModuleType(name)
                try:
                    with self.assertRaises(RuntimeError):
                        study.query_labels(study.DEV_SEEDS[0])
                finally:
                    del sys.modules[name]
            labels = study.query_labels(study.DEV_SEEDS[0])
        self.assertEqual(labels.shape, (10000,))
        self.assertEqual(labels.dtype, np.uint8)

    def test_score_is_the_only_caller_of_query_labels(self):
        callers = []
        for path in sorted(ROOT.glob("*.py")):
            if path.name in ("study.py", "score.py", "test_study.py"):
                continue
            # Match the function name itself, not the 'query_labels_supplied' flag.
            if re.search(r"\bquery_labels\b(?!_)", path.read_text()):
                callers.append(path.name)
        self.assertEqual(callers, [])


def _write_synthetic_job(root: Path, stage: str, logits: np.ndarray, labels: np.ndarray,
                         candidate_id: str = "synth", seed: int = None, n: int = 1000):
    seed = seed if seed is not None else study.DEV_SEEDS[0]
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "predictions").mkdir(parents=True, exist_ok=True)
    job = study.make_job(stage, seed, n, candidate_id, {"family": "mlp"}, 1)
    predictions_path = root / "predictions" / f"{job['id']}.npz"
    with predictions_path.open("wb") as stream:
        np.savez(stream, logits=logits, labels=labels)
    record = {
        "job": job,
        "metrics": {"fit_wall_seconds": 1.0, "epochs_completed": 1, "truncated": False,
                    "uses_query_images_unlabeled": False},
        "provenance": {"synthetic": True},
        "completed_at": study.utc(),
        "hardware": "test",
        "software": {"python": sys.version.split()[0]},
        "predictions_path": f"predictions/{job['id']}.npz",
        "predictions_sha256": study.sha(predictions_path),
        "output_sha256": {"logits": study.ahash(logits), "labels": study.ahash(labels)},
        "query_labels_supplied": False,
    }
    study.write_json(root / "results" / f"{job['id']}.json", record)
    return job["id"]


def _run_score(root: Path, *args):
    return subprocess.run([PYTHON, str(ROOT / "score.py"), "--root", str(root), *args],
                          cwd=str(ROOT), capture_output=True, text=True)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        _ensure_pool()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        # score.py refuses to freeze a final run before the selection rule exists.
        study.write_json(self.root / "selection.json",
                         {"candidates": [{"id": "synth", "config": {"family": "mlp"}}]})

    def _tied_logits(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(10000, 10)).astype(np.float32)
        # Rows 0..9 have an exact tie between classes 3 and 7 at the maximum.
        logits[:10, :] = -5.0
        logits[:10, 3] = 9.0
        logits[:10, 7] = 9.0
        return logits

    def test_empty_results_directory_is_handled(self):
        result = _run_score(self.root, "--stage", "dev")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No stage 'dev' results", result.stdout)
        scores = json.loads((self.root / "results" / "scores_dev.json").read_text())
        self.assertEqual(scores["job_count"], 0)
        self.assertEqual(scores["jobs"], [])

    def test_lowest_index_tie_break_accepted_and_enforced(self):
        logits = self._tied_logits()
        labels = np.argmax(logits, axis=1).astype(np.uint8)
        self.assertTrue(np.all(labels[:10] == 3))
        job_id = _write_synthetic_job(self.root, "dev", logits, labels)
        ok = _run_score(self.root, "--stage", "dev")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        scores = json.loads((self.root / "results" / "scores_dev.json").read_text())
        self.assertEqual(scores["job_count"], 1)
        row = scores["jobs"][0]
        self.assertEqual(row["id"], job_id)
        self.assertEqual(row["total"], 10000)
        with _without_learner_modules():
            truth = study.query_labels(study.DEV_SEEDS[0])
        self.assertEqual(row["correct"], int(np.count_nonzero(labels == truth)))
        # Highest-index tie-break must be rejected, with hashes kept consistent.
        bad = labels.copy()
        bad[:10] = 7
        _write_synthetic_job(self.root, "dev", logits, bad)
        failed = _run_score(self.root, "--stage", "dev")
        self.assertEqual(failed.returncode, 2)
        self.assertIn("argmax", failed.stderr)

    def test_tampered_prediction_file_is_rejected(self):
        logits = self._tied_logits()
        labels = np.argmax(logits, axis=1).astype(np.uint8)
        job_id = _write_synthetic_job(self.root, "dev", logits, labels)
        path = self.root / "predictions" / f"{job_id}.npz"
        logits[0, 0] = 123.0
        with path.open("wb") as stream:
            np.savez(stream, logits=logits, labels=labels)
        failed = _run_score(self.root, "--stage", "dev")
        self.assertEqual(failed.returncode, 2)
        self.assertIn("hash", failed.stderr)

    def test_corrupted_input_hash_is_rejected(self):
        logits = self._tied_logits()
        labels = np.argmax(logits, axis=1).astype(np.uint8)
        job_id = _write_synthetic_job(self.root, "dev", logits, labels)
        result_path = self.root / "results" / f"{job_id}.json"
        record = json.loads(result_path.read_text())
        record["job"]["input_sha256"]["train_x"] = "0" * 64
        study.write_json(result_path, record)
        failed = _run_score(self.root, "--stage", "dev")
        self.assertEqual(failed.returncode, 2)
        self.assertIn("input_sha256", failed.stderr)

    def test_final_stage_refuses_without_a_freeze(self):
        logits = self._tied_logits()
        labels = np.argmax(logits, axis=1).astype(np.uint8)
        job_id = _write_synthetic_job(self.root, "final", logits, labels,
                                      seed=study.FINAL_SEEDS[0])
        failed = _run_score(self.root, "--stage", "final")
        self.assertEqual(failed.returncode, 2)
        self.assertIn("freeze", failed.stderr)
        self.assertFalse((self.root / "results" / "scores_final.json").exists())
        # Freeze, then scoring succeeds.
        study.write_json(self.root / "plans" / "final.json", {"jobs": [{"id": job_id}]})
        frozen = _run_score(self.root, "--freeze-final", "--plan", "plans/final.json")
        self.assertEqual(frozen.returncode, 0, frozen.stderr)
        ok = _run_score(self.root, "--stage", "final")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        scores = json.loads((self.root / "results" / "scores_final.json").read_text())
        self.assertEqual(scores["job_count"], 1)
        self.assertTrue(scores["jobs"][0]["planned"])

    def test_final_stage_refuses_missing_and_altered_jobs(self):
        logits = self._tied_logits()
        labels = np.argmax(logits, axis=1).astype(np.uint8)
        job_id = _write_synthetic_job(self.root, "final", logits, labels,
                                      seed=study.FINAL_SEEDS[0])
        absent = "final-synth-s2026092002-n1000"
        study.write_json(self.root / "plans" / "final.json", {"jobs": [job_id, absent]})
        missing = _run_score(self.root, "--freeze-final", "--plan", "plans/final.json")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("missing result", missing.stderr)
        # Freeze only the present job, then alter its results file.
        study.write_json(self.root / "plans" / "final.json", {"jobs": [job_id]})
        self.assertEqual(_run_score(self.root, "--freeze-final",
                                    "--plan", "plans/final.json").returncode, 0)
        result_path = self.root / "results" / f"{job_id}.json"
        record = json.loads(result_path.read_text())
        record["metrics"]["epochs_completed"] = 99
        study.write_json(result_path, record)
        altered = _run_score(self.root, "--stage", "final")
        self.assertEqual(altered.returncode, 2)
        self.assertIn("after the freeze", altered.stderr)

    def test_aggregate_statistics(self):
        logits = self._tied_logits()
        labels = np.argmax(logits, axis=1).astype(np.uint8)
        for seed in study.DEV_SEEDS[:2]:
            _write_synthetic_job(self.root, "dev", logits, labels, seed=seed)
        ok = _run_score(self.root, "--stage", "dev")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        scores = json.loads((self.root / "results" / "scores_dev.json").read_text())
        self.assertEqual(scores["job_count"], 2)
        aggregate = scores["aggregate"]
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["count"], 2)
        self.assertEqual(aggregate[0]["candidate_id"], "synth")
        self.assertIsNotNone(aggregate[0]["sd_error_pct"])
        errors = [row["error_pct"] for row in scores["jobs"]]
        self.assertAlmostEqual(aggregate[0]["mean_error_pct"], float(np.mean(errors)), places=9)
        self.assertAlmostEqual(aggregate[0]["sd_error_pct"],
                               float(np.std(errors, ddof=1)), places=9)


    def test_selection_rule_is_required_before_freezing(self):
        logits = self._tied_logits()
        labels = np.argmax(logits, axis=1).astype(np.uint8)
        job_id = _write_synthetic_job(self.root, "final", logits, labels,
                                      seed=study.FINAL_SEEDS[0])
        study.write_json(self.root / "plans" / "final.json", {"jobs": [job_id]})
        (self.root / "selection.json").unlink()
        failed = _run_score(self.root, "--freeze-final", "--plan", "plans/final.json")
        self.assertEqual(failed.returncode, 2)
        self.assertIn("selection", failed.stderr)
        self.assertFalse((self.root / "predictions" / "final_freeze.json").exists())

    def test_refreeze_requires_a_flag_and_is_blocked_after_scoring(self):
        logits = self._tied_logits()
        labels = np.argmax(logits, axis=1).astype(np.uint8)
        first_id = _write_synthetic_job(self.root, "final", logits, labels,
                                        candidate_id="cand-a", seed=study.FINAL_SEEDS[0])
        second_id = _write_synthetic_job(self.root, "final", logits[::-1].copy(),
                                         np.argmax(logits[::-1], axis=1).astype(np.uint8),
                                         candidate_id="cand-b", seed=study.FINAL_SEEDS[0])
        freeze_path = self.root / "predictions" / "final_freeze.json"

        study.write_json(self.root / "plans" / "final.json", {"jobs": [first_id, second_id]})
        self.assertEqual(_run_score(self.root, "--freeze-final",
                                    "--plan", "plans/final.json").returncode, 0)
        # A second freeze of a different plan needs --refreeze ...
        study.write_json(self.root / "plans" / "other.json", {"jobs": [second_id]})
        blocked = _run_score(self.root, "--freeze-final", "--plan", "plans/other.json")
        self.assertEqual(blocked.returncode, 2)
        self.assertIn("--refreeze", blocked.stderr)
        self.assertEqual(json.loads(freeze_path.read_text())["job_count"], 2)
        # ... and with --refreeze the superseded freeze is archived.
        refrozen = _run_score(self.root, "--freeze-final", "--plan", "plans/other.json",
                              "--refreeze")
        self.assertEqual(refrozen.returncode, 0, refrozen.stderr)
        freeze = json.loads(freeze_path.read_text())
        self.assertEqual(freeze["job_count"], 1)
        self.assertIsNotNone(freeze["supersedes"])
        archived = self.root / freeze["supersedes"]["archived_path"]
        self.assertTrue(archived.exists())
        # Once the freeze has been scored no re-freeze is possible at all.
        scored = _run_score(self.root, "--stage", "final", "--allow-unplanned")
        self.assertEqual(scored.returncode, 0, scored.stderr)
        self.assertTrue(json.loads(freeze_path.read_text())["scored"])
        study.write_json(self.root / "plans" / "third.json", {"jobs": [first_id]})
        after = _run_score(self.root, "--freeze-final", "--plan", "plans/third.json",
                           "--refreeze")
        self.assertEqual(after.returncode, 2)
        self.assertIn("already been scored", after.stderr)

    def test_job_id_filter_neither_overwrites_nor_skips_the_final_table(self):
        logits = self._tied_logits()
        labels = np.argmax(logits, axis=1).astype(np.uint8)
        other = logits[::-1].copy()
        first_id = _write_synthetic_job(self.root, "final", logits, labels,
                                        candidate_id="cand-a", seed=study.FINAL_SEEDS[0])
        second_id = _write_synthetic_job(self.root, "final", other,
                                         np.argmax(other, axis=1).astype(np.uint8),
                                         candidate_id="cand-b", seed=study.FINAL_SEEDS[0])
        study.write_json(self.root / "plans" / "final.json", {"jobs": [first_id, second_id]})
        self.assertEqual(_run_score(self.root, "--freeze-final",
                                    "--plan", "plans/final.json").returncode, 0)
        full = _run_score(self.root, "--stage", "final")
        self.assertEqual(full.returncode, 0, full.stderr)
        canonical = self.root / "results" / "scores_final.json"
        before = canonical.read_text()
        self.assertEqual(json.loads(before)["job_count"], 2)

        filtered = _run_score(self.root, "--stage", "final", "--job-id", first_id)
        self.assertEqual(filtered.returncode, 0, filtered.stderr)
        self.assertEqual(canonical.read_text(), before)
        partial = json.loads((self.root / "results" / "scores_final_filtered.json").read_text())
        self.assertEqual(partial["job_count"], 1)
        self.assertEqual(partial["job_filter"], [first_id])

        # A filtered pass still requires every planned job to have a result.
        (self.root / "results" / f"{second_id}.json").unlink()
        incomplete = _run_score(self.root, "--stage", "final", "--job-id", first_id)
        self.assertEqual(incomplete.returncode, 2)
        self.assertIn("have no", incomplete.stderr)

    def test_dev_stage_result_carrying_a_final_seed_is_refused(self):
        logits = self._tied_logits()
        labels = np.argmax(logits, axis=1).astype(np.uint8)
        job_id = _write_synthetic_job(self.root, "final", logits, labels,
                                      candidate_id="cheat", seed=study.FINAL_SEEDS[0])
        record = json.loads((self.root / "results" / f"{job_id}.json").read_text())
        cheat_id = f"dev-cheat-s{study.FINAL_SEEDS[0]}-n1000"
        record["job"]["stage"] = "dev"
        record["job"]["id"] = cheat_id
        study.write_json(self.root / "results" / f"{cheat_id}.json", record)
        failed = _run_score(self.root, "--stage", "dev")
        self.assertEqual(failed.returncode, 2)
        self.assertIn("seeds", failed.stderr)
        self.assertFalse((self.root / "results" / "scores_dev.json").exists())

    def test_result_without_a_candidate_id_is_refused_cleanly(self):
        logits = self._tied_logits()
        labels = np.argmax(logits, axis=1).astype(np.uint8)
        job_id = _write_synthetic_job(self.root, "dev", logits, labels)
        path = self.root / "results" / f"{job_id}.json"
        record = json.loads(path.read_text())
        del record["job"]["candidate_id"]
        study.write_json(path, record)
        failed = _run_score(self.root, "--stage", "dev")
        self.assertEqual(failed.returncode, 2)
        self.assertIn("candidate_id", failed.stderr)
        self.assertNotIn("Traceback", failed.stderr)


class ProtocolDraftTests(unittest.TestCase):
    def test_draft_exists_and_is_consistent(self):
        draft = json.loads((ROOT / "protocol.draft.json").read_text())
        self.assertFalse(draft["test_labels_created"])
        self.assertEqual(draft["draws"]["dev_seeds"], study.DEV_SEEDS)
        self.assertEqual(draft["draws"]["final_seeds"], study.FINAL_SEEDS)
        self.assertEqual([entry["n"] for entry in draft["levels"]], study.LEVELS)
        self.assertEqual(draft["features"]["permutation_seed"], 20260923)
        self.assertEqual(draft["compute"]["per_fit_time_budget_seconds"], 1200)
        self.assertIn("to be frozen", draft["selection_rule"])
        self.assertEqual(draft["learner_contract"]["input_allowlist"],
                         ["train_x", "train_y", "query_x"])
        manifest = json.loads(study.DATA_MANIFEST_PATH.read_text())
        self.assertEqual(draft["data_manifest"]["pool_images_sha256"],
                         manifest["arrays"]["pool_images"]["sha256"])


if __name__ == "__main__":
    unittest.main()
