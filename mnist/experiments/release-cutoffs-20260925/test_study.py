"""Tests for the release-ladder data/protocol/runner/scoring layer.

    /tmp/penv/bin/python -m unittest test_study -v

The suite never imports ``kernels``/``neural``/``runner`` into its own process:
``study.query_labels`` refuses to run when a learner or the runner is loaded, and
that guard is itself under test.  Everything that needs the runner (its
``--validate-only`` preflight, its independent re-implementation of the draw)
runs in a subprocess.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
GPUMODE = Path("/Users/yaroslavvb/git/sutro-problems/gpumode")
PYTHON = sys.executable

import study  # noqa: E402
import release  # noqa: E402
import plan as plan_module  # noqa: E402


class hidden_learner_modules:
    """Read query labels as if this process had never imported a learner.

    ``study.query_labels`` refuses while 'kernels'/'neural'/'runner' are in
    ``sys.modules``, and this file is written to run in its own process (module
    docstring).  A shared collector -- ``pytest test_study.py test_kernels.py
    test_neural.py`` in ONE process -- imports test_kernels.py, which imports
    ``kernels``, so the three tests that must actually read labels would fail
    against a guard that is behaving exactly as designed.  Popping those entries
    for the duration of the read restores the precondition; it does not weaken
    the guard, which is still exercised directly (and with the real
    ``sys.modules`` contents) by ``test_guard_refuses_when_a_learner_is_loaded``
    and ``test_guard_refuses_a_forbidden_entrypoint``.
    """

    def __enter__(self):
        self._saved = {name: sys.modules.pop(name)
                       for name in study.FORBIDDEN_MODULES if name in sys.modules}
        return self

    def __exit__(self, *exception):
        sys.modules.update(self._saved)
        return False


def harness():
    """The real gpumode harness module (imported only inside the tests)."""
    if str(GPUMODE) not in sys.path:
        sys.path.insert(0, str(GPUMODE))
    import eval as harness_module
    return harness_module


class ReleaseIdentity(unittest.TestCase):
    """release.py must be bit-identical to the harness it copies."""

    def test_salts_and_version(self):
        source = harness()
        for name in ("LABEL_SALT", "UNIVERSE_SALT", "DRAW_SALT", "RELEASE_SALT"):
            self.assertEqual(getattr(release, name), getattr(source, name), name)
        self.assertEqual(release.HARNESS_VERSION_STRING, source.HARNESS_VERSION)
        digest = study.sha(GPUMODE / "eval.py")
        self.assertEqual(release.HARNESS_EVAL_SHA256, digest,
                         "eval.py changed; regenerate release.py and re-verify")

    def test_random_data_bit_identical(self):
        source = harness()
        rng = np.random.default_rng(0)
        for seed in (1, 2026092491, 7919):
            self.assertTrue(np.array_equal(
                release.haar_rotation(60, seed), source.haar_rotation(60, seed)))
            for count in (60000, 1234):
                mine = release.split_universes(count, seed, release.UNIVERSE_SALT)
                theirs = source.split_universes(count, seed, source.UNIVERSE_SALT)
                self.assertTrue(np.array_equal(mine[0], theirs[0]))
                self.assertTrue(np.array_equal(mine[1], theirs[1]))
        images = rng.random((400, 1, 9, 9), dtype=np.float64).astype(np.float32)
        for seed in (3, 2026092401):
            mine = release.release_map(images, seed, 60)
            theirs = source.release_map(images, seed, 60)
            for a, b in zip(mine, theirs):
                self.assertTrue(np.array_equal(a, b))
            transform = mine[2] @ mine[1]
            self.assertTrue(np.array_equal(
                release.apply_release(images, mine[0], transform),
                source.apply_release(images, theirs[0], theirs[2] @ theirs[1])))

    def test_real_draw_bit_identical(self):
        source = harness()
        pixels = study.pool_pixels()
        seed, n = study.DEV_SEEDS[0], 2236
        rows = study.train_rows(seed, n)
        mine = release.release_map(pixels[rows], seed, study.RELEASE_DIMS)
        theirs = source.release_map(pixels[rows], seed, study.RELEASE_DIMS)
        for a, b in zip(mine, theirs):
            self.assertTrue(np.array_equal(a, b))
        arrays = study.job_arrays(seed, n)
        expected = source.apply_release(pixels[rows], theirs[0], theirs[2] @ theirs[1])
        self.assertTrue(np.array_equal(arrays["train_z"], expected))
        self.assertEqual(arrays["train_z"].dtype, np.float32)


class DrawProtocol(unittest.TestCase):
    def test_nested_prefixes(self):
        for seed in (study.DEV_SEEDS[0], study.FINAL_SEEDS[0]):
            order, _ = study.draw_rows(seed)
            self.assertEqual(order.shape, (study.UNIVERSE_HALF,))
            self.assertEqual(len(set(order.tolist())), study.UNIVERSE_HALF)
            previous = None
            for n in study.LEVELS:
                rows = study.train_rows(seed, n)
                self.assertEqual(rows.shape, (n,))
                if previous is not None:
                    self.assertTrue(np.array_equal(rows[:previous.shape[0]], previous))
                previous = rows

    def test_query_fixed_and_disjoint(self):
        for seed in (study.DEV_SEEDS[0], study.DEV_SEEDS[1]):
            query = study.query_rows(seed)
            self.assertEqual(query.shape, (study.QUERY_COUNT,))
            self.assertEqual(len(set(query.tolist())), study.QUERY_COUNT)
            for n in study.LEVELS:
                self.assertTrue(np.array_equal(query, study.query_rows(seed)))
                self.assertEqual(np.intersect1d(study.train_rows(seed, n), query).size, 0)
            # and the query half is the harness's test universe
            _, test_universe = release.split_universes(
                study.POOL_COUNT, seed, release.UNIVERSE_SALT)
            self.assertTrue(set(query.tolist()) <= set(test_universe.tolist()))

    def test_different_seeds_differ(self):
        a, b = study.DEV_SEEDS
        self.assertFalse(np.array_equal(study.query_rows(a), study.query_rows(b)))
        self.assertFalse(np.array_equal(study.train_rows(a, 500), study.train_rows(b, 500)))

    def test_levels_are_the_geometric_ladder(self):
        # release-cutoffs-20260925: four levels per decade from 100 to 10,000
        expected = [round(100 * 10 ** (i / 4)) for i in range(9)]
        self.assertEqual(study.LEVELS, expected)
        self.assertEqual(study.LEVELS, [100, 178, 316, 562, 1000, 1778, 3162, 5623, 10000])
        self.assertAlmostEqual(study.LEVELS_EXACT[0], 100.0)
        self.assertAlmostEqual(study.LEVELS_EXACT[-1], 10000.0)


class ReleasedFeatures(unittest.TestCase):
    def test_train_rows_are_white(self):
        seed = study.DEV_SEEDS[0]
        for n in (500, 10000):
            arrays = study.job_arrays(seed, n)
            z = arrays["train_z"].astype(np.float64)
            self.assertEqual(z.shape, (n, study.RELEASE_DIMS))
            self.assertLess(float(np.abs(z.mean(0)).max()), 1e-4)
            covariance = np.cov(z, rowvar=False)   # ddof=1, as in release_map
            error = float(np.abs(covariance - np.eye(study.RELEASE_DIMS)).max())
            self.assertLess(error, 1e-5, f"n={n}: covariance is not the identity ({error})")

    def test_same_rotation_different_whitener(self):
        seed = study.DEV_SEEDS[0]
        mu_a, w_a, q_a, _ = study.release_transform(seed, 500)
        mu_b, w_b, q_b, _ = study.release_transform(seed, 10000)
        self.assertTrue(np.array_equal(q_a, q_b), "Q must depend only on the seed")
        self.assertTrue(np.array_equal(q_a, release.haar_rotation(study.RELEASE_DIMS, seed)))
        self.assertFalse(np.array_equal(w_a, w_b), "W must be refitted per level")
        self.assertFalse(np.array_equal(mu_a, mu_b), "mu must be refitted per level")
        other = study.release_transform(study.DEV_SEEDS[1], 500)[2]
        self.assertFalse(np.array_equal(q_a, other), "Q must differ between seeds")
        self.assertTrue(np.allclose(q_a @ q_a.T, np.eye(study.RELEASE_DIMS), atol=1e-10))

    def test_job_arrays_never_returns_query_labels(self):
        seed, n = study.DEV_SEEDS[1], 1057
        arrays = study.job_arrays(seed, n)
        self.assertEqual(set(arrays), {"train_z", "train_y", "query_z"})
        self.assertEqual(arrays["train_y"].shape, (n,))
        self.assertEqual(arrays["train_y"].dtype, np.uint8)
        self.assertEqual(arrays["query_z"].shape, (study.QUERY_COUNT, study.RELEASE_DIMS))
        with hidden_learner_modules():
            truth = study.query_labels(seed)
        for name, value in arrays.items():
            self.assertFalse(value.shape == truth.shape and value.dtype == truth.dtype
                             and np.array_equal(value, truth), name)
        pool_labels = np.load(study.POOL_LABELS_PATH)
        expected = pool_labels[study.train_rows(seed, n)].astype(np.uint8)
        self.assertTrue(np.array_equal(arrays["train_y"], expected))

    def test_query_features_depend_on_the_level(self):
        # Same query ROWS at every level, but mu and W are refitted, so the
        # released coordinates differ: a learner cannot reuse another level's z.
        seed = study.DEV_SEEDS[0]
        a = study.job_arrays(seed, 500)["query_z"]
        b = study.job_arrays(seed, 10000)["query_z"]
        self.assertFalse(np.array_equal(a, b))


class Jobs(unittest.TestCase):
    def test_stage_seed_binding(self):
        config = {"family": "krr"}
        job = study.make_job("dev", study.DEV_SEEDS[0], 500, "cand-a", config)
        self.assertEqual(job["id"], f"dev-cand-a-s{study.DEV_SEEDS[0]}-n500")
        self.assertEqual(job["device_kind"], "gpu")
        self.assertEqual(set(job["input_sha256"]), {"train_z", "train_y", "query_z"})
        with self.assertRaises(ValueError):
            study.make_job("dev", study.FINAL_SEEDS[0], 500, "cand-a", config)
        with self.assertRaises(ValueError):
            study.make_job("final", study.DEV_SEEDS[0], 500, "cand-a", config)
        with self.assertRaises(ValueError):
            study.make_job("probe", study.FINAL_SEEDS[3], 500, "cand-a", config)
        with self.assertRaises(ValueError):
            study.make_job("Dev", study.DEV_SEEDS[0], 500, "cand-a", config)
        with self.assertRaises(ValueError):
            study.make_job("dev", study.DEV_SEEDS[0], 500, "cand a", config)
        with self.assertRaises(ValueError):
            study.make_job("dev", study.DEV_SEEDS[0], 500, "cand-a", config,
                           device_kind="tpu")
        with self.assertRaises(ValueError):
            study.make_job("dev", study.DEV_SEEDS[0], 500, "cand-a", config,
                           time_budget_seconds=1201)
        # a probe stage on a dev seed is allowed
        study.make_job("probe", study.DEV_SEEDS[0], 500, "cand-a", config)

    def test_job_hashes_match_the_arrays(self):
        seed, n = study.DEV_SEEDS[0], 1057
        job = study.make_job("dev", seed, n, "cand-b", {"family": "krr"},
                             device_kind="cpu", extra={"estimated_seconds": 12.0})
        arrays = study.job_arrays(seed, n)
        for name, value in arrays.items():
            self.assertEqual(job["input_sha256"][name], study.ahash(value))
        self.assertEqual(job["train_rows_sha256"], study.ahash(study.train_rows(seed, n)))
        self.assertEqual(job["query_rows_sha256"], study.ahash(study.query_rows(seed)))
        self.assertEqual(job["estimated_seconds"], 12.0)
        with self.assertRaises(ValueError):
            study.make_job("dev", seed, n, "cand-b", {"family": "krr"},
                           extra={"seed": 1})

    def test_level_bounds(self):
        with self.assertRaises(ValueError):
            study.job_arrays(study.DEV_SEEDS[0], 60)      # <= release_dims
        with self.assertRaises(ValueError):
            study.job_arrays(study.DEV_SEEDS[0], 30001)   # beyond the train universe

    def test_plan_estimates(self):
        entry = {"id": "x", "config": {"family": "krr"},
                 "estimated_seconds": {"500": 20, "10000": 300}}
        self.assertAlmostEqual(plan_module.estimate_seconds(entry, 500, 1200), 20.0)
        self.assertAlmostEqual(plan_module.estimate_seconds(entry, 10000, 1200), 300.0)
        middle = plan_module.estimate_seconds(entry, 2236, 1200)
        self.assertTrue(20.0 < middle < 300.0)
        self.assertAlmostEqual(plan_module.estimate_seconds(entry, 10000, 100), 100.0)
        flat = {"id": "y", "config": {"family": "krr"}, "estimated_seconds": 55}
        self.assertAlmostEqual(plan_module.estimate_seconds(flat, 4729, 1200), 55.0)


class QueryLabelGuard(unittest.TestCase):
    def test_labels_are_correct_and_scoring_only(self):
        seed = study.DEV_SEEDS[0]
        with hidden_learner_modules():
            labels = study.query_labels(seed)
        self.assertEqual(labels.shape, (study.QUERY_COUNT,))
        self.assertEqual(labels.dtype, np.uint8)
        pool_labels = np.load(study.POOL_LABELS_PATH)
        self.assertTrue(np.array_equal(
            labels, pool_labels[study.query_rows(seed)].astype(np.uint8)))

    def test_guard_refuses_when_a_learner_is_loaded(self):
        import types
        with hidden_learner_modules():
            for name in ("kernels", "neural", "runner"):
                self.assertNotIn(name, sys.modules,
                                 f"{name} must not be visible to a label read")
            sys.modules["kernels"] = types.ModuleType("kernels")
            try:
                with self.assertRaises(RuntimeError):
                    study.query_labels(study.DEV_SEEDS[0])
            finally:
                del sys.modules["kernels"]
            study.query_labels(study.DEV_SEEDS[0])      # guard lifts again

    def test_guard_refuses_a_forbidden_entrypoint(self):
        """`python runner.py` is __main__, not 'runner' in sys.modules."""
        self.assertIn("runner.py", study.FORBIDDEN_ENTRYPOINTS)
        body = ("import sys\n"
                f"sys.path.insert(0, {str(ROOT)!r})\n"
                "import study\n"
                "try:\n"
                "    study.query_labels(study.DEV_SEEDS[0])\n"
                "except RuntimeError as error:\n"
                "    print('REFUSED', error)\n"
                "else:\n"
                "    print('LEAKED')\n")
        with tempfile.TemporaryDirectory() as directory:
            # The file NAME is what the guard keys on.
            script = Path(directory) / "runner.py"
            script.write_text(body)
            completed = subprocess.run(
                [PYTHON, str(script)], capture_output=True, text=True, timeout=300,
                env={**os.environ, "MNIST_POOL_CACHE": "/tmp/gpumode-pool"})
            self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
            self.assertIn("REFUSED", completed.stdout, completed.stdout)
            # ... while an ordinary scoring entry point is still allowed.
            allowed = Path(directory) / "score.py"
            allowed.write_text(body)
            completed = subprocess.run(
                [PYTHON, str(allowed)], capture_output=True, text=True, timeout=300,
                env={**os.environ, "MNIST_POOL_CACHE": "/tmp/gpumode-pool"})
            self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
            self.assertIn("LEAKED", completed.stdout, completed.stdout)

    def test_score_module_is_the_only_label_reader(self):
        # Only score.py may call query_labels() or touch pool_labels.npy.  The
        # runner is allowed to WRITE the receipt keys 'query_labels_opened' /
        # 'query_labels_supplied', which is why this matches the call form.
        for name in ("runner.py", "plan.py", "release.py"):
            text = (ROOT / name).read_text()
            self.assertNotIn("study.query_labels", text,
                             f"{name} must not call study.query_labels()")
            self.assertNotIn("POOL_LABELS", text, f"{name} must not load pool_labels")
            self.assertNotIn("pool_labels.npy'", text, f"{name} must not load pool_labels")
        for name in ("kernels.py", "neural.py"):
            text = (ROOT / name).read_text()
            for forbidden in ("study.query_labels(", "POOL_LABELS", "pool_labels",
                              "pool_pixels", "split_universes", "release_map("):
                self.assertNotIn(forbidden, text,
                                 f"{name} must not reach for {forbidden}")
        scorer = (ROOT / "score.py").read_text()
        self.assertIn("study.query_labels(", scorer)
        for forbidden in ("import runner", "import kernels", "import neural"):
            self.assertNotIn(forbidden, scorer, "score.py must stay learner-free")


class Manifest(unittest.TestCase):
    def test_manifest_verifies(self):
        manifest = study.prepare()
        self.assertEqual(manifest["arrays"]["pool_pixels"]["shape"],
                         [study.POOL_COUNT, study.PIXEL_COUNT])
        self.assertEqual(manifest["arrays"]["pool_labels"]["dtype"], "int64")
        self.assertFalse(manifest["query_labels_materialised"])
        self.assertEqual(manifest["protocol"]["release_dims"], 60)
        self.assertEqual(manifest["release_sha256"], study.sha(ROOT / "release.py"))

    def test_protocol_draft_is_complete(self):
        draft = json.loads((ROOT / "protocol.draft.json").read_text())
        self.assertEqual(draft["levels"], study.levels())
        self.assertEqual(draft["draws"]["final_seeds"], study.FINAL_SEEDS)
        # release-cutoffs-20260925 selects nothing: the recipe is fixed in advance, so the
        # frozen protocol replaces the candidate-selection rule with these four keys, and adds
        # the MLP timing rule, frozen before any fit ran.
        self.assertEqual(draft["status"], "frozen")
        self.assertEqual(set(draft["selection_rule"]), {"recipe", "cutoff", "reuse", "budget_fallback"})
        self.assertEqual(set(draft["mlp_timing_rule"]),
                         {"family", "sweep", "choice", "confirmation", "not_achievable"})
        self.assertEqual(draft["release"]["harness_eval_sha256"], release.HARNESS_EVAL_SHA256)


class RunnerPreflight(unittest.TestCase):
    """runner.py runs in a SUBPROCESS: importing it here would lock query_labels."""

    plan_path = ROOT / "plans" / "test-smoke.json"

    def setUp(self):
        self.ledger = ROOT / "budget-ledger.json"
        self.ledger_existed = self.ledger.exists()
        jobs = [study.make_job("smoke", study.DEV_SEEDS[0], 500, "test-krr",
                               {"family": "krr"}, device_kind="cpu",
                               time_budget_seconds=120,
                               extra={"estimated_seconds": 30.0}),
                study.make_job("smoke", study.DEV_SEEDS[0], 500, "test-mlp",
                               {"family": "mlp"}, device_kind="gpu",
                               time_budget_seconds=120,
                               extra={"estimated_seconds": 90.0})]
        study.write_json(self.plan_path, jobs)

    def tearDown(self):
        self.plan_path.unlink(missing_ok=True)
        (ROOT / "logs" / "test-smoke-preflight.json").unlink(missing_ok=True)

    def test_validate_only_creates_no_app_and_no_ledger(self):
        completed = subprocess.run(
            [PYTHON, str(ROOT / "runner.py"), "--plan", "plans/test-smoke.json",
             "--validate-only", "--gpus", "2"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=600,
            env={**os.environ, "MNIST_POOL_CACHE": "/tmp/gpumode-pool"})
        self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
        receipt = json.loads((ROOT / "logs" / "test-smoke-preflight.json").read_text())
        self.assertFalse(receipt["paid_compute_started"])
        self.assertFalse(receipt["new_reservation_created"])
        self.assertFalse(receipt["query_labels_opened"])
        self.assertEqual(receipt["max_concurrent_gpus"], 2)
        self.assertEqual(receipt["gpu"], "A100-40GB")
        self.assertEqual(receipt["jobs"], 2)
        # LPT over two workers with estimates 90 and 30 -> makespan 90 s.
        self.assertAlmostEqual(receipt["estimated_lpt_makespan_seconds"], 90.0)
        self.assertGreater(receipt["reservation_upper_usd"], 0.0)
        self.assertLess(receipt["reservation_upper_usd"],
                        receipt["available_worker_usd"])
        self.assertEqual(self.ledger.exists(), self.ledger_existed,
                         "--validate-only must not create the ledger")
        self.assertNotIn("ap-", completed.stdout)

    def test_runner_draw_matches_study(self):
        """runner.rebuild_arrays (the controller-side cross-check) == study.job_arrays."""
        script = (
            "import json,sys,numpy as np;"
            "sys.path.insert(0,'.');"
            "import runner, study;"
            "pool=np.load(study.POOL_PIXELS_PATH);"
            "out={};"
            "\nfor seed in study.DEV_SEEDS[:1] + study.FINAL_SEEDS[:1]:\n"
            "    for n in (500, 2236, 10000):\n"
            "        want = study.job_arrays(seed, n)\n"
            "        got, tr, qr = runner.rebuild_arrays(pool, seed, n, want['train_y'])\n"
            "        out[f'{seed}-{n}'] = {k: bool(np.array_equal(got[k], want[k])) for k in want}\n"
            "        out[f'{seed}-{n}']['rows'] = bool(np.array_equal(tr, study.train_rows(seed,n))"
            " and np.array_equal(qr, study.query_rows(seed)))\n"
            "print(json.dumps(out))")
        completed = subprocess.run(
            [PYTHON, "-c", script], capture_output=True, text=True, cwd=str(ROOT),
            timeout=600, env={**os.environ, "RELEASE_LADDER_NO_MODAL": "1",
                              "MNIST_POOL_CACHE": "/tmp/gpumode-pool"})
        self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(len(report), 6)
        for key, checks in report.items():
            self.assertTrue(all(checks.values()), f"{key}: {checks}")

    def test_container_gets_arrays_not_pixels(self):
        """The image must not carry the pixel pool and the worker must not refit.

        Recomputing the release in the container would (a) fail, because
        release_map is an eigendecomposition and LAPACK is not bit-reproducible
        across BLAS builds, and (b) let a learner invert z back to pixels.
        """
        text = (ROOT / "runner.py").read_text()
        self.assertNotIn("REMOTE_POOL_PIXELS", text)
        self.assertNotIn("add_local_file(str(POOL_PIXELS)", text)
        body = text.split("def _remote_body(", 1)[1].split("\n    def fit_a100(", 1)[0]
        for forbidden in ("rebuild_arrays", "release_map", "POOL_PIXELS", "np.load("):
            self.assertNotIn(forbidden, body,
                             f"the container body must not use {forbidden}")
        self.assertIn("unpack_arrays(payload)", body)

    def test_payload_round_trip_is_exact(self):
        script = (
            "import json,sys,numpy as np;"
            "sys.path.insert(0,'.');"
            "import runner, study;"
            "job=study.make_job('dev', study.DEV_SEEDS[0], 500, 'c', {'family':'krr'},"
            " extra={'estimated_seconds':1.0});"
            "payload,arrays=runner.job_payload(job, study);"
            "back=runner.unpack_arrays(payload);"
            "print(json.dumps({"
            "'exact': all(bool(np.array_equal(back[k],arrays[k])) and "
            "str(back[k].dtype)==str(arrays[k].dtype) for k in arrays),"
            "'hashes': all(runner.ahash(back[k])==job['input_sha256'][k] for k in back),"
            "'names': sorted(payload)}))")
        completed = subprocess.run(
            [PYTHON, "-c", script], capture_output=True, text=True, cwd=str(ROOT),
            timeout=600, env={**os.environ, "RELEASE_LADDER_NO_MODAL": "1",
                              "MNIST_POOL_CACHE": "/tmp/gpumode-pool"})
        self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertTrue(report["exact"])
        self.assertTrue(report["hashes"])
        self.assertEqual(report["names"], ["query_z", "train_y", "train_z"])

    def test_over_budget_fit_is_refused(self):
        """run_fit must not return a result that used more than its budget."""
        script = (
            "import sys,time,types,numpy as np;"
            "sys.path.insert(0,'.');"
            "import runner, study;"
            "job=study.make_job('dev', study.DEV_SEEDS[0], 500, 'c', {'family':'krr'},"
            " time_budget_seconds=1, extra={'estimated_seconds':1.0});"
            "_,arrays=runner.job_payload(job, study);"
            "slow=types.ModuleType('kernels');"
            "slow.fit_predict=lambda train_z, train_y, query_z, config, seed, device,"
            " deadline_unix: (time.sleep(7.0), {'logits': np.zeros((10000,10),"
            " dtype=np.float32), 'labels': np.zeros(10000, dtype=np.uint8),"
            " 'metrics': {}})[1];"
            "sys.modules['kernels']=slow;"
            "\ntry:\n"
            "    runner.run_fit(arrays, job, {}, time.time()+600, 'cpu')\n"
            "except TimeoutError as error:\n"
            "    print('REFUSED', error)\n"
            "else:\n"
            "    print('ACCEPTED')\n")
        completed = subprocess.run(
            [PYTHON, "-c", script], capture_output=True, text=True, cwd=str(ROOT),
            timeout=600, env={**os.environ, "RELEASE_LADDER_NO_MODAL": "1",
                              "MNIST_POOL_CACHE": "/tmp/gpumode-pool"})
        self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
        self.assertIn("REFUSED", completed.stdout, completed.stdout)

    def test_shutdown_grace_covers_the_verification_loop(self):
        """budget.py charges wall clock up to the verified stop, so the grace must
        cover the controller's own shutdown polling (2 + 7*15 = 107 s of sleeps
        plus up to nine `modal app` subprocesses at 60 s timeout each)."""
        script = ("import sys;sys.path.insert(0,'.');import runner;"
                  "print(runner.SHUTDOWN_GRACE_SECONDS)")
        completed = subprocess.run(
            [PYTHON, "-c", script], capture_output=True, text=True, cwd=str(ROOT),
            timeout=300, env={**os.environ, "RELEASE_LADDER_NO_MODAL": "1"})
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        self.assertGreaterEqual(int(completed.stdout.strip()), 200)

    def test_lpt_makespan(self):
        script = ("import sys;sys.path.insert(0,'.');import runner;"
                  "print(runner.lpt_makespan([10,9,8,7,6,5],3), "
                  "runner.lpt_makespan([100],4), runner.lpt_makespan([],2))")
        completed = subprocess.run(
            [PYTHON, "-c", script], capture_output=True, text=True, cwd=str(ROOT),
            timeout=300, env={**os.environ, "RELEASE_LADDER_NO_MODAL": "1"})
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        self.assertEqual(completed.stdout.split(), ["15.0", "100.0", "0.0"])


class Scoring(unittest.TestCase):
    """score.py guards (importing score.py here is safe: it loads no learner)."""

    def test_over_budget_record_is_not_scoreable(self):
        # Self-contained: an earlier version read a leftover smoke result out of
        # results/, so it skipped silently both in a clean tree and after the
        # smoke candidate ids changed -- i.e. exactly when it was needed.
        import score
        seed, n = study.DEV_SEEDS[0], study.LEVELS[0]
        job = study.make_job("smoke", seed, n, "budget-probe", {"family": "krr"},
                             time_budget_seconds=10, device_kind="cpu")
        logits = np.ascontiguousarray(
            np.random.default_rng(0).standard_normal((study.QUERY_COUNT, 10)),
            dtype=np.float32)
        labels = np.ascontiguousarray(np.argmax(logits, axis=1), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "predictions").mkdir()
            path = root / "predictions" / f"{job['id']}.npz"
            with path.open("wb") as handle:
                np.savez(handle, logits=logits, labels=labels)
            record = {
                "job": job,
                "metrics": {"measured_wall_seconds": 1.0},
                "predictions_path": f"predictions/{job['id']}.npz",
                "predictions_sha256": study.sha(path),
                "output_sha256": {"logits": study.ahash(logits),
                                  "labels": study.ahash(labels)},
                "query_labels_supplied": False,
            }
            score.verify_record(root, record)                 # inside budget: legal
            over = float(job["time_budget_seconds"]) + 60.0
            record["metrics"] = {"measured_wall_seconds": over}
            with self.assertRaises(score.ScoreError):
                score.verify_record(root, record)
            # With no measured_wall_seconds the check falls back to
            # job_wall_seconds, which must be policed identically.
            record["metrics"] = {}
            record["job_wall_seconds"] = over
            with self.assertRaises(score.ScoreError):
                score.verify_record(root, record)

    def test_final_ensemble_must_be_pre_declared(self):
        import score
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(score.ScoreError):          # no selection.json
                score._check_declared_ensemble(root, ["a", "b"], [0.5, 0.5], {})
            (root / "selection.json").write_text(json.dumps({"finalists": ["a", "b"]}))
            with self.assertRaises(score.ScoreError):          # nothing declared
                score._check_declared_ensemble(root, ["a", "b"], [0.5, 0.5], {})
            (root / "selection.json").write_text(json.dumps(
                {"final_ensemble": {"members": ["a", "b"], "weights": [0.65, 0.35],
                                    "temperatures": {"a": 4.0}}}))
            score._check_declared_ensemble(root, ["a", "b"], [0.65, 0.35], {"a": 4.0})
            for bad in (([0.5, 0.5], {"a": 4.0}),              # swept weights
                        ([0.65, 0.35], {}),                    # dropped temperature
                        ([0.65, 0.35], {"a": 3.0})):           # swept temperature
                with self.assertRaises(score.ScoreError):
                    score._check_declared_ensemble(root, ["a", "b"], *bad)
            with self.assertRaises(score.ScoreError):          # different members
                score._check_declared_ensemble(root, ["a", "c"], [0.65, 0.35], {"a": 4.0})


class Budget(unittest.TestCase):
    def test_self_test(self):
        completed = subprocess.run([PYTHON, str(ROOT / "budget.py"), "--self-test"],
                                   capture_output=True, text=True, timeout=300)
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        self.assertIn("PASS", completed.stdout)

    def test_rates_and_cap(self):
        import budget
        self.assertEqual(sorted(budget.WORKER_RATES), ["A100-40GB", "T4"])
        self.assertEqual(str(budget.WORKER_RATES["A100-40GB"]), "0.00067122")
        self.assertEqual(str(budget.WORKER_RATES["T4"]), "0.00025192")
        authorization = json.loads((ROOT / "authorization.json").read_text())
        self.assertEqual(authorization["total_modal_cap_usd"], 20)
        self.assertEqual(sorted(authorization["allowed_gpus"]), ["A100-40GB", "T4"])
        # eight A100 workers for 20 minutes must stay inside the usable allowance
        cost = 8 * 1200 * float(budget.WORKER_RATES["A100-40GB"])
        self.assertLess(cost, authorization["total_modal_cap_usd"]
                        - authorization["contingency_usd"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
