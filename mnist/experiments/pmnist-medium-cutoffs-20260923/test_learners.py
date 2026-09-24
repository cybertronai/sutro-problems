"""CPU unit tests for learners.py / runner.py / plan.py (no GPU, no Modal app).

    /tmp/pmnist-env/bin/python -m unittest -v test_learners.py
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import learners                                                   # noqa: E402

DEADLINE = 10 ** 10        # far future: no truncation


def synthetic(n=240, q=60, seed=0):
    """Linearly separable 81-feature problem in [0,1], 10 classes."""
    rng = np.random.default_rng(seed)
    directions = rng.random((10, 81)).astype(np.float32)
    y = rng.integers(0, 10, size=n).astype(np.uint8)
    x = np.clip(0.15 * rng.random((n, 81)).astype(np.float32)
                + 0.85 * directions[y], 0.0, 1.0)
    yq = rng.integers(0, 10, size=q).astype(np.uint8)
    xq = np.clip(0.15 * rng.random((q, 81)).astype(np.float32)
                 + 0.85 * directions[yq], 0.0, 1.0)
    return x, y, xq, yq


BASE_MLP = {'family': 'mlp', 'widths': [64], 'epochs': 20, 'lr': 0.01,
            'batch_size': 64, 'members': 1}



def _ledger_state():
    """Existence + sha256 of the study ledger; None when absent."""
    import hashlib
    path = ROOT / 'budget-ledger.json'
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

class TestMLP(unittest.TestCase):
    def test_training_loss_decreases(self):
        x, y, q, _ = synthetic()
        out = learners.fit_predict(x, y, q, BASE_MLP, 11, DEADLINE, 'cpu')
        history = out['metrics']['history']
        self.assertGreaterEqual(len(history), 2)
        self.assertLess(history[-1]['mean_member_loss'],
                        0.5 * history[0]['mean_member_loss'])
        self.assertEqual(out['metrics']['epochs_completed'], 20)
        self.assertFalse(out['metrics']['truncated'])
        self.assertEqual(out['logits'].shape, (q.shape[0], 10))
        self.assertEqual(out['logits'].dtype, np.float32)

    def test_batched_members_match_independent_runs(self):
        """K=3 batched == three members=1 runs at seed+1000*k (no stochastic aug)."""
        x, y, q, _ = synthetic(n=200, q=40, seed=3)
        config = {**BASE_MLP, 'epochs': 6, 'members': 3}
        batched = learners.fit_predict(x, y, q, config, 11, DEADLINE, 'cpu')
        singles = [learners.fit_predict(x, y, q, {**config, 'members': 1},
                                        11 + 1000 * k, DEADLINE, 'cpu')
                   for k in range(3)]
        mean_probability = np.mean([np.exp(s['logits'].astype(np.float64))
                                    for s in singles], axis=0)
        # log(mean prob) round-trips through float32, hence 1e-5 rather than 0.
        np.testing.assert_allclose(batched['logits'], np.log(mean_probability),
                                   atol=1e-5)
        np.testing.assert_array_equal(batched['labels'], learners.first_max_labels(
            np.log(mean_probability).astype(np.float32)))
        self.assertEqual(batched['metrics']['parameters'],
                         3 * singles[0]['metrics']['parameters'])
        self.assertFalse(np.allclose(singles[0]['logits'], singles[1]['logits']))

    def test_labels_are_first_max_argmax(self):
        x, y, q, _ = synthetic(n=120, q=30, seed=4)
        out = learners.fit_predict(x, y, q, {**BASE_MLP, 'epochs': 2}, 5, DEADLINE, 'cpu')
        np.testing.assert_array_equal(
            out['labels'], np.argmax(out['logits'], axis=1).astype(np.uint8))
        self.assertEqual(out['labels'].dtype, np.uint8)
        tied = np.zeros((3, 10), dtype=np.float32)
        tied[1, 7] = 1.0
        np.testing.assert_array_equal(learners.first_max_labels(tied),
                                      np.array([0, 7, 0], dtype=np.uint8))

    def test_permutation_equivariance_with_init_hook(self):
        """No family reads feature order: permuting data + init gives one function.

        A seeded dense init binds column j to *position* j, so plain reseeding is
        only equal in distribution. Reindexing the input-facing columns with the
        same permutation (the ``LadderAMLP.permute_inputs_`` trick) makes the two
        runs the same computation; they agree to ~1e-7 rather than bitwise only
        because permuted columns reorder the float summation in the first matmul.
        """
        x, y, q, _ = synthetic(n=200, q=40, seed=6)
        perm = np.random.default_rng(20260923).permutation(81)
        config = {**BASE_MLP, 'epochs': 5, 'members': 2, 'widths': [48, 32]}
        plain = learners.fit_predict(x, y, q, config, 11, DEADLINE, 'cpu')
        permuted = learners.fit_predict(
            x[:, perm], y, q[:, perm],
            {**config, 'init_input_permutation': perm.tolist()}, 11, DEADLINE, 'cpu')
        np.testing.assert_allclose(plain['logits'], permuted['logits'], atol=1e-4)
        np.testing.assert_array_equal(plain['labels'], permuted['labels'])

    def test_time_guard_truncates_without_raising(self):
        x, y, q, _ = synthetic(n=120, q=30, seed=7)
        out = learners.fit_predict(x, y, q, {**BASE_MLP, 'epochs': 500},
                                   11, time.time() - 5.0, 'cpu')
        self.assertTrue(out['metrics']['truncated'])
        self.assertIn(out['metrics']['epochs_completed'], (0, 1))
        self.assertEqual(out['logits'].shape, (30, 10))
        self.assertTrue(np.isfinite(out['logits']).all())

    def test_stochastic_options_run(self):
        x, y, q, _ = synthetic(n=160, q=20, seed=8)
        out = learners.fit_predict(
            x, y, q,
            {**BASE_MLP, 'epochs': 3, 'members': 2, 'dropout': 0.2,
             'input_noise_std': 0.1, 'mixup_alpha': 0.4, 'label_smoothing': 0.05,
             'ema_decay': 0.9, 'activation': 'gelu', 'warmup_fraction': 0.2,
             'weight_decay': 0.01},
            11, DEADLINE, 'cpu')
        self.assertTrue(out['metrics']['ema_used'])
        self.assertFalse(out['metrics']['uses_query_images_unlabeled'])
        self.assertTrue(np.isfinite(out['logits']).all())

    def test_ema_is_bias_corrected(self):
        """A high decay must not evaluate the random init (no warmup otherwise).

        Without bias correction ``decay**steps`` of the shadow is still the
        initialisation -- 0.92 at decay=0.9999 with the 800 steps a 100-epoch
        n=1000 fit takes -- and accuracy collapses.  With the correction the
        three decays agree.
        """
        x, y, q, yq = synthetic(n=240, q=120, seed=21)
        accuracies = {}
        for decay in (0.0, 0.999, 0.9999):
            out = learners.fit_predict(x, y, q, {**BASE_MLP, 'ema_decay': decay},
                                       11, DEADLINE, 'cpu')
            accuracies[decay] = float((out['labels'] == yq).mean())
            correction = out['metrics']['ema_bias_correction']
            if decay == 0.0:
                self.assertIsNone(correction)
                self.assertFalse(out['metrics']['ema_used'])
            else:
                self.assertTrue(out['metrics']['ema_used'])
                self.assertGreater(correction, 0.0)
                self.assertLess(correction, 1.0)
        for decay in (0.999, 0.9999):
            self.assertGreaterEqual(accuracies[decay], accuracies[0.0] - 0.1)

    def test_ema_with_zero_steps_keeps_live_weights(self):
        x, y, q, _ = synthetic(n=60, q=20, seed=22)
        out = learners.fit_predict(x, y, q, {**BASE_MLP, 'epochs': 500,
                                             'ema_decay': 0.999},
                                   11, time.time() - 5.0, 'cpu')
        self.assertTrue(np.isfinite(out['logits']).all())
        if out['metrics']['steps_completed'] == 0:
            self.assertFalse(out['metrics']['ema_used'])
            self.assertEqual(out['metrics']['ema_bias_correction'], 0.0)

    def test_autocast_bf16_runs_and_is_rejected_by_frozen_families(self):
        x, y, q, _ = synthetic(n=120, q=20, seed=14)
        out = learners.fit_predict(x, y, q, {**BASE_MLP, 'epochs': 2,
                                             'autocast_bf16': True},
                                   11, DEADLINE, 'cpu')
        self.assertEqual(out['logits'].dtype, np.float32)
        self.assertTrue(np.isfinite(out['logits']).all())
        with self.assertRaises(AssertionError):
            learners.fit_predict(x, y, q, {'family': 'ladder', 'epochs': 1,
                                           'decay_start_epoch': 1,
                                           'hidden_dims': [16],
                                           'autocast_bf16': True},
                                 7, DEADLINE, 'cpu')

    def test_rejects_unknown_config_key(self):
        with self.assertRaises(ValueError):
            learners.resolve_config({'family': 'mlp', 'nonsense': 1})
        with self.assertRaises(ValueError):
            learners.resolve_config({'family': 'nope'})


class TestLossPieces(unittest.TestCase):
    def test_soft_target_ce_equals_hard_ce(self):
        """mixup_alpha=0 keeps one-hot targets, so the soft CE is the hard CE."""
        torch.manual_seed(0)
        logits = torch.randn(4, 7, 10)
        y = torch.randint(0, 10, (4, 7))
        soft = learners._soft_cross_entropy(logits, learners._one_hot(y, 0.0))
        hard = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        self.assertAlmostEqual(float(soft), float(hard), places=6)
        smoothed = learners._soft_cross_entropy(logits, learners._one_hot(y, 0.1))
        reference = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1),
                                    label_smoothing=0.1)
        self.assertAlmostEqual(float(smoothed), float(reference), places=6)

    def _model(self):
        return learners.BatchedMLP([32], members=2, seed=11, device='cpu')

    def test_vat_loss_nonnegative_and_zero_at_eps_zero(self):
        model = self._model()
        xu = learners.normalize(torch.rand(2, 16, 81))
        generator = learners._make_generator('cpu', 5)
        vat = {**learners.VAT_DEFAULTS, 'eps': 1.5}
        value = learners.vat_loss(model, xu, vat, generator)
        self.assertGreaterEqual(float(value), 0.0)
        zero = learners.vat_loss(model, xu, {**learners.VAT_DEFAULTS, 'eps': 0.0},
                                 generator)
        self.assertEqual(float(zero), 0.0)

    def test_vat_shares_one_dropout_draw_across_passes(self):
        """With independent masks the eps=0 floor swamps the real VAT signal."""
        model = learners.BatchedMLP([64, 64], members=1, seed=3, device='cpu',
                                    dropout=0.2)
        xu = learners.normalize(torch.rand(1, 128, 81))
        generator = learners._make_generator('cpu', 1)
        zero = learners.vat_loss(model, xu, {**learners.VAT_DEFAULTS, 'eps': 0.0},
                                 generator)
        self.assertEqual(float(zero), 0.0)
        signal = learners.vat_loss(model, xu, {**learners.VAT_DEFAULTS, 'eps': 2.5},
                                   generator)
        self.assertGreater(float(signal), 0.0)

    def test_vat_gradient_flows(self):
        model = self._model()
        xu = learners.normalize(torch.rand(2, 16, 81))
        generator = learners._make_generator('cpu', 6)
        value = learners.vat_loss(model, xu, {**learners.VAT_DEFAULTS, 'eps': 2.0},
                                  generator)
        value.backward()
        grads = [p.grad for p in model.parameters()]
        self.assertTrue(all(g is not None for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)

    def test_vat_end_to_end_uses_query_flag(self):
        x, y, q, _ = synthetic(n=120, q=20, seed=9)
        out = learners.fit_predict(
            x, y, q, {**BASE_MLP, 'epochs': 2,
                      'vat': {'eps': 1.0, 'weight': 0.5, 'unlabeled': 'train+query'}},
            11, DEADLINE, 'cpu')
        self.assertTrue(out['metrics']['uses_query_images_unlabeled'])
        out_train = learners.fit_predict(
            x, y, q, {**BASE_MLP, 'epochs': 2, 'vat': {'eps': 1.0}},
            11, DEADLINE, 'cpu')
        self.assertFalse(out_train['metrics']['uses_query_images_unlabeled'])


class TestLadder(unittest.TestCase):
    def test_one_epoch_returns_logits(self):
        x, y, q, _ = synthetic(n=200, q=40, seed=10)
        out = learners.fit_predict(
            x, y, q, {'family': 'ladder', 'hidden_dims': [64, 32], 'epochs': 1,
                      'decay_start_epoch': 1, 'batch_size': 50, 'members': 1},
            7, DEADLINE, 'cpu')
        self.assertEqual(out['logits'].shape, (40, 10))
        self.assertEqual(out['metrics']['epochs_completed'], 1)
        self.assertEqual(out['metrics']['members_completed'], 1)
        self.assertTrue(np.isfinite(out['logits']).all())
        np.testing.assert_array_equal(
            out['labels'], np.argmax(out['logits'], axis=1).astype(np.uint8))

    def test_inputs_are_raw_zero_one_pixels(self):
        """The published noise_std / reconstruction weight assume /255 inputs.

        Rescaling to ``4*x-0.5`` multiplies the reconstruction penalty by ~12x
        relative to the cross entropy and turns the 0.3 corruption noise from
        1.26x into 0.32x the input SD, i.e. a different recipe from the
        published one this family vendors.
        """
        import ladder_model
        x, y, q, _ = synthetic(n=60, q=10, seed=30)
        seen = []
        original = ladder_model.LadderAMLP.loss

        def spy(self, x_labeled, y_labeled, x_unlabeled=None):
            seen.append((float(x_labeled.min()), float(x_labeled.max())))
            return original(self, x_labeled, y_labeled, x_unlabeled=x_unlabeled)

        with mock.patch.object(ladder_model.LadderAMLP, 'loss', spy):
            out = learners.fit_predict(
                x, y, q, {'family': 'ladder', 'hidden_dims': [16], 'epochs': 1,
                          'decay_start_epoch': 1, 'batch_size': 30, 'members': 1},
                7, DEADLINE, 'cpu')
        self.assertTrue(seen)
        self.assertGreaterEqual(min(low for low, _ in seen), 0.0)
        self.assertLessEqual(max(high for _, high in seen), 1.0)
        self.assertAlmostEqual(max(high for _, high in seen), float(x.max()), places=6)
        self.assertEqual(out['metrics']['normalization'],
                         'raw [0,1] pixels (published ladder recipe)')

    def test_untrained_member_is_not_averaged_in(self):
        """A member that ran zero epochs must be dropped, not voted with."""
        x, y, q, _ = synthetic(n=200, q=40, seed=31)
        config = {'family': 'ladder', 'hidden_dims': [32], 'epochs': 1,
                  'decay_start_epoch': 1, 'batch_size': 50, 'members': 2}

        class StopAfterFirstEpoch(learners.TimeGuard):
            calls = 0

            def should_stop(self):
                type(self).calls += 1
                return type(self).calls > 1

        with mock.patch.object(learners, 'TimeGuard', StopAfterFirstEpoch):
            truncated = learners.fit_predict(x, y, q, config, 7, DEADLINE, 'cpu')
        self.assertEqual(truncated['metrics']['member_epochs_completed'], [1, 0])
        self.assertEqual(truncated['metrics']['members_completed'], 1)
        self.assertTrue(truncated['metrics']['truncated'])
        solo = learners.fit_predict(x, y, q, {**config, 'members': 1}, 7,
                                    DEADLINE, 'cpu')
        np.testing.assert_allclose(truncated['logits'], solo['logits'], atol=1e-5)

    def test_transductive_reconstruction_stream_flag(self):
        x, y, q, _ = synthetic(n=200, q=40, seed=11)
        out = learners.fit_predict(
            x, y, q, {'family': 'ladder', 'hidden_dims': [32], 'epochs': 1,
                      'decay_start_epoch': 1, 'batch_size': 50,
                      'unlabeled': 'train+query'},
            7, DEADLINE, 'cpu')
        self.assertTrue(out['metrics']['uses_query_images_unlabeled'])

    def test_ragged_level_size_runs(self):
        """Study levels 1778/3162/5623 are not multiples of the batch size."""
        x, y, q, _ = synthetic(n=178, q=20, seed=12)
        out = learners.fit_predict(
            x, y, q, {'family': 'ladder', 'hidden_dims': [32], 'epochs': 1,
                      'decay_start_epoch': 1, 'batch_size': 100},
            7, DEADLINE, 'cpu')
        self.assertEqual(out['logits'].shape, (20, 10))


    def test_lr_factor_matches_vendored_schedule(self):
        """ladder_lr_factor reproduces ladder_model.learning_rate_at_epoch."""
        from ladder_model import LadderConfig, learning_rate_at_epoch
        config = LadderConfig(epochs=150, decay_start_epoch=100)
        for epoch in (0, 1, 99, 100, 125, 149, 150, 200):
            self.assertAlmostEqual(
                config.learning_rate * learners.ladder_lr_factor(epoch, 150, 100),
                learning_rate_at_epoch(epoch, config), places=12)
        # epochs == decay_start would divide by zero in the vendored helper.
        self.assertEqual(learners.ladder_lr_factor(0, 1, 1), 1.0)


def permuted_pool(n, q, seed=0):
    """Real permuted 9x9 study rows: topology.recover_layout needs image data.

    Synthetic i.i.d. vectors have no spatial statistics at all, so the layout
    search on them is meaningless (and currently raises inside scipy).
    """
    images = np.load(ROOT / 'raw/pool_images.npy', mmap_mode='r')
    labels = np.load(ROOT / 'raw/pool_labels.npy', mmap_mode='r')
    perm = np.load(ROOT / 'raw/feature_permutation.npy')
    rng = np.random.default_rng(seed)
    rows = rng.choice(images.shape[0], size=n + q, replace=False)
    x = np.ascontiguousarray(np.asarray(images[rows])[:, perm], dtype=np.float32)
    y = np.ascontiguousarray(np.asarray(labels[rows]), dtype=np.uint8)
    return x[:n], y[:n], x[n:], y[n:]


class TestTopoCNN(unittest.TestCase):
    def _data(self, n=300, q=40, seed=13):
        if not (ROOT / 'raw/pool_images.npy').exists():
            self.skipTest('raw/pool_images.npy not built yet')
        return permuted_pool(n, q, seed)

    def test_guarded_when_topology_missing(self):
        x, y, q, _ = self._data()
        config = {'family': 'topo_cnn', 'epochs': 1, 'member_seeds': [101],
                  'layout_max_seconds': 5.0}
        if (ROOT / 'topology.py').exists():
            out = learners.fit_predict(x, y, q, config, 11, DEADLINE, 'cpu')
            self.assertEqual(out['logits'].shape, (q.shape[0], 10))
            self.assertFalse(out['metrics']['uses_query_images_unlabeled'])
            self.assertEqual(len(out['metrics']['layout']), 81)
            self.assertIn('qap_objective', out['metrics']['layout_diagnostics'])
            self.assertGreater(out['metrics']['layout_recovery_seconds'], 0.0)
        else:
            with self.assertRaises(RuntimeError) as caught:
                learners.fit_predict(x, y, q, config, 11, DEADLINE, 'cpu')
            self.assertIn('topology', str(caught.exception))

    def test_members_are_combined_as_float64_mean_logits(self):
        """cutoff-calibration-20260920/analyze.py averages logits, not softmax.

        Mixing in a different combination rule would move ~0.2 pp of labels and
        contaminate the paired comparison against the spatial CNN.
        """
        if not (ROOT / 'topology.py').exists():
            self.skipTest('topology.py not written yet')
        x, y, q, _ = self._data(n=200, q=30, seed=15)
        config = {'family': 'topo_cnn', 'epochs': 1, 'member_seeds': [101, 102],
                  'layout_max_seconds': 5.0}
        out = learners.fit_predict(x, y, q, config, 11, DEADLINE, 'cpu')
        self.assertEqual(out['metrics']['logit_kind'], 'mean_logit_float64')
        self.assertEqual(out['metrics']['members_completed'], 2)
        # Mean logits are unnormalised: log-mean-prob rows would all be <= 0.
        self.assertGreater(float(out['logits'].max()), 0.0)

    def test_untrained_member_is_dropped(self):
        if not (ROOT / 'topology.py').exists():
            self.skipTest('topology.py not written yet')
        x, y, q, _ = self._data(n=200, q=30, seed=16)
        config = {'family': 'topo_cnn', 'epochs': 1, 'member_seeds': [101, 102],
                  'layout_max_seconds': 5.0}

        class StopAfterFirstEpoch(learners.TimeGuard):
            calls = 0

            def should_stop(self):
                type(self).calls += 1
                return type(self).calls > 1

        with mock.patch.object(learners, 'TimeGuard', StopAfterFirstEpoch):
            out = learners.fit_predict(x, y, q, config, 11, DEADLINE, 'cpu')
        self.assertEqual(out['metrics']['member_epochs_completed'], [1, 0])
        self.assertEqual(out['metrics']['members_completed'], 1)
        self.assertTrue(out['metrics']['truncated'])
        solo = learners.fit_predict(x, y, q, {**config, 'member_seeds': [101]},
                                    11, DEADLINE, 'cpu')
        np.testing.assert_allclose(out['logits'], solo['logits'], atol=1e-5)

    def test_schedule_rule_matches_calibration(self):
        self.assertEqual(learners._cnn_schedule_epochs(1000), 125)
        self.assertEqual(learners._cnn_schedule_epochs(10000), 100)
        self.assertEqual(learners._cnn_schedule_epochs(1778), 100)

    def test_frozen_cnn_config_matches_vendored_learner(self):
        import spatial_learner
        self.assertEqual(learners.CNN09_CONFIG, spatial_learner.CONFIG)


class TestRunnerAndPlan(unittest.TestCase):
    def test_runner_imports_without_credentials_or_dispatch(self):
        ledger_before = _ledger_state()
        import runner
        self.assertEqual(runner.APP_NAME, 'pmnist-medium-cutoffs-20260923')
        self.assertLessEqual(runner.MAX_GPUS, 8)
        import budget
        self.assertLessEqual(runner.MAX_GPUS, budget.MAX_CONTAINERS)
        self.assertEqual(_ledger_state(), ledger_before,
                         'Importing runner must not create or change the ledger')
        self.assertTrue(callable(runner.fit_job.remote))

    def test_container_gets_no_study_module_and_no_dataset_seed(self):
        import runner
        self.assertNotIn('study.py', runner.SOURCE_FILES)
        self.assertNotIn('study.py', runner.ADDED_FILES)
        self.assertNotIn('study.py', runner.required_sources(
            [{'config': {'family': 'topo_cnn'}}]))
        source = (ROOT / 'runner.py').read_text()
        # study.py carries FEATURE_PERMUTATION_SEED and the draw rule; the seed
        # alone would let the container reconstruct the query rows.
        self.assertNotIn('block_network=True', source)  # blob-store inputs need network
        self.assertIn("if key != 'seed'", source)
        self.assertIn("assert 'seed' not in job", source)

    def test_image_installs_topology_dependencies(self):
        """topology.py imports scipy at module level; the base image has none."""
        source = (ROOT / 'runner.py').read_text()
        self.assertIn("pip_install('numpy==2.2.6', 'scipy==1.15.3')", source)
        self.assertIn('from scipy.optimize import linear_sum_assignment',
                      (ROOT / 'topology.py').read_text())

    def test_preflight_rejects_a_bad_config_locally(self):
        if not (ROOT / 'study.py').exists():
            self.skipTest('study.py not written yet')
        if not (ROOT / 'raw/data_manifest.json').exists():
            self.skipTest('raw/data_manifest.json not built yet')
        import runner
        sys.path.insert(0, str(ROOT))
        import study
        bad = study.make_job('dev', study.DEV_SEEDS[0], 1000, 'bad',
                             {'family': 'mlp', 'nonsense': 1}, 11)
        authorization = json.loads((ROOT / 'authorization.json').read_text())
        with self.assertRaises(ValueError):
            runner.preflight([bad], [], {}, ROOT / 'plans/smoke.json', authorization,
                             60, study, [])

    def test_plan_builder_rejects_a_bad_candidate(self):
        import plan
        with self.assertRaises(ValueError):
            plan._candidates([{'id': 'bad', 'config': {'family': 'mlp',
                                                       'nonsense': 1}}])
        with self.assertRaises(ValueError):
            plan._candidates([{'id': 'bad', 'config': {'family': 'nope'}}])

    def test_validate_only_on_smoke_plan(self):
        ledger_before = _ledger_state()
        if not (ROOT / 'study.py').exists():
            self.skipTest('study.py not written yet')
        if not (ROOT / 'raw/data_manifest.json').exists():
            self.skipTest('raw/data_manifest.json not built yet')
        if not (ROOT / 'plans/smoke.json').exists():
            subprocess.run([sys.executable, str(ROOT / 'plan.py'), 'smoke'],
                           cwd=ROOT, check=True, capture_output=True, text=True)
        completed = subprocess.run(
            [sys.executable, str(ROOT / 'runner.py'), '--plan', 'plans/smoke.json',
             '--validate-only'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
        receipt = json.loads(completed.stdout.strip().splitlines()[-1])
        if receipt['missing_sources']:
            # topology.py is written by another agent; preflight reports it as
            # missing and dispatch refuses, but the receipt is still produced.
            self.assertFalse(receipt['passed'])
            self.assertIn('topology.py', receipt['missing_sources'])
        else:
            self.assertTrue(receipt['passed'])
        self.assertFalse(receipt['new_reservation_created'])
        self.assertFalse(receipt['paid_compute_started'])
        self.assertEqual(_ledger_state(), ledger_before,
                         '--validate-only must not reserve budget')


if __name__ == '__main__':
    unittest.main()


class TestNormalizationOption(unittest.TestCase):
    def _data(self):
        rng = np.random.default_rng(3)
        x = rng.random((96, 81), dtype=np.float32) * np.linspace(0, 1, 81, dtype=np.float32)
        y = (x[:, :10].sum(1) > x[:, 10:20].sum(1)).astype(np.uint8)
        q = rng.random((32, 81), dtype=np.float32)
        return x, y, q

    def test_standardize_runs_and_differs_from_default(self):
        import learners, time
        x, y, q = self._data()
        base = dict(family='mlp', widths=[32], epochs=2, members=1)
        a = learners.fit_predict(x, y, q, base, 5, time.time() + 60, 'cpu')
        b = learners.fit_predict(x, y, q, {**base, 'normalization': 'standardize'}, 5,
                                 time.time() + 60, 'cpu')
        self.assertEqual(a['logits'].shape, b['logits'].shape)
        self.assertTrue(np.isfinite(b['logits']).all())
        self.assertFalse(np.allclose(a['logits'], b['logits']))
        self.assertIn('training-set statistics', b['metrics']['normalization'])
        self.assertIn('4*x-0.5', a['metrics']['normalization'])

    def test_unknown_normalization_rejected(self):
        import learners, time
        x, y, q = self._data()
        with self.assertRaises(ValueError):
            learners.fit_predict(x, y, q, dict(family='mlp', widths=[8], epochs=1,
                                               normalization='zscore'), 5, time.time() + 60, 'cpu')
