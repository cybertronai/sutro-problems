"""Dataset-independent checks for training integrity and model compatibility.

These tests require PyTorch, but neither MNIST downloads, GPUs, nor W&B access.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mnist.models import build_model, candidate_configs
from mnist.predict import predict
from mnist.train import evaluate, split_indices, train_one


class _OfflineTestRun:
    """Keep training tests independent of W&B credentials and networking."""

    def __init__(self):
        self.url = "https://example.invalid/test-run"
        self.id = "test-run"
        self.summary = {}

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


class ValidationSplitTests(unittest.TestCase):
    def test_exact_disjoint_complete_stratification(self):
        # Fractional per-class targets exercise largest-remainder allocation.
        for counts in ([93, 99, 107, 101, 98, 103, 97, 104, 102, 96],
                       [3, 1, 17, 24, 6, 33, 21, 12, 8, 19],
                       [17, 0, 0, 9, 0, 35, 0, 3, 0, 11]):
            with self.subTest(counts=counts):
                labels = np.repeat(np.arange(10), counts)
                original = labels.copy()
                fit, val = split_indices(labels)
                self.assertEqual(len(val), round(len(labels) * .2))
                self.assertEqual(len(fit) + len(val), len(labels))
                self.assertEqual(len(np.intersect1d(fit, val)), 0)
                np.testing.assert_array_equal(
                    np.sort(np.concatenate([fit, val])), np.arange(len(labels))
                )
                allocations = np.bincount(labels[val], minlength=10)
                expected = np.asarray(counts) * .2
                self.assertTrue(np.all(allocations >= np.floor(expected)))
                self.assertTrue(np.all(allocations <= np.ceil(expected)))
                np.testing.assert_array_equal(labels, original)

    def test_repeatable_seed_changes_membership(self):
        labels = np.repeat(np.arange(10), 100)
        first = split_indices(labels, seed=4701)
        repeat = split_indices(labels, seed=4701)
        different = split_indices(labels, seed=4702)
        for left, right in zip(first, repeat):
            np.testing.assert_array_equal(left, right)
        self.assertFalse(np.array_equal(np.sort(first[1]), np.sort(different[1])))
        # Different split seeds preserve per-class allocations.
        np.testing.assert_array_equal(
            np.bincount(labels[first[1]], minlength=10),
            np.bincount(labels[different[1]], minlength=10),
        )


class ModelCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_every_candidate_supports_training_and_single_image_inference(self):
        for tier, size in (("small", 3), ("medium", 9)):
            configs = candidate_configs(tier)
            self.assertEqual(len({c["trial_id"] for c in configs}), len(configs))
            self.assertEqual({c["architecture"] for c in configs}, {"linear", "mlp", "cnn"})
            for config in configs:
                with self.subTest(tier=tier, trial=config["trial_id"]):
                    torch.manual_seed(7)
                    model = build_model(config, size)
                    model.train()
                    x = torch.randn(3, 1, size, size)
                    logits = model(x)
                    self.assertEqual(tuple(logits.shape), (3, 10))
                    loss = F.cross_entropy(logits, torch.tensor([1, 3, 7]))
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                    gradients = [p.grad for p in model.parameters() if p.requires_grad]
                    self.assertTrue(all(g is not None for g in gradients))
                    self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))
                    self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                    model.eval()
                    with torch.inference_mode():
                        result = model(x[:1])
                    self.assertEqual(tuple(result.shape), (1, 10))
                    self.assertTrue(torch.isfinite(result).all().item())

    def test_small_cnn_never_pools(self):
        for config in candidate_configs("small"):
            if config["architecture"] == "cnn":
                model = build_model(config, 3)
                self.assertFalse(any(isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d,
                                                         nn.AdaptiveAvgPool2d))
                                     for layer in model.modules()))
                features = model.features(torch.randn(2, 1, 3, 3))
                self.assertEqual(tuple(features.shape[-2:]), (3, 3))

    def test_evaluation_does_not_update_batchnorm_or_enable_dropout(self):
        config = dict(architecture="cnn", width=4, depth=2, dropout=.5, pooling="none")
        torch.manual_seed(19)
        model = build_model(config, 3)
        model.train()
        model(torch.randn(8, 1, 3, 3))
        initial_buffers = {key: value.clone() for key, value in model.named_buffers()}
        x = torch.randn(11, 1, 3, 3)
        y = torch.arange(11) % 10
        metrics, predictions = evaluate(model, x, y)
        repeat_metrics, repeat_predictions = evaluate(model, x, y)
        self.assertFalse(model.training)
        self.assertEqual(metrics, repeat_metrics)
        np.testing.assert_array_equal(predictions, repeat_predictions)
        for key, value in model.named_buffers():
            torch.testing.assert_close(value, initial_buffers[key], rtol=0, atol=0)
        with torch.inference_mode():
            logits = model(x)
        self.assertAlmostEqual(metrics["loss"], F.cross_entropy(logits, y).item(), places=6)
        self.assertAlmostEqual(metrics["accuracy"], (logits.argmax(1) == y).float().mean().item(), places=6)


class TrainingIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_validation_values_do_not_change_normalization_or_learned_weights(self):
        config = dict(trial_id="synthetic-cnn", architecture="cnn", width=4, depth=2,
                      dropout=.2, lr=.003, weight_decay=.001, batch_size=4, epochs=2)
        x = torch.linspace(0, 1, 8 * 9).reshape(8, 1, 3, 3)
        y = torch.arange(8)
        models, results = [], []
        with tempfile.TemporaryDirectory() as directory, patch(
            "mnist.train.wandb.init", side_effect=lambda **kwargs: _OfflineTestRun()
        ):
            for index, validation_value in enumerate((0., 100.)):
                model, _, result = train_one(
                    config, tier="small", seed=13, phase="search", x=x, y=y,
                    val=(torch.full((4, 1, 3, 3), validation_value), torch.arange(4)),
                    output=Path(directory) / str(index), tracking={},
                )
                models.append(model)
                results.append(result)
        for result in results:
            self.assertEqual(result["normalization"]["mean"], x.mean().item())
            self.assertEqual(result["normalization"]["std"], x.std(unbiased=False).item())
        self.assertNotEqual(results[0]["last"]["val/loss"], results[1]["last"]["val/loss"])
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)
            if key.endswith("num_batches_tracked"):
                self.assertEqual(value.item(), 4)  # Two training batches × two epochs.

    def test_short_refit_keeps_search_schedule_and_checkpoint_normalization(self):
        config = dict(trial_id="synthetic-linear", architecture="linear", width=0, depth=0,
                      dropout=0., lr=.01, weight_decay=.001, batch_size=4, epochs=4)
        x = torch.linspace(0, 1, 8 * 9).reshape(8, 1, 3, 3)
        y = torch.arange(8)
        with tempfile.TemporaryDirectory() as directory, patch(
            "mnist.train.wandb.init", side_effect=lambda **kwargs: _OfflineTestRun()
        ):
            model, _, result = train_one(
                config, tier="small", seed=17, phase="final", x=x, y=y, val=None,
                output=Path(directory), tracking={}, epochs=2, schedule_epochs=4,
            )
            self.assertIsNone(result["best"])
            self.assertEqual(result["epochs"], 2)
            self.assertEqual(result["schedule_epochs"], 4)
            self.assertEqual(len(result["history"]), 2)
            expected_lr = config["lr"] * (.02 + .98 * (1 + math.cos(math.pi / 4)) / 2)
            self.assertAlmostEqual(result["history"][1]["learning_rate"], expected_lr, places=12)
            checkpoint_path = Path(directory) / result["checkpoint"]
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["normalization"], result["normalization"])
            self.assertEqual(checkpoint["epochs"], 2)
            restored = build_model(checkpoint["config"], checkpoint["image_size"])
            restored.load_state_dict(checkpoint["state_dict"])
            restored.eval()
            model.eval()
            norm = checkpoint["normalization"]
            standardized = (x - norm["mean"]) / norm["std"]
            with torch.inference_mode():
                torch.testing.assert_close(model(standardized), restored(standardized), rtol=0, atol=0)
                expected_predictions = model(standardized).argmax(1).numpy()
            dataset_path = Path(directory) / "synthetic.npz"
            predictions_path = Path(directory) / "predictions.npz"
            # The inference path must work without access to any labels.
            np.savez_compressed(dataset_path, test_images=x.numpy())
            predict(checkpoint_path, dataset_path, predictions_path)
            with np.load(predictions_path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["predictions"], expected_predictions)


if __name__ == "__main__":
    unittest.main()
