"""Focused mathematical checks for the Ladder port; no dataset downloads."""
from copy import deepcopy
import unittest

import torch
from torch.nn import functional as F

from model import LadderAMLP, LadderConfig, NodewiseAMLP, learning_rate_at_epoch


class LadderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def small_model(self, noise=0.3):
        return LadderAMLP(9, 3, LadderConfig(noise_std=noise),
                          hidden_dims=(7, 5)).double()

    def test_nodewise_combinator_matches_independent_mlps(self):
        module = NodewiseAMLP(5).double()
        lateral, vertical = torch.randn(2, 8, 5, dtype=torch.float64)
        independent = []
        for coordinate in range(5):
            h = torch.stack((vertical[:, coordinate], lateral[:, coordinate],
                             vertical[:, coordinate] * lateral[:, coordinate]), -1)
            for index, (weight, bias) in enumerate(zip(module.weights, module.biases)):
                h = h @ weight[coordinate] + bias[coordinate]
                if index < len(module.weights) - 1:
                    h = F.leaky_relu(h, negative_slope=0.1)
            independent.append(h.squeeze(-1))
        torch.testing.assert_close(module(lateral, vertical),
                                   torch.stack(independent, -1), atol=1e-15, rtol=1e-12)

    def test_full_objective_and_logits_are_permutation_equivalent(self):
        original = self.small_model(noise=0)
        permutation = torch.tensor([8, 2, 4, 1, 5, 0, 6, 3, 7])
        permuted = deepcopy(original).permute_inputs_(permutation)
        x, xu = torch.randn(2, 20, 9, dtype=torch.float64)
        y = torch.arange(20) % 3
        loss = original.loss(x, y, xu)
        other_loss = permuted.loss(x[:, permutation], y, xu[:, permutation])
        torch.testing.assert_close(loss, other_loss, atol=1e-9, rtol=1e-11)
        loss.backward()
        other_loss.backward()
        torch.testing.assert_close(original.encoder[0].linear.weight.grad[:, permutation],
                                   permuted.encoder[0].linear.weight.grad,
                                   atol=1e-9, rtol=1e-9)
        original.eval()
        permuted.eval()
        torch.testing.assert_close(original(x), permuted(x[:, permutation]),
                                   atol=1e-12, rtol=1e-10)

    def test_eval_is_noise_free_batch_independent_and_read_only(self):
        model = self.small_model()
        train_x = torch.randn(103, 9, dtype=torch.float64)
        model.calibrate_bn(train_x, batch_size=20)
        model.eval()
        state = deepcopy(model.state_dict())
        x = torch.randn(17, 9, dtype=torch.float64)
        expected = model(x)
        torch.testing.assert_close(expected, model(x), rtol=0, atol=0)
        split = torch.cat([model(part) for part in x.split(3)], dim=0)
        torch.testing.assert_close(expected, split, atol=1e-12, rtol=1e-10)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_calibration_matches_reference_batch_statistics(self):
        model = self.small_model()
        x = torch.randn(100, 9, dtype=torch.float64)
        seed = 29
        indices = torch.randperm(100, generator=torch.Generator().manual_seed(seed))
        means = [[] for _ in model.encoder]
        variances = [[] for _ in model.encoder]
        for part in indices.split(20):
            h = x[part]
            for i, layer in enumerate(model.encoder):
                raw = layer.linear(h)
                mean, var = raw.mean(0), raw.var(0, unbiased=False)
                means[i].append(mean)
                variances[i].append(var * 20 / 19)
                h = layer.activate((raw - mean) / torch.sqrt(var + layer.eps))
        model.calibrate_bn(x, batch_size=20, shuffle_seed=seed)
        for i, layer in enumerate(model.encoder):
            torch.testing.assert_close(layer.running_mean, torch.stack(means[i]).mean(0))
            torch.testing.assert_close(layer.running_var, torch.stack(variances[i]).mean(0))

    def test_noise_and_gradients_and_adam_update(self):
        model = self.small_model()
        x, xu = torch.randn(2, 32, 9, dtype=torch.float64)
        y = torch.arange(32) % 3
        first = model.loss_components(x, y, xu)
        second = model.loss_components(x, y, xu)
        self.assertFalse(torch.equal(first["noisy_logits"], second["noisy_logits"]))
        torch.testing.assert_close(first["loss"], first["cross_entropy"] +
                                   2000 * first["reconstruction_mse"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        first["loss"].backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertTrue(torch.isfinite(model.loss(x, y, xu)))

    def test_paper_schedule(self):
        self.assertEqual(learning_rate_at_epoch(0), 0.002)
        self.assertEqual(learning_rate_at_epoch(99), 0.002)
        self.assertEqual(learning_rate_at_epoch(100), 0.002)
        self.assertAlmostEqual(learning_rate_at_epoch(125), 0.001)
        self.assertAlmostEqual(learning_rate_at_epoch(149), 0.00004)
        self.assertEqual(learning_rate_at_epoch(150), 0)


if __name__ == "__main__":
    unittest.main()
