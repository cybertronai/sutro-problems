"""PyTorch port of the fully supervised AMLP[2,2] Ladder configuration.

Pezeshki et al., ICML 2016, Table 2; supplement Tables 4 and 5.
https://proceedings.mlr.press/v48/pezeshki16.html
Details not specified there follow CuriousAI/ladder commit 5a8daa1.
See research/model-protocol.md for provenance and explicit deviations.

The model accepts vector inputs: neither spatial positions nor the permutation
are used. ``forward`` returns clean logits; ``loss`` uses a noisy encoder and
an AMLP decoder. Call ``calibrate_bn`` on TRAINING inputs before final testing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LadderConfig:
    hidden_dims: tuple[int, ...] = (1000, 500, 250, 250, 250)
    noise_std: float = 0.3
    input_reconstruction_weight: float = 2000.0
    combinator_hidden_dims: tuple[int, ...] = (2, 2)
    combinator_init_std: float = 0.025
    bn_eps: float = 1e-10
    batch_size: int = 100
    learning_rate: float = 0.002
    epochs: int = 150
    decay_start_epoch: int = 100
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8


DEFAULT_CONFIG = asdict(LadderConfig())


def learning_rate_at_epoch(epoch: int, config: LadderConfig | None = None) -> float:
    """Zero-based epoch: 100 constant-LR epochs, then 50 decaying epochs.

As with the reference after-epoch scheduler, the LR reaches zero immediately
after the final epoch (epoch=150). Epochs 100..149 use 1.00..0.02 times base LR.
    """
    cfg = config or LadderConfig()
    fraction = max(0.0, min(1.0, (cfg.epochs - epoch) /
                           (cfg.epochs - cfg.decay_start_epoch)))
    return cfg.learning_rate * fraction


def _normalize(x: Tensor, eps: float) -> tuple[Tensor, Tensor, Tensor]:
    variance, mean = torch.var_mean(x, dim=0, unbiased=False)
    return (x - mean) * torch.rsqrt(variance + eps), mean, variance


class NodewiseAMLP(nn.Module):
    """Independent 3 -> 2 -> 2 -> 1 MLP at every coordinate, vectorized.

Input coordinates are [vertical, lateral, vertical*lateral]. There is no
parameter sharing between nodes, layers, or examples' coordinate positions.
    """

    def __init__(self, size: int, hidden_dims: Sequence[int] = (2, 2),
                 init_std: float = 0.025):
        super().__init__()
        dims = (3, *hidden_dims, 1)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(size, n_in, n_out))
            for n_in, n_out in zip(dims[:-1], dims[1:])])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(size, n_out)) for n_out in dims[1:]])
        for weight in self.weights:
            nn.init.normal_(weight, std=init_std)

    def forward(self, lateral: Tensor, vertical: Tensor) -> Tensor:
        h = torch.stack((vertical, lateral, vertical * lateral), dim=-1)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            h = torch.einsum("bni,nio->bno", h, weight) + bias
            if index < len(self.weights) - 1:
                h = F.leaky_relu(h, negative_slope=0.1)
        return h.squeeze(-1)


class EncoderLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, *, top: bool, eps: float):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=False)
        nn.init.normal_(self.linear.weight, std=n_in ** -0.5)
        self.beta = nn.Parameter(torch.zeros(n_out))
        self.gamma = nn.Parameter(torch.ones(n_out)) if top else None
        self.top, self.eps = top, eps
        self.register_buffer("running_mean", torch.zeros(n_out))
        self.register_buffer("running_var", torch.ones(n_out))

    def activate(self, z: Tensor) -> Tensor:
        h = z + self.beta
        if self.top:
            return h * self.gamma
        return F.relu(h)


class LadderAMLP(nn.Module):
    """Fully supervised, input-reconstruction-only AMLP Ladder network.

``x_unlabeled`` is an independently shuffled batch from the SAME training
set. Passing it reproduces the official paired labeled/unlabeled streams.
Omitting it reuses ``x`` for both losses, a cheaper explicitly marked variant.
    """

    def __init__(self, input_dim: int = 784, num_classes: int = 10,
                 config: LadderConfig | None = None, *,
                 hidden_dims: Sequence[int] | None = None):
        super().__init__()
        self.config = config or LadderConfig()
        self.input_dim, self.num_classes = input_dim, num_classes
        dims = (input_dim, *(hidden_dims if hidden_dims is not None else
                             self.config.hidden_dims), num_classes)
        self.dims = tuple(dims)
        self.encoder = nn.ModuleList([
            EncoderLayer(n_in, n_out, top=i == len(dims) - 2,
                         eps=self.config.bn_eps)
            for i, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:]))])
        # decoder[i] maps reconstructed layer i+1 to layer i.
        self.decoder = nn.ModuleList([
            nn.Linear(n_in, n_out, bias=False)
            for n_in, n_out in zip(dims[1:], dims[:-1])])
        for layer in self.decoder:
            nn.init.normal_(layer.weight, std=layer.in_features ** -0.5)
        self.combinators = nn.ModuleList([
            NodewiseAMLP(size, self.config.combinator_hidden_dims,
                         self.config.combinator_init_std) for size in dims])
        self.register_buffer("bn_batches", torch.zeros(()))

    def _vector(self, x: Tensor) -> Tensor:
        x = x.flatten(start_dim=1)
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {x.shape[1]}")
        return x

    @torch.no_grad()
    def _update_clean_stats(self, x: Tensor) -> None:
        # Match reference counter=1 initially, clamped to 10 for EMA updates.
        self.bn_batches.add_(1).clamp_(max=10)
        momentum = self.bn_batches.reciprocal()
        h = x
        for layer in self.encoder:
            z, mean, var = _normalize(layer.linear(h), layer.eps)
            layer.running_mean.lerp_(mean, momentum)
            correction = x.shape[0] / max(x.shape[0] - 1, 1)
            layer.running_var.lerp_(var * correction, momentum)
            h = layer.activate(z)

    def forward(self, x: Tensor) -> Tensor:
        """Clean encoder logits, using frozen training statistics in eval mode."""
        h = self._vector(x)
        for layer in self.encoder:
            raw = layer.linear(h)
            if self.training:
                z, _, _ = _normalize(raw, layer.eps)
            else:
                z = (raw - layer.running_mean) * torch.rsqrt(
                    layer.running_var + layer.eps)
            h = layer.activate(z)
        return h

    def loss_components(self, x: Tensor, y: Tensor,
                        x_unlabeled: Tensor | None = None, *,
                        update_running_stats: bool = True) -> dict[str, Tensor]:
        """Return differentiable total/CE/reconstruction losses and noisy logits.

No clean encoder autograd graph is needed: all hidden reconstruction weights
are zero in the published fully supervised AMLP configuration.
        """
        x = self._vector(x)
        if x.shape[0] < 2:
            raise ValueError("Ladder training requires at least 2 examples per batch")
        if update_running_stats:
            self._update_clean_stats(x)
        paired = x_unlabeled is not None
        reconstruction_target = self._vector(x_unlabeled) if paired else x
        if reconstruction_target.shape[0] < 2:
            raise ValueError("Reconstruction batch requires at least 2 examples")
        label_count = x.shape[0]
        h = torch.cat((x, reconstruction_target), dim=0) if paired else x
        h = h + torch.randn_like(h) * self.config.noise_std
        lateral = [h[label_count:] if paired else h]
        for layer in self.encoder:
            raw = layer.linear(h)
            if paired:
                # Reference normalizes labeled/unlabeled streams separately.
                zl, _, _ = _normalize(raw[:label_count], layer.eps)
                zu, _, _ = _normalize(raw[label_count:], layer.eps)
                z = torch.cat((zl, zu), dim=0)
            else:
                z, _, _ = _normalize(raw, layer.eps)
            z = z + torch.randn_like(z) * self.config.noise_std
            lateral.append(z[label_count:] if paired else z)
            h = layer.activate(z)
        noisy_logits = h[:label_count] if paired else h
        top_logits = h[label_count:] if paired else h
        # The top decoder receives the normalized noisy classifier softmax.
        vertical, _, _ = _normalize(F.softmax(top_logits, dim=-1),
                                     self.config.bn_eps)
        reconstruction = self.combinators[-1](lateral[-1], vertical)
        for index in range(len(self.decoder) - 1, -1, -1):
            vertical, _, _ = _normalize(self.decoder[index](reconstruction),
                                         self.config.bn_eps)
            reconstruction = self.combinators[index](lateral[index], vertical)
        ce = F.cross_entropy(noisy_logits, y.long())
        mse = F.mse_loss(reconstruction, reconstruction_target)
        penalty = self.config.input_reconstruction_weight * mse
        return {"loss": ce + penalty, "cross_entropy": ce,
                "reconstruction_mse": mse, "reconstruction_penalty": penalty,
                "noisy_logits": noisy_logits}

    def loss(self, x: Tensor, y: Tensor, x_unlabeled: Tensor | None = None,
             *, update_running_stats: bool = True) -> Tensor:
        return self.loss_components(x, y, x_unlabeled,
                                    update_running_stats=update_running_stats)["loss"]

    @torch.no_grad()
    def calibrate_bn(self, train_x: Tensor, batch_size: int = 100, *,
                     shuffle_seed: int = 1) -> None:
        """Reference-style clean BN calibration using training data only.

Average minibatch means and unbiased within-minibatch variances over one
pass. Deeper layers use minibatch-normalized activations, matching the actual
CuriousAI FinalTestMonitoring graph. Evaluation uses the resulting fixed stats.
Input can live on CPU; batches are moved to the model's device.
        """
        if len(train_x) < 2 or batch_size < 2:
            raise ValueError("BN calibration requires at least 2 training samples")
        device = next(self.parameters()).device
        generator = torch.Generator(device="cpu").manual_seed(shuffle_seed)
        indices = torch.randperm(len(train_x), generator=generator)
        means = [torch.zeros_like(layer.running_mean) for layer in self.encoder]
        variances = [torch.zeros_like(layer.running_var) for layer in self.encoder]
        batches = 0
        for batch_indices in indices.split(batch_size):
            if len(batch_indices) < 2:
                continue
            batch_indices = batch_indices.to(train_x.device)
            h = self._vector(train_x[batch_indices].to(device))
            batches += 1
            correction = len(h) / (len(h) - 1)
            for index, layer in enumerate(self.encoder):
                z, mean, var = _normalize(layer.linear(h), layer.eps)
                means[index].add_(mean)
                variances[index].add_(var * correction)
                h = layer.activate(z)
        for index, layer in enumerate(self.encoder):
            layer.running_mean.copy_(means[index] / batches)
            layer.running_var.copy_(variances[index] / batches)
        self.bn_batches.fill_(min(batches, 10))

    @torch.no_grad()
    def permute_inputs_(self, permutation: Tensor) -> "LadderAMLP":
        """Reindex all input-facing parameters for x[:, permutation].

Useful for checking permutation equivariance of the complete objective;
the dataset pipeline must still use the same permutation on train and test.
        """
        permutation = permutation.to(self.encoder[0].linear.weight.device)
        if permutation.numel() != self.input_dim or not torch.equal(
                permutation.sort().values, torch.arange(self.input_dim,
                                                        device=permutation.device)):
            raise ValueError("Expected a permutation of input coordinate indices")
        self.encoder[0].linear.weight.copy_(
            self.encoder[0].linear.weight[:, permutation].clone())
        self.decoder[0].weight.copy_(self.decoder[0].weight[permutation].clone())
        for parameter in self.combinators[0].parameters():
            parameter.copy_(parameter[permutation].clone())
        return self
