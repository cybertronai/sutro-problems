"""Fixed, portable training recipes for Aminist 21 validation.

The only callable benchmark interface is ``train_predict``. Each call constructs
fresh model(s), optimizer(s), and seed-only shuffle/augmentation schedules. Query
images are evaluated only after all member training has finished. No query labels
are accepted, and there is no per-dataset selection or hyperparameter tuning.

The CNN recipes are native PyTorch references, not a bitwise port of Sutro's
ordered-arithmetic 97%-accuracy submission. The reversible core reconstructs its
activations during backward; its classifier and other training state are ordinary.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any

# cuBLAS reads this while creating its handles. Set it before importing torch;
# already-running CUDA callers must have selected a deterministic workspace.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

RECIPE_VERSION = 1
RECIPES: dict[str, dict[str, Any]] = {
    "linear-sgd": {
        "family": "linear", "epochs": 8, "learning_rate": 0.1,
        "batch_size": 128, "momentum": 0.9, "weight_decay": 0.0,
        "augmentation": "none", "members": 1,
    },
    "mlp64-sgd": {
        "family": "mlp", "hidden_widths": [64], "epochs": 8,
        "learning_rate": 0.05, "batch_size": 128, "momentum": 0.9,
        "weight_decay": 0.0001, "augmentation": "none", "members": 1,
    },
    "mlp256-sgd": {
        "family": "mlp", "hidden_widths": [256, 256], "epochs": 12,
        "learning_rate": 0.03, "batch_size": 128, "momentum": 0.9,
        "weight_decay": 0.0001, "augmentation": "none", "members": 1,
    },
    "cnn16-sgd": {
        "family": "cnn", "width": 16, "depth": 2, "head_width": 64,
        "epochs": 8, "learning_rate": 0.03, "batch_size": 128,
        "momentum": 0.9, "weight_decay": 0.0001,
        "augmentation": "mild_affine", "members": 1,
    },
    "cnn32-ensemble3": {
        "family": "cnn", "width": 32, "depth": 3, "head_width": 128,
        "epochs": 8, "learning_rate": 0.03, "batch_size": 128,
        "momentum": 0.9, "weight_decay": 0.0001,
        "augmentation": "mild_affine", "members": 3,
    },
    "reversible82-sgd": {
        "family": "reversible", "depth": 2, "state_width": 82,
        "alpha": 0.5, "epochs": 2, "learning_rate": 0.1,
        "batch_size": 128, "momentum": 0.9, "weight_decay": 0.0,
        "augmentation": "none", "members": 1,
    },
}


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class _Reconstruct(torch.autograd.Function):
    """Additive coupling with one saved endpoint and weight references."""

    @staticmethod
    def forward(ctx, inputs, alpha, *weights):
        left, right = inputs.chunk(2, dim=1)
        for index in range(0, len(weights), 2):
            left = left + alpha * F.linear(F.relu(right), weights[index])
            right = right + alpha * F.linear(F.relu(left), weights[index + 1])
        output = torch.cat((left, right), dim=1)
        ctx.alpha = alpha
        ctx.save_for_backward(output, *weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        output, *weights = ctx.saved_tensors
        left, right = output.detach().chunk(2, dim=1)
        dleft, dright = gradient.chunk(2, dim=1)
        parameter_gradients = [None] * len(weights)
        for index in range(len(weights) - 2, -1, -2):
            with torch.enable_grad():
                branch_input = left.detach().requires_grad_(True)
                branch = ctx.alpha * F.linear(F.relu(branch_input), weights[index + 1])
                derivative, weight_gradient = torch.autograd.grad(
                    branch, (branch_input, weights[index + 1]), dright)
            right = right - branch.detach()
            dleft = dleft + derivative
            parameter_gradients[index + 1] = weight_gradient
            with torch.enable_grad():
                branch_input = right.detach().requires_grad_(True)
                branch = ctx.alpha * F.linear(F.relu(branch_input), weights[index])
                derivative, weight_gradient = torch.autograd.grad(
                    branch, (branch_input, weights[index]), dleft)
            left = left - branch.detach()
            dright = dright + derivative
            parameter_gradients[index] = weight_gradient
        return (torch.cat((dleft, dright), dim=1), None, *parameter_gradients)


class ReversibleClassifier(nn.Module):
    """82-coordinate reversible core; non-reversible 10-class linear head."""

    def __init__(self, depth: int = 2, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.weights = nn.ParameterList(
            nn.Parameter(torch.empty(41, 41)) for _ in range(2 * depth))
        for weight in self.weights:
            nn.init.normal_(weight, mean=0.0, std=0.05 / math.sqrt(41))
        self.head = nn.Linear(82, 10, bias=True)

    def core(self, inputs, reconstruct: bool = True):
        if reconstruct and torch.is_grad_enabled():
            return _Reconstruct.apply(inputs, self.alpha, *self.weights)
        left, right = inputs.chunk(2, dim=1)
        for index in range(0, len(self.weights), 2):
            left = left + self.alpha * F.linear(F.relu(right), self.weights[index])
            right = right + self.alpha * F.linear(F.relu(left), self.weights[index + 1])
        return torch.cat((left, right), dim=1)

    def forward(self, images):
        inputs = F.pad(images.flatten(1), (0, 1), value=0.0)
        return self.head(self.core(inputs))


def _build_model(recipe: dict[str, Any], seed: int) -> nn.Module:
    # Initialize on CPU for a device-independent random stream. This avoids
    # perturbing caller RNG state and does not copy any previous checkpoint.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        if recipe["family"] == "reversible":
            return ReversibleClassifier(recipe["depth"], recipe["alpha"])
        if recipe["family"] in ("linear", "mlp"):
            layers: list[nn.Module] = [nn.Flatten()]
            width = 81
            for next_width in recipe.get("hidden_widths", []):
                layers.extend([nn.Linear(width, next_width), nn.ReLU()])
                width = next_width
            layers.append(nn.Linear(width, 10))
        elif recipe["family"] == "cnn":
            layers = []
            channels = 1
            for _ in range(recipe["depth"]):
                layers.extend([nn.Conv2d(channels, recipe["width"], 3,
                                         padding=1, bias=False), nn.ReLU()])
                channels = recipe["width"]
            layers.extend([nn.Flatten(), nn.Linear(channels * 81, recipe["head_width"]),
                           nn.ReLU(), nn.Linear(recipe["head_width"], 10)])
        else:
            raise ValueError(f"Unknown model family {recipe['family']!r}")
        model = nn.Sequential(*layers)
        for layer in model.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(layer.weight, a=0, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        return model


def _configure_device(device: str) -> torch.device:
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use device='cpu'")
    if device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":16:8", ":4096:8"):
        raise RuntimeError("Launch a fresh process with CUBLAS_WORKSPACE_CONFIG=:16:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device(device)


def _validate_inputs(train_images, train_labels, test_images):
    arrays = tuple(np.asarray(value) for value in (train_images, train_labels, test_images))
    train, labels, query = arrays
    for name, images in (("train_images", train), ("test_images", query)):
        if images.ndim != 4 or images.shape[1:] != (1, 9, 9) or len(images) == 0:
            raise ValueError(f"{name} must have nonempty shape (N,1,9,9)")
        if images.dtype != np.float32:
            raise TypeError(f"{name} must have dtype float32")
        if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError(f"{name} pixels must be finite and in [0,1]")
    if labels.shape != (len(train),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("train_labels must be an integer vector matching train_images")
    if labels.min() < 0 or labels.max() > 9:
        raise ValueError("train_labels must be in 0..9")
    return train, labels, query


def _affine_schedule(rng: np.random.Generator, count: int) -> np.ndarray:
    """Seed-only destination-to-source maps; shifts expressed in pixels."""
    angle = rng.uniform(-8.0, 8.0, count) * (math.pi / 180.0)
    scale = rng.uniform(0.94, 1.06, count)
    shift = rng.uniform(-0.35, 0.35, (count, 2))
    active = rng.random(count) < 0.5
    theta = np.zeros((count, 2, 3), dtype=np.float32)
    theta[:, 0, 0] = scale * np.cos(angle)
    theta[:, 0, 1] = -scale * np.sin(angle)
    theta[:, 1, 0] = scale * np.sin(angle)
    theta[:, 1, 1] = scale * np.cos(angle)
    theta[:, :, 2] = shift * (2.0 / 9.0)
    theta[~active] = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    return theta


def train_predict(train_images, train_labels, test_images, config):
    """Train from scratch and return ``(int64 predictions, JSON metadata)``.

    ``config`` accepts only ``recipe``, ``seed`` (default 11), and ``device``
    (default CPU). Hyperparameters are fixed in RECIPES, not dataset-dependent.
    Numerical reproducibility is expected within the same software/hardware
    environment; cross-device bitwise equality is not promised.
    """
    config = dict(config)
    unknown = set(config) - {"recipe", "seed", "device"}
    if unknown:
        raise ValueError(f"Fixed baseline config does not accept {sorted(unknown)}")
    recipe_id = config.get("recipe")
    if recipe_id not in RECIPES:
        raise ValueError(f"Unknown recipe {recipe_id!r}; choose one of {list(RECIPES)}")
    seed = config.get("seed", 11)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0,2**31)")
    seed = int(seed)
    recipe = RECIPES[recipe_id]
    train, labels, query = _validate_inputs(train_images, train_labels, test_images)
    device = _configure_device(config.get("device", "cpu"))
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    x = torch.from_numpy(np.ascontiguousarray(train)).to(device)
    y = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)).to(device)
    normalized = x * 4.0 - 0.5
    models = []
    members = []
    batch_size = recipe["batch_size"]
    # Original three-member seed convention generalized to a caller seed.
    member_seeds = [seed + 11 * index for index in range(recipe["members"])]
    for member_seed in member_seeds:
        model = _build_model(recipe, member_seed).to(device)
        initial_hash = _state_hash(model)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=recipe["learning_rate"],
                                    momentum=recipe["momentum"],
                                    weight_decay=recipe["weight_decay"], foreach=False)
        # Shuffling matches the reversible seed-only PCG64 convention. Affine
        # draws use their own stream so augmentation does not change shuffling.
        order_rng = np.random.Generator(np.random.PCG64(member_seed))
        affine_rng = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([member_seed, 20260922])))
        order_hashes, affine_hashes = [], []
        for _ in range(recipe["epochs"]):
            order = order_rng.permutation(len(train)).astype(np.int64)
            order_hashes.append(_hash_array(order))
            order_device = torch.from_numpy(order).to(device)
            if recipe["augmentation"] == "mild_affine":
                theta_array = _affine_schedule(affine_rng, len(train))
                affine_hashes.append(_hash_array(theta_array))
                theta = torch.from_numpy(theta_array).to(device)
            for first in range(0, len(train), batch_size):
                last = min(first + batch_size, len(train))
                indices = order_device[first:last]
                if recipe["augmentation"] == "mild_affine":
                    raw_batch = x.index_select(0, indices)
                    # No gradients are required through the input warp.
                    with torch.no_grad():
                        grid = F.affine_grid(theta[first:last], raw_batch.shape, align_corners=False)
                        batch = F.grid_sample(raw_batch, grid, mode="bilinear",
                                              padding_mode="zeros", align_corners=False)
                        batch = batch * 4.0 - 0.5
                else:
                    batch = normalized.index_select(0, indices)
                target = y.index_select(0, indices)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(batch), target, reduction="mean")
                loss.backward()
                optimizer.step()
        model.eval()
        members.append({
            "seed": member_seed,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "initial_state_sha256": initial_hash,
            "final_state_sha256": _state_hash(model),
            "epoch_order_sha256": order_hashes,
            "epoch_affine_sha256": affine_hashes,
            "training_updates": recipe["epochs"] * math.ceil(len(train) / batch_size),
            "final_batch_training_loss": float(loss.detach().cpu()),
        })
        models.append(model)
        del optimizer
    # All training has ended before any query-dependent model operation.
    predictions = np.empty(len(query), dtype=np.int64)
    logits_hash = hashlib.sha256()
    with torch.no_grad():
        for first in range(0, len(query), 512):
            last = min(first + 512, len(query))
            batch = torch.from_numpy(np.ascontiguousarray(query[first:last])).to(device)
            batch = batch * 4.0 - 0.5
            scores = torch.zeros((last - first, 10), dtype=torch.float32, device=device)
            for model in models:
                scores.add_(model(batch))
            cpu_scores = scores.cpu().numpy()
            if not np.isfinite(cpu_scores).all():
                raise FloatingPointError("Nonfinite prediction scores")
            logits_hash.update(cpu_scores.tobytes())
            predictions[first:last] = cpu_scores.argmax(axis=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metadata = {
        "recipe_id": recipe_id, "recipe_version": RECIPE_VERSION,
        "recipe": dict(recipe), "seed": seed, "member_seeds": member_seeds,
        "recipe_sha256": hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
        "device": str(device), "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "deterministic_algorithms": True, "tf32": False,
        "normalization": "4*x-0.5", "loss": "mean_cross_entropy",
        "precision": "float32", "query_batch_size": 512,
        "ensemble": "ordered float32 logit sum; lowest-class argmax ties",
        "n_train": len(train), "n_query": len(query),
        "input_sha256": dict(zip(("train_images", "train_labels", "test_images"),
                                  map(_hash_array, (train, labels, query)))),
        "predictions_sha256": _hash_array(predictions),
        "logits_sha256": logits_hash.hexdigest(), "members": members,
        "elapsed_seconds": elapsed,
        "timing_scope": "training, transfers, hashes and inference; excludes input validation",
        "query_inference_after_all_training": True,
    }
    return predictions, metadata
