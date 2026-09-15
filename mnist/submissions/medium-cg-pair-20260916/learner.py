"""Frozen NumPy-filter FP32 CG pair from server v80's bench_a100.CGPair.

This module never accepts test labels. CPU convolution is chunked to bound its
temporary feature map; the matrix construction, CG updates and score fusion
follow the measured server model. CUDA uses the same chunk size by default.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())


def filters(count: int = 512, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64(seed))
    weights = (generator.standard_normal((count, 9)) / 3).astype(np.float32)
    biases = (generator.standard_normal(count) * 0.1).astype(np.float32)
    return weights, biases


def zs(scores: torch.Tensor) -> torch.Tensor:
    """The source uses PyTorch's default sample standard deviation."""
    return (scores - scores.mean(1, keepdim=True)) / scores.std(1, keepdim=True)


class CGPair:
    def __init__(self, x, y, q, *, device="cpu", conv_batch_size=None):
        self.device = torch.device(device)
        self.conv_batch_size = (
            CONFIG["conv_batch_size"] if conv_batch_size is None else conv_batch_size
        )
        if self.conv_batch_size < 1:
            raise ValueError("conv_batch_size must be positive")
        weights, biases = filters(CONFIG["filters"], CONFIG["filter_seed"])
        self.w = torch.tensor(weights, device=self.device, dtype=torch.float32).view(
            CONFIG["filters"], 1, 3, 3
        )
        self.b = torch.tensor(biases, device=self.device, dtype=torch.float32)
        self.x = torch.tensor(x, device=self.device, dtype=torch.float32)
        self.q = torch.tensor(q, device=self.device, dtype=torch.float32)
        self.Y = torch.full((len(y), 10), -1.0, device=self.device, dtype=torch.float32)
        self.Y[
            torch.arange(len(y), device=self.device),
            torch.tensor(y, device=self.device, dtype=torch.int64),
        ] = 1

    def cg(self, matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Literal FP32 source recurrence: no convergence or denominator guards."""
        inverse_diagonal = 1.0 / matrix.diagonal()
        solution = torch.zeros_like(rhs)
        residual = rhs.clone()
        preconditioned = residual * inverse_diagonal[:, None]
        direction = preconditioned.clone()
        rz = (residual * preconditioned).sum(0)
        for _ in range(CONFIG["iterations"]):
            product = matrix @ direction
            alpha = rz / (direction * product).sum(0)
            solution = solution + alpha * direction
            residual = residual - alpha * product
            preconditioned = residual * inverse_diagonal[:, None]
            next_rz = (residual * preconditioned).sum(0)
            direction = preconditioned + (next_rz / rz) * direction
            rz = next_rz
        return solution

    def feats(self, pixels: torch.Tensor) -> torch.Tensor:
        # Split on examples only; each example sees every frozen filter.
        chunks = [
            F.avg_pool2d(
                F.relu(F.conv2d(chunk.reshape(-1, 1, 9, 9), self.w, self.b, padding=1)),
                3,
            ).flatten(1)
            for chunk in pixels.split(self.conv_batch_size)
        ]
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)

    def run(self, *, stage=None, return_scores=False):
        if stage is None:
            stage = lambda name: None
        u = torch.asin(torch.sqrt(self.x.clamp(0, 1)))
        uq = torch.asin(torch.sqrt(self.q.clamp(0, 1)))
        stage("pixel_transform")
        p, pq = self.feats(u), self.feats(uq)
        mean = p.mean(0)
        p = p - mean
        pq = pq - mean
        stage("convolution_features_and_centering")
        gram = p.T @ p
        gram.diagonal().add_(CONFIG["ridge_lambda_mean_diagonal"] * gram.diagonal().mean())
        rhs = p.T @ self.Y
        stage("ridge_system_assembly")
        kernel = torch.exp(-CONFIG["rbf_gamma"] * torch.cdist(u, u) ** 2)
        kernel.diagonal().add_(CONFIG["rbf_lambda"])
        query_kernel = torch.exp(-CONFIG["rbf_gamma"] * torch.cdist(uq, u) ** 2)
        stage("rbf_system_assembly")
        weights = self.cg(gram, rhs)
        stage("ridge_cg")
        dual = self.cg(kernel, self.Y)
        stage("rbf_cg")
        ridge_scores = pq @ weights
        rbf_scores = query_kernel @ dual
        combined = zs(ridge_scores) + zs(rbf_scores)
        predictions = combined.argmax(1)
        stage("score_and_predict")
        if return_scores:
            return predictions, (ridge_scores, rbf_scores, combined)
        return predictions


@torch.no_grad()
def train_predict(x, y, q, *, device="cpu", conv_batch_size=None, return_details=False):
    """Fit afresh and predict; detailed score finiteness is checked by the runner.

    With return_details=True, returns (NumPy int64 predictions, details). The
    details include CPU score arrays for validation outside the timed region.
    ``total_seconds`` includes all tensor preparation and host output copies.
    """
    device = torch.device(device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    start = previous = time.perf_counter()
    stages = {}

    def stage(name):
        nonlocal previous
        synchronize()
        now = time.perf_counter()
        stages[name] = now - previous
        previous = now

    if np.shape(x)[1:] != (81,) or np.shape(q)[1:] != (81,):
        raise ValueError("Expected training and query pixels with shape (N, 81)")
    if np.shape(y) != (len(x),):
        raise ValueError("Expected one training label per training image")
    if np.any(np.asarray(y) < 0) or np.any(np.asarray(y) > 9):
        raise ValueError("Training labels must be integers in [0, 9]")
    model = CGPair(x, y, q, device=device, conv_batch_size=conv_batch_size)
    stage("tensor_and_filter_preparation")
    predicted, scores = model.run(stage=stage, return_scores=True)
    predictions = predicted.cpu().numpy().copy()
    score_arrays = tuple(score.cpu().numpy().copy() for score in scores) if return_details else ()
    stage("host_output_transfer")
    elapsed = time.perf_counter() - start
    if return_details:
        return predictions, {
            "total_seconds": elapsed,
            "stage_seconds": stages,
            "score_arrays": score_arrays,
        }
    return predictions
