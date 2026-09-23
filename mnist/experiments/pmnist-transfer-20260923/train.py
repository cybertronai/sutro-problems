"""Fixed-recipe training. The learner never receives evaluation labels."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import LadderAMLP


DEFAULT_CONFIG = {
    "seed": 11, "permutation_seed": 20260923,
    "epochs": 150, "decay_start": 100, "batch_size": 100,
    "learning_rate": 0.002, "device": "cuda", "cuda_graph": True,
    "independent_reconstruction_stream": True,
    "checkpoint_selection": "last_fixed_epoch",
}


def train_predict(train_images, train_labels, test_images, config):
    predictions, metadata, _ = fit(train_images, train_labels, test_images, config)
    return predictions, metadata


def fit(train_images, train_labels, test_images, config, deadline=None):
    config = {**DEFAULT_CONFIG, **config}
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    n, d = len(train_images), int(np.prod(train_images.shape[1:]))
    batch = config["batch_size"]
    if n % batch or n < batch:
        raise ValueError("Training count must be a positive multiple of batch_size")
    permutation = np.random.default_rng(config["permutation_seed"]).permutation(d)
    # One deterministic feature permutation, shared by every train/query row.
    x = torch.as_tensor(train_images.reshape(n, d)[:, permutation].copy(), device=device)
    y = torch.as_tensor(train_labels.copy(), dtype=torch.long, device=device)
    model = LadderAMLP(input_dim=d, num_classes=10).to(device)
    model.train()
    use_graph = config["cuda_graph"] and device.type == "cuda"
    lr = torch.tensor(config["learning_rate"], device=device) if use_graph else config["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, capturable=use_graph,
                                 betas=(0.9, 0.999), eps=1e-8)
    history = []
    xb, yb, ub = x[:batch].clone(), y[:batch].clone(), x[-batch:].clone()

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(xb, yb, x_unlabeled=ub)
        loss.backward()
        optimizer.step()
        return loss

    graph = None
    if use_graph:
        # Warmup/capture is erased, including optimizer moments and RNG state.
        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        rng = torch.cuda.get_rng_state()
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(warmup)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = step()
        with torch.no_grad():
            model.load_state_dict(initial)
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        value.zero_()
        torch.cuda.set_rng_state(rng)
        del initial
    training_start = time.monotonic()
    for epoch in range(config["epochs"]):
        if deadline is not None and time.time() > deadline - 30:
            raise TimeoutError("Study deadline reached; incomplete fits are never scored")
        factor = min(1.0, (config["epochs"] - epoch) /
                     (config["epochs"] - config["decay_start"]))
        learning_rate = config["learning_rate"] * factor
        if use_graph:
            lr.fill_(learning_rate)
        else:
            optimizer.param_groups[0]["lr"] = learning_rate
        order = torch.randperm(n, device=device)
        reconstruction_order = (torch.randperm(n, device=device)
                                if config["independent_reconstruction_stream"] else order)
        for start in range(0, n, batch):
            xb.copy_(x[order[start:start+batch]])
            yb.copy_(y[order[start:start+batch]])
            ub.copy_(x[reconstruction_order[start:start+batch]])
            if graph is not None:
                graph.replay()
            else:
                loss = step()
        if epoch == 0 or (epoch+1) % 10 == 0 or epoch+1 == config["epochs"]:
            value = float(loss.detach().cpu())
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite training objective")
            row = {"epoch": epoch+1, "last_minibatch_loss": value,
                   "learning_rate": learning_rate,
                   "elapsed_training_seconds": time.monotonic()-training_start}
            history.append(row)
            print(json.dumps(row), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.monotonic()-training_start
    model.calibrate_bn(x, batch_size=batch)
    model.eval()
    logits = []
    with torch.inference_mode():
        for start in range(0, len(test_images), 1000):
            query = test_images[start:start+1000].reshape(-1,d)[:,permutation].copy()
            logits.append(model(torch.as_tensor(query, device=device)).cpu().numpy())
    logits = np.concatenate(logits)
    predictions = logits.argmax(1).astype(np.int64)
    checkpoint = io.BytesIO()
    torch.save({"state_dict": model.state_dict(), "config": config,
                "input_dim": d, "permutation": permutation,
                "epochs_completed": config["epochs"]}, checkpoint)
    metadata = {"config": config, "input_dim": d, "train_count": n,
                "query_count": len(test_images), "history": history,
                "epochs_completed": config["epochs"], "training_seconds": training_seconds,
                "total_seconds": time.monotonic()-started,
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
                "permutation": permutation.tolist(),
                "permutation_sha256": hashlib.sha256(permutation.astype("<i8").tobytes()).hexdigest(),
                "precision": "float32; TF32 disabled", "cuda_graph": use_graph,
                "fresh_weights_and_optimizer": True,
                "query_labels_supplied": False,
                "checkpoint_selection": "final epoch, fixed before query evaluation"}
    return predictions, metadata, {"checkpoint": checkpoint.getvalue(), "logits": logits}
