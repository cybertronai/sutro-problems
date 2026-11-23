"""Reuse the frozen 97.758% MNIST learning procedure on new image datasets.

Inputs are already resized float32 images in [0, 1]. This module does not read
test labels, load checkpoints, tune hyperparameters, or carry learned state
between calls. ``draw_seed`` identifies dataset sampling only: the original
member/initialization/augmentation seeds remain 11, 22, and 33 on every draw.

The source under reference/ is copied byte-for-byte from the original commit.
Run on the original pinned CUDA image for exact numerical reproduction.
"""
from __future__ import annotations

import gc
import hashlib
import importlib
import json
import numbers
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference"
IMAGE_REF = (
    "ghcr.io/ab-10/wikitext-bench@"
    "sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f"
)


def array_hash(array):
    """Hash the little-endian, contiguous array payload as in the reference."""
    array = np.asarray(array)
    array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


def verify_reference():
    """Check the immutable code copies before importing or executing them."""
    provenance = json.loads((REFERENCE / "provenance.json").read_text())
    for name, expected in provenance["files"].items():
        actual = hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen reference source changed: {name}")
    return provenance


def _reference_trainer():
    # Preserve the original modules' unqualified imports without editing them.
    # Refuse a name collision instead of silently importing a different backend.
    backend = REFERENCE / "ordered_backend"
    for name in ("network", "schedule", "ops", "cpu_ref", "bn_ops", "bn_ref"):
        loaded = sys.modules.get(name)
        if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() != backend / f"{name}.py":
            raise RuntimeError(f"Conflicting module {name}; run the learner in an isolated process")
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    return importlib.import_module("network").Trainer


def _validate_inputs(train_images, train_labels, test_images, draw_seed):
    if not isinstance(draw_seed, numbers.Integral):
        raise TypeError("draw_seed must be an integer dataset-draw identifier")
    arrays = {}
    for name, value in (("train_images", train_images), ("test_images", test_images)):
        value = np.asarray(value)
        if value.dtype != np.float32 or value.ndim != 4 or value.shape[1:] != (1, 9, 9) or len(value) == 0:
            raise ValueError(f"{name} must be a nonempty float32 array of shape (N,1,9,9)")
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError(f"{name} must contain finite pixels in [0,1]")
        arrays[name] = np.ascontiguousarray(value)
    labels = np.asarray(train_labels)
    if labels.shape != (len(arrays["train_images"]),) or labels.dtype.kind not in "iu":
        raise ValueError("train_labels must be an integer vector with one label per training image")
    if labels.min() < 0 or labels.max() > 9:
        raise ValueError("train_labels must be remapped to class IDs 0 through 9")
    arrays["train_labels"] = np.ascontiguousarray(labels, dtype=np.int64)
    return arrays


def train_predict(train_images, train_labels, test_images, draw_seed, device="cuda"):
    """Return (int64 predictions, JSON-serializable reproducibility metadata).

    Arbitrary positive train/test counts are supported, including final partial
    minibatches. The architecture, update rule, eight epochs, three member seeds,
    and original ordered FP32 arithmetic remain fixed. Only row counts change.
    """
    started = time.perf_counter()
    provenance = verify_reference()
    arrays = _validate_inputs(train_images, train_labels, test_images, draw_seed)
    import torch
    import triton

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The original ordered Triton procedure requires a CUDA GPU")
    Trainer = _reference_trainer()
    config = json.loads((REFERENCE / "config.json").read_text())
    n_train, n_test = len(arrays["train_images"]), len(arrays["test_images"])
    config["n_train"], config["n_test"] = n_train, n_test
    members, compiled_hashes = [], {}
    ensemble_logits = np.zeros((n_test, 10), dtype=np.float32)
    invocation_seconds = 0.0

    with torch.cuda.device(target):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        for seed in config["member_seeds"]:
            trainer = Trainer(arrays["train_images"], arrays["train_labels"], config, seed)
            trainer.prepare(arrays["test_images"])
            torch.cuda.synchronize()
            invoked = time.perf_counter()
            trainer.invoke()  # Resets all parameters, velocities, and schedule position.
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - invoked
            invocation_seconds += elapsed
            output = trainer.outputs()
            scores = output["scores"]
            if scores.shape != (n_test, 10) or scores.dtype != np.float32 or not np.isfinite(scores).all():
                raise RuntimeError(f"Invalid final logits from member seed {seed}")
            # This is the unchanged ordered FP32 accumulation used by the
            # original accuracy runner, equivalent to reference/ensemble.py.
            ensemble_logits = np.add(ensemble_logits, scores, dtype=np.float32)
            members.append({
                "seed": seed,
                "epochs": config["epochs"],
                "fresh_state": True,
                "minibatches_per_epoch": (n_train + config["batch_size"] - 1) // config["batch_size"],
                "final_batch_size": ((n_train - 1) % config["batch_size"]) + 1,
                "parameter_count": sum(a.size for a in output["parameters"].values()),
                "initial_parameter_sha256": {name: array_hash(a) for name, a in trainer.initial_arrays.items()},
                "final_parameter_sha256": {name: array_hash(a) for name, a in output["parameters"].items()},
                "final_buffer_sha256": {name: array_hash(a) for name, a in output["buffers"].items()},
                "final_velocity_sha256": {name: array_hash(a) for name, a in output["velocities"].items()},
                "logits_sha256": array_hash(scores),
                "schedule_manifests": trainer.schedule_manifests,
                "training_prediction_seconds": elapsed,
            })
            for name, kernel in trainer.network.backend.compiled.items():
                ptx = kernel.asm["ptx"]
                if re.search(r"\bfma(?:\.[a-z0-9]+)*\.f32\b", ptx) or ".ftz." in ptx or re.search(r"\b(?:mma|wmma)\.", ptx):
                    raise RuntimeError(f"Compiled kernel changed the declared FP32 arithmetic: {name}")
                compiled_hashes[name] = hashlib.sha256(ptx.encode()).hexdigest()
            print(f"Draw {draw_seed}, member {seed}: completed {config['epochs']} fresh epochs", flush=True)
            del trainer, output, scores
            gc.collect()
            torch.cuda.empty_cache()
        predictions = ensemble_logits.argmax(axis=1).astype(np.int64)
        hardware = {"gpu": torch.cuda.get_device_name(target)}

    metadata = {
        "schema_version": 1,
        "draw_seed": int(draw_seed),
        "config": config,
        "source_commit": provenance["source_commit"],
        "reference_sha256": provenance["files"],
        "learner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {name: array_hash(a) for name, a in arrays.items()},
        "training_examples": n_train,
        "test_examples": n_test,
        "member_seeds": config["member_seeds"],
        "members": members,
        "test_labels_supplied": False,
        "fresh_state_per_member": True,
        "checkpoint_loaded": False,
        "draw_seed_role": "Dataset sampling identifier only; member RNG seeds stay fixed at 11,22,33",
        "ensemble_arithmetic": "Ordered FP32 addition starting from positive zero, member seeds11/22/33; ascending argmax",
        "predictions_sha256": array_hash(predictions),
        "ensemble_logits_sha256": array_hash(ensemble_logits),
        "training_seconds": invocation_seconds,
        "training_seconds_scope": "Sum of fresh initialization, eight training epochs, and test inference per member; excludes preparation/compilation, hashing, and CPU ensemble",
        "elapsed_including_preparation_seconds": time.perf_counter() - started,
        "ptx_sha256": compiled_hashes,
        "ptx_has_fp32_fma": False,
        "ptx_has_ftz": False,
        "ptx_has_tensorcore": False,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__), "triton": str(triton.__version__), "cuda": torch.version.cuda},
        "hardware": hardware,
    }
    return predictions, metadata


def verify_original_control(predictions, metadata):
    """Check exact original MNIST draw00 reproduction using hashes, no labels."""
    control = json.loads((REFERENCE / "control.json").read_text())
    checked = []
    for name in ("input_sha256", "predictions_sha256", "ensemble_logits_sha256", "member_seeds"):
        if metadata[name] != control[name]:
            raise AssertionError(f"Original MNIST control mismatch: {name}")
        checked.append(name)
    if array_hash(predictions) != control["predictions_sha256"]:
        raise AssertionError("Prediction payload disagrees with original control")
    for actual, expected in zip(metadata["members"], control["members"]):
        for key, value in expected.items():
            if actual[key] != value:
                raise AssertionError(f"Original MNIST member {expected['seed']} mismatch: {key}")
            checked.append(f"seed{expected['seed']}/{key}")
    return {"passed": True, "dataset_seed": control["dataset_seed"], "checked": checked,
            "correct": control["correct"], "total": control["test_count"]}
