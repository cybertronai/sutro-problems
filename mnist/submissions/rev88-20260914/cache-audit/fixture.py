"""Isolated cache-profile fixture; imports, never edits, the frozen learner.

Set CUBLAS_WORKSPACE_CONFIG in the subprocess environment before starting.
The input archive contains training images/labels and query images only.
Named storage is a lower bound on allocations: CUDA graph private pools and
library workspaces need not have a surviving Python tensor reference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import torch

import learner
import energy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(task):
    """Count actual underlying CUDA storages once, retaining every alias name."""
    tensors = []
    for name, parameter in task.model.named_parameters():
        tensors.append(("parameters." + name, parameter))
        if parameter.grad is not None:
            tensors.append(("gradients." + name, parameter.grad))
    for group in ("initial", "velocity"):
        tensors.extend((f"{group}.{i}", tensor)
                       for i, tensor in enumerate(getattr(task, group)))
    # Includes all currently tensor-valued task attributes; list/module aliases
    # above are explicit so shared storage can never inflate the total.
    tensors.extend((name, value) for name, value in vars(task).items()
                   if isinstance(value, torch.Tensor))
    named, storages = [], {}
    for name, tensor in tensors:
        if not tensor.is_cuda:
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key not in storages:
            storages[key] = {"storage_id": len(storages), "device": str(tensor.device),
                             "address": storage.data_ptr(), "bytes": storage.nbytes(),
                             "aliases": []}
        record = storages[key]
        record["aliases"].append(name)
        named.append({"name": name, "storage_id": record["storage_id"],
                      "shape": list(tensor.shape), "stride": list(tensor.stride()),
                      "dtype": str(tensor.dtype), "storage_offset": tensor.storage_offset(),
                      "logical_bytes": tensor.numel() * tensor.element_size()})
    total = sum(record["bytes"] for record in storages.values())
    allocated = torch.cuda.memory_allocated()
    if total > allocated:
        raise AssertionError("Named storage exceeds allocator accounting")
    return {"named_tensors": named, "unique_storages": list(storages.values()),
            "unique_named_storage_bytes": total, "allocator_allocated_bytes": allocated,
            "allocated_bytes_without_named_storage_reference": allocated - total,
            "scope": "Actual CUDA storages reachable through named task tensors, model parameters/gradients, initial copies and momentum. Counts aliases once. Unnamed graph-pool and library allocations are not assigned to named buffers.",
            "is_per_kernel_working_set": False, "cache_residency_established": False}


def profiler_call(name):
    status = getattr(torch.cuda.cudart(), name)()
    if int(status) != 0:
        raise RuntimeError(f"{name} failed: {status}")


def compare_hashes(before, after):
    keys = ("input_hashes", "initial_parameter_sha256", "epoch_permutation_sha256",
            "final_parameter_sha256", "final_velocity_sha256", "state_vector_sha256",
            "predictions_sha256", "scores_sha256")
    for key in keys:
        if before[key] != after[key]:
            raise AssertionError("Fresh invocation changed " + key)
    return list(keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("measure", "profile"), required=True)
    parser.add_argument("--scope", choices=("task", "training"), default="task")
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--energy", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.energy and (args.mode != "measure" or args.scope != "task"):
        parser.error("--energy requires --mode measure --scope task")
    config = json.loads(args.config.read_text())
    if int(config["epochs"]) != 2:
        raise ValueError("This fixture preserves the frozen two-epoch procedure")
    with np.load(args.input, allow_pickle=False) as archive:
        if set(archive.files) != {"train_images", "train_labels", "test_images"}:
            raise ValueError("Input must contain exactly the three allowed arrays; no query labels")
        train, labels, query = (archive[key].copy() for key in
                                ("train_images", "train_labels", "test_images"))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")
    torch.cuda.init()
    torch.cuda.synchronize()
    entry_allocated = torch.cuda.memory_allocated()
    entry_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    task = learner.prepare_task(train, labels, query, config)
    torch.cuda.synchronize()
    after_prepare = torch.cuda.memory_allocated()
    task.run()
    torch.cuda.synchronize()
    # Capture before diagnostics, profiler warmup, or optional energy work.
    memory = {"allocated_before_prepare_bytes": entry_allocated,
              "reserved_before_prepare_bytes": entry_reserved,
              "allocated_after_prepare_bytes": after_prepare,
              "prepare_and_invoke_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "allocated_after_invoke_bytes": torch.cuda.memory_allocated(),
              "reserved_after_invoke_bytes": torch.cuda.memory_reserved(),
              "prepare_and_invoke_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "scope": "Fresh task preparation including CUDA graph capture plus one complete reset/train/predict invocation; tensor allocator only, excluding driver/context. Recorded before output diagnostics and profiling warmup."}
    memory["added_peak_over_entry_allocated_bytes"] = memory["prepare_and_invoke_peak_allocated_bytes"] - entry_allocated
    storage_inventory = inventory(task)
    _, reference = task.outputs()
    profile = None
    if args.mode == "profile":
        for _ in range(2):
            task.run()
        torch.cuda.synchronize()
        if args.scope == "training":
            task.reset()
            torch.cuda.synchronize()
        profiler_call("cudaProfilerStart")
        try:
            if args.scope == "task":
                for _ in range(args.repeats):
                    task.run()
            else:
                # Exactly the original trajectory: one reset outside range,
                # then the existing two resident epoch graphs and their orders.
                for epoch in range(task.config["epochs"]):
                    task.order.copy_(task.orders[epoch])
                    task.epoch_graph.replay()
            torch.cuda.synchronize()
        finally:
            profiler_call("cudaProfilerStop")
        if args.scope == "training":
            # Obtain comparable predictions without including inference in range.
            task.predict_graph.replay()
            torch.cuda.synchronize()
        profile = {"scope": args.scope, "warmup_complete_tasks": 2,
                   "requested_repeats": args.repeats,
                   "fresh_tasks_in_range": args.repeats if args.scope == "task" else 0,
                   "fresh_training_sequences_in_range": args.repeats if args.scope == "task" else 1,
                   "epochs_per_sequence": task.config["epochs"],
                   "training_scope_ignores_repeats": args.scope == "training",
                   "reset_inside_range": args.scope == "task",
                   "prediction_inside_range": args.scope == "task",
                   "epoch_order_copies_inside_range": True,
                   "output_serialization_inside_range": False,
                   "cpu_validation_inside_range": False,
                   "synchronized_before_stop": True}
    measured_energy = energy.measure(task, trials=3) if args.energy else None
    predictions, metadata = task.outputs()
    matched = compare_hashes(reference, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output.with_name(args.output.stem + ".predictions.npy")
    np.save(prediction_path, predictions, allow_pickle=False)
    result = {"schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "mode": args.mode, "requested_scope": args.scope, "config": config,
              "input_archive_sha256": sha(args.input), "config_file_sha256": sha(args.config),
              "source_sha256": {"fixture.py": sha(__file__), "learner.py": sha(learner.__file__),
                                "energy.py": sha(energy.__file__)},
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "environment_was_set_externally": True,
              "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                           "cuda": torch.version.cuda, "numpy": np.__version__},
              "gpu": torch.cuda.get_device_name(), "cuda_device_index": torch.cuda.current_device(),
              "memory": memory, "buffer_inventory": storage_inventory,
              "cache_reference": {"bytes": 40 * 1024**2, "unit": "40 MiB",
                                  "meaning": "A100 L2 nominal capacity reference, not measured cache residency or a per-kernel working set"},
              "profile": profile, "energy": measured_energy, "metadata": metadata,
              "same_process_reference_hashes_matched": matched,
              "frozen_qualification_comparison": "Must be performed independently by the orchestrator",
              "predictions_file": prediction_path.name,
              "predictions_file_sha256": sha(prediction_path),
              "test_labels_read": False}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Completed {args.mode}/{args.scope}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
