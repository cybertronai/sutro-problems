"""Bounded architecture search, seed replication, and untouched-test evaluation.

All tuning uses one fixed stratified 80/20 split of the supplied training set.
Finalists repeat initialization/minibatch seeds, not the validation split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from mnist.models import build_model, candidate_configs


def split_indices(labels, seed=4701):
    """Stratify with largest-remainder allocation to exactly 20% validation."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(labels, minlength=10)
    target = int(round(len(labels) * .2))
    allocations = np.floor(counts * .2).astype(int)
    order = np.argsort(-(counts * .2 - allocations), kind="stable")
    allocations[order[:target - allocations.sum()]] += 1
    fit, val = [], []
    for label in range(10):
        idx = rng.permutation(np.flatnonzero(labels == label))
        val.extend(idx[:allocations[label]])
        fit.extend(idx[allocations[label]:])
    return rng.permutation(fit), rng.permutation(val)


def evaluate(model, x, y):
    model.eval()
    loss, correct, predictions = 0., 0, []
    with torch.inference_mode():
        for start in range(0, len(x), 1024):
            logits = model(x[start:start + 1024])
            batch_y = y[start:start + 1024]
            loss += F.cross_entropy(logits, batch_y, reduction="sum").item()
            pred = logits.argmax(1)
            correct += (pred == batch_y).sum().item()
            predictions.append(pred.cpu().numpy())
    return {"loss": loss / len(x), "accuracy": correct / len(x)}, np.concatenate(predictions)


def train_one(config, *, tier, seed, phase, x, y, val, output, tracking,
              epochs=None, schedule_epochs=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if x.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_model(config, x.shape[-1]).to(x.device)
    # Statistics are derived from the fit portion only, or all train for refit.
    mean, std = x.mean().item(), x.std(unbiased=False).item()
    std = max(std, 1e-6)
    normalized = (x - mean) / std
    vx, vy = ((val[0] - mean) / std, val[1]) if val is not None else (None, None)
    count_epochs = int(epochs or config["epochs"])
    schedule_epochs = int(schedule_epochs or config["epochs"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=schedule_epochs, eta_min=config["lr"] * .02)
    name = f"{tier}-{phase}-{config['trial_id']}-s{seed}"
    run = wandb.init(**tracking, name=name, job_type=phase,
                     tags=[tier, phase, config["architecture"]],
                     config={**config, "tier": tier, "seed": seed, "phase": phase,
                             "dataset_sha256": os.environ.get("MNIST_DATASET_SHA256"),
                             "training_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "models_code_sha256": hashlib.sha256(Path(__file__).with_name("models.py").read_bytes()).hexdigest(),
                             "fit_examples": len(x), "val_examples": len(vy) if vy is not None else 0,
                             "actual_epochs": count_epochs, "schedule_epochs": schedule_epochs,
                             "normalization_mean": mean, "normalization_std": std,
                             "optimizer": "AdamW", "scheduler": "cosine_min_2pct",
                             "augmentation": "none", "precision": "float32",
                             "validation_split_seed": 4701,
                             "dataset_seed": int(os.environ.get("MNIST_DATASET_SEED", "20260910"))},
                     settings=wandb.Settings(quiet=True, disable_git=True,
                                             x_disable_stats=True))
    run.define_metric("epoch")
    for key in ("train/*", "val/*", "learning_rate", "elapsed_seconds"):
        run.define_metric(key, step_metric="epoch")
    parameters = sum(p.numel() for p in model.parameters())
    run.summary.update({"parameters": parameters, "model": repr(model)})
    history, best = [], None
    started = time.monotonic()
    for epoch in range(1, count_epochs + 1):
        model.train()
        permutation = torch.randperm(len(x), device=x.device)
        lr = optimizer.param_groups[0]["lr"]
        for idx in permutation.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(normalized[idx]), y[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_metrics, _ = evaluate(model, normalized, y)
        row = {"epoch": epoch, "learning_rate": lr,
               "elapsed_seconds": time.monotonic() - started,
               **{f"train/{k}": v for k, v in train_metrics.items()}}
        if val is not None:
            val_metrics, _ = evaluate(model, vx, vy)
            row.update({f"val/{k}": v for k, v in val_metrics.items()})
            rank = (-row["val/accuracy"], row["val/loss"])
            if best is None or rank < (-best["val/accuracy"], best["val/loss"]):
                best = dict(row)
        if not all(np.isfinite(v) for v in row.values()):
            run.finish(exit_code=1)
            raise RuntimeError(f"Nonfinite metric in {name}")
        run.log(row)
        history.append(row)
    result = {"name": name, "config": config, "seed": seed, "phase": phase,
              "parameters": parameters, "epochs": count_epochs,
              "schedule_epochs": schedule_epochs, "normalization": {"mean": mean, "std": std},
              "best": best, "last": history[-1], "history": history,
              "wandb_url": run.url, "wandb_id": run.id}
    if best:
        run.summary.update({"best_val_accuracy": best["val/accuracy"],
                            "best_val_loss": best["val/loss"], "best_epoch": best["epoch"]})
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    if phase == "final":
        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint = checkpoint_dir / f"{name}.pt"
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "config": config, "image_size": x.shape[-1], "normalization": result["normalization"],
                    "epochs": count_epochs, "seed": seed}, checkpoint)
        result["checkpoint"] = str(checkpoint.relative_to(output))
    print(json.dumps({"run": name, "best_val_accuracy": best["val/accuracy"] if best else None,
                      "seconds": round(time.monotonic() - started, 2), "url": run.url}), flush=True)
    return model, run, result


def run_suite(tier, data_dir, output, entity="yaroslavvb", project="sutro-mnist-tiers",
              group="mnist-competition-v2", smoke=False, device="cuda"):
    output, data_dir = Path(output), Path(data_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    archive_path = data_dir / f"{tier}.npz"
    os.environ["MNIST_DATASET_SHA256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest["tiers"][tier]["sha256"] != os.environ["MNIST_DATASET_SHA256"]:
        raise ValueError(f"Dataset archive does not match its manifest: {archive_path}")
    archived_manifest = output / "dataset_manifest.json"
    if archived_manifest.exists() and json.loads(archived_manifest.read_text()) != manifest:
        raise ValueError(f"Output contains results from a different dataset: {output}")
    archived_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.environ["MNIST_DATASET_SEED"] = str(manifest["seed"])
    # Do not load test arrays during tuning.
    with np.load(archive_path, allow_pickle=False) as archive:
        images, labels = archive["train_images"], archive["train_labels"]
        source_indices = archive["train_indices"]
    fit, validation = split_indices(labels)
    (output / "validation_split.json").write_text(json.dumps({
        "seed": 4701, "fit_positions": fit.tolist(), "val_positions": validation.tolist(),
        "fit_source_indices": source_indices[fit].tolist(),
        "val_source_indices": source_indices[validation].tolist()}, indent=2) + "\n")
    x = torch.from_numpy(images).to(device)
    y = torch.from_numpy(labels).long().to(device)
    tracking = {"entity": entity, "project": project, "group": group,
                "dir": str(output), "mode": "online"}
    candidates = candidate_configs(tier)
    if smoke:
        candidates = [{**c, "epochs": 3} for c in (candidates[1], candidates[-1])]
    runs = []
    for config in candidates:
        model, run, result = train_one(config, tier=tier, seed=11, phase="search",
            x=x[fit], y=y[fit], val=(x[validation], y[validation]), output=output, tracking=tracking)
        run.finish()
        runs.append(result)
        del model
    ranked = sorted(runs, key=lambda r: (-r["best"]["val/accuracy"], r["best"]["val/loss"], r["parameters"]))
    finalists = ranked[:3]
    if not smoke:
        for candidate in finalists:
            for seed in (22, 33):
                model, run, result = train_one(candidate["config"], tier=tier, seed=seed,
                    phase="replicate", x=x[fit], y=y[fit], val=(x[validation], y[validation]),
                    output=output, tracking=tracking)
                run.finish()
                runs.append(result)
                del model
    aggregation = []
    for candidate in finalists:
        reps = [r for r in runs if r["config"]["trial_id"] == candidate["config"]["trial_id"]]
        aggregation.append({"trial_id": candidate["config"]["trial_id"], "config": candidate["config"],
            "mean_val_accuracy": statistics.mean(r["best"]["val/accuracy"] for r in reps),
            "mean_val_loss": statistics.mean(r["best"]["val/loss"] for r in reps),
            "val_accuracy_std": statistics.stdev(r["best"]["val/accuracy"] for r in reps) if len(reps)>1 else 0.,
            "selected_epochs": int(statistics.median(r["best"]["epoch"] for r in reps)),
            "parameters": candidate["parameters"], "seeds": [r["seed"] for r in reps]})
    selection = sorted(aggregation, key=lambda r: (-r["mean_val_accuracy"], r["mean_val_loss"], r["parameters"]))[0]
    # Freeze model and stopping epoch before opening the tier's test examples.
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    finals = []
    if not smoke:
        with np.load(archive_path, allow_pickle=False) as archive:
            tx = torch.from_numpy(archive["test_images"]).to(device)
            ty = torch.from_numpy(archive["test_labels"]).long().to(device)
        for seed in (101, 102, 103):
            model, run, result = train_one(selection["config"], tier=tier, seed=seed, phase="final",
                x=x, y=y, val=None, output=output, tracking=tracking,
                epochs=selection["selected_epochs"], schedule_epochs=selection["config"]["epochs"])
            norm = result["normalization"]
            test, predictions = evaluate(model, (tx-norm["mean"])/norm["std"], ty)
            result["test"] = test
            run.log({f"test/{k}": v for k, v in test.items()})
            run.summary.update({f"test_{k}": v for k, v in test.items()})
            run.summary.update({"selection_mean_val_accuracy": selection["mean_val_accuracy"],
                                "test_examples": len(ty), "checkpoint_selection": "fixed_before_test"})
            artifact = wandb.Artifact(f"mnist-{tier}-seed-{seed}", type="model",
                                      metadata={"config": selection["config"], "test": test})
            artifact.add_file(output / result["checkpoint"])
            run.log_artifact(artifact)
            run.finish()
            np.savez_compressed(output / f"predictions-seed-{seed}.npz", predictions=predictions,
                                labels=ty.cpu().numpy())
            (output / f"{result['name']}.json").write_text(json.dumps(result, indent=2) + "\n")
            finals.append(result)
            print(json.dumps({"tier": tier, "seed": seed, "test": test}), flush=True)
            del model
    summary = {"tier": tier, "smoke": smoke, "group": group, "entity": entity, "project": project,
        "dataset_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "dataset_seed": manifest["seed"], "dataset_profile": manifest.get("profile"),
        "hardware": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "wandb_version": wandb.__version__, "python_version": platform.python_version(),
        "selection": selection, "finalists": aggregation,
        "search_runs": [{k:v for k,v in r.items() if k != "history"} for r in runs],
        "final_runs": [{k:v for k,v in r.items() if k != "history"} for r in finals]}
    if finals:
        values = [r["test"]["accuracy"] for r in finals]
        summary.update({"test_accuracy_mean": statistics.mean(values),
                        "test_accuracy_std": statistics.stdev(values)})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["small", "medium"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("mnist/data"))
    parser.add_argument("--output", type=Path,
                        help="Output directory; defaults to mnist/results/local/<tier>")
    parser.add_argument("--entity", default="yaroslavvb")
    parser.add_argument("--project", default="sutro-mnist-tiers")
    parser.add_argument("--group", default="mnist-competition-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("mnist/results/local") / args.tier
    run_suite(**vars(args))
