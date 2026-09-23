"""Independently audit all 56 fits and replay two checkpoints without scoring.

Uses NumPy/stdlib to regenerate split and pixel permutations, validate metadata,
and check predictions against saved logits. A separate CPU process loads each
selected checkpoint with the frozen model source and reproduces 100 logits.
The original MNIST test-label file is never opened. Transfer labels are selected
only at training indices for input-hash checks; no query-label scoring occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_SUITE = Path(os.environ.get("AMINIST21_ROOT", "/Users/yaroslavvb/git/aminist-21-validation"))
NAMES = ["kmnist", "emnist_letters_aj", "qmnist_recovered", "fashion_mnist", "cifar10"]
EXPECTED_CONFIG = {
    "seed": 11, "permutation_seed": 20260923, "epochs": 150, "decay_start": 100,
    "batch_size": 100, "learning_rate": 0.002, "device": "cuda", "cuda_graph": True,
    "independent_reconstruction_stream": True, "checkpoint_selection": "last_fixed_epoch",
}
IMAGE = "ghcr.io/ab-10/wikitext-bench@sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    a = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    return hashlib.sha256(a.tobytes()).hexdigest()


def split_indices(count, draw):
    # Separate implementation: never call the training harness splitter.
    seed = np.random.SeedSequence(20261101 + draw).spawn(2)[0]
    indices = np.random.Generator(np.random.PCG64(seed)).permutation(count)
    return indices[:10000], indices[10000:20000]


def pixel_permutation(dim):
    return np.random.Generator(np.random.PCG64(20260923)).permutation(dim)


def official_arrays(query_only=False):
    folder = ROOT.parent / "official-mnist-sample-curve-20260922/raw/source"
    specs = {
        "test_images": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3", 16, (10000, 1, 28, 28)),
    }
    if not query_only:
        specs.update({
            "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873", 16, (60000, 1, 28, 28)),
            "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432", 8, (60000,)),
        })
    result = {}
    for key, (filename, md5, offset, shape) in specs.items():
        raw = (folder / filename).read_bytes()
        require(hashlib.md5(raw).hexdigest() == md5, "Wrong official MNIST source: " + filename)
        values = np.frombuffer(gzip.decompress(raw), dtype=np.uint8, offset=offset).reshape(shape)
        result[key] = values.astype(np.int64) if key == "train_labels" else values.astype(np.float32) / 255
    return result


def pool_spec(suite, name):
    manifest = read(suite / "datasets/manifest.json")
    return next(row for row in manifest["datasets"] if row["id"] == name)


def expected_parameter_count(dim):
    dims = [dim, 1000, 500, 250, 250, 250, 10]
    # Encoder/decoder matrices + betas + top gamma + nodewise 3→2→2→1 combinators.
    return 2 * sum(a * b for a, b in zip(dims[:-1], dims[1:])) + sum(dims[1:]) + 10 + 17 * sum(dims)


def audit_record(path, sources, input_hashes, dataset, draw, train_count, dim, indices=None):
    meta = read(path)
    detail, provenance = meta["metadata"], meta["provenance"]
    prefix = f"{dataset}/{draw}: "
    require(meta["config"] == detail["config"] == EXPECTED_CONFIG, prefix + "configuration differs")
    require(meta["adapter"] == "train.py:train_predict", prefix + "adapter differs")
    require(meta["adapter_source_sha256"] == sources["train.py"], prefix + "training source differs")
    require(provenance["sources"] == sources, prefix + "mixed sources")
    require(provenance["dataset"] == dataset and provenance["draw"] == draw, prefix + "identity differs")
    require(meta["input_sha256"] == provenance["input_sha256"] == input_hashes, prefix + "input hashes differ")
    require(meta["query_labels_supplied"] is False and detail["query_labels_supplied"] is False, prefix + "label contract differs")
    require(detail["fresh_weights_and_optimizer"] is True, prefix + "fresh initialization not declared")
    require(detail["input_dim"] == dim and detail["train_count"] == train_count and detail["query_count"] == 10000, prefix + "shape/count mismatch")
    require(detail["epochs_completed"] == 150 and detail["cuda_graph"] is True, prefix + "training incomplete/procedure changed")
    require(detail["precision"] == "float32; TF32 disabled", prefix + "precision changed")
    require(detail["checkpoint_selection"] == "final epoch, fixed before query evaluation", prefix + "checkpoint selection changed")
    require(detail["trainable_parameters"] == expected_parameter_count(dim), prefix + "architecture parameter count differs")
    require(meta["container_image"] == IMAGE, prefix + "container image differs")
    perm = pixel_permutation(dim)
    require(np.array_equal(np.asarray(detail["permutation"]), perm), prefix + "pixel permutation differs")
    require(detail["permutation_sha256"] == array_sha(perm), prefix + "pixel permutation hash differs")
    if indices is not None:
        fit, query = indices
        require(provenance["dataset_seed"] == 20261101 + draw, prefix + "dataset seed differs")
        require(provenance["train_indices_sha256"] == array_sha(fit), prefix + "training row hash differs")
        require(provenance["query_indices_sha256"] == array_sha(query), prefix + "query row hash differs")
        require(np.intersect1d(fit, query).size == 0, prefix + "train/query row overlap")
    history = detail["history"]
    require([row["epoch"] for row in history] == [1] + list(range(10, 151, 10)), prefix + "training history incomplete")
    for row in history:
        expected_lr = 0.002 * min(1.0, (150 - (row["epoch"] - 1)) / 50)
        require(abs(row["learning_rate"] - expected_lr) < 1e-12, prefix + "learning-rate history differs")
        require(np.isfinite(row["last_minibatch_loss"]), prefix + "nonfinite training loss")
    require(all(a["elapsed_training_seconds"] < b["elapsed_training_seconds"] for a, b in zip(history[:-1], history[1:])), prefix + "nonmonotone history times")
    require(meta["adapter_wall_seconds"] > 0 and detail["training_seconds"] > 0, prefix + "invalid timings")
    prediction_path = path.with_suffix(".npz")
    logit_path = path.with_name(path.stem + "-logits.npz")
    checkpoint_path = path.with_suffix(".pt")
    require(file_sha(prediction_path) == meta["prediction_file_sha256"], prefix + "predictions file changed")
    require(file_sha(logit_path) == meta["logits_file_sha256"], prefix + "logits file changed")
    require(file_sha(checkpoint_path) == meta["checkpoint_sha256"], prefix + "checkpoint changed")
    with np.load(prediction_path, allow_pickle=False) as f:
        require(f.files == ["predictions"], prefix + "unexpected prediction archive fields")
        predictions = f["predictions"]
    with np.load(logit_path, allow_pickle=False) as f:
        require(f.files == ["logits"], prefix + "unexpected logit archive fields")
        logits = f["logits"]
    require(predictions.shape == (10000,) and predictions.dtype == np.int64, prefix + "invalid prediction shape/type")
    require(logits.shape == (10000, 10) and logits.dtype == np.float32 and np.isfinite(logits).all(), prefix + "invalid logits")
    require(np.array_equal(logits.argmax(1), predictions), prefix + "saved predictions differ from logits")
    require(array_sha(predictions) == meta["predictions_sha256"], prefix + "prediction array hash differs")
    return {"dataset": dataset, "draw": draw, "metadata_sha256": file_sha(path),
            "input_sha256": input_hashes, "checkpoint_sha256": file_sha(checkpoint_path),
            "predictions_sha256": array_sha(predictions), "permutation_sha256": array_sha(perm),
            "completed_at_utc": meta["completed_at_utc"], "epochs_completed": 150,
            "gpu": meta["hardware"]["gpu"], "software": meta["software"]}


def checkpoint_replay(study, suite, record_path, tolerance):
    """Child-process branch: no training and no reading of query labels."""
    import torch
    meta = read(record_path)
    require(file_sha(record_path.with_suffix(".pt")) == meta["checkpoint_sha256"], "Checkpoint changed before replay")
    source_path = study / "source/model.py"
    require(file_sha(source_path) == meta["provenance"]["sources"]["model.py"], "Model source changed before replay")
    spec = importlib.util.spec_from_file_location("frozen_ladder_model", source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    checkpoint = torch.load(record_path.with_suffix(".pt"), map_location="cpu")
    require(checkpoint["config"] == EXPECTED_CONFIG and checkpoint["epochs_completed"] == 150, "Checkpoint config/epoch differs")
    dim = meta["metadata"]["input_dim"]
    require(checkpoint["input_dim"] == dim, "Checkpoint input dimension differs")
    perm = pixel_permutation(dim)
    require(np.array_equal(checkpoint["permutation"], perm), "Checkpoint permutation differs")
    dataset = meta["provenance"]["dataset"]
    if dataset == "official_mnist_60000_10000":
        query_images = official_arrays(query_only=True)["test_images"][:100]
    else:
        ps = pool_spec(suite, dataset)
        with np.load(suite / "data/pools" / ps["filename"], allow_pickle=False) as archive:
            _, indices = split_indices(ps["count"], meta["provenance"]["draw"])
            query_images = archive["images"][indices[:100]]
    model = module.LadderAMLP(input_dim=dim, num_classes=10)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    require(all(torch.isfinite(tensor).all().item() for tensor in model.state_dict().values()), "Checkpoint contains nonfinite state")
    require(sum(p.numel() for p in model.parameters()) == expected_parameter_count(dim), "Checkpoint parameter count differs")
    model.eval()
    values = query_images.reshape(100, dim)[:, perm].copy()
    with torch.inference_mode():
        reproduced = model(torch.from_numpy(values)).numpy()
    with np.load(record_path.with_name(record_path.stem + "-logits.npz"), allow_pickle=False) as f:
        original = f["logits"][:100]
    differences = np.abs(reproduced - original)
    max_abs = float(differences.max())
    prediction_mismatches = int(np.sum(reproduced.argmax(1) != original.argmax(1)))
    require(np.isfinite(reproduced).all() and max_abs <= tolerance, f"Checkpoint replay error {max_abs} exceeds tolerance {tolerance}")
    require(prediction_mismatches == 0, "Checkpoint replay predictions differ")
    return {"dataset": dataset, "draw": meta["provenance"]["draw"], "query_examples": 100,
            "logit_values": 1000, "device": "cpu", "python": platform.python_version(),
            "numpy": np.__version__, "torch": str(torch.__version__),
            "max_absolute_logit_error": max_abs, "mean_absolute_logit_error": float(differences.mean()),
            "absolute_tolerance": tolerance, "predictions_mismatched": prediction_mismatches,
            "checkpoint_sha256": file_sha(record_path.with_suffix(".pt")),
            "frozen_model_source_sha256": file_sha(source_path)}


def audit(study, suite, cpu_python, tolerance):
    execution = read(study / "execution.json")
    require(execution["status"] == "complete", "Wait for all planned training jobs to complete")
    transfer = study / "transfer"
    plan = read(transfer / "plan.json")
    official_plan = read(study / "official/plan.json")
    require(plan["config"] == official_plan["config"] == EXPECTED_CONFIG, "Frozen plan differs from intended configuration")
    require(plan["datasets"] == NAMES and plan["draws"] == list(range(11)), "Frozen task/draw selection differs")
    require(plan["seeds"] == list(range(20261101, 20261112)), "Frozen draw seeds differ")
    require(plan["train_count"] == plan["query_count"] == 10000, "Frozen sample counts differ")
    require(plan["fresh_weights_per_draw"] is True and plan["query_labels_supplied"] is False and official_plan["test_labels_supplied"] is False, "Frozen input/init contract differs")
    sources = official_plan["sources"]
    require(plan["sources"] == sources and set(sources) == {"model.py", "train.py", "run_study.py"}, "Frozen source set differs")
    for filename, digest in sources.items():
        require(file_sha(study / "source" / filename) == digest, "Frozen source changed: " + filename)
    manifest_path = suite / "datasets/manifest.json"
    require(file_sha(manifest_path) == plan["dataset_manifest_sha256"], "Transfer dataset manifest differs")
    frozen = read(transfer / "prediction-manifest.json")
    require(frozen["plan_sha256"] == file_sha(transfer / "plan.json"), "Frozen plan hash differs")
    require(frozen["dataset_manifest_sha256"] == file_sha(manifest_path), "Frozen dataset hash differs")
    require(frozen["datasets"] == NAMES and frozen["draws"] == list(range(11)) and frozen["complete_v1_suite"] is False, "Prediction manifest scope differs")
    expected_keys = [(name, draw) for name in NAMES for draw in range(11)]
    require([(r["dataset"], r["draw"]) for r in frozen["entries"]] == expected_keys, "Prediction manifest has missing/extra/reordered rows")
    for entry in frozen["entries"]:
        require(entry["dataset_seed"] == 20261101 + entry["draw"], "Frozen entry seed differs")
        for field in ("metadata", "prediction"):
            require(file_sha(transfer / entry[field + "_path"]) == entry[field + "_sha256"], "Frozen file changed")
    records = []
    arrays = official_arrays()
    official_hashes = {key: array_sha(values) for key, values in arrays.items()}
    require(official_plan["provenance"]["input_sha256"] == official_hashes, "Official plan inputs differ")
    records.append(audit_record(study / "official/mnist.json", sources, official_hashes,
                                "official_mnist_60000_10000", 0, 60000, 784))
    del arrays
    for name in NAMES:
        ps = pool_spec(suite, name)
        data_path = suite / "data/pools" / ps["filename"]
        require(file_sha(data_path) == ps["sha256"], "Prepared pool changed: " + name)
        with np.load(data_path, allow_pickle=False) as archive:
            images = archive["images"]
            native_hashes = archive["example_hashes"]
            require(images.shape == (ps["count"], 1, 9, 9), "Prepared pool shape differs")
            for draw in range(11):
                fit, query = split_indices(ps["count"], draw)
                input_hashes = {"train_images": array_sha(images[fit]),
                                "train_labels": array_sha(archive["labels"][fit]),
                                "test_images": array_sha(images[query])}
                require(len(np.unique(native_hashes[np.r_[fit, query]])) == 20000, "Exact native-image overlap")
                records.append(audit_record(transfer / name / f"draw-{draw:02d}.json", sources,
                    input_hashes, name, draw, 10000, 81, (fit, query)))
        del images, native_hashes
    require(len(records) == 56, "Wrong number of fits")
    started = datetime.fromisoformat(execution["started_at_utc"])
    frozen_at = datetime.fromisoformat(frozen["frozen_at_utc"])
    finished = datetime.fromisoformat(execution["finished_at_utc"])
    require(started <= frozen_at <= finished, "Execution and prediction-freeze timestamps disagree")
    require(all(started <= datetime.fromisoformat(row["completed_at_utc"]) <= frozen_at for row in records),
            "Some model completion falls outside the declared execution and global freeze")
    replays = []
    for path in (study / "official/mnist.json", transfer / "kmnist/draw-00.json"):
        process = subprocess.run([str(cpu_python), str(Path(__file__).resolve()), "--study", str(study),
            "--suite", str(suite), "--child-checkpoint", str(path), "--absolute-tolerance", str(tolerance)],
            check=True, capture_output=True, text=True)
        replays.append(json.loads(process.stdout))
    return {"audited_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "auditor_source_sha256": file_sha(__file__), "frozen_training_sources": sources,
        "transfer_dataset_manifest_sha256": file_sha(manifest_path),
        "transfer_prediction_manifest_sha256": file_sha(transfer / "prediction-manifest.json"),
        "execution_record_sha256": file_sha(study / "execution.json"),
        "official_plan_sha256": file_sha(study / "official/plan.json"),
        "transfer_plan_sha256": file_sha(transfer / "plan.json"),
        "fits_verified": 56, "predictions_verified_against_logits": 560000,
        "configuration": EXPECTED_CONFIG, "permutation_algorithm": "NumPy Generator(PCG64(20260923)).permutation(input_dim)",
        "draw_algorithm": "NumPy PCG64(SeedSequence(20261101+draw).spawn(2)[0]), first10000 train/next10000 query",
        "original_mnist_test_labels_opened": False, "query_labels_scored": False,
        "training_label_hashes_verified": True, "fresh_initialization_declarations_verified": True,
        "limitations": ["Fresh initialization is checked as a declaration plus the frozen source implementation; this audit does not rerun training.",
                       "Checkpoint replay uses CPU/PyTorch 2.2.2 versus GPU/PyTorch 2.5.1 and a smaller inference batch; numerical tolerance is recorded, not bitwise equivalence.",
                       "Only the first 100 queries of the official checkpoint and KMNIST draw 0 are independently inferred; all 560,000 saved predictions are checked against saved logits."],
        "checkpoint_replays": replays, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "results/all")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--cpu-python", type=Path, default=DEFAULT_SUITE / ".venv-cpu/bin/python")
    parser.add_argument("--absolute-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child-checkpoint", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(0 < args.absolute_tolerance <= 1e-4, "Replay tolerance must be positive and no larger than 1e-4")
    if args.child_checkpoint:
        print(json.dumps(checkpoint_replay(args.study, args.suite, args.child_checkpoint, args.absolute_tolerance), allow_nan=False))
        return
    result = audit(args.study, args.suite, args.cpu_python, args.absolute_tolerance)
    target = args.output or args.study / "audit.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "fits_verified": result["fits_verified"],
                      "checkpoint_replays": result["checkpoint_replays"]}, indent=2))


if __name__ == "__main__":
    main()
