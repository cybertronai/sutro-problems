"""RevNet-4 with a lossless ReLU lift on MNIST-medium, 3% error target.

The learner is the 13 Sep 2026 reversible-network study's selected model,
unchanged: the files under source/ are byte-identical to the code that scored
97.42% on that study's confirmation draws. This runner only supplies the
current dataset protocol (generator profile medium-error-targets-v1), the
prediction freeze, official scoring and the A100 measurement.

Local phases (any Python with numpy and modal; run from the repository root):
    python mnist/submissions/revnet4-lift-20261123/run.py protocol
    python mnist/submissions/revnet4-lift-20261123/run.py prepare
    python mnist/submissions/revnet4-lift-20261123/run.py score
Modal phases (A100-40GB):
    modal run mnist/submissions/revnet4-lift-20261123/run.py --phase selftest
    modal run mnist/submissions/revnet4-lift-20261123/run.py --phase fit
    modal run mnist/submissions/revnet4-lift-20261123/run.py --phase energy
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import modal

HERE = Path(__file__).resolve().parent
NAME = "revnet4-lift-20261123"
ARCHITECTURE = "rev32_b4_lift_relu"
SEEDS = list(range(2026112300, 2026112311))  # eleven predeclared dataset seeds
TARGET_ERROR_PERCENT = 3
REQUIRED_CORRECT = 106_700  # ceil(110,000 * 0.97)
IMAGE_REF = ("ghcr.io/ab-10/wikitext-bench@"
             "sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f")
SOURCE_FILES = [
    "reversible-memory/ordered_learner.py",
    "reversible-memory/ordered_models.py",
    "reversible-refinement/models.py",
    "reversible-refinement/extra_ops.py",
    "reversible-refinement/protocol.json",
    "procedure-transfer/learner.py",
    "procedure-transfer/reference/config.json",
    "procedure-transfer/reference/ensemble.py",
    "procedure-transfer/reference/provenance.json",
    "procedure-transfer/reference/ordered_backend/network.py",
    "procedure-transfer/reference/ordered_backend/ops.py",
    "procedure-transfer/reference/ordered_backend/schedule.py",
    "procedure-transfer/reference/ordered_backend/cpu_ref.py",
    "procedure-transfer/reference/ordered_backend/bn_ops.py",
    "procedure-transfer/reference/ordered_backend/bn_ref.py",
]

image = (modal.Image.from_registry(IMAGE_REF).apt_install("gcc")
         .pip_install("numpy==2.2.6", "nvidia-ml-py==12.560.30")
         .env({"PYTHONPATH": "/workspace/reversible-memory:/workspace/reversible-refinement:/workspace/tools",
               "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}))
if modal.is_local():
    image = (image.add_local_dir(HERE / "source", "/workspace")
             .add_local_file(HERE / "nvml_energy.py", "/workspace/tools/nvml_energy.py"))
app = modal.App(f"sutro-mnist-{NAME}")
GPU = dict(image=image, gpu="A100-40GB", cpu=4, memory=16384, retries=0)


# ---------------------------------------------------------------- helpers
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    """SHA-256 of the little-endian C-order payload (the generator's and learner's rule)."""
    import numpy as np
    array = np.asarray(array)
    array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    """Write once; an existing file must hold the same value."""
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"Refusing to overwrite a different frozen file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def repo_root():
    return HERE.parents[2]


def data_root():
    return repo_root() / "mnist" / "data" / NAME


def draw_dir(index):
    return data_root() / f"draw-{index:02d}"


def source_hashes():
    return {name: sha(HERE / "source" / name) for name in SOURCE_FILES}


def check_protocol():
    protocol = read(HERE / "protocol.json")
    if protocol["source_sha256"] != source_hashes():
        raise RuntimeError("Learner source differs from the predeclared protocol")
    if protocol["runner_sha256"]["nvml_energy.py"] != sha(HERE / "nvml_energy.py"):
        raise RuntimeError("Energy harness differs from the predeclared protocol")
    if protocol["dataset_seeds"] != SEEDS:
        raise RuntimeError("Dataset seeds differ from the predeclared protocol")
    return protocol


def load_inputs(index):
    """The learner's arrays for one draw. Test labels are never read here."""
    import numpy as np
    manifest = read(draw_dir(index) / "manifest.json")
    tier = manifest["tiers"]["medium"]
    if sha(draw_dir(index) / "medium.npz") != tier["sha256"]:
        raise RuntimeError(f"medium.npz for draw {index} differs from its manifest")
    with np.load(draw_dir(index) / "medium.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("train_images", "train_labels", "test_images")}
    for key, value in arrays.items():
        if array_hash(value) != tier["arrays"][key]["sha256_c_order_little_endian"]:
            raise RuntimeError(f"Draw {index} {key} hash differs from the generator manifest")
    return arrays, manifest


def encode(arrays):
    return {key: {"bytes": value.tobytes(), "dtype": str(value.dtype), "shape": list(value.shape)}
            for key, value in arrays.items()}


def decode(payload):
    import numpy as np
    return {key: np.frombuffer(value["bytes"], dtype=value["dtype"]).reshape(value["shape"]).copy()
            for key, value in payload.items()}


def container_info():
    import platform
    info = {"python": platform.python_version()}
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,power.limit,clocks.max.sm",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as error:  # noqa: BLE001 - diagnostic only
        info["nvidia_smi_error"] = repr(error)
    return info


# ---------------------------------------------------------------- remote functions
@app.function(**GPU, max_containers=1, scaledown_window=10, timeout=900)
def selftest(payload):
    """One short epoch on 2,048 training images to check image, kernels and PTX policy."""
    import re
    import numpy as np
    import torch
    import triton
    started = time.time()
    from ordered_learner import config_for, previous_learner
    from ordered_models import OrderedReconstructionTrainer
    previous = previous_learner()
    previous.verify_reference()
    previous._reference_trainer()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    arrays = decode(payload)
    x, y = arrays["train_images"], arrays["train_labels"]
    config = config_for(ARCHITECTURE)
    config.update(n_train=len(x), n_test=256, epochs=1)
    trainer = OrderedReconstructionTrainer(x, y, config, 11)
    trainer.prepare(x[:256])
    torch.cuda.synchronize()
    began = time.perf_counter()
    trainer.invoke()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - began
    output = trainer.outputs()
    finite = bool(np.isfinite(output["scores"]).all()) and all(
        np.isfinite(value).all() for value in output["parameters"].values())
    ptx_ok = True
    for name, kernel in trainer.network.backend.compiled.items():
        assembly = kernel.asm["ptx"]
        if (re.search(r"\bfma(?:\.[a-z0-9]+)*\.f32\b", assembly) or ".ftz." in assembly
                or re.search(r"\b(?:mma|wmma)\.", assembly)):
            ptx_ok = False
    return {"passed": finite and ptx_ok, "finite": finite, "ptx_policy_ok": ptx_ok,
            "kernels": len(trainer.network.backend.compiled), "one_epoch_seconds": seconds,
            "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "triton": str(triton.__version__), "numpy": np.__version__, "cuda": torch.version.cuda,
            "container": container_info(), "container_seconds": time.time() - started}


@app.function(**GPU, max_containers=4, scaledown_window=30, timeout=1200)
def fit(payload, context):
    """Train the frozen three-member ensemble on one draw and predict its test images."""
    import numpy as np
    started = time.time()
    arrays = decode(payload)
    if {key: array_hash(value) for key, value in arrays.items()} != context["input_sha256"]:
        raise RuntimeError("Input arrays changed in transit")
    from ordered_learner import train_predict
    try:
        predictions, metadata = train_predict(**arrays, draw_seed=context["dataset_seed"],
                                              architecture=ARCHITECTURE)
    except RuntimeError as error:
        if "Nonfinite" not in str(error):
            raise
        return {"status": "diverged", "error": str(error), "context": context,
                "test_labels_supplied": False, "container": container_info(),
                "container_seconds": time.time() - started}, None
    metadata.update(status="completed", context=context, container=container_info(),
                    container_seconds=time.time() - started)
    stream = io.BytesIO()
    np.save(stream, predictions.astype(np.int64))
    return metadata, stream.getvalue()


@app.function(**GPU, max_containers=1, scaledown_window=10, timeout=1800)
def measure_energy(payload, context):
    """Paired-idle NVML energy of complete fresh tasks on one draw.

    A task resets and trains all three members, predicts every test image and
    forms the ordered FP32 ensemble and argmax on the GPU. Preparation (schedule
    upload, Triton compilation, CUDA-graph capture) is excluded, as in the
    original Ordered ConvNets and Rev88 measurements. The matched non-reversible
    ensemble (the three-conv CNN with the same kernels, schedules and recipe) is
    measured the same way in the same container as a control.
    """
    import numpy as np
    import torch
    started = time.time()
    from nvml_energy import measure
    from ordered_learner import config_for, previous_learner
    from ordered_models import OrderedReconstructionTrainer
    previous = previous_learner()
    previous.verify_reference()
    Trainer = previous._reference_trainer()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    arrays = decode(payload)
    if {key: array_hash(value) for key, value in arrays.items()} != context["input_sha256"]:
        raise RuntimeError("Input arrays changed in transit")
    x, y, q = arrays["train_images"], arrays["train_labels"], arrays["test_images"]

    class Task:
        def __init__(self, trainers):
            self.trainers = trainers
            self.ensemble = torch.zeros((len(q), 10), dtype=torch.float32, device="cuda")
            self.predictions = torch.empty(len(q), dtype=torch.int64, device="cuda")

        def run(self):
            for trainer in self.trainers:
                trainer.invoke()
            self.ensemble.zero_()
            for trainer in self.trainers:  # member order 11, 22, 33, as in the learner
                self.ensemble.add_(trainer.output_scores)
            self.predictions.copy_(torch.argmax(self.ensemble, dim=1))

        def outputs(self):
            predictions = self.predictions.cpu().numpy().copy()
            return predictions, {
                "predictions_sha256": array_hash(predictions),
                "ensemble_logits_sha256": array_hash(self.ensemble.cpu().numpy()),
                "member_logits_sha256": [array_hash(t.output_scores.cpu().numpy()) for t in self.trainers]}

    def prepared(make):
        trainers = []
        for seed in (11, 22, 33):
            trainer = make(seed)
            trainer.prepare(q)
            trainers.append(trainer)
        torch.cuda.synchronize()
        return trainers

    config = config_for(ARCHITECTURE)
    config.update(n_train=len(x), n_test=len(q))
    reversible = Task(prepared(lambda seed: OrderedReconstructionTrainer(x, y, config, seed)))
    result = {"reversible": measure(reversible, trials=3, target_seconds=5.0)}
    pred = result["reversible"]["after_metadata"]["predictions_sha256"]
    result["reversible_predictions_match_frozen_fit"] = pred == context["frozen_predictions_sha256"]
    del reversible
    torch.cuda.empty_cache()
    cnn_config = json.loads(Path("/workspace/procedure-transfer/reference/config.json").read_text())
    cnn_config.update(n_train=len(x), n_test=len(q))
    control = Task(prepared(lambda seed: Trainer(x, y, cnn_config, seed)))
    result["cnn_control"] = measure(control, trials=3, target_seconds=5.0)
    result.update(context=context, container=container_info(), container_seconds=time.time() - started)
    return result


# ---------------------------------------------------------------- local phases
def phase_protocol():
    protocol = {
        "submission": NAME,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "tier": "MNIST-medium",
        "error_target_percent": TARGET_ERROR_PERCENT,
        "required_correct_of_110000": REQUIRED_CORRECT,
        "dataset_generator": {"module": "mnist.code.data", "profile": "medium-error-targets-v1",
                              "generator_sha256": sha(repo_root() / "mnist/code/data.py")},
        "dataset_seeds": SEEDS,
        "draws": 11, "train_count": 10000, "test_count": 10000,
        "learner": {
            "architecture": ARCHITECTURE,
            "description": "Lossless ReLU lift (input kept as channel 0 + 31 learned channels), "
                           "4 additive coupling blocks on 16+16 channels (alpha 0.25), "
                           "dense 2592-128-10 head; backward reconstructs block inputs",
            "entry_point": "source/reversible-memory/ordered_learner.py:train_predict",
            "member_seeds": [11, 22, 33],
            "training_randomness": "Held fixed across draws: member seeds 11, 22, 33 set "
                                   "initialization and augmentation/shuffle schedules",
            "recipe": "8 epochs, batch 128, SGD lr 0.03 momentum 0.9 weight decay 1e-4, "
                      "mild affine augmentation, ordered FP32 Triton kernels (no FMA, FTZ or tensor cores)",
            "ensemble": "Ordered FP32 sum of the three members' logits, then first-index argmax",
            "selection_history": "Architecture and recipe chosen on 13 Sep 2026 from exploratory "
                                 "sweeps on other draws of the same 60,000-image pool; see history/",
        },
        "source_sha256": source_hashes(),
        "runner_sha256": {"nvml_energy.py": sha(HERE / "nvml_energy.py")},
        "evaluation": "All eleven prediction files are saved and hashed in prediction-freeze.json "
                      "before any test label is read; then mnist.code.evaluate --tier medium "
                      "--error-target 3 scores each draw and the pass uses sum(correct)/110000.",
        "divergence_policy": "A non-finite member ends that draw as a recorded failure; "
                             "no rerun with other seeds. Eleven completed draws are required.",
        "a100": {"modal_gpu": "A100-40GB", "image": IMAGE_REF,
                 "energy": "Paired-idle NVML (nvml_energy.py, copied unchanged from "
                           "rev88-20260914/energy.py) on draw 00, three trials",
                 "control": "Matched non-reversible CNN ensemble measured the same way"},
    }
    write_new(HERE / "protocol.json", protocol)
    print("Wrote", HERE / "protocol.json")


def phase_prepare():
    check_protocol()
    raw = data_root() / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, seed in enumerate(SEEDS):
        out = draw_dir(index)
        if not (out / "manifest.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            if not (out / "raw").exists():
                (out / "raw").symlink_to(raw, target_is_directory=True)
            subprocess.run([sys.executable, "-m", "mnist.code.data", "--profile",
                            "medium-error-targets-v1", "--seed", str(seed), "--output", str(out)],
                           cwd=repo_root(), check=True)
            for extra in ("large.npz", "small.npz"):  # not used; the manifest keeps their hashes
                (out / extra).unlink(missing_ok=True)
        manifest = read(out / "manifest.json")
        if manifest["seed"] != seed or manifest["profile"] != "medium-error-targets-v1":
            raise RuntimeError(f"Draw {index} manifest disagrees with the protocol")
        (HERE / "draws").mkdir(exist_ok=True)
        shutil.copyfile(out / "manifest.json", HERE / "draws" / f"draw-{index:02d}-manifest.json")
        _, manifest = load_inputs(index)
        tier = manifest["tiers"]["medium"]
        rows.append({"draw_index": index, "dataset_seed": seed, "medium_npz_sha256": tier["sha256"],
                     "input_sha256": {key: tier["arrays"][key]["sha256_c_order_little_endian"]
                                      for key in ("train_images", "train_labels", "test_images")},
                     "train_indices_sha256": tier["arrays"]["train_indices"]["sha256_c_order_little_endian"],
                     "test_indices_sha256": tier["arrays"]["test_indices"]["sha256_c_order_little_endian"],
                     "class_histograms": tier["class_histograms"]})
    write_new(HERE / "draws" / "draw-manifest.json",
              {"profile": "medium-error-targets-v1", "draws": rows})
    print(f"Prepared {len(rows)} draws under {data_root()}")


def phase_score():
    import numpy as np
    check_protocol()
    freeze = read(HERE / "prediction-freeze.json")
    for name, digest in freeze["sha256"].items():
        if sha(HERE / name) != digest:
            raise RuntimeError(f"Frozen file changed after the freeze: {name}")
    rows = []
    for index, seed in enumerate(SEEDS):
        result = read(HERE / "results" / f"draw-{index:02d}.json")
        if result.get("status") != "completed":
            raise RuntimeError(f"Draw {index} did not complete; no 11-draw summary is possible")
        output = HERE / "scores" / f"draw-{index:02d}.json"
        output.parent.mkdir(exist_ok=True)
        subprocess.run([sys.executable, "-m", "mnist.code.evaluate", "--tier", "medium",
                        "--error-target", str(TARGET_ERROR_PERCENT), "--data-dir", str(draw_dir(index)),
                        "--predictions", str(HERE / "predictions" / f"draw-{index:02d}.npy"),
                        "--output", str(output)], cwd=repo_root(), check=True, capture_output=True)
        score = read(output)
        rows.append({"draw_index": index, "dataset_seed": seed, "member_seeds": [11, 22, 33],
                     "correct": score["correct"], "total": score["total"],
                     "accuracy_percent": 100 * score["correct"] / score["total"],
                     "training_and_prediction_seconds": result["training_seconds"]})
    accuracy = np.array([row["accuracy_percent"] for row in rows])
    correct = sum(row["correct"] for row in rows)
    total = sum(row["total"] for row in rows)
    bands = {f"{band}%": correct >= {2: 107_800, 3: 106_700, 5: 104_500, 8: 101_200, 12: 96_800}[band]
             for band in (2, 3, 5, 8, 12)}
    summary = {"draws": rows, "correct": correct, "total": total,
               "aggregate_accuracy_percent": 100 * correct / total,
               "mean_accuracy_percent": float(accuracy.mean()),
               "sample_sd_pp": float(accuracy.std(ddof=1)),
               "mean_error_percent": float(100 - accuracy.mean()),
               "error_target_percent": TARGET_ERROR_PERCENT, "required_correct": REQUIRED_CORRECT,
               "meets_target": correct >= REQUIRED_CORRECT, "meets_bands": bands,
               "scored_after_freeze": freeze["frozen_at_utc"],
               "evaluator": "mnist.code.evaluate --tier medium --error-target 3"}
    write_new(HERE / "accuracy.json", summary)
    print(json.dumps({k: summary[k] for k in ("correct", "total", "mean_accuracy_percent",
                                              "sample_sd_pp", "meets_target")}, indent=1))


# ---------------------------------------------------------------- Modal phases
@app.local_entrypoint()
def main(phase: str):
    protocol = check_protocol()
    rows = {row["draw_index"]: row for row in read(HERE / "draws" / "draw-manifest.json")["draws"]}
    if phase == "selftest":
        arrays, _ = load_inputs(0)
        subset = {"train_images": arrays["train_images"][:2048], "train_labels": arrays["train_labels"][:2048]}
        result = selftest.remote(encode(subset))
        write_new(HERE / "evidence" / "selftest.json", result)
        print(json.dumps(result, indent=1))
        if not result["passed"]:
            raise SystemExit("Self-test failed")
    elif phase == "fit":
        if not (HERE / "evidence" / "selftest.json").exists():
            raise SystemExit("Run the self-test first")
        if (HERE / "prediction-freeze.json").exists():
            raise SystemExit("Predictions are already frozen")

        def jobs():
            for index, seed in enumerate(SEEDS):
                if (HERE / "results" / f"draw-{index:02d}.json").exists():
                    continue
                arrays, _ = load_inputs(index)
                context = {"draw_index": index, "dataset_seed": seed,
                           "input_sha256": rows[index]["input_sha256"],
                           "protocol_sha256": sha(HERE / "protocol.json")}
                yield encode(arrays), context

        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "predictions").mkdir(exist_ok=True)
        for metadata, payload in fit.starmap(jobs(), order_outputs=False):
            index = metadata["context"]["draw_index"]
            if payload is not None:
                path = HERE / "predictions" / f"draw-{index:02d}.npy"
                path.write_bytes(payload)
                metadata["prediction_file_sha256"] = sha(path)
            write_new(HERE / "results" / f"draw-{index:02d}.json", metadata)
            print(f"draw {index:02d}: {metadata.get('status')} "
                  f"({metadata.get('training_seconds', 0):.1f} s training); labels unopened", flush=True)
        names = [f"results/draw-{i:02d}.json" for i in range(11)]
        names += [f"predictions/draw-{i:02d}.npy" for i in range(11)
                  if (HERE / "predictions" / f"draw-{i:02d}.npy").exists()]
        write_new(HERE / "prediction-freeze.json",
                  {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                   "protocol_sha256": sha(HERE / "protocol.json"),
                   "sha256": {name: sha(HERE / name) for name in names},
                   "test_labels_read_before_freeze": False})
        print("Froze", len(names), "files")
    elif phase == "energy":
        frozen = read(HERE / "results" / "draw-00.json")
        arrays, _ = load_inputs(0)
        context = {"draw_index": 0, "dataset_seed": SEEDS[0], "input_sha256": rows[0]["input_sha256"],
                   "frozen_predictions_sha256": frozen["predictions_sha256"],
                   "protocol_sha256": sha(HERE / "protocol.json")}
        result = measure_energy.remote(encode(arrays), context)
        write_new(HERE / "evidence" / "energy.json", result)
        for key in ("reversible", "cnn_control"):
            summary = result[key]["summary"]
            print(key, {k: round(v["mean"], 1) for k, v in summary.items()})
        print("predictions match frozen fit:", result["reversible_predictions_match_frozen_fit"])
    else:
        raise SystemExit(f"Unknown Modal phase {phase!r}; local phases run with python")
    del protocol


if __name__ == "__main__":
    phases = {"protocol": phase_protocol, "prepare": phase_prepare, "score": phase_score}
    if len(sys.argv) != 2 or sys.argv[1] not in phases:
        raise SystemExit(f"usage: python {Path(__file__).name} {{{'|'.join(phases)}}}")
    phases[sys.argv[1]]()
