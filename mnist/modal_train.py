"""Run the two bounded suites on at most two A100s; download all results."""
from pathlib import Path
import io
import json
import netrc
import os
import tarfile

import modal

HERE = Path(__file__).resolve().parent
IMAGE_REF = ("ghcr.io/ab-10/wikitext-bench@"
             "sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f")
image = (modal.Image.from_registry(IMAGE_REF)
         .pip_install("wandb==0.30.0", "numpy==2.2.6")
         .env({"PYTHONPATH": "/workspace", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
               "WANDB_SILENT": "true"})
         .add_local_dir(HERE, "/workspace/mnist",
                        ignore=[".venv", "results", "wandb", "__pycache__", "*.log", "data",
                                "data-reference-20260910"]))
for filename in ("small.npz", "medium.npz", "manifest.json"):
    image = image.add_local_file(HERE / "data" / filename, f"/workspace/mnist/data/{filename}")


def configured_secret():
    key = os.environ.get("WANDB_API_KEY")
    if not key and modal.is_local():
        auth = netrc.netrc().authenticators("api.wandb.ai")
        key = auth[2] if auth else None
    return modal.Secret.from_dict({"WANDB_API_KEY": key}) if key else modal.Secret.from_dict({})


app = modal.App("sutro-mnist-tiers")


@app.function(image=image, cpu=2, memory=4096, timeout=120, retries=0)
def verify_remote():
    import unittest
    import torch
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.discover("/workspace/mnist", pattern="test_*.py", top_level_dir="/workspace")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("MNIST unit tests failed")
    return {"tests": result.testsRun, "torch_version": str(torch.__version__)}


@app.local_entrypoint()
def checks():
    print(verify_remote.remote())


@app.function(image=image, gpu="A100-40GB", cpu=4, memory=8192,
              secrets=[configured_secret()], min_containers=0, max_containers=2,
              buffer_containers=0, scaledown_window=2, retries=0,
              timeout=2400, startup_timeout=300)
def train_remote(tier: str, entity: str, project: str, group: str, smoke: bool):
    from mnist.train import run_suite
    import unittest
    suite = unittest.defaultTestLoader.discover("/workspace/mnist", pattern="test_*.py", top_level_dir="/workspace")
    result = unittest.TextTestRunner().run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("MNIST unit tests failed")
    output = Path("/tmp/mnist-results") / tier
    summary = run_suite(tier, "/workspace/mnist/data", output, entity, project, group, smoke)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and "wandb" not in path.relative_to(output).parts:
                archive.add(path, arcname=f"{tier}/{path.relative_to(output)}")
    return tier, summary, buffer.getvalue()


@app.local_entrypoint()
def main(smoke: bool = False, entity: str = "yaroslavvb", project: str = "sutro-mnist-tiers",
         group: str = "mnist-competition-v2", tier: str = "both"):
    tiers = ["small", "medium"] if tier == "both" else [tier]
    if any(t not in ("small", "medium") for t in tiers):
        raise ValueError("tier must be small, medium, or both")
    output = HERE / "results" / ("smoke" if smoke else group)
    if not smoke and any((output / t / "summary.json").exists() for t in tiers):
        raise FileExistsError("Completed results already exist; choose a new --group")
    output.mkdir(parents=True, exist_ok=True)
    import shutil
    manifest = json.loads((HERE / "data" / "manifest.json").read_text())
    archived_manifest = output / "dataset_manifest.json"
    if archived_manifest.exists() and json.loads(archived_manifest.read_text()) != manifest:
        raise ValueError(f"Output contains results from a different dataset: {output}")
    shutil.copy2(HERE / "data" / "manifest.json", archived_manifest)
    source = output / "source"
    source.mkdir(exist_ok=True)
    for path in [*HERE.glob("*.py"), HERE / "requirements.txt"]:
        shutil.copy2(path, source / path.name)
    for name, summary, payload in train_remote.starmap(
            [(t, entity, project, group + ("-smoke" if smoke else ""), smoke) for t in tiers],
            order_outputs=False):
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            archive.extractall(output, filter="data")
        print(json.dumps({"tier": name, "output": str(output / name),
                          "selection": summary["selection"],
                          "test_accuracy_mean": summary.get("test_accuracy_mean")}, indent=2))
