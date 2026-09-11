"""Version the competition datasets or finished report in the W&B project."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

from mnist.report import load_dataset_manifest, load_results, verify_dataset_files


def publish(data_dir, results=None, entity="yaroslavvb", project="sutro-mnist-tiers"):
    data_dir = Path(data_dir)
    if results is None:
        manifest_path = data_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for tier in ("small", "medium", "large"):
            if not (data_dir / f"{tier}.npz").is_file():
                raise ValueError(f"Dataset archive missing: {data_dir / f'{tier}.npz'}")
        verify_dataset_files(data_dir, manifest)
        source = Path(__file__).resolve().parent
    else:
        results = Path(results)
        suites = load_results(results)
        manifest = load_dataset_manifest(results, data_dir,
                                         {tier: suite["summary"] for tier, suite in suites.items()})
        source = results / "source"
        if not (source / "train.py").is_file() or not (source / "requirements.txt").is_file():
            raise ValueError(f"Archived training source is required to publish results: {source}")
    with wandb.init(entity=entity, project=project,
                    dir=str(Path(__file__).resolve().parent),
                    name="competition-datasets" if results is None else "competition-results",
                    job_type="dataset" if results is None else "report",
                    config={"dataset_seed": manifest["seed"], "dataset_profile": manifest.get("profile"),
                            "tiers": manifest["tiers"]},
                    settings=wandb.Settings(disable_git=True)) as run:
        if results is None:
            artifact = wandb.Artifact("mnist-competition-tiers", type="dataset", metadata=manifest)
            artifact.add_file(data_dir / "manifest.json")
            for tier in ("small", "medium", "large"):
                artifact.add_file(data_dir / f"{tier}.npz")
            run.log_artifact(artifact, aliases=["latest", f"seed-{manifest['seed']}"])
        else:
            columns = ["tier", "model", "parameters", "val_accuracy_mean", "test_accuracy_mean", "test_accuracy_seed_std"]
            table = wandb.Table(columns=columns)
            artifact = wandb.Artifact("mnist-competition-results", type="report",
                                      metadata={"dataset_manifest": manifest,
                                                "group": suites["small"]["summary"]["group"]})
            with artifact.new_file("dataset_manifest.json", mode="w") as stream:
                json.dump(manifest, stream, indent=2, sort_keys=True)
                stream.write("\n")
            for tier in ("small", "medium"):
                summary = suites[tier]["summary"]
                selected = summary["selection"]
                table.add_data(tier, selected["config"]["architecture"], selected["parameters"],
                               selected["mean_val_accuracy"], summary["test_accuracy_mean"], summary["test_accuracy_std"])
                run.summary.update({f"{tier}/test_accuracy_mean": summary["test_accuracy_mean"],
                                    f"{tier}/test_accuracy_seed_std": summary["test_accuracy_std"]})
                for path in sorted((results / tier).glob("*.json")):
                    artifact.add_file(path, name=f"{tier}/{path.name}")
            run.log({"baseline_comparison": table})
            for path in sorted((results / "report").glob("*")):
                if path.is_file():
                    artifact.add_file(path, name=f"report/{path.name}")
                    if path.suffix == ".png":
                        run.log({path.stem: wandb.Image(str(path))})
            for path in sorted((results / "source").glob("*")):
                if path.is_file():
                    artifact.add_file(path, name=f"source/{path.name}")
            run.log_artifact(artifact)
        # Results retain their archived training code; dataset publication snapshots current code.
        code = wandb.Artifact("mnist-competition-source", type="code")
        for path in sorted(source.glob("*.py")):
            code.add_file(path)
        requirements = source / "requirements.txt"
        code.add_file(requirements)
        run.log_artifact(code)
        url = run.url
    print(url)
    return url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("mnist/data"))
    parser.add_argument("--results", type=Path)
    parser.add_argument("--entity", default="yaroslavvb")
    parser.add_argument("--project", default="sutro-mnist-tiers")
    publish(**vars(parser.parse_args()))
