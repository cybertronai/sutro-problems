"""Render an evidence-based report from two completed MNIST tuning suites.

Usage: python -m mnist.report --results mnist/results/GROUP
The default output directory is RESULTS/report. This reads local JSON only;
it neither imports torch nor changes or uploads any experiment results.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


TIERS = ("small", "medium", "large")
COLORS = {"linear": "#707070", "mlp": "#0072B2", "cnn": "#D55E00"}
LABELS = {"linear": "Linear", "mlp": "MLP", "cnn": "CNN"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_dataset_manifest(results: Path, data_dir: Path, summaries: dict) -> dict:
    """Prefer archived provenance and require it to identify every result archive."""
    archived = [path for path in [results / "dataset_manifest.json",
                *(results / tier / "dataset_manifest.json" for tier in summaries)]
                if path.is_file()]
    path = archived[0] if archived else data_dir / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Dataset manifest required to describe these results: {path}")
    manifest = read_json(path)
    if any(read_json(other) != manifest for other in archived[1:]):
        raise ValueError("Archived dataset manifests disagree")
    if "seed" not in manifest or any(tier not in manifest.get("tiers", {}) for tier in TIERS):
        raise ValueError(f"Incomplete dataset manifest: {path}")
    for tier in TIERS:
        info = manifest["tiers"][tier]
        if any(not isinstance(info.get(key), int) or info[key] < 1
               for key in ("size", "train_count", "test_count")) or not info.get("sha256"):
            raise ValueError(f"Incomplete {tier} dataset provenance: {path}")
    for tier, summary in summaries.items():
        if summary.get("dataset_sha256") != manifest["tiers"][tier]["sha256"]:
            raise ValueError(f"{tier} results do not match dataset manifest: {path}")
        if ("dataset_seed" in summary and summary["dataset_seed"] != manifest["seed"] or
                "dataset_profile" in summary and
                summary["dataset_profile"] != manifest.get("profile")):
            raise ValueError(f"{tier} results disagree with dataset seed/profile: {path}")
    return manifest


def verify_dataset_files(data_dir: Path, manifest: dict) -> None:
    """Never illustrate or publish results using another dataset's arrays."""
    for tier in TIERS:
        path = data_dir / f"{tier}.npz"
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != manifest["tiers"][tier]["sha256"]:
                raise ValueError(f"Dataset archive does not match result provenance: {path}")


def load_results(results: Path) -> dict:
    """Require complete final evaluations and their matching winner histories."""
    suites = {}
    for tier in ("small", "medium"):
        directory = results / tier
        path = directory / "summary.json"
        if not path.is_file():
            raise ValueError(f"Completed summary not found: {path}")
        summary = read_json(path)
        if summary.get("tier") != tier or summary.get("smoke"):
            raise ValueError(f"Expected a completed non-smoke {tier} suite: {path}")
        finals = summary.get("final_runs", [])
        if len(finals) != 3 or {r["seed"] for r in finals} != {101, 102, 103}:
            raise ValueError(f"Expected final refits for seeds 101, 102, 103: {path}")
        for run in finals:
            if "test" not in run or not np.isfinite(run["test"]["accuracy"]):
                raise ValueError(f"Missing finite final test accuracy: {run['name']}")
        values = [r["test"]["accuracy"] for r in finals]
        for key, expected in (("test_accuracy_mean", statistics.mean(values)),
                              ("test_accuracy_std", statistics.stdev(values))):
            if not np.isclose(summary[key], expected):
                raise ValueError(f"Summary {key} disagrees with final runs: {path}")
        chosen = summary["selection"]["trial_id"]
        matching = [r for r in summary["search_runs"]
                    if r["config"]["trial_id"] == chosen]
        if len(matching) != 3 or {r["seed"] for r in matching} != {11, 22, 33}:
            raise ValueError(f"Expected selected configuration seeds 11, 22, 33: {path}")
        histories = []
        for run in sorted(matching, key=lambda r: r["seed"]):
            full = read_json(directory / f"{run['name']}.json")
            history = full.get("history", [])
            if not history or full["config"] != run["config"] or full["seed"] != run["seed"]:
                raise ValueError(f"Missing or inconsistent history: {run['name']}")
            epochs = [row["epoch"] for row in history]
            if epochs != list(range(1, run["epochs"] + 1)):
                raise ValueError(f"Incomplete epoch history: {run['name']}")
            metrics = [[row[key] for key in ("train/loss", "val/loss",
                                            "train/accuracy", "val/accuracy")]
                       for row in history]
            if not np.isfinite(metrics).all():
                raise ValueError(f"Nonfinite training history: {run['name']}")
            histories.append(full)
        if len({len(run["history"]) for run in histories}) != 1:
            raise ValueError(f"Selected configuration histories have different lengths: {path}")
        suites[tier] = {"summary": summary, "winner_histories": histories}
    groups = {v["summary"]["group"] for v in suites.values()}
    if len(groups) != 1:
        raise ValueError("The two tier summaries have different experiment groups")
    return suites


def training_curves(suites: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row, (tier, suite) in enumerate(suites.items()):
        selection = suite["summary"]["selection"]
        histories = suite["winner_histories"]
        epochs = np.array([r["epoch"] for r in histories[0]["history"]])
        for col, metric in enumerate(("loss", "accuracy")):
            ax = axes[row, col]
            for split, color in (("train", COLORS["mlp"]), ("val", COLORS["cnn"])):
                values = np.array([[r[f"{split}/{metric}"] for r in run["history"]]
                                   for run in histories])
                if metric == "accuracy":
                    values *= 100
                mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
                ax.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=.14,
                                linewidth=0)
                ax.plot(epochs, mean, color=color, linewidth=1.8,
                        label="Training" if split == "train" else "Validation")
            ax.axvline(selection["selected_epochs"], color="#444444", linewidth=1,
                       linestyle="--", label=f"Refit epoch ({selection['selected_epochs']})")
            size = suite["dataset"]["size"]
            ax.set_title(f"MNIST {tier} ({size}×{size}) · "
                         f"{LABELS[selection['config']['architecture']]}", loc="left")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-entropy loss" if metric == "loss" else "Accuracy (%)")
            ax.set_xlim(1, max(2, int(epochs[-1])))
            if metric == "accuracy":
                ax.set_ylim(max(0, ax.get_ylim()[0]), 100.5)
            else:
                ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
            ax.legend(loc="best", fontsize=9)
    fig.suptitle("Selected models: training and validation curves", fontsize=15)
    fig.supxlabel("Means and ±1 sample SD over seeds 11, 22, 33 on the same fixed validation split.\n"
                  "Bands show seed variability, not confidence intervals. Test data are evaluated only after final refits.",
                  fontsize=10)
    fig.savefig(output / "training_curves.png", dpi=180)
    plt.close(fig)


def search_plot(suites: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(left=.07, right=.99, top=.85, bottom=.29, wspace=.22)
    for ax, (tier, suite) in zip(axes, suites.items()):
        summary = suite["summary"]
        initial = [r for r in summary["search_runs"] if r["phase"] == "search"]
        for architecture, color in COLORS.items():
            runs = [r for r in initial if r["config"]["architecture"] == architecture]
            ax.scatter([r["parameters"] for r in runs],
                       [r["best"]["val/accuracy"] * 100 for r in runs],
                       color=color, s=45, alpha=.8, edgecolors="white", linewidths=.5,
                       zorder=3)
        for finalist in summary["finalists"]:
            ax.errorbar(finalist["parameters"], finalist["mean_val_accuracy"] * 100,
                        yerr=finalist["val_accuracy_std"] * 100, fmt="s", markersize=7,
                        markerfacecolor="none", color=COLORS[finalist["config"]["architecture"]],
                        capsize=4, linewidth=1.2, zorder=4)
        selected = summary["selection"]
        ax.scatter([selected["parameters"]], [selected["mean_val_accuracy"] * 100],
                   s=175, marker="*", c="#E69F00", edgecolors="#222222", linewidths=.7,
                   zorder=5)
        ax.annotate(selected["trial_id"],
                    (selected["parameters"], selected["mean_val_accuracy"] * 100),
                    xytext=(-6, -17), textcoords="offset points", ha="right", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("Trainable parameters (log scale)")
        ax.set_ylabel("Best validation accuracy (%)")
        ax.set_title(f"MNIST {tier} · {len(initial)} candidates", loc="left")
        ax.grid(alpha=.2)
        ax.margins(x=.16, y=.18)
    handles = [Line2D([], [], marker="o", linestyle="", color=color,
                      label=LABELS[name]) for name, color in COLORS.items()]
    handles += [Line2D([], [], marker="s", linestyle="", markerfacecolor="none", color="#444444",
                       label="Finalist mean ± seed SD"),
                Line2D([], [], marker="*", linestyle="", color="#E69F00",
                       markeredgecolor="#222222", markersize=11, label="Selected mean")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015),
               ncol=5, fontsize=9)
    fig.suptitle("Architecture search on validation data", fontsize=15)
    fig.text(.5, .12, "Circles: best checkpoint from seed 11. Squares: mean of per-seed best checkpoints (11, 22, 33).\n"
             "Error bars show seed variability, not confidence intervals. Selection never uses test accuracy.",
             ha="center", fontsize=9)
    fig.savefig(output / "search_results.png", dpi=180)
    plt.close(fig)


def dataset_examples(data_dir: Path, output: Path, manifest: dict) -> bool:
    """Show the same source digit at all three resolutions, if data are present."""
    if not all((data_dir / f"{tier}.npz").is_file() for tier in TIERS):
        return False
    with np.load(data_dir / "small.npz", allow_pickle=False) as archive:
        labels, indices = archive["train_labels"], archive["train_indices"]
        positions = [np.flatnonzero(labels == digit) for digit in range(10)]
        if any(len(found) == 0 for found in positions):
            raise ValueError("The small training tier must contain all ten digits")
        source_indices = indices[[found[0] for found in positions]]
    fig, axes = plt.subplots(3, 10, figsize=(12, 4.7))
    fig.subplots_adjust(left=.10, right=.99, top=.84, bottom=.12, wspace=.07, hspace=.17)
    for row, tier in enumerate(TIERS):
        info = manifest["tiers"][tier]
        with np.load(data_dir / f"{tier}.npz", allow_pickle=False) as archive:
            lookup = {int(source): pos for pos, source in enumerate(archive["train_indices"])}
            if any(int(source) not in lookup for source in source_indices):
                raise ValueError("Dataset tiers must share the selected source examples")
            selected_positions = [lookup[int(source)] for source in source_indices]
            if not np.array_equal(archive["train_labels"][selected_positions], np.arange(10)):
                raise ValueError("Dataset labels differ across matched source examples")
            images = archive["train_images"][selected_positions]
        for digit in range(10):
            ax = axes[row, digit]
            ax.imshow(images[digit, 0], cmap="gray", interpolation="nearest", vmin=0, vmax=1)
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(str(digit), fontsize=11)
            if digit == 0:
                ax.set_ylabel(f"{tier.capitalize()}\n{info['size']}×{info['size']}",
                              rotation=0, labelpad=36, va="center", fontsize=10)
    fig.suptitle("The same handwritten digits at each dataset resolution", fontsize=14)
    fig.text(.54, .055, "First training example of each digit in MNIST small, matched by original MNIST source index.\n"
             "Nearest-neighbor display; all panels use the same grayscale range and physical size.",
             ha="center", fontsize=9)
    fig.savefig(output / "dataset_examples.png", dpi=180)
    plt.close(fig)
    return True


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def dispersion(mean: float, sd: float) -> str:
    return f"{mean * 100:.2f} ± {sd * 100:.2f}"


def architecture_text(config: dict, image_size: int) -> str:
    architecture = config["architecture"]
    if architecture == "linear":
        return f"Flatten {image_size}×{image_size} pixels, then linear → 10 logits."
    if architecture == "mlp":
        return (f"Flatten → {config['depth']} hidden layers of width {config['width']}, "
                f"each with GELU and dropout {config['dropout']:g} → linear 10 logits.")
    pooling = ("Max-pool 2×2 after the second convolution (or the only convolution). "
               if config.get("pooling", "none") == "max" else "No pooling. ")
    return (f"{config['depth']} same-padded 3×3 convolutions of width {config['width']}, "
            "each without convolution bias and followed by BatchNorm and GELU. "
            + pooling + f"Flatten → linear {config['width'] * 4} → GELU → "
            f"dropout {config['dropout']:g} → linear 10 logits.")


def wandb_link(run: dict, label: str | None = None) -> str:
    url = run.get("wandb_url")
    return f"[{label or run['name']}]({url})" if url else "Unavailable in saved result"


def markdown_report(suites: dict, output: Path, manifest: dict,
                    has_examples: bool = False) -> None:
    first = next(iter(suites.values()))["summary"]
    count = sum(len(s["summary"]["search_runs"]) + len(s["summary"]["final_runs"])
                for s in suites.values())
    lines = ["# MNIST dataset tiers and tuned baselines", "",
             f"Experiment group: `{first['group']}`. {count} completed training runs across small and medium.", "",
             f"Dataset profile: `{manifest.get('profile', 'legacy (format version 1)')}`; "
             f"dataset seed: `{manifest['seed']}`. Dataset details below come from the manifest "
             "whose archive hashes match these results.", "",
             "## Datasets", "",
             "| Tier | Image size | Training examples | Test examples | Tuning fit / validation |",
             "|---|---:|---:|---:|---:|"]
    for tier in TIERS:
        info = manifest["tiers"][tier]
        val_count = round(info["train_count"] * .2)
        split = f"{info['train_count'] - val_count:,} / {val_count:,}" if tier in suites else "Not tuned"
        lines.append(f"| MNIST {tier} | {info['size']}×{info['size']} | {info['train_count']:,} | "
                     f"{info['test_count']:,} | {split} |")
    sampling = manifest["sampling"]
    lines += ["", f"Sampling: {sampling['selection']}. Dataset seed: {manifest['seed']}. "
              "This report tunes models only for small and medium.", ""]
    for tier in TIERS:
        info = manifest["tiers"][tier]
        train_source = info.get("train_source", sampling.get("train_source"))
        test_source = info.get("test_source", sampling.get("test_source"))
        if not train_source or not test_source:
            raise ValueError(f"Manifest lacks source-split provenance for {tier}")
        source_names = {"train": "official MNIST training split (60,000 examples)",
                        "test": "official MNIST test split (10,000 examples)"}
        train_source = source_names.get(train_source, train_source)
        test_source = source_names.get(test_source, test_source)
        lines.append(f"- MNIST {tier}: training source: {train_source}; test source: {test_source}.")
    lines += ["",
              "Pixels are float32 in [0, 1]. Downsampling uses exact separable box-area averaging, "
              "including fractional input-pixel overlap. Each run standardizes using only the examples "
              "it fits. The final refit computes normalization from the entire training tier.", "",
              "The download mirror and source checksums follow the "
              "[torchvision MNIST implementation](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py).",
              "", "## Results", "",
              "| Tier | Selected architecture | Parameters | Validation accuracy (%) | Final test accuracy (%) | Refit epochs |",
              "|---|---|---:|---:|---:|---:|"]
    for tier, suite in suites.items():
        s, selected = suite["summary"], suite["summary"]["selection"]
        lines.append(f"| MNIST {tier} | {LABELS[selected['config']['architecture']]} | {selected['parameters']:,} | "
                     f"{dispersion(selected['mean_val_accuracy'], selected['val_accuracy_std'])} | "
                     f"{dispersion(s['test_accuracy_mean'], s['test_accuracy_std'])} | {selected['selected_epochs']} |")
    lines += ["", "Values are mean ± sample standard deviation across three training seeds. "
              "Validation uses seeds 11, 22, 33 on one fixed split; final test uses independently refitted "
              "seeds 101, 102, 103 on the same test examples. These are measures of seed variability, "
              "not confidence intervals or uncertainty over newly sampled datasets. Final models are "
              "scored separately; no ensemble is used. Validation accuracy is a tuning result, not "
              "an unbiased performance estimate.", "", "## Selection protocol", "",
              "1. Hold out a fixed, stratified 20% of each training tier (split seed 4701).",
              "2. Run all predefined architecture and hyperparameter candidates with seed 11. Rank by "
              "best validation accuracy, then validation loss, then parameter count.",
              "3. Repeat the best three candidates with seeds 22 and 33. Select by mean per-seed best "
              "validation accuracy, then mean validation loss, then parameter count.",
              "4. Freeze the selected configuration and the median best-checkpoint epoch before opening "
              "test arrays. Refit on the entire training tier with seeds 101, 102, 103; evaluate each "
              "model on test once after its fixed stopping epoch.", "",
              "All runs use AdamW, a cosine learning-rate schedule ending at 2% of initial learning rate, "
              "float32, deterministic PyTorch algorithms, and no data augmentation. Final refits keep "
              "the original schedule duration and stop at the selected epoch. Search varies architecture, "
              "width, depth, dropout, learning rate, weight decay, batch size, and (for medium CNNs) pooling.", "",
              "Training and validation loss/accuracy are logged every epoch; test metrics are logged "
              "only at the end of final refits. W&B records the run configuration, scalar histories, "
              "and final model artifacts through its "
              "[logging API](https://docs.wandb.ai/models/track/log).", ""]
    for tier, suite in suites.items():
        summary = suite["summary"]
        selected, config = summary["selection"], summary["selection"]["config"]
        counts = Counter(r["phase"] for r in summary["search_runs"] + summary["final_runs"])
        lines += [f"## MNIST {tier}", "", f"Selected trial: `{selected['trial_id']}`.", "",
                  architecture_text(config, manifest["tiers"][tier]["size"]), "",
                  f"Learning rate `{config['lr']:g}`; weight decay `{config['weight_decay']:g}`; "
                  f"batch size `{config['batch_size']}`; search duration `{config['epochs']}` epochs; "
                  f"final refit `{selected['selected_epochs']}` epochs.", "",
                  f"Completed runs: {counts['search']} initial candidates, {counts['replicate']} finalist "
                  f"replications, and {counts['final']} final refits.", "",
                  "### Finalist comparison", "",
                  "| Trial | Parameters | Validation accuracy (%) | Mean validation loss | Median best epoch |",
                  "|---|---:|---:|---:|---:|"]
        for finalist in sorted(summary["finalists"], key=lambda r: (-r["mean_val_accuracy"], r["mean_val_loss"], r["parameters"])):
            mark = " **selected**" if finalist["trial_id"] == selected["trial_id"] else ""
            lines.append(f"| `{finalist['trial_id']}`{mark} | {finalist['parameters']:,} | "
                         f"{dispersion(finalist['mean_val_accuracy'], finalist['val_accuracy_std'])} | "
                         f"{finalist['mean_val_loss']:.4f} | {finalist['selected_epochs']} |")
        lines += ["", "### Selected validation runs", "",
                  "| Seed | Best validation accuracy | Validation loss at best checkpoint | Best epoch | W&B |",
                  "|---:|---:|---:|---:|---|"]
        for run in suite["winner_histories"]:
            best = run["best"]
            lines.append(f"| {run['seed']} | {percent(best['val/accuracy'])} | {best['val/loss']:.4f} | "
                         f"{best['epoch']} | {wandb_link(run, 'Run')} |")
        lines += ["", "### Final test evaluations", "",
                  "| Seed | Test accuracy | Test cross-entropy loss | Refit epochs | W&B |",
                  "|---:|---:|---:|---:|---|"]
        for run in sorted(summary["final_runs"], key=lambda r: r["seed"]):
            lines.append(f"| {run['seed']} | {percent(run['test']['accuracy'])} | {run['test']['loss']:.4f} | "
                         f"{run['epochs']} | {wandb_link(run, 'Run and model artifact')} |")
        lines += ["", "### All tuning runs", "",
                  "| Trial | Phase | Seed | Parameters | Best validation accuracy | Best epoch | W&B |",
                  "|---|---|---:|---:|---:|---:|---|"]
        for run in summary["search_runs"]:
            lines.append(f"| `{run['config']['trial_id']}` | {run['phase']} | {run['seed']} | "
                         f"{run['parameters']:,} | {percent(run['best']['val/accuracy'])} | "
                         f"{run['best']['epoch']} | {wandb_link(run, 'Run')} |")
        lines += ["", "### Provenance", "",
                  f"Dataset archive SHA-256: `{summary['dataset_sha256']}`.", "",
                  f"Hardware: {summary['hardware']}. Python {summary['python_version']}; "
                  f"PyTorch {summary['torch_version']}; NumPy {summary['numpy_version']}; "
                  f"W&B {summary['wandb_version']}.", ""]
    lines += ["## Plots and experiment workspace", "",
              "The curve panels show epoch-aligned means and ±1 sample standard deviation for the "
              "selected configuration's three validation runs. They exclude final refit curves, "
              "which use more training data and have no validation set. Dashed vertical lines mark "
              "the epoch chosen for final refitting. The search plot compares seed-11 best checkpoints "
              "and three-seed finalist summaries; test accuracy plays no role in either plot.", "",
              f"[W&B project](https://wandb.ai/{quote(first['entity'], safe='')}/{quote(first['project'], safe='')})", "",
              "![Selected model training and validation curves](training_curves.png)", "",
              "![Validation architecture search](search_results.png)", ""]
    if has_examples:
        lines += ["![Matched handwritten digits at all three tier resolutions](dataset_examples.png)", ""]
    (output / "report.md").write_text("\n".join(lines))


def generate(results: Path, output: Path | None = None,
             data_dir: Path = Path("mnist/data")) -> Path:
    results = Path(results)
    output = Path(output) if output is not None else results / "report"
    suites = load_results(results)
    manifest = load_dataset_manifest(results, Path(data_dir),
                                     {tier: suite["summary"] for tier, suite in suites.items()})
    verify_dataset_files(Path(data_dir), manifest)
    for tier, suite in suites.items():
        suite["dataset"] = manifest["tiers"][tier]
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"}):
        training_curves(suites, output)
        search_plot(suites, output)
        has_examples = dataset_examples(Path(data_dir), output, manifest)
    markdown_report(suites, output, manifest, has_examples=has_examples)
    return output / "report.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Experiment-group directory containing small/ and medium/")
    parser.add_argument("--output", type=Path, default=None,
                        help="Report directory (default: RESULTS/report)")
    parser.add_argument("--data-dir", type=Path, default=Path("mnist/data"),
                        help="Dataset tier NPZ directory; example panel skipped when absent")
    args = parser.parse_args()
    print(generate(args.results, args.output, args.data_dir))


if __name__ == "__main__":
    main()
