"""Compare frozen/scored transfer results with matched historical reference draws.

This reads score files only: no learning, query-label access or model selection.
Run after all planned candidate predictions have been frozen and scored.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


REFERENCE_RECIPES = (
    "linear-sgd", "mlp64-sgd", "mlp256-sgd", "cnn16-sgd",
    "cnn32-ensemble3", "reversible82-sgd",
)
CANONICAL_DATASETS = (
    "kmnist", "emnist_letters_aj", "emnist_letters_kt", "emnist_balanced_aj",
    "emnist_digits", "emnist_mnist", "qmnist_recovered", "k49_10",
    "kannada_digits", "devanagari_digits", "madbase", "notmnist_large",
    "fashion_mnist", "svhn", "cifar10",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, message):
    if not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError(message)


def validate_scores(score):
    """Recount every supplied aggregate from its score counts, without labels."""
    rows = score["datasets"]
    names = [row["dataset"] for row in rows]
    if not names or names != [name for name in CANONICAL_DATASETS if name in names]:
        raise ValueError("Dataset rows must be a unique canonical-order selection")
    if "dataset_order" in score and score["dataset_order"] != names:
        raise ValueError("Score dataset_order differs from dataset rows")
    result = {}
    for row in rows:
        draws = row["draws"]
        ids = [d["draw"] for d in draws]
        if not ids or ids != sorted(set(ids)) or any(type(d) is not int or d not in range(11) for d in ids):
            raise ValueError("Draw indices must be unique sorted integers 0..10")
        values = []
        for draw in draws:
            if draw["total"] != 10000 or type(draw["correct"]) is not int or not 0 <= draw["correct"] <= 10000:
                raise ValueError("Every scored draw must contain 10,000 valid query predictions")
            if draw.get("dataset_seed", 20261101 + draw["draw"]) != 20261101 + draw["draw"]:
                raise ValueError("Draw seed does not match the v1 suite")
            value = 100 * draw["correct"] / draw["total"]
            close(draw["accuracy_percent"], value, "Draw accuracy differs from counts")
            values.append(value)
        if "mean_accuracy_percent" in row:
            close(row["mean_accuracy_percent"], statistics.mean(values), "Mean accuracy differs from counts")
        if "sample_sd_pp" in row:
            close(row["sample_sd_pp"], statistics.stdev(values) if len(values) > 1 else 0, "Sample SD differs from counts")
        if "correct" in row and row["correct"] != sum(d["correct"] for d in draws):
            raise ValueError("Aggregate correct count differs from draws")
        if "total" in row and row["total"] != sum(d["total"] for d in draws):
            raise ValueError("Aggregate total count differs from draws")
        result[row["dataset"]] = {d["draw"]: d for d in draws}
    if "fifteen_numbers" in score:
        if len(score["fifteen_numbers"]) != len(rows):
            raise ValueError("Accuracy vector length differs from dataset rows")
        for mean, row in zip(score["fifteen_numbers"], rows):
            close(mean, statistics.mean(d["accuracy_percent"] for d in row["draws"]), "Accuracy vector differs")
    return result


def summarize(values):
    return {
        "mean_accuracy_percent": statistics.mean(values),
        "sample_sd_pp": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_draw_accuracy_percent": min(values),
        "max_draw_accuracy_percent": max(values),
    }


def compare(candidate_path, references, output, label):
    candidate_path, references, output = map(Path, (candidate_path, references, output))
    candidate = read(candidate_path)
    cand_rows = validate_scores(candidate)
    ref_rows = {}
    provenance = {
        "candidate_scores": {"path": str(candidate_path.resolve()), "sha256": sha(candidate_path)},
        "reference_scores": {},
        "comparison_source_sha256": sha(__file__),
    }
    for filename in ("plan.json", "prediction-manifest.json"):
        path = candidate_path.parent / filename
        if path.exists():
            provenance[filename] = {"path": str(path.resolve()), "sha256": sha(path)}
    for recipe in REFERENCE_RECIPES:
        path = references / recipe / "scores.json"
        ref_rows[recipe] = validate_scores(read(path))
        provenance["reference_scores"][recipe] = {"path": str(path.resolve()), "sha256": sha(path)}
    records = []
    for name, draws in cand_rows.items():
        draw_ids = list(draws)
        candidate_values = [draws[d]["accuracy_percent"] for d in draw_ids]
        record = {"dataset": name, "draws": draw_ids, "candidate": summarize(candidate_values), "references": {}}
        for recipe, by_dataset in ref_rows.items():
            for draw_id in draw_ids:
                candidate_label_sha = draws[draw_id].get("test_labels_sha256")
                reference_label_sha = by_dataset[name][draw_id].get("test_labels_sha256")
                if candidate_label_sha and reference_label_sha and candidate_label_sha != reference_label_sha:
                    raise ValueError("Candidate and reference query labels differ for matched draw")
            values = [by_dataset[name][d]["accuracy_percent"] for d in draw_ids]
            deltas = [a-b for a, b in zip(candidate_values, values)]
            record["references"][recipe] = {
                **summarize(values),
                "candidate_minus_reference_mean_pp": statistics.mean(deltas),
                "paired_draw_delta_sample_sd_pp": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "paired_draw_deltas_pp": deltas,
            }
        records.append(record)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_label": label,
        "candidate_config": candidate.get("config"),
        "complete_fifteen_dataset_suite": len(records) == 15 and all(r["draws"] == list(range(11)) for r in records),
        "all_selected_tasks_have_eleven_draws": all(r["draws"] == list(range(11)) for r in records),
        "comparison_basis": "Exactly matched task IDs and draw indices; fixed v1 9x9 inputs and 10000/10000 draws",
        "caveats": [
            "Reference CNNs use the original ordered 9x9 grid, whereas the candidate may use a fixed pixel permutation. This is a comparison of complete procedures, not an architecture-only or compute-matched ablation.",
            "These transfer tasks repartition curated pools; they are not official-test 28x28 pMNIST results.",
            "Models train from fresh initialization on each draw; this measures training-procedure transfer, not transfer of learned MNIST weights.",
            "Draws overlap and datasets share ancestry. Draw SD and paired-delta SD are descriptive, not confidence intervals or significance tests.",
            "A subset of tasks is an incomplete v1 fifteen-number suite even when all eleven draws are run for each selected task.",
        ],
        "provenance": provenance,
        "datasets": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fields = ["dataset", "draw_count", "reference", "candidate_mean_accuracy_percent", "candidate_sample_sd_pp",
              "reference_mean_accuracy_percent", "reference_sample_sd_pp", "candidate_minus_reference_mean_pp", "paired_draw_delta_sample_sd_pp"]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            for recipe, ref in row["references"].items():
                writer.writerow({"dataset": row["dataset"], "draw_count": len(row["draws"]), "reference": recipe,
                    "candidate_mean_accuracy_percent": row["candidate"]["mean_accuracy_percent"],
                    "candidate_sample_sd_pp": row["candidate"]["sample_sd_pp"],
                    "reference_mean_accuracy_percent": ref["mean_accuracy_percent"],
                    "reference_sample_sd_pp": ref["sample_sd_pp"],
                    "candidate_minus_reference_mean_pp": ref["candidate_minus_reference_mean_pp"],
                    "paired_draw_delta_sample_sd_pp": ref["paired_draw_delta_sample_sd_pp"]})
    lines = [f"# {label}: transfer comparison", "",
        "Accuracy is mean ± sample SD in percentage points. Reference rows use exactly the same draw indices as the candidate.", "",
        "| Dataset | Draws | Candidate | " + " | ".join(REFERENCE_RECIPES) + " |",
        "|---|---:|---:|" + "---:|" * len(REFERENCE_RECIPES)]
    for row in records:
        fmt = lambda x: f"{x['mean_accuracy_percent']:.2f} ± {x['sample_sd_pp']:.2f}"
        lines.append(f"| {row['dataset']} | {len(row['draws'])} | {fmt(row['candidate'])} | " +
                     " | ".join(fmt(row["references"][r]) for r in REFERENCE_RECIPES) + " |")
    lines += ["", "Positive differences below favor the candidate. These are paired descriptive differences across the matched draws.", "",
        "| Dataset | " + " | ".join(REFERENCE_RECIPES) + " |",
        "|---|" + "---:|" * len(REFERENCE_RECIPES)]
    for row in records:
        lines.append(f"| {row['dataset']} | " + " | ".join(
            f"{row['references'][r]['candidate_minus_reference_mean_pp']:+.2f}" for r in REFERENCE_RECIPES) + " |")
    lines += ["", *("- " + caveat for caveat in result["caveats"]), "",
        f"All selected tasks use all eleven draws: {result['all_selected_tasks_have_eleven_draws']}. Complete fifteen-task suite: {result['complete_fifteen_dataset_suite']}.", "",
        "The JSON records score-file, source, and available plan/prediction-manifest hashes. No model fitting or query-label reading occurs in this comparison script.", ""]
    (output / "comparison.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--references", type=Path, default=Path("/Users/yaroslavvb/git/aminist-21-validation/reference-results"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", default="Fixed-permutation candidate")
    args = parser.parse_args()
    result = compare(args.scores, args.references, args.output, args.label)
    print(json.dumps({"datasets": len(result["datasets"]), "complete_fifteen_dataset_suite": result["complete_fifteen_dataset_suite"],
                      "all_selected_tasks_have_eleven_draws": result["all_selected_tasks_have_eleven_draws"]}))


if __name__ == "__main__":
    main()
