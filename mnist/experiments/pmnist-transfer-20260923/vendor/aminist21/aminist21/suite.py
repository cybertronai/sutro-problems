"""Run adapters, freeze every prediction, then independently score all fifteen tasks."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from .common import ROOT, DATASETS, SEEDS, sha, array_sha, read, write, utc, split_indices
from .data import load_pool, learner_inputs, manifest


def validate_selection(names, draws):
    if names != [name for name in DATASETS if name in names] or not names:
        raise ValueError("Datasets must be a unique, nonempty canonical-order selection")
    if draws != sorted(set(draws)) or not draws or any(type(d) is not int or d not in range(11) for d in draws):
        raise ValueError("Draws must be unique sorted integers in 0..10")
    return names == DATASETS and draws == list(range(11))


def load_predictions(path):
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["predictions"]:
            raise ValueError("Prediction archive must contain exactly one predictions array")
        values = archive["predictions"]
    if values.shape != (10000,) or values.dtype != np.int64 or np.any((values < 0)|(values > 9)):
        raise ValueError("Invalid prediction array")
    return values


def freeze_predictions(output, names, draws):
    output = Path(output)
    complete = validate_selection(names, draws)
    plan = read(output/"plan.json")
    if plan["datasets"] != names or plan["draws"] != draws:
        raise ValueError("Prediction selection differs from plan")
    entries = []
    for name in names:
        for draw in draws:
            stem = f"{name}/draw-{draw:02d}"
            meta, pred = output/(stem+".json"), output/(stem+".npz")
            info = read(meta)
            if sha(pred) != info["prediction_file_sha256"]:
                raise ValueError(f"Prediction archive changed: {stem}")
            values = load_predictions(pred)
            if array_sha(values) != info["predictions_sha256"]:
                raise ValueError(f"Prediction array changed: {stem}")
            entries.append({"dataset":name, "draw":draw, "dataset_seed":SEEDS[draw],
                            "metadata_path":str(meta.relative_to(output)), "metadata_sha256":sha(meta),
                            "prediction_path":str(pred.relative_to(output)), "prediction_sha256":sha(pred)})
    frozen = {"frozen_at_utc":utc(), "dataset_manifest_sha256":sha(ROOT/"datasets/manifest.json"),
              "plan_sha256":sha(output/"plan.json"), "datasets":names, "draws":draws,
              "complete_v1_suite": complete, "entries":entries}
    write(output/"prediction-manifest.json", frozen)
    return frozen


def run_local(adapter, config, output, data_dir, names=DATASETS, draws=list(range(11))):
    from .worker import resolve_adapter
    output = Path(output)
    _, source = resolve_adapter(adapter)
    plan = {"adapter":adapter, "adapter_source_sha256":sha(source), "config":config,
            "dataset_manifest_sha256":sha(ROOT/"datasets/manifest.json"),
            "datasets":names, "draws":draws, "seeds":[SEEDS[d] for d in draws],
            "train_count":10000, "query_count":10000,
            "fresh_subprocess_per_draw":True, "query_labels_supplied":False}
    if (output/"plan.json").exists():
        if read(output/"plan.json") != plan:
            raise ValueError("Existing output plan differs; use another output directory")
    else:
        write(output/"plan.json", plan)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)+os.pathsep+env.get("PYTHONPATH", "")
    for name in names:
        pool = load_pool(name, data_dir)
        for draw in draws:
            target = output/name/f"draw-{draw:02d}.json"
            arrays, _, _ = learner_inputs(pool, draw)
            expected = {key:array_sha(value) for key,value in arrays.items()}
            if target.exists():
                old = read(target)
                if (old["input_sha256"] != expected or old["config"] != config or
                    old["adapter_source_sha256"] != plan["adapter_source_sha256"] or
                    sha(target.with_suffix(".npz")) != old["prediction_file_sha256"]):
                    raise ValueError(f"Existing run does not match: {target}")
                continue
            with tempfile.TemporaryDirectory(prefix="aminist21-input-") as folder:
                folder = Path(folder)
                np.savez_compressed(folder/"input.npz", **arrays)
                write(folder/"config.json", config)
                subprocess.run([sys.executable,"-m","aminist21.worker","--adapter",adapter,
                    "--input",str(folder/"input.npz"),"--config",str(folder/"config.json"),
                    "--output",str(target.resolve())], check=True, env=env)
            result = read(target)
            if result["input_sha256"] != expected or result["adapter_source_sha256"] != plan["adapter_source_sha256"]:
                raise ValueError("Adapter source/input changed during execution")
            print(f"Trained {name} draw {draw+1}/11", flush=True)
        del pool
    freeze_predictions(output, names, draws)
    return score(output, data_dir)


def score(output, data_dir):
    output = Path(output)
    frozen = read(output/"prediction-manifest.json")
    plan = read(output/"plan.json")
    complete = validate_selection(frozen["datasets"], frozen["draws"])
    if (plan["datasets"] != frozen["datasets"] or plan["draws"] != frozen["draws"] or
        type(frozen["complete_v1_suite"]) is not bool or frozen["complete_v1_suite"] != complete):
        raise ValueError("Frozen selection/completeness differs from plan")
    if frozen["dataset_manifest_sha256"] != sha(ROOT/"datasets/manifest.json") or frozen["plan_sha256"] != sha(output/"plan.json"):
        raise ValueError("Frozen data/plan manifest changed")
    expected_keys = [(name, draw) for name in frozen["datasets"] for draw in frozen["draws"]]
    if [(r["dataset"],r["draw"]) for r in frozen["entries"]] != expected_keys:
        raise ValueError("Prediction manifest has missing, duplicate or reordered entries")
    # Verify the entire frozen set before opening labels for any scoring.
    for row in frozen["entries"]:
        for field in ("metadata", "prediction"):
            if sha(output/row[field+"_path"]) != row[field+"_sha256"]:
                raise ValueError(f"Frozen {field} changed")
    results = []
    for name in frozen["datasets"]:
        pool = load_pool(name, data_dir)
        rows = []
        for entry in [r for r in frozen["entries"] if r["dataset"] == name]:
            draw = entry["draw"]
            if entry["dataset_seed"] != SEEDS[draw]:
                raise ValueError("Frozen draw seed differs")
            inputs, fit, query = learner_inputs(pool, draw)
            metadata = read(output/entry["metadata_path"])
            if metadata["input_sha256"] != {key:array_sha(value) for key,value in inputs.items()}:
                raise ValueError(f"Scoring inputs differ from training: {name}/{draw}")
            if metadata["config"] != plan["config"] or metadata["query_labels_supplied"] is not False:
                raise ValueError("Learner configuration/input contract differs")
            if metadata["adapter_source_sha256"] != plan["adapter_source_sha256"]:
                raise ValueError("Mixed adapter sources")
            if np.intersect1d(fit,query).size or len(np.unique(pool["example_hashes"][np.r_[fit,query]])) != 20000:
                raise ValueError("Training and query rows/native images overlap")
            predictions = load_predictions(output/entry["prediction_path"])
            labels = pool["labels"][query]
            if predictions.shape != labels.shape or predictions.dtype != np.int64 or np.any((predictions<0)|(predictions>9)):
                raise ValueError("Invalid prediction labels")
            if array_sha(predictions) != metadata["predictions_sha256"]:
                raise ValueError("Prediction array hash differs")
            confusion = np.bincount(labels*10+predictions, minlength=100).reshape(10,10)
            correct = int(np.trace(confusion))
            rows.append({"draw":draw,"dataset_seed":SEEDS[draw],"correct":correct,"total":10000,
                         "accuracy_percent":correct/100,"confusion_matrix":confusion.tolist(),
                         "test_labels_sha256":array_sha(labels),
                         "adapter_wall_seconds":metadata["adapter_wall_seconds"]})
        values = np.array([r["accuracy_percent"] for r in rows])
        sd = float(values.std(ddof=1)) if len(values)>1 else 0.0
        results.append({"dataset":name,"draw_count":len(rows),"correct":sum(r["correct"] for r in rows),
                        "total":sum(r["total"] for r in rows),"mean_accuracy_percent":float(values.mean()),
                        "sample_sd_pp":sd,"min_draw_accuracy_percent":float(values.min()),
                        "max_draw_accuracy_percent":float(values.max()),
                        "descriptive_mean_plus_minus_2sd_percent":[max(0,float(values.mean())-2*sd),min(100,float(values.mean())+2*sd)],
                        "mean_adapter_wall_seconds":float(np.mean([r["adapter_wall_seconds"] for r in rows])),"draws":rows})
        del pool
    result = {"scored_at_utc":utc(),"prediction_manifest_sha256":sha(output/"prediction-manifest.json"),
              "complete_v1_suite":complete,"adapter":plan["adapter"],"config":plan["config"],
              "metric":"Mean accuracy percent across eleven independent draws, in canonical dataset order",
              "sd_interpretation":"Sample SD across draws; descriptive, not an acceptance criterion or confidence interval",
              "dataset_order":[r["dataset"] for r in results],
              "fifteen_numbers":[r["mean_accuracy_percent"] for r in results],"datasets":results}
    write(output/"scores.json", result)
    with (output/"scores.csv").open("w",newline="") as stream:
        fields=["dataset","mean_accuracy_percent","sample_sd_pp","min_draw_accuracy_percent","max_draw_accuracy_percent","correct","total","mean_adapter_wall_seconds"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({key:row[key] for key in fields} for row in results)
    return result
