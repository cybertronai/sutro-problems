"""Verify and score prediction files for the permutation-invariant MNIST-medium study.

This is the ONLY module in the study that imports or calls ``study.query_labels``.
Nothing here may ever be imported by a learner, a runner, or anything shipped into
a Modal container.

Usage
-----
  python score.py --stage dev
  python score.py --stage dev --job-id dev-mlp-a-s2026092301-n1000
  python score.py --freeze-final --plan plans/final.json
  python score.py --stage final

Every scored job is re-verified end to end before its labels are touched:
  * predictions/<id>.npz bytes hash to results/<id>.json:predictions_sha256
  * logits/labels content hashes match output_sha256
  * labels == argmax(logits) with ties resolved to the lowest class index
  * the job's train/query index and input array hashes match a fresh regeneration
    from study.job_arrays(seed, n)
For --stage final a prediction freeze (predictions/final_freeze.json) covering every
planned job is mandatory, and any post-freeze change to a result or prediction file
aborts scoring.  The freeze requires selection.json to exist, cannot be replaced
without --refreeze (which archives the superseded freeze), and can never be replaced
at all once final labels have been read against it.  A --job-id spot check still
requires every planned job to have a result and writes results/scores_<stage>_filtered.json
instead of the canonical table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

import study

FREEZE_NAME = "final_freeze.json"
SELECTION_NAME = "selection.json"
METRIC_KEYS = (
    "training_seconds", "fit_wall_seconds", "epochs_completed", "truncated",
    "uses_query_images_unlabeled", "train_accuracy", "device", "device_name",
)


class ScoreError(RuntimeError):
    pass


def _load_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception as error:  # pragma: no cover - surfaced with context below
        raise ScoreError(f"{path}: unreadable JSON ({error})") from error


def _argmax_lowest(logits: np.ndarray) -> np.ndarray:
    """argmax over axis 1 with ties resolved to the lowest class index."""
    return np.argmax(logits, axis=1).astype(np.uint8)


def verify_record(root: Path, record: dict) -> dict:
    """Check one results/<id>.json against its prediction file and the protocol."""
    job = record.get("job")
    if not isinstance(job, dict):
        raise ScoreError("result record has no 'job' object")
    job_id = job.get("id")
    candidate_id = job.get("candidate_id")
    if not isinstance(candidate_id, str) or not study._CANDIDATE_RE.match(candidate_id):
        raise ScoreError(f"{job_id}: job has no usable candidate_id ({candidate_id!r})")
    permitted = study.allowed_seeds(job.get("stage"))
    if permitted is not None and int(job.get("seed", -1)) not in permitted:
        raise ScoreError(
            f"{job_id}: stage {job.get('stage')!r} may only use seeds {list(permitted)}, "
            f"found {job.get('seed')!r}; refusing to read query labels for it")
    # Ad-hoc stages (integration harnesses, probes) are scoreable, but -- exactly
    # as study.make_job requires -- they may never touch a final seed, otherwise
    # the final freeze could be bypassed by relabelling the stage.
    if permitted is None and int(job.get("seed", -1)) in study.FINAL_SEEDS:
        raise ScoreError(
            f"{job_id}: stage {job.get('stage')!r} is not a protocol stage and may not "
            f"use final seed {job.get('seed')!r}; refusing to read query labels for it")
    if record.get("query_labels_supplied") is not False:
        raise ScoreError(f"{job_id}: query_labels_supplied must be present and false")
    predictions_path = root / record["predictions_path"]
    if not predictions_path.exists():
        raise ScoreError(f"{job_id}: missing prediction file {predictions_path}")
    actual_file_sha = study.sha(predictions_path)
    if actual_file_sha != record.get("predictions_sha256"):
        raise ScoreError(f"{job_id}: predictions file hash {actual_file_sha} != recorded "
                         f"{record.get('predictions_sha256')}")
    with np.load(predictions_path) as payload:
        missing = {"logits", "labels"} - set(payload.files)
        if missing:
            raise ScoreError(f"{job_id}: prediction npz missing arrays {sorted(missing)}")
        logits = np.ascontiguousarray(payload["logits"])
        labels = np.ascontiguousarray(payload["labels"])
    if logits.dtype != np.float32 or logits.shape != (study.QUERY_COUNT, 10):
        raise ScoreError(f"{job_id}: logits must be float32 ({study.QUERY_COUNT},10), "
                         f"found {logits.dtype} {logits.shape}")
    if labels.dtype != np.uint8 or labels.shape != (study.QUERY_COUNT,):
        raise ScoreError(f"{job_id}: labels must be uint8 ({study.QUERY_COUNT},), "
                         f"found {labels.dtype} {labels.shape}")
    if not np.all(np.isfinite(logits)):
        raise ScoreError(f"{job_id}: logits contain non-finite values")
    recorded = record.get("output_sha256") or {}
    for name, array in (("logits", logits), ("labels", labels)):
        digest = study.ahash(array)
        if digest != recorded.get(name):
            raise ScoreError(f"{job_id}: output_sha256[{name}] mismatch ({digest} != "
                             f"{recorded.get(name)})")
    expected = _argmax_lowest(logits)
    if not np.array_equal(labels, expected):
        bad = int(np.count_nonzero(labels != expected))
        raise ScoreError(f"{job_id}: {bad} labels differ from argmax(logits) with "
                         "lowest-index tie-breaking")
    seed, n = int(job["seed"]), int(job["n"])
    arrays = study.job_arrays(seed, n)
    for name, array in arrays.items():
        digest = study.ahash(array)
        if digest != (job.get("input_sha256") or {}).get(name):
            raise ScoreError(f"{job_id}: input_sha256[{name}] does not match a fresh "
                             f"regeneration of job_arrays({seed},{n})")
    if study.ahash(study.train_indices(seed, n)) != job.get("train_indices_sha256"):
        raise ScoreError(f"{job_id}: train_indices_sha256 mismatch")
    if study.ahash(study.query_indices(seed)) != job.get("query_indices_sha256"):
        raise ScoreError(f"{job_id}: query_indices_sha256 mismatch")
    return {"job": job, "labels": labels, "logits": logits,
            "predictions_sha256": actual_file_sha}


def _plan_job_ids(plan) -> list:
    """Accept a list of ids, a list of job dicts, or {'jobs'|'job_ids': [...]}"""
    if isinstance(plan, dict):
        for key in ("jobs", "job_ids", "plan"):
            if key in plan:
                plan = plan[key]
                break
        else:
            raise ScoreError("plan JSON must contain 'jobs' or 'job_ids'")
    if not isinstance(plan, list) or not plan:
        raise ScoreError("plan must be a non-empty list of job ids or job objects")
    ids = []
    for entry in plan:
        if isinstance(entry, str):
            ids.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("id"), str):
            ids.append(entry["id"])
        else:
            raise ScoreError(f"plan entry is neither an id nor a job object: {entry!r}")
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ScoreError(f"plan lists duplicate job ids: {duplicates}")
    return ids


def freeze_final(root: Path, plan_path: Path, refreeze: bool = False,
                 selection_path=SELECTION_NAME) -> dict:
    """Record the prediction hashes of every planned final job. Never scores."""
    plan_path = Path(plan_path)
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    if not plan_path.exists():
        raise ScoreError(f"missing plan {plan_path}")
    selection = Path(selection_path)
    if not selection.is_absolute():
        selection = root / selection
    if not selection.exists():
        raise ScoreError(
            f"refusing to freeze without the frozen selection rule {selection}; "
            "write selection.json (the candidates chosen on dev seeds) first")
    freeze_path = root / "predictions" / FREEZE_NAME
    previous = None
    if freeze_path.exists():
        previous = _load_json(freeze_path)
        if previous.get("scored") is True:
            raise ScoreError(
                f"{freeze_path} has already been scored against the final query labels; "
                "refusing to re-freeze (the final result is fixed)")
        if not refreeze:
            raise ScoreError(
                f"{freeze_path} already exists (frozen at {previous.get('frozen_at_utc')}); "
                "pass --refreeze to supersede it")
    job_ids = _plan_job_ids(_load_json(plan_path))
    entries = {}
    for job_id in job_ids:
        result_path = root / "results" / f"{job_id}.json"
        if not result_path.exists():
            raise ScoreError(f"cannot freeze: missing result {result_path}")
        record = _load_json(result_path)
        if (record.get("job") or {}).get("stage") != "final":
            raise ScoreError(f"cannot freeze: {job_id} is not a final-stage job")
        predictions_path = root / record["predictions_path"]
        if not predictions_path.exists():
            raise ScoreError(f"cannot freeze: missing prediction file {predictions_path}")
        actual = study.sha(predictions_path)
        if actual != record.get("predictions_sha256"):
            raise ScoreError(f"cannot freeze: {job_id} prediction hash disagrees with its result")
        entries[job_id] = {
            "predictions_path": record["predictions_path"],
            "predictions_sha256": actual,
            "predictions_bytes": predictions_path.stat().st_size,
            "result_sha256": study.sha(result_path),
        }
    supersedes = None
    if previous is not None:
        archive_stamp = str(previous.get("frozen_at_utc") or "unknown").replace(":", "")
        archive = freeze_path.with_name(f"final_freeze.{archive_stamp}.json")
        suffix = 1
        while archive.exists():
            archive = freeze_path.with_name(f"final_freeze.{archive_stamp}.{suffix}.json")
            suffix += 1
        supersedes = {
            "archived_path": str(archive.relative_to(root)),
            "frozen_at_utc": previous.get("frozen_at_utc"),
            "sha256": study.sha(freeze_path),
            "job_count": previous.get("job_count"),
        }
        study.write_json(archive, previous)
    freeze = {
        "schema_version": 1,
        "frozen_at_utc": study.utc(),
        "plan_path": str(plan_path.relative_to(root)) if plan_path.is_relative_to(root) else str(plan_path),
        "plan_sha256": study.sha(plan_path),
        "selection_path": str(selection.relative_to(root)) if selection.is_relative_to(root) else str(selection),
        "selection_sha256": study.sha(selection),
        "job_count": len(entries),
        "jobs": entries,
        "scored": False,
        "supersedes": supersedes,
        "note": "Prediction hashes frozen before any query label was read.",
    }
    study.write_json(freeze_path, freeze)
    return freeze


def _discover(root: Path, stage: str, job_ids=None) -> list:
    results_dir = root / "results"
    if not results_dir.exists():
        return []
    paths = []
    for path in sorted(results_dir.glob("*.json")):
        # runner.py writes results/<job_id>-failed.json when a fit crashes: it is
        # not a result record (no predictions) and its stored job id deliberately
        # differs from the filename.  Skipping it stops one failed job from
        # blocking the whole stage from being scored.  A real result file always
        # ends in "-n<level>.json", so this can never hide one.
        if path.name.startswith("scores_") or path.name.endswith("-failed.json"):
            continue
        record = _load_json(path)
        job = record.get("job") or {}
        if job.get("stage") != stage:
            continue
        if job_ids is not None and job.get("id") not in job_ids:
            continue
        if job.get("id") != path.stem:
            raise ScoreError(f"{path}: job id {job.get('id')!r} does not match the filename")
        paths.append((path, record))
    return paths


def score(root: Path, stage: str, job_ids=None, allow_unplanned: bool = False) -> dict:
    root = Path(root)
    found = _discover(root, stage, job_ids)
    freeze = None
    if stage == "final":
        freeze_path = root / "predictions" / FREEZE_NAME
        if not freeze_path.exists():
            raise ScoreError(
                f"refusing to score stage 'final' without {freeze_path}; run "
                "`score.py --freeze-final --plan plans/final.json` first")
        freeze = _load_json(freeze_path)
        planned = freeze.get("jobs") or {}
        if not planned:
            raise ScoreError(f"{freeze_path} lists no jobs")
        # Completeness and freeze checks always cover EVERY final result on disk,
        # never just the --job-id subset: a spot check must not be able to reveal
        # candidates one at a time while the rest of the plan is absent.
        every = _discover(root, stage) if job_ids is not None else found
        have = {record["job"]["id"] for _, record in every}
        missing = sorted(set(planned) - have)
        if missing:
            raise ScoreError(f"refusing to score: {len(missing)} planned final jobs have no "
                             f"result file, e.g. {missing[:5]}")
        unplanned = sorted(have - set(planned))
        if unplanned and not allow_unplanned:
            raise ScoreError(f"refusing to score final jobs absent from the freeze: {unplanned[:5]} "
                             "(re-freeze with an updated plan, or pass --allow-unplanned)")
        if job_ids is not None:
            outside = sorted(set(job_ids) - set(planned))
            if outside and not allow_unplanned:
                raise ScoreError(f"--job-id lists jobs absent from the freeze: {outside[:5]}")
        for path, record in every:
            job_id = record["job"]["id"]
            entry = planned.get(job_id)
            if entry is None:
                continue
            if study.sha(path) != entry.get("result_sha256"):
                raise ScoreError(f"{job_id}: results file changed after the freeze")
            if record.get("predictions_sha256") != entry.get("predictions_sha256"):
                raise ScoreError(f"{job_id}: prediction hash changed after the freeze")

    rows = []
    label_cache = {}
    for path, record in found:
        verified = verify_record(root, record)
        job = verified["job"]
        seed = int(job["seed"])
        if seed not in label_cache:
            label_cache[seed] = study.query_labels(seed)
        truth = label_cache[seed]
        correct = int(np.count_nonzero(verified["labels"] == truth))
        metrics = record.get("metrics") or {}
        row = {
            "id": job["id"],
            "candidate_id": str(job["candidate_id"]),
            "seed": seed,
            "n": int(job["n"]),
            "correct": correct,
            "total": int(truth.shape[0]),
            "error_pct": 100.0 * (truth.shape[0] - correct) / truth.shape[0],
            "predictions_sha256": verified["predictions_sha256"],
            "result_sha256": study.sha(path),
            "metrics": {key: metrics[key] for key in METRIC_KEYS if key in metrics},
        }
        if freeze is not None:
            row["planned"] = job["id"] in (freeze.get("jobs") or {})
        rows.append(row)
    rows.sort(key=lambda row: (row["candidate_id"], row["n"], row["seed"]))

    aggregate = []
    keys = sorted({(row["candidate_id"], row["n"]) for row in rows})
    for candidate_id, n in keys:
        errors = [row["error_pct"] for row in rows if row["candidate_id"] == candidate_id and row["n"] == n]
        mean = float(np.mean(errors))
        sd = float(np.std(errors, ddof=1)) if len(errors) > 1 else None
        aggregate.append({
            "candidate_id": candidate_id, "n": n, "count": len(errors),
            "mean_error_pct": mean, "sd_error_pct": sd,
            "sem_error_pct": (sd / math.sqrt(len(errors))) if sd is not None else None,
            "min_error_pct": float(min(errors)), "max_error_pct": float(max(errors)),
        })
    aggregate.sort(key=lambda entry: (entry["mean_error_pct"], entry["candidate_id"], entry["n"]))

    # A filtered pass never overwrites the canonical table analysis.py consumes.
    scores_name = f"scores_{stage}_filtered.json" if job_ids else f"scores_{stage}.json"
    freeze_path = root / "predictions" / FREEZE_NAME
    if freeze is not None:
        # Final labels have now been read: seal the freeze so it can never be
        # replaced by another plan and scored again.
        freeze["scored"] = True
        freeze["scored_at_utc"] = study.utc()
        freeze["scores_path"] = f"results/{scores_name}"
        study.write_json(freeze_path, freeze)
    scores = {
        "schema_version": 1,
        "stage": stage,
        "scored_at_utc": study.utc(),
        "job_filter": sorted(job_ids) if job_ids else None,
        "data_manifest_sha256": study.sha(study.DATA_MANIFEST_PATH),
        "freeze_sha256": study.sha(freeze_path) if freeze is not None else None,
        "scores_path": f"results/{scores_name}",
        "job_count": len(rows),
        "jobs": rows,
        "aggregate": aggregate,
    }
    study.write_json(root / "results" / scores_name, scores)
    return scores


def render_table(scores: dict) -> str:
    aggregate = scores["aggregate"]
    if not aggregate:
        return f"No stage '{scores['stage']}' results found; nothing scored."
    lines = ["| candidate | n | seeds | mean error % | sd % | min % | max % |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for entry in aggregate:
        sd = "-" if entry["sd_error_pct"] is None else f"{entry['sd_error_pct']:.3f}"
        lines.append(
            f"| {entry['candidate_id']} | {entry['n']} | {entry['count']} | "
            f"{entry['mean_error_pct']:.3f} | {sd} | {entry['min_error_pct']:.3f} | "
            f"{entry['max_error_pct']:.3f} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage",
                        help="dev | final | any ad-hoc lowercase stage name "
                             "(only 'final' requires a prediction freeze)")
    parser.add_argument("--job-id", action="append", default=None,
                        help="restrict scoring to these job ids (repeatable)")
    parser.add_argument("--freeze-final", action="store_true",
                        help="write predictions/final_freeze.json from --plan; never scores")
    parser.add_argument("--plan", default="plans/final.json")
    parser.add_argument("--selection", default=SELECTION_NAME,
                        help="frozen selection rule that must exist before a final freeze")
    parser.add_argument("--refreeze", action="store_true",
                        help="supersede an existing, never-scored freeze (archives the old one)")
    parser.add_argument("--allow-unplanned", action="store_true",
                        help="score final results that are absent from the freeze")
    parser.add_argument("--root", default=str(study.ROOT),
                        help="directory holding results/, predictions/ and plans/ (tests only)")
    arguments = parser.parse_args(argv)
    root = Path(arguments.root).resolve()
    try:
        if arguments.freeze_final:
            if arguments.stage is not None:
                raise ScoreError("--freeze-final never scores; drop --stage")
            freeze = freeze_final(root, arguments.plan, refreeze=arguments.refreeze,
                                  selection_path=arguments.selection)
            print(f"Froze {freeze['job_count']} final prediction files at {freeze['frozen_at_utc']}.")
            return 0
        if arguments.stage is None:
            raise ScoreError("--stage dev|final is required (or --freeze-final)")
        if not study._STAGE_RE.match(arguments.stage):
            raise ScoreError(f"--stage {arguments.stage!r} is not a valid stage name")
        scores = score(root, arguments.stage, set(arguments.job_id) if arguments.job_id else None,
                       allow_unplanned=arguments.allow_unplanned)
    except ScoreError as error:
        print(f"score.py: {error}", file=sys.stderr)
        return 2
    print(render_table(scores))
    print(f"\n{scores['job_count']} job(s) scored -> {root / scores['scores_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
