"""Verify and score prediction files for the release-cutoffs study (copied from
release-ladder-20260924 unchanged).

This is the ONLY module in the study that imports or calls ``study.query_labels``.
Nothing here may ever be imported by a learner, by ``runner.py``, or by anything
shipped into a Modal container.

Usage
-----
  python score.py --stage dev
  python score.py --stage dev --job-id dev-krr-a-s2026092491-n500
  python score.py --stage dev --ensemble krr-a+mlp-b --weights 0.5,0.5
  python score.py --freeze-final --plan plans/final.json
  python score.py --stage final

Every scored job is re-verified end to end before its labels are touched:
  * predictions/<id>.npz bytes hash to results/<id>.json:predictions_sha256
  * logits/labels content hashes match output_sha256
  * labels == argmax(logits) with ties resolved to the lowest class index
  * the job's train/query row hashes and input array hashes match a fresh
    regeneration from study.job_arrays(seed, n)
For ``--stage final`` a prediction freeze (predictions/final_freeze.json) covering
every planned job is mandatory, and any post-freeze change to a result or
prediction file aborts scoring.  The freeze requires selection.json to exist,
cannot be replaced without ``--refreeze`` (which archives the superseded freeze),
and can never be replaced at all once final labels have been read against it.

ENSEMBLES
---------
``--ensemble A+B --weights wA,wB`` combines two candidates' ALREADY SAVED logits
for every (seed, level) both have, WITHOUT reading a single query label: each
candidate's logits are turned into probabilities with ``softmax(logits * T)``
(``T`` is the candidate's ``config['ensemble_temperature']``, default 1.0 --
kernel-ridge outputs are regression scores, not calibrated logits, so their
temperature matters), the probabilities are averaged with the given weights and
the mixture's log-probability is written as a normal prediction file for the
synthetic candidate ``ens-<A>+<B>`` in the derived stage ``<stage>-ensemble``.
Those records are then verified and scored by exactly the same code path as a
single recipe.  On final seeds every constituent job must appear unchanged in
the final prediction freeze, so an ensemble cannot evade it.
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
    "training_seconds", "fit_wall_seconds", "measured_wall_seconds",
    "epochs_completed", "truncated",
    "uses_query_features_unlabeled", "uses_query_images_unlabeled",
    "train_accuracy", "device", "device_name", "members",
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
            raise ScoreError(
                f"{job_id}: input_sha256[{name}] does not match a fresh regeneration of "
                f"job_arrays({seed},{n}). release_map is an eigendecomposition and LAPACK "
                "is not bit-reproducible across BLAS thread counts or builds, so re-run "
                "this with the same numpy and the same OPENBLAS_NUM_THREADS/OMP_NUM_THREADS "
                "as the process that wrote the plan before concluding the result is bad")
    if study.ahash(study.train_rows(seed, n)) != job.get("train_rows_sha256"):
        raise ScoreError(f"{job_id}: train_rows_sha256 mismatch")
    if study.ahash(study.query_rows(seed)) != job.get("query_rows_sha256"):
        raise ScoreError(f"{job_id}: query_rows_sha256 mismatch")
    # The per-fit budget is part of the claim being measured, so an over-budget
    # fit is not scoreable.  runner.run_fit refuses to return one; this catches a
    # record produced by an older runner or edited by hand.
    budget = job.get("time_budget_seconds")
    metrics = record.get("metrics") or {}
    wall = metrics.get("measured_wall_seconds")
    if wall is None:
        wall = record.get("job_wall_seconds")
    if budget is not None and wall is not None and float(wall) > float(budget) + 5.0:
        raise ScoreError(f"{job_id}: fit used {float(wall):.1f}s, over its "
                         f"time_budget_seconds={budget}")
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
        # blocking the whole stage from being scored.
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


# ------------------------------------------------------------------ ensembles ---
def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(scores, dtype=np.float64) * float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _member_temperature(job: dict, overrides: dict) -> float:
    candidate = str(job["candidate_id"])
    if candidate in overrides:
        return float(overrides[candidate])
    value = (job.get("config") or {}).get("ensemble_temperature", 1.0)
    return float(value)


def _check_declared_ensemble(root: Path, members, weights, temperatures) -> dict:
    """A final ensemble must match the one pre-declared in selection.json.

    Without this, the weight grid and the temperatures could be swept against the
    final query labels, which is exactly what
    ``selection_rule()['no_test_guided_changes']`` forbids.
    """
    selection_path = root / SELECTION_NAME
    if not selection_path.exists():
        raise ScoreError(f"refusing to build a final ensemble without {selection_path}")
    declared = (_load_json(selection_path) or {}).get("final_ensemble")
    if not isinstance(declared, dict):
        raise ScoreError(
            f"{selection_path} declares no 'final_ensemble' object; the ensemble "
            "finalist (members, weights, temperatures) must be pre-registered there")
    want_members = [str(m) for m in (declared.get("members") or [])]
    if want_members != [str(m) for m in members]:
        raise ScoreError(f"selection.json declares final_ensemble members {want_members}, "
                         f"not {[str(m) for m in members]}")
    raw = declared.get("weights")
    if raw is None or len(raw) != len(members):
        raise ScoreError("selection.json must declare final_ensemble.weights")
    total = float(sum(float(w) for w in raw))
    if total <= 0:
        raise ScoreError("selection.json declares degenerate final_ensemble.weights")
    want_weights = [float(w) / total for w in raw]
    if any(abs(a - b) > 1e-9 for a, b in zip(want_weights, list(weights))):
        raise ScoreError(f"selection.json declares final_ensemble weights {want_weights}, "
                         f"not {list(weights)}")
    want_temperatures = {str(k): float(v)
                         for k, v in (declared.get("temperatures") or {}).items()}
    got = {str(k): float(v) for k, v in dict(temperatures or {}).items()}
    if want_temperatures != got:
        raise ScoreError(f"selection.json declares final_ensemble temperatures "
                         f"{want_temperatures}, not {got}")
    return declared


def build_ensemble(root: Path, stage: str, spec: str, weights=None,
                   temperatures=None) -> dict:
    """Write synthetic ``ens-A+B`` results from saved logits.  Reads NO labels."""
    members = [part for part in str(spec).split("+") if part]
    if len(members) != 2:
        raise ScoreError(f"--ensemble takes exactly two candidates 'A+B', got {spec!r}")
    if stage not in ("dev", "final", "smoke"):
        raise ScoreError("--ensemble applies to a base stage ('dev', 'final' or 'smoke')")
    derived = study.ENSEMBLE_STAGE[stage]
    weights = [1.0, 1.0] if weights is None else [float(w) for w in weights]
    if len(weights) != 2 or min(weights) < 0 or sum(weights) <= 0:
        raise ScoreError("--weights must be two non-negative numbers that do not both vanish")
    total = sum(weights)
    weights = [w / total for w in weights]
    overrides = dict(temperatures or {})
    candidate_id = "ens-" + "+".join(members)

    freeze = None
    if stage == "final":
        freeze_path = root / "predictions" / FREEZE_NAME
        if not freeze_path.exists():
            raise ScoreError(
                "refusing to build a final ensemble before the constituent predictions "
                f"are frozen; run --freeze-final first ({freeze_path} is absent)")
        freeze = _load_json(freeze_path)
        if freeze.get("scored") is True:
            raise ScoreError(
                "the final freeze has already been scored against the query labels; "
                "an ensemble built now would be chosen with those labels in hand "
                "(selection_rule.no_test_guided_changes)")
        _check_declared_ensemble(root, members, weights, overrides)

    by_member = {}
    for member in members:
        by_member[member] = {}
        for path, record in _discover(root, stage):
            job = record.get("job") or {}
            if job.get("candidate_id") != member:
                continue
            if freeze is not None:
                entry = (freeze.get("jobs") or {}).get(job["id"])
                if entry is None:
                    raise ScoreError(f"{job['id']} is not in the final freeze; refusing "
                                     "to use it in a final ensemble")
                if study.sha(path) != entry.get("result_sha256"):
                    raise ScoreError(f"{job['id']}: results file changed after the freeze")
                if record.get("predictions_sha256") != entry.get("predictions_sha256"):
                    raise ScoreError(f"{job['id']}: prediction hash changed after the freeze")
            by_member[member][(int(job["seed"]), int(job["n"]))] = record
        if not by_member[member]:
            raise ScoreError(f"no stage {stage!r} results found for candidate {member!r}")

    shared = sorted(set(by_member[members[0]]) & set(by_member[members[1]]))
    if not shared:
        raise ScoreError(f"{members[0]} and {members[1]} share no (seed, level) results "
                         f"in stage {stage!r}")
    written = []
    for seed, n in shared:
        mixture = np.zeros((study.QUERY_COUNT, 10), dtype=np.float64)
        constituents, used_temperatures = [], {}
        wall, truncated, transductive = 0.0, False, False
        for weight, member in zip(weights, members):
            record = by_member[member][(seed, n)]
            verified = verify_record(root, record)
            temperature = _member_temperature(verified["job"], overrides)
            used_temperatures[member] = temperature
            mixture += weight * _softmax(verified["logits"], temperature)
            metrics = record.get("metrics") or {}
            wall += float(metrics.get("fit_wall_seconds") or 0.0)
            truncated = truncated or bool(metrics.get("truncated"))
            transductive = transductive or bool(
                metrics.get("uses_query_features_unlabeled")
                or metrics.get("uses_query_images_unlabeled"))
            constituents.append({
                "candidate_id": member, "job_id": verified["job"]["id"],
                "weight": weight, "temperature": temperature,
                "predictions_sha256": verified["predictions_sha256"],
                "result_sha256": study.sha(root / "results" / f"{verified['job']['id']}.json"),
            })
        logits = np.ascontiguousarray(np.log(np.maximum(mixture, 1e-300)), dtype=np.float32)
        labels = _argmax_lowest(logits)
        config = {"family": "ensemble", "members": members, "weights": weights,
                  "temperatures": used_temperatures, "rule": "weighted mean of "
                  "softmax(logits*T), reported as log-probability"}
        job = study.make_job(derived, seed, n, candidate_id, config,
                             learner_seed=0, time_budget_seconds=1, device_kind="cpu")
        # Rebuilding an ensemble with the SAME weights/temperatures is idempotent;
        # silently re-mixing a final ensemble with different ones after the
        # constituents were frozen would be a post-hoc choice, so it is refused.
        existing_path = root / "results" / f"{job['id']}.json"
        if derived == "final-ensemble" and existing_path.exists():
            existing = (_load_json(existing_path).get("job") or {}).get("config")
            if existing != job["config"]:
                raise ScoreError(
                    f"{job['id']} already exists with different ensemble weights or "
                    "temperatures; a final ensemble may not be re-mixed after the fact")
        predictions = root / "predictions" / f"{job['id']}.npz"
        temporary = predictions.with_suffix(".npz.part")
        with temporary.open("wb") as handle:
            np.savez(handle, logits=logits, labels=labels)
        temporary.replace(predictions)
        record = {
            "job": job,
            "metrics": {"fit_wall_seconds": wall, "truncated": truncated,
                        "uses_query_features_unlabeled": transductive,
                        "members": [c["job_id"] for c in constituents]},
            "provenance": {"built_by": "score.py --ensemble", "built_at": study.utc(),
                           "base_stage": stage, "constituents": constituents,
                           "freeze_sha256": (study.sha(root / "predictions" / FREEZE_NAME)
                                             if freeze is not None else None)},
            "completed_at": study.utc(), "device": "cpu", "learner_module": "score.py",
            "predictions_path": str(predictions.relative_to(root)),
            "predictions_sha256": study.sha(predictions),
            "output_sha256": {"logits": study.ahash(logits), "labels": study.ahash(labels)},
            "output_specs": {"logits": {"shape": list(logits.shape), "dtype": "float32"},
                             "labels": {"shape": list(labels.shape), "dtype": "uint8"}},
            "query_labels_supplied": False,
        }
        study.write_json(root / "results" / f"{job['id']}.json", record)
        written.append(job["id"])
    return {"stage": derived, "candidate_id": candidate_id, "weights": weights,
            "jobs": written, "count": len(written)}


# --------------------------------------------------------------------- scoring ---
def score(root: Path, stage: str, job_ids=None, allow_unplanned: bool = False) -> dict:
    root = Path(root)
    found = _discover(root, stage, job_ids)
    freeze = None
    unplanned = []
    if stage in ("final", "final-ensemble"):
        freeze_path = root / "predictions" / FREEZE_NAME
        if not freeze_path.exists():
            raise ScoreError(
                f"refusing to score stage {stage!r} without {freeze_path}; run "
                "`score.py --freeze-final --plan plans/final.json` first")
        freeze = _load_json(freeze_path)
    if stage == "final-ensemble":
        # A derived final stage is only as frozen as its constituents.
        entries = freeze.get("jobs") or {}
        for _, record in found:
            constituents = (record.get("provenance") or {}).get("constituents") or []
            if not constituents:
                raise ScoreError(f"{record['job']['id']}: a final ensemble must record "
                                 "its constituents")
            for member in constituents:
                if member.get("job_id") not in entries:
                    raise ScoreError(f"{record['job']['id']}: constituent "
                                     f"{member.get('job_id')} is not in the final freeze")
                if member.get("predictions_sha256") != entries[member["job_id"]].get(
                        "predictions_sha256"):
                    raise ScoreError(f"{record['job']['id']}: constituent "
                                     f"{member.get('job_id')} changed after the freeze")
    if stage == "final":
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
        rows.append({
            "id": job["id"],
            "candidate_id": str(job["candidate_id"]),
            "seed": seed,
            "n": int(job["n"]),
            "correct": correct,
            "total": int(truth.shape[0]),
            "error_pct": 100.0 * (truth.shape[0] - correct) / truth.shape[0],
            "device_kind": job.get("device_kind"),
            "predictions_sha256": verified["predictions_sha256"],
            "result_sha256": study.sha(path),
            "metrics": {key: metrics[key] for key in METRIC_KEYS if key in metrics},
        })
        if job["id"] in unplanned:
            rows[-1]["unplanned"] = True
    rows.sort(key=lambda row: (row["candidate_id"], row["n"], row["seed"]))

    aggregate = []
    keys = sorted({(row["candidate_id"], row["n"]) for row in rows})
    for candidate_id, n in keys:
        matching = [row for row in rows if row["candidate_id"] == candidate_id and row["n"] == n]
        errors = [row["error_pct"] for row in matching]
        mean = float(np.mean(errors))
        sd = float(np.std(errors, ddof=1)) if len(errors) > 1 else None
        total = sum(row["total"] for row in matching)
        wrong = sum(row["total"] - row["correct"] for row in matching)
        pooled = 100.0 * wrong / total
        aggregate.append({
            "candidate_id": candidate_id, "n": n, "count": len(errors),
            "mean_error_pct": mean, "sd_error_pct": sd,
            "sem_error_pct": (sd / math.sqrt(len(errors))) if sd is not None else None,
            "pooled_error_pct": pooled,
            "pooled_queries": total, "pooled_wrong": wrong,
            "pooled_binomial_se_pct": 100.0 * math.sqrt(
                max(pooled / 100.0 * (1 - pooled / 100.0), 0.0) / total),
            "min_error_pct": float(min(errors)), "max_error_pct": float(max(errors)),
        })
    aggregate.sort(key=lambda entry: (entry["n"], entry["pooled_error_pct"], entry["candidate_id"]))

    # A filtered pass never overwrites the canonical table.
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
        "unplanned_jobs": list(unplanned),
        "unplanned_disclosure": (
            "scored with --allow-unplanned: the jobs listed in 'unplanned_jobs' were "
            "produced after the final prediction freeze and are NOT part of the frozen "
            "final plan; they may not be used in the reported thresholds without "
            "disclosing them as post-freeze additions"
        ) if unplanned else None,
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
    lines = ["| candidate | n | seeds | pooled error % | mean % | sd % | min % | max % |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for entry in aggregate:
        sd = "-" if entry["sd_error_pct"] is None else f"{entry['sd_error_pct']:.3f}"
        lines.append(
            f"| {entry['candidate_id']} | {entry['n']} | {entry['count']} | "
            f"{entry['pooled_error_pct']:.3f} | {entry['mean_error_pct']:.3f} | {sd} | "
            f"{entry['min_error_pct']:.3f} | {entry['max_error_pct']:.3f} |")
    return "\n".join(lines)


def _parse_temperatures(text):
    if not text:
        return {}
    out = {}
    for part in str(text).split(","):
        if not part.strip():
            continue
        name, _, value = part.partition("=")
        if not _:
            raise ScoreError("--temperatures takes 'candidate=value' pairs")
        out[name.strip()] = float(value)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage",
                        help="dev | final | dev-ensemble | final-ensemble | any ad-hoc "
                             "lowercase stage name (only 'final' requires a freeze)")
    parser.add_argument("--job-id", action="append", default=None,
                        help="restrict scoring to these job ids (repeatable)")
    parser.add_argument("--freeze-final", action="store_true",
                        help="write predictions/final_freeze.json from --plan; never scores")
    parser.add_argument("--plan", default="plans/final.json")
    parser.add_argument("--selection", default=SELECTION_NAME)
    parser.add_argument("--refreeze", action="store_true")
    parser.add_argument("--allow-unplanned", action="store_true")
    parser.add_argument("--ensemble", default=None,
                        help="combine two candidates' saved logits, e.g. 'krr-a+mlp-b'")
    parser.add_argument("--weights", default=None,
                        help="two comma-separated non-negative weights (default 0.5,0.5)")
    parser.add_argument("--temperatures", default=None,
                        help="override per-candidate softmax temperatures, "
                             "e.g. 'krr-a=4.0,mlp-b=1.0'")
    parser.add_argument("--build-only", action="store_true",
                        help="with --ensemble: write the synthetic results, read no labels")
    parser.add_argument("--root", default=str(study.ROOT))
    arguments = parser.parse_args(argv)
    root = Path(arguments.root).resolve()
    try:
        if arguments.freeze_final:
            if arguments.stage is not None:
                raise ScoreError("--freeze-final never scores; drop --stage")
            freeze = freeze_final(root, arguments.plan, refreeze=arguments.refreeze,
                                  selection_path=arguments.selection)
            print(f"Froze {freeze['job_count']} final prediction files at "
                  f"{freeze['frozen_at_utc']}.")
            return 0
        if arguments.stage is None:
            raise ScoreError("--stage dev|final is required (or --freeze-final)")
        if not study._STAGE_RE.match(arguments.stage):
            raise ScoreError(f"--stage {arguments.stage!r} is not a valid stage name")
        stage = arguments.stage
        if arguments.ensemble:
            weights = ([float(v) for v in arguments.weights.split(",")]
                       if arguments.weights else None)
            built = build_ensemble(root, stage, arguments.ensemble, weights,
                                   _parse_temperatures(arguments.temperatures))
            print(json.dumps(built, indent=2, sort_keys=True))
            if arguments.build_only:
                return 0
            stage = built["stage"]
        scores = score(root, stage, set(arguments.job_id) if arguments.job_id else None,
                       allow_unplanned=arguments.allow_unplanned)
    except ScoreError as error:
        print(f"score.py: {error}", file=sys.stderr)
        return 2
    print(render_table(scores))
    print(f"\n{scores['job_count']} job(s) scored -> {root / scores['scores_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
