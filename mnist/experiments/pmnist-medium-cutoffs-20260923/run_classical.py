#!/usr/bin/env python
"""Run the CPU classical baselines against a study plan.  No GPU, no Modal.

    /tmp/pmnist-env/bin/python run_classical.py --make-plan \
        --candidates plans/classical_candidates.json \
        --levels 1000,10000 --seeds 2026092301,2026092302 --out plans/classical.json

    /tmp/pmnist-env/bin/python run_classical.py --plan plans/classical.json

Results are written in the study-wide contract shared with the GPU runner:
``results/<job_id>.json`` plus ``predictions/<job_id>.npz``.  Query labels are
never loaded here; ``study.job_arrays`` only ever hands back query *images*.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent
STAGE_DEFAULT = "dev"
LEARNER_SEED_DEFAULT = 11


def _study():
    """Import ``study`` lazily so this module loads before the builder lands."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import study  # noqa: PLC0415
    return study


def _sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(array):
    canonical = np.ascontiguousarray(
        array.astype(array.dtype.newbyteorder("<"), copy=False))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _software():
    import scipy
    import sklearn
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _selection_ids(path):
    """Candidate ids committed in selection.json (accepts 'id' or 'candidate_id')."""
    payload = json.loads(Path(path).read_text())
    entries = payload["candidates"] if isinstance(payload, dict) else payload
    ids = set()
    for entry in entries:
        identifier = entry.get("id", entry.get("candidate_id"))
        if identifier is None:
            raise ValueError(f"{path}: every selection entry needs an id")
        ids.add(str(identifier))
    return ids


def make_plan(candidates_path, levels, seeds, out_path,
              stage=STAGE_DEFAULT, learner_seed=LEARNER_SEED_DEFAULT,
              time_budget_seconds=1200, selection_path=None):
    """Build a classical plan, with the same protocol guards plan.py applies.

    Model selection may only touch dev seeds and final runs may only touch
    candidates already committed in selection.json -- otherwise a 'dev' plan
    over final seeds would be scored (score.py only demands a prediction freeze
    for stage 'final') and the winner could be picked against final query labels.
    """
    study = _study()
    stage = str(stage)
    if stage not in ("dev", "final"):
        raise ValueError(f"stage must be 'dev' or 'final', got {stage!r}")
    permitted = (study.allowed_seeds(stage) if hasattr(study, "allowed_seeds")
                 else (study.DEV_SEEDS if stage == "dev" else study.FINAL_SEEDS))
    levels = [int(value) for value in levels]
    seeds = [int(value) for value in seeds]
    for level in levels:
        if level not in study.LEVELS:
            raise ValueError(f"{level} is not a study level {list(study.LEVELS)}")
    for seed in seeds:
        if seed not in set(permitted):
            raise ValueError(f"{seed} is not a {stage} seed {list(permitted)}")
    candidates = json.loads(Path(candidates_path).read_text())
    if stage == "final":
        selection_path = Path(selection_path or (ROOT / "selection.json"))
        if not selection_path.exists():
            raise ValueError(f"stage 'final' needs a committed {selection_path}")
        committed = _selection_ids(selection_path)
        for candidate in candidates:
            if str(candidate["candidate_id"]) not in committed:
                raise ValueError(
                    f"candidate {candidate['candidate_id']!r} is not in "
                    f"{selection_path}; final runs are restricted to the "
                    "committed selection")
    jobs = []
    for seed in seeds:
        for level in levels:
            for candidate in candidates:
                config = {k: v for k, v in candidate.items() if k != "candidate_id"}
                jobs.append(study.make_job(stage, int(seed), int(level),
                                           candidate["candidate_id"], config,
                                           int(learner_seed),
                                           time_budget_seconds=time_budget_seconds))
    _write_json(out_path, jobs)
    return jobs


def _source_hashes():
    return {name: _sha_file(ROOT / name)
            for name in ("classical.py", "study.py", "canonical_data.py")}


def _already_done(job, results_dir, predictions_dir, verbose=True):
    """True only when the stored record was produced by *this* job and code.

    Job ids carry no config hash (the study fixes the id format), so comparing
    the input arrays alone would let an edited config, learner seed, time budget
    or an edited classical.py be silently skipped, leaving results/ disagreeing
    with the plan that supposedly produced it.
    """
    path = Path(results_dir) / f"{job['id']}.json"
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    reason = None
    if record.get("job") != job:
        reason = "job specification (config/seed/budget/inputs) changed"
    else:
        stored = (record.get("provenance") or {}).get("source_sha256") or {}
        current = _source_hashes()
        changed = sorted(name for name, digest in current.items()
                         if stored.get(name) != digest)
        if changed:
            reason = f"source changed since it ran: {', '.join(changed)}"
    if reason is not None:
        if verbose:
            print(f"rerunning {job['id']}: {reason}", flush=True)
        return False
    predictions = Path(predictions_dir) / f"{job['id']}.npz"
    if not predictions.exists():
        return False
    return _sha_file(predictions) == record.get("predictions_sha256")


def _record_path(path):
    """Path as score.py will resolve it: relative to ROOT when it lives there."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_job(job, results_dir, predictions_dir):
    study = _study()
    import classical

    arrays = study.job_arrays(job["seed"], job["n"])
    if set(arrays) != {"train_x", "train_y", "query_x"}:
        raise RuntimeError("study.job_arrays returned unexpected keys")
    observed = {name: study.ahash(arrays[name]) for name in ("train_x", "train_y", "query_x")}
    if observed != job["input_sha256"]:
        raise RuntimeError(f"input hash mismatch for {job['id']}: {observed} != "
                           f"{job['input_sha256']}")

    started = time.time()
    output = classical.fit_predict(arrays["train_x"], arrays["train_y"],
                                   arrays["query_x"], job["config"],
                                   job["learner_seed"])
    elapsed = time.time() - started

    logits = np.ascontiguousarray(output["logits"], dtype=np.float32)
    labels = np.ascontiguousarray(output["labels"], dtype=np.uint8)
    if logits.shape != (arrays["query_x"].shape[0], 10):
        raise RuntimeError(f"bad logits shape {logits.shape}")
    if not np.array_equal(labels, np.argmax(logits, axis=1).astype(np.uint8)):
        raise RuntimeError("labels must be the first-maximum argmax of logits")

    predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_dir / f"{job['id']}.npz"
    temporary = predictions_path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez(stream, logits=logits, labels=labels)
    temporary.replace(predictions_path)

    metrics = dict(output["metrics"])
    # 'fit_wall_seconds' is the study-wide key (score.py's METRIC_KEYS allowlist
    # drops anything else); 'wall_seconds' is kept as a legacy alias.
    metrics["fit_wall_seconds"] = elapsed
    metrics["wall_seconds"] = elapsed
    metrics["truncated"] = bool(metrics.get("truncated", False)) or \
        elapsed > job.get("time_budget_seconds", 1200)

    record = {
        "job": job,
        "metrics": metrics,
        "provenance": {
            "source_sha256": _source_hashes(),
            "runner": "run_classical.py",
            "runner_sha256": _sha_file(ROOT / "run_classical.py"),
        },
        "completed_at": study.utc(),
        "hardware": platform.platform(),
        "software": _software(),
        "predictions_path": _record_path(predictions_path),
        "predictions_sha256": _sha_file(predictions_path),
        "output_sha256": {"logits": _array_hash(logits), "labels": _array_hash(labels)},
        "query_labels_supplied": False,
    }
    _write_json(Path(results_dir) / f"{job['id']}.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--make-plan", action="store_true")
    parser.add_argument("--candidates", type=Path,
                        default=ROOT / "plans" / "classical_candidates.json")
    parser.add_argument("--levels", default="1000,10000")
    parser.add_argument("--seeds", default="2026092301,2026092302")
    parser.add_argument("--stage", default=STAGE_DEFAULT, choices=("dev", "final"))
    parser.add_argument("--selection", type=Path, default=ROOT / "selection.json",
                        help="committed selection.json, required for --stage final")
    parser.add_argument("--learner-seed", type=int, default=LEARNER_SEED_DEFAULT)
    parser.add_argument("--time-budget-seconds", type=int, default=1200)
    parser.add_argument("--out", type=Path, default=ROOT / "plans" / "classical.json")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "predictions")
    parser.add_argument("--only", default=None, help="substring filter on job ids")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.make_plan:
        levels = [int(v) for v in str(arguments.levels).split(",") if v.strip()]
        seeds = [int(v) for v in str(arguments.seeds).split(",") if v.strip()]
        jobs = make_plan(arguments.candidates, levels, seeds, arguments.out,
                         arguments.stage, arguments.learner_seed,
                         arguments.time_budget_seconds,
                         selection_path=arguments.selection)
        print(f"wrote {len(jobs)} jobs -> {arguments.out}")
        return 0

    if arguments.plan is None:
        parser.error("--plan is required unless --make-plan is given")
    jobs = json.loads(Path(arguments.plan).read_text())
    if arguments.only:
        jobs = [job for job in jobs if arguments.only in job["id"]]
    pending = [job for job in jobs
               if not _already_done(job, arguments.results_dir, arguments.predictions_dir)]
    print(f"{len(jobs)} jobs, {len(pending)} pending")
    if arguments.dry_run:
        for job in pending:
            print("pending", job["id"])
        return 0
    failures = []
    for index, job in enumerate(pending, 1):
        started = time.time()
        try:
            record = run_job(job, arguments.results_dir, arguments.predictions_dir)
        except Exception as error:                      # one bad job, not the batch
            failures.append({"id": job["id"], "error": f"{type(error).__name__}: {error}",
                             "traceback": traceback.format_exc(),
                             "seconds": time.time() - started})
            print(f"[{index}/{len(pending)}] {job['id']} FAILED after "
                  f"{time.time() - started:.1f}s: {type(error).__name__}: {error}",
                  flush=True)
            traceback.print_exc()
            continue
        print(f"[{index}/{len(pending)}] {job['id']} "
              f"cv={record['metrics'].get('cv_accuracy')} "
              f"{time.time() - started:.1f}s", flush=True)
    if failures:
        log = Path(arguments.results_dir).resolve().parent / "logs" / "classical_failures.json"
        _write_json(log, {"written_at": _study().utc(), "failures": failures})
        print(f"{len(failures)} job(s) failed; details in {log}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
