"""Saved-record audit adapted from the published hybrid's reviewer verifier.
These are author execution records, not new independent reviewer execution.
Never qualify incomplete sets or replace failed runs. Data is read only.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("mnist", ROOT / "mnist.py")
mnist = importlib.util.module_from_spec(spec)
sys.modules["mnist"] = mnist
spec.loader.exec_module(mnist)


def audit(record_dir, source_path, difficulty=5, record_filename=None, function="classify"):
    source = source_path.read_bytes()
    sha = hashlib.sha256(source).hexdigest()
    mnist.check_source(source, function, source_path.name)
    band = mnist.DIFFICULTY[difficulty][0]
    expected_filename = record_filename or source_path.name
    rows = []
    for number in range(1, 4):
        p = record_dir / f"run-{number}.json"
        raw = p.read_bytes()
        result = json.loads(raw)
        record = result.get("record")
        row = {"run": number, "raw_sha256": hashlib.sha256(raw).hexdigest(),
               "returncode": result["returncode"], "passed": False}
        if record is None:
            row.update(record_present=False, stderr=result.get("stderr"), calls=0)
            rows.append(row)
            continue
        calls = [mnist.Call(**c) for c in record.get("calls", [])]
        digits = [c for c in calls if c.dataset == "mnist"]
        complete = (len(calls) == mnist.MNIST_CALLS + mnist.HOLDOUT_CALLS and
                    len(digits) == mnist.MNIST_CALLS and record["holdout"] in mnist.FOREIGN and
                    sum(c.dataset == record["holdout"] for c in calls) == mnist.HOLDOUT_CALLS)
        valid = all(c.total == mnist.TEST and 0 <= c.correct <= c.total and
                    all(math.isfinite(v) and v >= 0 for v in (c.ms, c.parent_ms, c.overhead_ms))
                    and c.ms <= mnist.MAX_CALL_MS for c in calls)
        checks = {"source_identity": record["source_sha256"] == sha,
                  "filename": record["file"] == expected_filename,
                  "entry_point": record["function"] == function,
                  "canonical_scorer_version": record["version"] == mnist.VERSION,
                  "band": record["band_bp"] == band,
                  "sandbox": record["sandboxed"] is True,
                  "complete_calls": complete, "valid_calls": valid,
                  "successful_return": result["returncode"] == 0,
                  "record_problems_empty": not record["problems"]}
        correct, total = sum(c.correct for c in digits), sum(c.total for c in digits)
        problems = score = summary = energy_rebuilt = None
        if complete and valid:
            problems, score, summary = mnist.judge(calls, band)
            checks["canonical_judge_passes"] = not problems
            checks["canonical_problems_exact"] = problems == record["problems"]
            checks["ranked_exact"] = (score == record["ranked_ms"] if not problems else record["ranked_ms"] is None)
            checks["summary_exact"] = summary == record["summary"]
        else:
            checks.update(canonical_judge_passes=False, ranked_exact=False)
        energy = record.get("energy")
        if energy and complete and valid:
            energy_rebuilt = mnist.energy_summary(energy["windows"], energy["device"], band,
                                                 statistics.median(c.ms for c in digits))
            checks["energy_reconstruction_exact"] = json.loads(json.dumps(energy_rebuilt)) == energy
            checks["energy_passes"] = not energy_rebuilt["problems"] and energy_rebuilt["mj_per_call"] is not None
        else:
            checks.update(energy_reconstruction_exact=False, energy_passes=False)
        row.update(checks=checks, recorded_filename=record["file"], calls=len(calls), mnist_calls=len(digits),
                   holdout_calls=sum(c.dataset == record["holdout"] for c in calls),
                   mnist_correct=correct, mnist_total=total,
                   observed_mnist_error_pct=100 * (total-correct) / total if total else None,
                   mnist_error_pct=100 * (total-correct) / total if total and complete and valid else None,
                   accuracy_gate_margin_labels=correct - mnist.required_correct(total, band) if complete and valid else None,
                   canonical_judge_applicable=complete and valid,
                   canonical_judge_skip_reason=None if complete and valid else
                     f"Saved {len(calls)} timed calls ({len(digits)} MNIST and {sum(c.dataset == record['holdout'] for c in calls)} holdout); canonical qualification requires all {mnist.MNIST_CALLS + mnist.HOLDOUT_CALLS} calls ({mnist.MNIST_CALLS} MNIST and {mnist.HOLDOUT_CALLS} holdout). Partial counts cannot be judged as a qualification run.",
                   canonical_problems=problems, record_problems=record["problems"],
                   ranked_ms=record["ranked_ms"], rebuilt_ranked_ms=score,
                   energy_mj=energy.get("mj_per_call") if energy else None,
                   energy_reconstruction_applicable=energy_rebuilt is not None,
                   energy_reconstruction_skip_reason=None if energy_rebuilt is not None else
                     "Complete valid timed-call record and saved energy windows are required; incomplete warm-up failure has neither a ranked result nor a qualifying energy measurement.",
                   energy_problems=energy_rebuilt["problems"] if energy_rebuilt else None,
                   device=energy["device"] if energy else record["device"],
                   score_passed=all(v for k, v in checks.items() if not k.startswith("energy_")),
                   energy_passed=(checks.get("energy_reconstruction_exact", False)
                                  and checks.get("energy_passes", False)),
                   passed=all(v for k, v in checks.items() if not k.startswith("energy_")))
        if result["returncode"] != 0:
            row["stderr"] = result.get("stderr")
        rows.append(row)
    passed = sum(r["passed"] for r in rows)
    total = sum(r.get("mnist_total", 0) for r in rows)
    correct = sum(r.get("mnist_correct", 0) for r in rows)
    return {"source_sha256": sha, "source_bytes": len(source), "difficulty": difficulty,
            "source_file": str(source_path), "function": function, "expected_record_filename": expected_filename,
            "filename_scope": "Author sources may be packaged under submission.py after original execution; exact source hash, not filename alone, establishes identity.",
            "scorer_sha256": hashlib.sha256((ROOT / "mnist.py").read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256((ROOT / "run_modal.py").read_bytes()).hexdigest(),
            "required_runs": 3, "passed_runs": passed, "energy_valid_runs": sum(r.get("energy_passed", False) for r in rows), "runs": rows,
            "median_ranked_ms": statistics.median(r["ranked_ms"] for r in rows) if passed == 3 else None,
            "median_energy_mj": statistics.median(r["energy_mj"] for r in rows) if passed == 3 and all(r.get("energy_passed") for r in rows) else None,
            "mnist_correct": correct, "mnist_total": total,
            "aggregate_mnist_error_pct": 100 * (total-correct) / total if total else None,
            "aggregate_scope": "Only saved observed MNIST calls, including partial failed runs; descriptive accuracy, never qualification of incomplete runs. Each full run must independently qualify.",
            "distinct_energy_gpu_uuids": len({r["device"].get("uuid") for r in rows if r.get("energy_mj") is not None and r["device"].get("uuid")}),
            "scope": "Independent canonical judge and complete energy-window reconstruction of frozen raw records; fresh execution is a separate evidence set."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("record_dir", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-filename")
    parser.add_argument("--difficulty", type=int, default=5)
    parser.add_argument("--function", default="classify")
    args = parser.parse_args()
    report = audit(args.record_dir, args.source, difficulty=args.difficulty, record_filename=args.record_filename, function=args.function)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "runs"},indent=2))
    for r in report["runs"]:
        print(json.dumps(r))
    sys.exit(0 if report["passed_runs"] == 3 else 1)
