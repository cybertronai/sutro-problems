"""Reconstruct saved official records from a JSON output directory; local only, no GPU.

    python verify_final.py official-candidate-3 [--source candidate.py]

Reads the manifest.json and run-N.json files validate_final.py writes (the old
validate_official.py directories work too) and rebuilds every check the scorer
made: source hash, band, call counts, holdout, judge-ranked medians, energy
medians and distinct GPU UUIDs. Checks that need information the directory does
not have are reported as n/a instead of failing.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import mnist


def check_run(result, source_sha, source_name, band):
    record = result.get("record")
    checks, facts = {}, {}
    if record is None:
        return {"record present": False}, {}
    checks["run returned 0"] = result["returncode"] == 0
    checks["no problems"] = not record["problems"]
    checks["source hash"] = record["source_sha256"] == source_sha if source_sha else None
    checks["file name"] = record["file"] == source_name if source_name else None
    checks["band"] = record["band_bp"] == band if band else None
    checks["sandboxed"] = record["sandboxed"] is True
    calls = [mnist.Call(**c) for c in record["calls"]]
    mnist_calls = [c for c in calls if c.dataset == "mnist"]
    holdout_calls = [c for c in calls if c.dataset != "mnist"]
    checks["15 calls (11+4)"] = len(calls) == 15 and len(mnist_calls) == 11 and len(holdout_calls) == 4
    checks["holdout"] = (record["holdout"] in mnist.FOREIGN
                         and sum(c.dataset == record["holdout"] for c in holdout_calls) == 4)
    judge_band = band or record.get("band_bp")
    if calls and judge_band:
        problems, score, summary = mnist.judge(calls, judge_band)
        checks["judge rebuild"] = (not problems and record["ranked_ms"] is not None
                                   and math.isclose(score, record["ranked_ms"], abs_tol=1e-9, rel_tol=0))
        facts["score"] = score
    else:
        checks["judge rebuild"] = False
    facts["mnist_error"] = (1 - sum(c.correct for c in mnist_calls) / max(1, sum(c.total for c in mnist_calls)))
    facts["holdout_error"] = (1 - sum(c.correct for c in holdout_calls) /
                              max(1, sum(c.total for c in holdout_calls))) if holdout_calls else None
    energy = record.get("energy") or {}
    device = energy.get("device") or {}
    facts["device"], facts["uuid"] = device.get("name"), device.get("uuid")
    if energy.get("mj_per_call") is not None:
        timed_median = statistics.median(c.ms for c in mnist_calls)
        rebuilt = mnist.energy_summary(energy["windows"], energy["device"], record["band_bp"], timed_median)
        checks["energy rebuild"] = (not energy.get("problems") and not rebuilt["problems"]
                                    and math.isclose(rebuilt["mj_per_call"], energy["mj_per_call"],
                                                     abs_tol=1e-9, rel_tol=0))
        facts["energy"] = rebuilt["mj_per_call"]
    else:
        checks["energy rebuild"] = None
        facts["energy"] = None
    facts["function"] = record["function"]
    facts["holdout_name"] = record["holdout"]
    return checks, facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source", type=Path, help="candidate the records must match, if not in the manifest")
    args = parser.parse_args()

    manifest_path = args.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    paths = sorted(args.directory.glob("run-*.json"))
    if not paths:
        parser.error(f"no run-*.json in {args.directory}")
    results = [json.loads(p.read_text()) for p in paths]

    if args.source:
        source_bytes = args.source.read_bytes()
        source_sha, source_name = hashlib.sha256(source_bytes).hexdigest(), args.source.name
        if manifest.get("source_sha256"):
            assert manifest["source_sha256"] == source_sha, "--source does not match the manifest's candidate"
    elif manifest.get("source"):
        source_bytes = manifest["source"].encode()
        source_sha, source_name = hashlib.sha256(source_bytes).hexdigest(), None
        assert source_sha == manifest["source_sha256"], "manifest source and hash disagree"
    else:
        source_sha, source_name = None, None

    difficulty = manifest.get("difficulty")
    band = mnist.DIFFICULTY[difficulty][0] if difficulty in mnist.DIFFICULTY else None
    if difficulty is not None:
        assert band, f"manifest difficulty {difficulty} has no band"
        assert manifest["runs"] == len(results), f"manifest says {manifest['runs']} runs, found {len(results)}"

    print(f"{args.directory}: {len(results)} runs, difficulty {difficulty}, "
          f"source {(source_name or 'from manifest')} sha {(source_sha or 'unknown')[:8]}")
    print(f"{'run':>3}  {'device':<23} {'gpu':<12} {'ranked ms':>12} {'energy mJ':>12} "
          f"{'MNIST err':>9} {'holdout err':>11}  status")
    failures, all_checks, scores, energies, uuids = [], [], [], [], []
    for index, result in enumerate(results, 1):
        checks, facts = check_run(result, source_sha, source_name, band)
        all_checks.append(checks)
        scores.append(facts.get("score", result.get("record", {}).get("ranked_ms")))
        energies.append(facts.get("energy"))
        uuids.append(facts.get("uuid"))
        bad = [name for name, ok in checks.items() if ok is False]
        failures += [f"run {index}: {name}" for name in bad]
        gpu = (facts.get("uuid") or "-")
        gpu = gpu[4:12] if gpu.startswith("GPU-") else gpu
        energy_text = f"{facts['energy']:,.3f}" if facts.get("energy") is not None else "-"
        holdout = f"{100 * facts['holdout_error']:.2f}% {facts['holdout_name']}" if facts.get("holdout_error") is not None else "-"
        print(f"{index:>3}  {facts.get('device') or '-':<23} {gpu:<12} {facts.get('score', float('nan')):>12,.6f} "
              f"{energy_text:>12} {100 * facts.get('mnist_error', 0):>8.2f}% {holdout:>11}  "
              f"{'PASS' if not bad else 'FAIL'}")

    labels = list(all_checks[0])
    for label in labels:
        states = [c.get(label) for c in all_checks]
        message = ("n/a" if all(s is None for s in states) else
                   "FAIL" if any(s is False for s in states) else f"OK ({sum(s is True for s in states)}/{len(states)})")
        print(f"  {label}: {message}")
    known_uuids = [u for u in uuids if u]
    if len(known_uuids) == len(results):
        distinct = len(set(known_uuids)) == len(results)
        print(f"  distinct GPU UUIDs: {'OK' if distinct else 'FAIL'} ({len(set(known_uuids))} of {len(results)})")
        failures += [] if distinct else ["distinct GPU UUIDs"]
    else:
        print(f"  distinct GPU UUIDs: n/a (recorded in {len(known_uuids)} of {len(results)} runs)")
    median_score = statistics.median(s for s in scores if s is not None) if any(s is not None for s in scores) else None
    known_energies = [e for e in energies if e is not None]
    print(f"  ranked medians: median {median_score:,.6f} ms" if median_score is not None else "  ranked medians: n/a")
    if known_energies:
        print(f"  energy medians: median {statistics.median(known_energies):,.6f} mJ over {len(known_energies)} runs")
    if failures:
        print("FAILED: " + "; ".join(failures))
        sys.exit(1)
    print(f"Verified {len(results)}/{len(results)} saved records: no disqualification problems; "
          "local reconstruction only, not an independent GPU reproduction.")


if __name__ == "__main__":
    main()
