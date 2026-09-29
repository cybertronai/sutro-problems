"""Check frozen source and reconstruct saved evidence; no GPU or Modal required."""

import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
SCORER = HERE.parents[1]
sys.path.insert(0, str(SCORER))
import mnist

SOURCE_SHA256 = "fc0ddef0d3dba8f00fc44f9cd576548c2a5184c5cb88741767d33d6f10b38012"
SCORER_SHA256 = "1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064"


def main():
    source = (HERE / "kernel_pcg.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256, "candidate changed"
    assert hashlib.sha256((SCORER / "mnist.py").read_bytes()).hexdigest() == SCORER_SHA256, "scorer changed"
    mnist.check_source(source, "classify", "kernel_pcg.py")
    assert not mnist.review_flags(source)
    times, energies, boards = [], [], set()
    for index in (1, 2, 3):
        path = HERE / "evidence" / f"run-{index}.json"
        result = json.loads(path.read_text())
        record = result["record"]
        assert result["returncode"] == 0 and not record["problems"]
        assert record["source_sha256"] == SOURCE_SHA256
        assert record["sandboxed"] is True and record["band_bp"] == 340
        assert record["device"] == "NVIDIA A100-SXM4-80GB"
        assert record["file"] == "kernel_pcg.py" and record["function"] == "classify"
        calls = [mnist.Call(**c) for c in record["calls"]]
        assert len(calls) == 15 and sum(c.dataset == "mnist" for c in calls) == 11
        assert record["holdout"] in mnist.FOREIGN
        assert sum(c.dataset == record["holdout"] for c in calls) == 4
        problems, score, _ = mnist.judge(calls, 340)
        assert not problems and math.isclose(score, record["ranked_ms"], abs_tol=1e-9, rel_tol=0)
        energy = record["energy"]
        timed_median = statistics.median(c.ms for c in calls if c.dataset == "mnist")
        rebuilt = mnist.energy_summary(energy["windows"], energy["device"], 340, timed_median)
        assert not energy["problems"] and not rebuilt["problems"]
        assert math.isclose(rebuilt["mj_per_call"], energy["mj_per_call"], abs_tol=1e-9, rel_tol=0)
        window = next(w for w in energy["windows"] if w["name"] == "method")
        assert window["calls"] == len(window["ms"]) == len(window["correct"])
        boards.add(energy["device"]["uuid"])
        times.append(score)
        energies.append(rebuilt["mj_per_call"])
        print(f"run {index}: {score:.6f} ms, {rebuilt['mj_per_call']:.6f} mJ; passes")
    assert len(boards) == 3, "expected three distinct recorded boards"
    print(f"Median: {statistics.median(times):.6f} ms, {statistics.median(energies):.6f} mJ")
    print("Saved evidence verified; this is not an independent GPU reproduction.")


if __name__ == "__main__":
    main()
