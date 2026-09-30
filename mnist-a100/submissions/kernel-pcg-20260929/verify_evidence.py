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
BASELINE_SHA256 = "24453851a3faec1808bb157296ac3bab21f8c88b170dea11352bd98dbd6d0ba5"


def check_result(result, source_hash, filename, function, device):
    record = result["record"]
    assert result["returncode"] == 0 and not record["problems"]
    assert record["source_sha256"] == source_hash
    assert record["sandboxed"] is True and record["band_bp"] == 340
    assert record["device"] == device
    assert record["file"] == filename and record["function"] == function
    calls = [mnist.Call(**c) for c in record["calls"]]
    assert len(calls) == 15 and sum(c.dataset == "mnist" for c in calls) == 11
    assert record["holdout"] in mnist.FOREIGN
    assert sum(c.dataset == record["holdout"] for c in calls) == 4
    problems, score, _ = mnist.judge(calls, 340)
    assert not problems and math.isclose(score, record["ranked_ms"], abs_tol=1e-9, rel_tol=0)
    energy = record["energy"]
    assert energy["device"]["name"] == device
    timed_median = statistics.median(c.ms for c in calls if c.dataset == "mnist")
    rebuilt = mnist.energy_summary(energy["windows"], energy["device"], 340, timed_median)
    assert not energy["problems"] and not rebuilt["problems"]
    assert math.isclose(rebuilt["mj_per_call"], energy["mj_per_call"], abs_tol=1e-9, rel_tol=0)
    window = next(w for w in energy["windows"] if w["name"] == "method")
    assert window["calls"] == len(window["ms"]) == len(window["correct"])
    return score, rebuilt["mj_per_call"], energy["device"]["uuid"]


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
        score, mj, uuid = check_result(result, SOURCE_SHA256, "kernel_pcg.py", "classify",
                                       "NVIDIA A100-SXM4-80GB")
        boards.add(uuid)
        times.append(score)
        energies.append(mj)
        print(f"run {index}: {score:.6f} ms, {mj:.6f} mJ; passes")
    assert len(boards) == 3, "expected three distinct recorded boards"
    print(f"Median: {statistics.median(times):.6f} ms, {statistics.median(energies):.6f} mJ")

    comparison = HERE / "evidence" / "same-board"
    manifest = json.loads((comparison / "manifest.json").read_text())
    order = ["baseline", "ours", "ours", "baseline", "baseline", "ours"]
    assert manifest["order"] == order and manifest["scorer_sha256"] == SCORER_SHA256
    methods = manifest["methods"]
    for name, expected_hash in [("baseline", BASELINE_SHA256), ("ours", SOURCE_SHA256)]:
        assert methods[name]["sha256"] == expected_hash
        assert hashlib.sha256(methods[name]["source"].encode()).hexdigest() == expected_hash
    expected_entries = {"baseline": ("mlpg_k4_w256_s800_b512.py", "mlp"),
                        "ours": ("kernel_pcg.py", "classify")}
    paired = {name: [] for name in expected_entries}
    boards = set()
    for index, name in enumerate(order, 1):
        result = json.loads((comparison / f"run-{index}-{name}.json").read_text())
        assert result["index"] == index and result["method"] == name
        assert result["scorer_sha256"] == SCORER_SHA256
        assert result["source_sha256"] == methods[name]["sha256"]
        filename, function = expected_entries[name]
        score, mj, uuid = check_result(result, methods[name]["sha256"], filename, function,
                                       "NVIDIA A100 80GB PCIe")
        assert result["before"]["uuid"] == result["after"]["uuid"] == uuid
        assert result["before"]["power_limit_w"] == result["after"]["power_limit_w"] == 300.0
        assert result["record"]["energy"]["device"]["power_limit_w"] == 300.0
        boards.add(uuid)
        paired[name].append((score, mj))
        print(f"same-board {index} ({name}): {score:.6f} ms, {mj:.6f} mJ; passes")
    assert len(boards) == 1, "comparison changed physical GPUs"
    medians = {name: tuple(statistics.median(v[i] for v in values) for i in (0, 1))
               for name, values in paired.items()}
    print(f"Same-board speedup: {medians['baseline'][0] / medians['ours'][0]:.3f}x; "
          f"energy reduction: {100 * (1 - medians['ours'][1] / medians['baseline'][1]):.2f}%")
    print("Saved evidence verified; this is not an independent GPU reproduction.")


if __name__ == "__main__":
    main()
