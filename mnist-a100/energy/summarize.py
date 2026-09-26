"""Tables for energy/README.md from the saved runs, recomputing every energy from its raw windows.

    python energy/summarize.py            # from mnist-a100/: prints the tables, writes results/summary.json

Each results/<entry>/run-N.json is what `run_modal.py --json` saved: the scorer's stdout and its
record (every timed call, and the seven energy windows with the counter's joules and seconds).
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mnist  # noqa: E402

# directory under results/, what it is, the difficulty it was run at, its source
ENTRIES = [
    ("fast_mlp", "CUDA-graph MLP, 60-256-256-10, 200 steps (PR #96)", 1, "entries/fast_mlp.py"),
    ("mlp_k1_w1024_s100_b512", "eager MLP, 60-1024-1024-10, 100 steps", 1, "entries/mlp_k1_w1024_s100_b512.py"),
    ("example", "`example.py`: eager MLP, 60-1024-1024-10, 400 steps", 1, "../example.py"),
    ("mlpg_k4_w256_s800_b512", "CUDA-graph ensemble, 4 x 60-256-256-10, 800 steps", 2,
     "entries/mlpg_k4_w256_s800_b512.py"),
    ("mlp_k16_w1024_s400_b512", "eager ensemble, 16 x 60-1024-1024-10, 400 steps", 2,
     "entries/mlp_k16_w1024_s400_b512.py"),
]


def load(entry):
    """One row per saved run; each energy is recomputed from the raw windows and must match."""
    rows = []
    for path in sorted((HERE / "results" / entry).glob("run-*.json"), key=lambda p: int(p.stem.split("-")[1])):
        result = json.loads(path.read_text())
        record = result.get("record") or {}
        energy = record.get("energy") or {}
        if energy.get("windows"):
            again = mnist.energy_summary(energy["windows"], energy["device"], record["band_bp"],
                                         energy["timed_mnist_ms_median"])
            assert again["problems"] == energy["problems"], path
            assert (again["mj_per_call"] is None) == (energy["mj_per_call"] is None), path
            assert again["mj_per_call"] is None or abs(again["mj_per_call"] - energy["mj_per_call"]) < 1e-6, path
        calls = [c for c in record.get("calls", []) if c["dataset"] == "mnist"]
        rows.append({"run": path.stem, "passed": result["returncode"] == 0 and record.get("ranked_ms") is not None,
                     "ranked_ms": record.get("ranked_ms"), "gpu": record.get("device"), "energy": energy,
                     "problems": record.get("problems", []), "holdout": record.get("holdout"),
                     "correct": sum(c["correct"] for c in calls), "total": sum(c["total"] for c in calls)})
    return rows


def number(value, digits=0):
    return "" if value is None else f"{value:,.{digits}f}"


def main():
    summary = []
    table = ["| Method | Difficulty | Runs passed | ms per call | mJ per call above idle | W above idle | MNIST |",
             "| --- | :-: | :-: | ---: | ---: | ---: | ---: |"]
    detail = ["| Method | Run | GPU | ms per call | mJ per call | Energy window | Idle W | Round trip mJ | "
              "Reference J/TFLOP |",
              "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for entry, label, difficulty, source in ENTRIES:
        rows = load(entry)
        if not rows:
            continue
        passed = [r for r in rows if r["passed"]]
        measured = [r for r in passed if r["energy"].get("mj_per_call") is not None]
        mj = [r["energy"]["mj_per_call"] for r in measured]
        watts = [r["energy"]["mj_per_call"] / r["energy"]["ms_median"] for r in measured]  # mJ per ms is W
        ms = [r["ranked_ms"] for r in passed]
        correct, total = sum(r["correct"] for r in passed), sum(r["total"] for r in passed)
        item = {"entry": entry, "label": label, "difficulty": difficulty, "source": source, "runs": len(rows),
                "passed": len(passed), "ranked_ms": ms, "ranked_ms_median": statistics.median(ms) if ms else None,
                "mj_per_call": mj, "mj_per_call_median": statistics.median(mj) if mj else None,
                "above_idle_w_median": statistics.median(watts) if watts else None,
                "mnist_accuracy": correct / total if total else None,
                "gpus": [r["energy"].get("device", {}).get("name") for r in measured]}
        summary.append(item)
        table.append(f"| {label} | {difficulty} | {len(passed)} of {len(rows)} | {number(item['ranked_ms_median'], 1)} | "
                     f"{number(item['mj_per_call_median'])} | {number(item['above_idle_w_median'])} | "
                     + (f"{item['mnist_accuracy']:.2%} |" if total else " |"))
        for r in rows:
            e = r["energy"]
            if e.get("windows"):
                ref = e["reference"]
                value = number(e["mj_per_call"]) if e["mj_per_call"] is not None else "not measured: " + "; ".join(e["problems"])
                detail.append(f"| {label} | {r['run']} | {e['device'].get('name')} | {number(r['ranked_ms'], 1)} | {value} | "
                              f"{e['calls']:,} calls, {e['seconds']:.1f} s, {e['ms_median']:,.1f} ms each | "
                              f"{e['idle_w']:.1f} | {e['control_mj_per_call']:.1f} | {ref['j_per_tflop']:.2f} |")
            else:
                reason = e.get("reason") or "; ".join(r["problems"]) or "no record"
                detail.append(f"| {label} | {r['run']} | {r['gpu']} | {number(r['ranked_ms'], 1)} | "
                              f"not measured: {reason} | | | | |")
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(table) + "\n\n" + "\n".join(detail))


if __name__ == "__main__":
    main()
