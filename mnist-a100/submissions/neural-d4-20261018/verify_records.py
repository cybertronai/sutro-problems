import hashlib
import json
import math
from pathlib import Path
import runpy
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, name):
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8), name)


def verify(folder, scorer):
    result = read(folder / "results.json")
    api = runpy.run_path(str(scorer), run_name="saved_score_verification")
    band = api["DIFFICULTY"][result["difficulty"]][0]
    output = []
    for profile in result["profiles"]:
        source = folder / profile["source"]
        require(digest(source) == profile["source_sha256"], "source hash mismatch")
        require(len(source.read_bytes()) == profile["source_bytes"], "source size mismatch")
        api["check_source"](source.read_bytes(), result["function"], source.name)
        record_sha = profile.get("record_source_sha256", profile["source_sha256"])
        record_size = profile.get("record_source_bytes", profile["source_bytes"])
        flags = [flag.replace(f"{profile['source_bytes']:,} bytes,", f"{record_size:,} bytes,")
                 for flag in api["review_flags"](source.read_bytes())]
        require(len(profile["runs"]) == 3, "expected three runs")
        expected_paths = [f"evidence/{profile['id']}/run-{i}.json" for i in range(1, 4)]
        require([r["record"] for r in profile["runs"]] == expected_paths, "run inventory mismatch")
        rows = []
        for saved in profile["runs"]:
            path = folder / saved["record"]
            raw = read(path)
            record = raw["record"]
            require(raw["returncode"] == 0, "failed run")
            require(raw["review_flags"] == flags, "source review flags mismatch")
            require(record["source_sha256"] == record_sha, "run source mismatch")
            require(record["file"] == source.name, "source filename mismatch")
            require(record["sandboxed"] is True, "unsandboxed run")
            require(record["version"] == api["VERSION"], "scorer version mismatch")
            require(record["function"] == result["function"], "entry point mismatch")
            require(record["band_bp"] == band, "difficulty mismatch")
            require("A100" in record["device"] and "80GB" in record["device"], "unexpected gpu")
            calls = [api["Call"](**c) for c in record["calls"]]
            digit = [c for c in calls if c.dataset == "mnist"]
            hold = [c for c in calls if c.dataset != "mnist"]
            require(len(digit) == api["MNIST_CALLS"] and len(hold) == api["HOLDOUT_CALLS"],
                    "incomplete protocol")
            require(all(c.total == api["TEST"] and 0 <= c.correct <= c.total for c in calls),
                    "invalid label counts")
            require(all(math.isfinite(v) and v >= 0 for c in calls
                        for v in (c.ms, c.parent_ms, c.overhead_ms)), "invalid timing")
            require({c.dataset for c in hold} == {record["holdout"]}
                    and record["holdout"] in ("fashion", "kmnist"), "holdout mismatch")
            problems, ranked, summary = api["judge"](calls, band)
            require(not problems and not record["problems"], f"disqualified: {problems}")
            require(summary == record["summary"], "scorer summary mismatch")
            close(ranked, record["ranked_ms"], "ranked time mismatch")
            energy = record["energy"]
            rebuilt = api["energy_summary"](energy["windows"], energy["device"], band,
                                             statistics.median(c.ms for c in digit))
            require(not rebuilt["problems"] and not energy["problems"], "energy disqualified")
            require(json.loads(json.dumps(rebuilt)) == energy, "energy reconstruction mismatch")
            correct, total = sum(c.correct for c in digit), sum(c.total for c in digit)
            row = {"ranked_ms": ranked, "energy_mj": rebuilt["mj_per_call"],
                   "error_pct": 100 * (total - correct) / total,
                   "holdout_accuracy_pct": 100 * sum(c.correct for c in hold) / sum(c.total for c in hold)}
            require(record["holdout"] == saved["holdout"], "saved holdout mismatch")
            for name, value in row.items():
                close(value, saved[name], f"saved {name} mismatch")
            rows.append({**row, "correct": correct, "total": total})
        close(statistics.median(r["ranked_ms"] for r in rows), profile["median_ms"], "median time mismatch")
        close(statistics.median(r["energy_mj"] for r in rows), profile["median_energy_mj"], "median energy mismatch")
        close(100 * sum(r["total"] - r["correct"] for r in rows) / sum(r["total"] for r in rows),
              profile["pooled_error_pct"], "pooled error mismatch")
        output.append(f"{profile['id']}: 3/3 saved runs passed, {profile['median_ms']:.9f} ms")
    return output


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    print("\n".join(verify(folder, folder.parents[1] / "mnist.py")))
