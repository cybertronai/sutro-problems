"""Independently check saved PCANet evidence without importing PyTorch or NVML.

Usage: python verify_results.py --results generated/rerun-01 --raw /path/to/raw
The optional raw directory enables independent accuracy and input-hash checks.
No files are downloaded. JSON is written to stdout; any failed check exits 1.
Reference disagreements are disclosed separately from repeated-run stability.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

import numpy as np


SOURCES = (
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873", 60000, True),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432", 60000, False),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3", 10000, True),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c", 10000, False),
)


class Checker:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.checked = 0

    def require(self, condition, message):
        self.checked += 1
        if not condition:
            self.errors.append(message)

    def near(self, actual, expected, name):
        self.require(
            isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7),
            f"{name}: recorded {actual!r}; recomputed {expected!r}",
        )

    def integer(self, actual, expected, name):
        self.require(type(actual) is int and actual == expected,
                     f"{name}: expected integer {expected}, got {actual!r}")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(), parse_constant=reject_constant,
                      object_pairs_hook=unique_pairs)


def verify_data(check, recorded, raw):
    check.require(set(recorded) == {row[0] for row in SOURCES},
                  "data manifest must cover exactly four canonical MNIST sources")
    test_labels = None
    for name, expected_md5, count, images in SOURCES:
        entry = recorded[name]
        check.require(entry["md5"] == expected_md5, f"{name}: canonical MD5")
        check.require(entry["shape"] == ([count, 28, 28] if images else [count]),
                      f"{name}: expected official full-split shape")
        check.require(entry["dtype"] == ("float32" if images else "int64"),
                      f"{name}: expected normalized image / integer label dtype")
        for key in ("sha256", "array_sha256"):
            check.require(isinstance(entry[key], str) and len(entry[key]) == 64
                          and all(c in "0123456789abcdef" for c in entry[key]),
                          f"{name}: valid {key}")
        if raw is None:
            continue
        contents = (raw / name).read_bytes()
        check.require(hashlib.md5(contents).hexdigest() == expected_md5,
                      f"{name}: raw file must match canonical MNIST")
        check.require(sha256(contents) == entry["sha256"], f"{name}: raw SHA256")
        payload = gzip.decompress(contents)
        expected_header = (2051, count, 28, 28) if images else (2049, count)
        offset = 4 * len(expected_header)
        check.require(struct.unpack(">" + "I" * len(expected_header), payload[:offset]) == expected_header,
                      f"{name}: IDX header")
        check.integer(len(payload), offset + count * (784 if images else 1),
                      f"{name}: IDX payload length")
        array = np.frombuffer(payload, dtype=np.uint8, offset=offset)
        if images:
            array = array.reshape(count, 28, 28).astype(np.float32) / np.float32(255)
        else:
            array = array.astype(np.int64)
            check.require(bool(np.all((array >= 0) & (array <= 9))), f"{name}: digit labels")
        check.require(sha256(array.tobytes()) == entry["array_sha256"],
                      f"{name}: independently reconstructed normalized array hash")
        if name == "t10k-labels-idx1-ubyte.gz":
            test_labels = array
    if raw is None:
        check.warnings.append("No --raw directory: accuracy counts and input hashes have not been independently checked against MNIST bytes.")
    return test_labels


def predictions(check, path):
    value = np.load(path, allow_pickle=False)
    check.require(isinstance(value, np.ndarray) and value.dtype == np.dtype("int64")
                  and value.shape == (10000,), f"{path.name}: int64 vector of exactly 10,000 labels")
    check.require(bool(np.all((value >= 0) & (value <= 9))), f"{path.name}: all labels in 0..9")
    return value


def verify(results, raw=None):
    check = Checker()
    summary = {"results_directory": str(results), "raw_directory": str(raw) if raw else None,
               "runs": {}, "independent_accuracy_check": raw is not None}
    try:
        doc = read_json(results / "results.json")
        execution = read_json(results / "execution.json")
        check.integer(execution["returncode"], 0, "execution return code")
        check.require(doc.get("completed") is True, "run must have completed all measurement and control windows")
        for name in ("run.log", "nvidia-smi.txt"):
            check.require((results / name).is_file() and (results / name).stat().st_size > 0,
                          f"missing or empty required evidence: {name}")
        check.require(doc["hardware"]["gpu"] == "NVIDIA A100-SXM4-40GB", "measurement must use the specified A100 model")
        check.require(bool(doc["hardware"]["uuid"]) and bool(doc["hardware"]["driver"]), "GPU identity and driver must be recorded")
        check.require(set(doc["software"]) >= {"python", "torch", "cuda", "cudnn", "numpy"}, "required software versions")
        check.require(doc["monitor_errors"] == [], "power/process monitor errors must be empty")
        for name in ("model.py", "cuda_kernel.py", "data.py", "validate.py"):
            source = Path(__file__).resolve().parent / name
            check.require(doc["source_sha256"].get(name) == sha256(source.read_bytes()),
                          f"executed source differs from packaged {name}")
        ownership = doc["process_ownership"]
        check.require(ownership["initial"] == [], "GPU was occupied before this job")
        probe = ownership["during_probe"]
        check.require(len(probe) == 1 and probe[0]["pid"] == ownership["nvml_pid"], "unique measured GPU process identity")
        if len(probe) == 1:
            check.require(probe[0]["bytes"] - sum(p["bytes"] for p in ownership["before_probe"]) >= 128 * 1024 * 1024,
                          "PID ownership allocation probe")
            check.require(probe[0]["bytes"] - sum(p["bytes"] for p in ownership["after_release"]) >= 128 * 1024 * 1024,
                          "PID ownership release probe")
        config = doc["config"]
        for key, value in {"L1": 8, "L2": 5, "blocks": 9, "block_size": 14,
                           "stride": 7, "features": 2304, "mixture_components_per_class": 8}.items():
            check.integer(config[key], value, "config." + key)
        check.require(config["tf32_matmul"] is True and config["tf32_cudnn"] is True,
                      "expected recorded TF32 configuration")
        dims = config["dimensions"]
        check.require(isinstance(dims, list) and len(dims) == len(set(dims)) and 100 in dims,
                      "unique dimensions must include the requested K=100 model")
        check.require(set(doc["runs"]) == {str(K) for K in dims}, "all configured models must have complete evidence")
        nrounds, repeats = config["rounds"], config["repeats"]
        check.require(type(nrounds) is int and nrounds >= 3, "at least three complete measurement rounds")
        check.require(type(repeats) is int and repeats > 1, "measurement windows must repeat complete tasks")
        labels = verify_data(check, doc["data"], raw)

        samples = sorted(doc["samples"], key=lambda s: s["t"])
        ts = np.asarray([s["t"] for s in samples], dtype=np.float64)
        ps = np.asarray([s["power_w"] for s in samples], dtype=np.float64)
        check.require(len(ts) >= 2 and bool(np.all(np.isfinite(ts)))
                      and bool(np.all(np.diff(ts) > 0)), "power samples require distinct finite timestamps")
        check.require(bool(np.all(np.isfinite(ps) & (ps > 0))), "power samples must be finite and positive")
        by_name = {}
        previous_end, previous_energy = -math.inf, -1
        for row in doc["intervals"]:
            name = row["name"]
            check.require(name not in by_name, "duplicate interval: " + name)
            by_name[name] = row
            start, end = row["start"], row["end"]
            dt = end["t"] - start["t"]
            check.require(dt > 0 and start["t"] >= previous_end, f"{name}: positive, ordered, nonoverlapping interval")
            check.require(type(start["energy_mj"]) is int and type(end["energy_mj"]) is int
                          and end["energy_mj"] >= start["energy_mj"] >= previous_energy,
                          f"{name}: monotonic integer NVML cumulative energy")
            previous_end, previous_energy = end["t"], end["energy_mj"]
            check.require(ts[0] <= start["t"] and ts[-1] >= end["t"], f"{name}: samples bracket both boundaries")
            inner = ts[(ts > start["t"]) & (ts < end["t"])]
            check.require(len(inner) >= 2, f"{name}: enough interior power samples")
            times = np.r_[start["t"], inner, end["t"]]
            power = np.interp(times, ts, ps)
            sampled_j = float(np.sum(np.diff(times) * (power[1:] + power[:-1]) * .5))
            counter_j = (end["energy_mj"] - start["energy_mj"]) / 1000
            check.near(row["seconds"], dt, name + ".seconds")
            for method, joules in (("counter", counter_j), ("sampled", sampled_j)):
                check.near(row[method + "_j"], joules, name + "." + method + "_j")
                check.near(row[method + "_w"], joules / dt, name + "." + method + "_w")

        sensor = doc["sensor_check"]
        idle, load = by_name["sensor-idle"], by_name["sensor-load"]
        relative = abs(load["counter_j"] - load["sampled_j"]) / load["counter_j"]
        check.near(sensor["idle_w"], idle["counter_w"], "sensor idle power")
        check.near(sensor["load_w"], load["counter_w"], "sensor load power")
        check.near(sensor["counter_sampled_relative_difference"], relative, "sensor counter/power disagreement")
        check.require(load["counter_w"] > max(150, idle["counter_w"] + 50) and relative < .1,
                      "sensor control must show substantial load and agreement within 10%")
        expected_intervals = {"sensor-idle", "sensor-load", "control-before", "control-idle-only", "control-after"}

        for K in dims:
            run = doc["runs"][str(K)]
            validation = run["validation"]
            pred = predictions(check, results / f"predictions-k{K}.npy")
            ref = predictions(check, results / f"reference-predictions-k{K}.npy")
            check.require(sha256(pred.tobytes()) == validation["prediction_sha256"], f"K={K}: saved prediction hash")
            check.integer(validation["total"], 10000, f"K={K}: total")
            correct = int(np.count_nonzero(pred == labels)) if labels is not None else validation["correct"]
            check.integer(validation["correct"], correct, f"K={K}: correct predictions")
            check.require(0 <= correct <= 10000, f"K={K}: valid correct count")
            check.integer(validation["errors"], 10000 - correct, f"K={K}: error count")
            check.near(validation["accuracy_pct"], correct / 100, f"K={K}: accuracy percentage")
            check.require(validation["passes_99_percent"] is (correct >= 9900), f"K={K}: exact 99% threshold flag")
            if K == 100:
                check.require(correct >= 9900, "requested K=100 model must achieve at least 9,900/10,000 correct")
            check.require(validation["repeat_prediction_matches"] == [10000, 10000], f"K={K}: repeated learner outputs must be stable")
            agreement = int(np.count_nonzero(pred == ref))
            check.integer(validation["reference_prediction_matches"], agreement, f"K={K}: saved full-reference prediction agreement")
            if labels is not None:
                check.integer(validation["reference_correct"], int(np.count_nonzero(ref == labels)), f"K={K}: reference accuracy")
            if agreement != 10000:
                check.warnings.append(f"K={K}: {10000 - agreement} predictions differ from the PyTorch feature reference. TF32/convolution arithmetic can cause threshold changes; this is not exact reference equivalence.")
            if K == dims[0]:
                check.require(9000 < validation["rotated_train_label_changed_predictions"] <= 10000,
                              f"K={K}: reported training-label perturbation must change over 9,000 predictions")
            check.require(len(run["rounds"]) == nrounds, f"K={K}: every configured measurement round must be present")
            for r, row in enumerate(run["rounds"]):
                check.integer(row["round"], r, f"K={K}: ordered round index")
                check.integer(row["repeats"], repeats, f"K={K} round {r}: repeat count")
                check.integer(row["last_prediction_matches"], 10000, f"K={K} round {r}: final measured predictions")
                names = [f"k{K}-r{r}-{phase}" for phase in ("before", "active", "after")]
                expected_intervals.update(names)
                before, active, after = [by_name[name] for name in names]
                check.require(before["seconds"] >= config["idle_seconds"] and after["seconds"] >= config["idle_seconds"],
                              f"K={K} round {r}: paired idle windows meet their configured duration")
                check.near(row["active_seconds"], active["seconds"], f"K={K} round {r}: active seconds")
                check.near(row["task_ms"], active["seconds"] * 1000 / repeats, f"K={K} round {r}: milliseconds/task")
                check.near(row["gross_j"], active["counter_j"] / repeats, f"K={K} round {r}: gross joules/task")
                for method in ("counter", "sampled"):
                    idle_w = (before[method + "_w"] + after[method + "_w"]) / 2
                    adjusted = (active[method + "_j"] - idle_w * active["seconds"]) / repeats
                    check.near(row["idle_w_" + method], idle_w, f"K={K} round {r}: {method} paired idle")
                    check.near(row["adjusted_j_" + method], adjusted, f"K={K} round {r}: {method} adjusted joules/task")
                    if adjusted <= 0:
                        check.warnings.append(f"K={K} round {r}: nonpositive {method} idle-adjusted energy; retained with its sign.")
                if abs(active["counter_j"] - active["sampled_j"]) / active["counter_j"] > .1:
                    check.errors.append(f"K={K} round {r}: active counter and integrated power disagree by more than 10%")
            medians = {key: statistics.median(row[key] for row in run["rounds"])
                       for key in ("task_ms", "gross_j", "adjusted_j_counter", "adjusted_j_sampled")}
            for key, value in medians.items():
                check.near(run["summary"][key], value, f"K={K}: median {key}")
            check.require(0 < run["peak_allocated_bytes"] <= run["peak_reserved_bytes"], f"K={K}: peak GPU memory accounting")
            summary["runs"][str(K)] = {"correct": correct, "total": 10000, "reference_prediction_matches": agreement,
                                        "median_task_ms": medians["task_ms"], "median_gross_mj": medians["gross_j"] * 1000,
                                        "median_counter_above_idle_mj": medians["adjusted_j_counter"] * 1000,
                                        "median_sampled_above_idle_mj": medians["adjusted_j_sampled"] * 1000}

        check.require(set(by_name) == expected_intervals, "exactly the configured sensor, measurement and idle-control intervals must be present")
        before, sham, after = [by_name[name] for name in ("control-before", "control-idle-only", "control-after")]
        for method in ("counter", "sampled"):
            net = sham[method + "_j"] - .5 * (before[method + "_w"] + after[method + "_w"]) * sham["seconds"]
            check.near(doc["idle_only_control"][method + "_net_j"], net, "idle-only control " + method)
        summary["idle_only_control_net_j"] = doc["idle_only_control"]
        feature = doc["feature_validation"]
        check.integer(feature["images"], 1536, "feature check image count")
        check.integer(feature["values"], 1536 * 2304, "feature check value count")
        check.require(feature["nondefault_stream_checked"] is True, "nondefault CUDA stream check must have run")
        check.require(type(feature["mismatched_values"]) is int and 0 <= feature["mismatched_values"] <= feature["values"], "valid feature mismatch count")
        maximum = feature["max_abs_difference"]
        check.require(isinstance(maximum, (int, float)) and math.isfinite(maximum) and 0 <= maximum <= 1,
                      "feature differences must be finite and within normalized feature range")
        check.require((feature["mismatched_values"] == 0) == (maximum == 0), "feature mismatch count and maximum difference must agree")
        if feature["mismatched_values"]:
            check.warnings.append("Feature cross-check records differences. TF32 reference and the fused FP32-FMA convolution are not established as bitwise equivalent; raw feature arrays were not retained.")
        summary["feature_validation"] = feature
        check.warnings.append("Repeated-run stability, training-label perturbation, process monitoring and feature checks are retained scalar observations; this offline checker cannot replay them. Only the last measured prediction vector in each round was compared during execution.")
        check.warnings.append("NVML energy counter and sampled-power integration share GPU sensors. Above-idle energy excludes host/platform power and the documented setup stages.")
    except (OSError, ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, struct.error) as exc:
        check.errors.append(f"Missing, malformed or inconsistent evidence: {type(exc).__name__}: {exc}")
    summary.update(checks=check.checked, passed=not check.errors, errors=check.errors, limitations=check.warnings)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    args = parser.parse_args()
    result = verify(args.results, args.raw)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
