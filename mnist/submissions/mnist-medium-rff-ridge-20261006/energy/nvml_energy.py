"""Idle-adjusted NVML energy measurement for the RFF-ridge MNIST-medium method.

Run this on the machine whose energy you want to report. It measures one
complete training-and-prediction cycle of the frozen learner from
[../rff_ridge.py](../rff_ridge.py), device resident, warm repeated execution.

Method
------
Follows the audit harness of the upstream PCA-QDA submission
(`submissions/medium-pca-qda-20260915/energy-audit/independent_measure.py`):

  * pynvml, device 0; the cumulative energy counter
    (`nvmlDeviceGetTotalEnergyConsumption`) is stamped at interval EDGES only
  * a monitor thread samples board power, clocks, temperature, pstate and
    utilization every 50 ms
  * every active window is bracketed by idle-before and idle-after windows
  * two independent readouts per interval:
        counter          (E_end - E_begin), joules
        integrated_power trapezoid of sampled power over [t_begin, t_end]
  * idle-adjusted energy per task
        net J/task = [active_J - mean(idle_before_W, idle_after_W) * active_seconds] / repeats
  * sham controls run idle-only windows with zero repeats and are published as
    the subtraction noise; negative values are retained, not clipped

Scope and exclusions
--------------------
IN scope:  the full train + predict cycle for one draw, on the device, after
           warm-up, repeated so the window is at least `--min-active-seconds`.
OUT of scope and reported separately: dataset loading, host<->device transfer,
           allocation, and CUDA graph capture.

This is NOT cold end-to-end energy and NOT an external wall-socket measurement.
One draw is one independent task; a full eleven-draw protocol costs
11 x the per-draw figure because no state crosses draws.

Usage
-----
    pip install pynvml torch numpy
    python nvml_energy.py --data-root <dir with data-<seed>> --D 2448 \
        --out results/energy-rff-cuda-f32p-chunk2048-D2448.json
    python nvml_energy.py --data-root <dir> --idle-baseline \
        --out results/idle_baseline.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rff_ridge as R  # noqa: E402


# --------------------------------------------------------------------------- #
# NVML core
# --------------------------------------------------------------------------- #
class Nvml:
    def __init__(self, device=0):
        import pynvml as nv
        self.nv = nv
        nv.nvmlInit()
        self.handle = nv.nvmlDeviceGetHandleByIndex(device)
        self.samples = []
        self.stop = threading.Event()
        self.errors = []

    def energy_mj(self):
        return int(self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle))

    def sample(self):
        nv = self.nv
        row = {
            "t": time.perf_counter(),
            "power_w": nv.nvmlDeviceGetPowerUsage(self.handle) / 1000.0,
            "sm_clock_mhz": nv.nvmlDeviceGetClockInfo(self.handle, nv.NVML_CLOCK_SM),
            "mem_clock_mhz": nv.nvmlDeviceGetClockInfo(self.handle, nv.NVML_CLOCK_MEM),
            "temperature_c": nv.nvmlDeviceGetTemperature(self.handle, nv.NVML_TEMPERATURE_GPU),
            "pstate": nv.nvmlDeviceGetPerformanceState(self.handle),
        }
        u = nv.nvmlDeviceGetUtilizationRates(self.handle)
        row["gpu_util_percent"] = u.gpu
        row["mem_util_percent"] = u.memory
        self.samples.append(row)

    def monitor(self):
        while not self.stop.is_set():
            try:
                self.sample()
            except Exception as exc:
                self.errors.append(repr(exc))
            self.stop.wait(0.05)

    def stamp(self):
        t0 = time.perf_counter()
        e = self.energy_mj()
        t1 = time.perf_counter()
        return {"t": (t0 + t1) / 2, "energy_mj": e, "read_seconds": t1 - t0}

    def interval(self, name, reps, seconds, fn, is_cuda):
        import torch
        if is_cuda:
            torch.cuda.synchronize()
        self.sample()
        begin = self.stamp()
        if reps:
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            if is_cuda:
                a.record()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            wall = time.perf_counter() - t0
            if is_cuda:
                b.record()
                torch.cuda.synchronize()
                gpu_ms = a.elapsed_time(b) / reps
            else:
                gpu_ms = None
        else:
            time.sleep(seconds)
            wall = seconds
            gpu_ms = None
        end = self.stamp()
        self.sample()
        rows = sorted(self.samples, key=lambda r: r["t"])
        ts = np.array([r["t"] for r in rows])
        ps = np.array([r["power_w"] for r in rows])
        inner = (ts > begin["t"]) & (ts < end["t"])
        st = np.r_[begin["t"], ts[inner], end["t"]]
        sp = np.interp(st, ts, ps)
        joules = float(np.trapezoid(sp, st))
        elapsed = end["t"] - begin["t"]
        counter = (end["energy_mj"] - begin["energy_mj"]) / 1000.0
        return {"name": name, "repeats": reps, "seconds": elapsed,
                "wall_ms_per_task": wall * 1000.0 / max(reps, 1),
                "gpu_ms_per_task": gpu_ms,
                "counter_j": counter, "integrated_power_j": joules,
                "counter_w": counter / elapsed, "integrated_power_w": joules / elapsed,
                "sample_count": int(inner.sum()),
                "mean_util_percent": float(np.mean(
                    [r["gpu_util_percent"] for r in rows])) if inner.any() else None,
                "begin": begin, "end": end}

    def hardware(self):
        nv = self.nv
        sm = mem = None
        try:
            import torch
            pr = torch.cuda.get_device_properties(0)
            sm, mem = pr.multi_processor_count, pr.total_memory
        except Exception:
            pass
        return {"name": str(nv.nvmlDeviceGetName(self.handle)),
                "uuid": str(nv.nvmlDeviceGetUUID(self.handle)),
                "driver": str(nv.nvmlSystemGetDriverVersion()),
                "vbios": str(nv.nvmlDeviceGetVbiosVersion(self.handle)),
                "power_limit_w": nv.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0,
                "temperature_c": nv.nvmlDeviceGetTemperature(self.handle, nv.NVML_TEMPERATURE_GPU),
                "sm_count": sm, "memory_bytes": mem,
                "reported_compute_process_count": len(
                    nv.nvmlDeviceGetComputeRunningProcesses(self.handle))}


def software():
    import torch
    import pynvml
    ver = getattr(pynvml, "__version__", None)
    if ver is None:
        try:
            from importlib.metadata import version
            ver = version("nvidia-ml-py")
        except Exception:
            ver = "unknown"
    return {"python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "pynvml": ver}


# --------------------------------------------------------------------------- #
# idle baseline
# --------------------------------------------------------------------------- #
def idle_baseline(out, windows=40, window_seconds=1.0):
    """Idle-only energy, sampled with NO CUDA context open in this process."""
    m = Nvml()
    rows = []
    for i in range(windows):
        rows.append(m.interval(f"idle-{i}", 0, window_seconds, None, False))
    doc = {"experiment": "idle-baseline", "hardware": m.hardware(),
           "software": software(), "windows": len(rows),
           "window_seconds": window_seconds,
           "note": "no CUDA context open in the sampling process; "
                   "cumulative energy counter stamped at 1 s edges",
           "intervals": rows}
    for method in ("counter", "integrated_power"):
        w = [r[method + "_w"] for r in rows]
        doc[method + "_w"] = {"min": min(w), "median": statistics.median(w),
                              "mean": float(np.mean(w)), "max": max(w),
                              "stdev": float(np.std(w, ddof=1))}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"counter_median_w": doc["counter_w"]["median"],
                      "counter_stdev_w": doc["counter_w"]["stdev"]}), flush=True)


# --------------------------------------------------------------------------- #
# active measurement
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--D", type=int, default=2448)
    ap.add_argument("--draw", type=int, default=20261001)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--gram-dtype", default="float64")
    ap.add_argument("--f32-products", action="store_true", default=True)
    ap.add_argument("--no-f32-products", dest="f32_products", action="store_false")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--shams", type=int, default=2)
    ap.add_argument("--idle-seconds", type=float, default=10.0)
    ap.add_argument("--min-active-seconds", type=float, default=6.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--idle-baseline", action="store_true")
    ap.add_argument("--baseline-windows", type=int, default=40)
    a = ap.parse_args()

    if a.idle_baseline:
        idle_baseline(a.out, windows=a.baseline_windows)
        return

    import torch
    is_cuda = a.device.startswith("cuda")

    t_load0 = time.perf_counter()
    m = Nvml()
    npz = Path(a.data_root) / f"data-{a.draw}" / "medium.npz"
    Xtr, ytr, Xte, tidx, teidx = R.load_draw(npz)
    val_ov, train_ov = R.check_no_test_overlap(tidx, teidx)
    assert val_ov == 0 and train_ov == 0, (val_ov, train_ov)
    load_s = time.perf_counter() - t_load0

    def fn():
        return R.fit_predict_torch(Xtr, ytr, Xte, a.D, device=a.device,
                                   dtype=a.dtype, chunk=a.chunk,
                                   gram_dtype=a.gram_dtype,
                                   f32_products=a.f32_products)

    for _ in range(a.warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    fn()
    if is_cuda:
        torch.cuda.synchronize()
    probe_s = time.perf_counter() - t0
    reps = max(1, int(round(a.min_active_seconds / max(probe_s, 1e-6))))

    peak = None
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated())

    doc = {
        "variant": f"rff-{a.device}-{a.dtype}-chunk{a.chunk}-D{a.D}"
                   + (f"-gram{a.gram_dtype}" if a.gram_dtype else "")
                   + ("-f32products" if a.f32_products else ""),
        "draw": a.draw, "D": a.D, "learner_seed": R.LEARNER_SEED,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": m.hardware(), "software": software(),
        "env": {k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "config": {"dtype": a.dtype, "chunk": a.chunk, "gram_dtype": a.gram_dtype,
                   "f32_products": a.f32_products},
        "load_seconds": round(load_s, 3),
        "probe_task_seconds": round(probe_s, 4),
        "repeats_per_round": reps, "rounds": a.rounds, "shams": a.shams,
        "idle_seconds_per_window": a.idle_seconds,
        "peak_cuda_allocated_bytes": peak,
        "val_test_index_overlap_rows": val_ov,
        "train_test_index_overlap_rows": train_ov,
        "measurement_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "learner_sha256": hashlib.sha256(
            Path(__file__).resolve().parent.parent.joinpath("rff_ridge.py")
            .read_bytes()).hexdigest(),
        "scope": ("one complete training-and-prediction cycle, device resident, warm "
                  "repeated execution; dataset loading, host<->device transfer, "
                  "allocation and CUDA graph capture are excluded and reported "
                  "separately; not cold end-to-end, not a wall-socket measurement"),
        "intervals": [], "comparisons": [], "samples": [],
    }

    def flush():
        doc["samples"] = m.samples
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")

    order = []
    for r in range(a.rounds):
        order.append((f"{r}-idle-before", 0, a.idle_seconds))
        order.append((f"{r}-active", reps, 0))
        order.append((f"{r}-idle-after", 0, a.idle_seconds))
        for _ in range(a.shams):
            order.append((f"{r}-sham-idle", 0, a.idle_seconds))

    monitor = threading.Thread(target=m.monitor, daemon=True)
    monitor.start()
    try:
        rows = {}
        for name, r, s in order:
            rows[name] = m.interval(name, r, s, fn, is_cuda)
            doc["intervals"].append(rows[name])
            print(json.dumps({"interval": name,
                              "counter_w": round(rows[name]["counter_w"], 2),
                              "power_w": round(rows[name]["integrated_power_w"], 2),
                              "gpu_ms": rows[name]["gpu_ms_per_task"]}), flush=True)
            flush()
        for r in range(a.rounds):
            before, active, after = (rows[f"{r}-idle-before"], rows[f"{r}-active"],
                                     rows[f"{r}-idle-after"])
            comp = {"kind": "active", "repeats": reps, "round": r,
                    "wall_ms_per_task": active["wall_ms_per_task"],
                    "gpu_ms_per_task": active["gpu_ms_per_task"]}
            for method in ("counter", "integrated_power"):
                idle = (before[method + "_w"] + after[method + "_w"]) / 2.0
                comp[method + "_gross_mj_per_task"] = active[method + "_j"] * 1000.0 / reps
                comp[method + "_idle_adjusted_mj_per_task"] = (
                    (active[method + "_j"] - idle * active["seconds"]) * 1000.0 / reps)
                comp[method + "_idle_before_w"] = before[method + "_w"]
                comp[method + "_idle_after_w"] = after[method + "_w"]
            doc["comparisons"].append(comp)
            print(json.dumps({"comparison": comp}), flush=True)
        for r in range(a.rounds):
            doc["comparisons"].append(
                {"kind": "sham", "round": r, "nominal_tasks": reps,
                 "counter_noise_mj_per_nominal_task":
                     rows[f"{r}-sham-idle"]["counter_j"] * 1000.0 / reps,
                 "integrated_power_noise_mj_per_nominal_task":
                     rows[f"{r}-sham-idle"]["integrated_power_j"] * 1000.0 / reps})
        act = [c for c in doc["comparisons"] if c["kind"] == "active"]
        for method in ("counter", "integrated_power"):
            vals = [c[method + "_idle_adjusted_mj_per_task"] for c in act]
            doc[method + "_idle_adjusted_mj_per_task_median"] = statistics.median(vals)
            doc[method + "_idle_adjusted_mj_per_task_all"] = vals
        doc["gpu_ms_per_task_median"] = statistics.median(
            [c["gpu_ms_per_task"] for c in act]) if act[0]["gpu_ms_per_task"] is not None else None
        doc["wall_ms_per_task_median"] = statistics.median(
            [c["wall_ms_per_task"] for c in act])
        doc["idle_baseline_w"] = {
            "counter": statistics.median([c["counter_idle_before_w"] for c in act]
                                         + [c["counter_idle_after_w"] for c in act]),
            "integrated_power": statistics.median(
                [c["integrated_power_idle_before_w"] for c in act]
                + [c["integrated_power_idle_after_w"] for c in act])}
        doc["monitor_errors"] = m.errors
        doc["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        flush()
        print(json.dumps({"variant": doc["variant"],
                          "counter_median_mj": doc["counter_idle_adjusted_mj_per_task_median"],
                          "power_median_mj": doc["integrated_power_idle_adjusted_mj_per_task_median"],
                          "gpu_ms": doc["gpu_ms_per_task_median"],
                          "wall_ms": doc["wall_ms_per_task_median"],
                          "peak_cuda_bytes": peak}), flush=True)
    except Exception as exc:
        doc["variant_failed"] = repr(exc)
        doc["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        flush()
        print(json.dumps({"variant": doc["variant"], "failed": repr(exc)}), flush=True)
        raise SystemExit(3)
    finally:
        m.stop.set()
        monitor.join()
        flush()


if __name__ == "__main__":
    main()