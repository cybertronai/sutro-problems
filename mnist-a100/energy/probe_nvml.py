"""What the A100's NVML energy telemetry can and cannot resolve, measured on one Modal A100-80GB.

    python energy/probe_nvml.py --out energy/results/probe.json     # from mnist-a100/, needs Modal

Runs as root in the scoring image (run_modal.py) and reads NVML through ctypes, as mnist.py does:

1. how often the cumulative energy counter and the power reading change;
2. idle power with no CUDA context, with an idle context, and with that context's process
   stopped (SIGSTOP), which is how the scorer measures idle;
3. an FP32 4096 x 4096 matmul reference: TFLOP/s and joules per 10^12 FLOPs above idle;
4. thirty isolated ~60 ms matmul bursts, each read on its own from the counter, against the
   energy the same work costs inside a long window. This is why the scorer measures energy
   over a 20 s window of back-to-back calls instead of per call.
"""

import argparse
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if modal.is_local():
    sys.path.insert(0, str(HERE.parent))
    from run_modal import image  # the scoring image: CUDA 13.3, Python 3.13, torch 2.12.0
else:
    image = None

app = modal.App("mnist-a100-nvml-probe", image=image)

CHILD = r"""
import os, sys, time, torch
torch.backends.cuda.matmul.allow_tf32 = False
a = torch.randn(4096, 4096, device="cuda"); b = torch.randn(4096, 4096, device="cuda")
out = torch.empty_like(a)
torch.cuda.synchronize()
w = sys.stdout.buffer
w.write(b"R"); w.flush()
for line in sys.stdin:
    cmd, _, arg = line.strip().partition(" ")
    if cmd == "matmul":          # run matmuls for arg seconds; reply with the count
        n, end = 0, time.perf_counter() + float(arg)
        while time.perf_counter() < end:
            for _ in range(8):
                torch.mm(a, b, out=out)
            torch.cuda.synchronize()
            n += 8
        w.write(f"{n}\n".encode()); w.flush()
    elif cmd == "burst":         # exactly arg matmuls, then reply
        for _ in range(int(arg)):
            torch.mm(a, b, out=out)
        torch.cuda.synchronize()
        w.write(b"D\n"); w.flush()
    elif cmd == "quit":
        break
"""


@app.function(gpu="A100-80GB", timeout=900, single_use_containers=True)
def probe() -> dict:
    import ctypes
    import os
    import random
    import signal
    import statistics
    import subprocess
    import time

    lib = ctypes.CDLL("libnvidia-ml.so.1")
    assert lib.nvmlInit_v2() == 0
    handle = ctypes.c_void_p()
    assert lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) == 0

    def energy_mj():
        value = ctypes.c_ulonglong()
        assert lib.nvmlDeviceGetTotalEnergyConsumption(handle, ctypes.byref(value)) == 0
        return value.value

    def uint(name, *args):
        value = ctypes.c_uint()
        assert getattr(lib, name)(handle, *args, ctypes.byref(value)) == 0
        return value.value

    def text(name, size=96, device=True):
        buffer = ctypes.create_string_buffer(size)
        assert (getattr(lib, name)(handle, buffer, size) if device else getattr(lib, name)(buffer, size)) == 0
        return buffer.value.decode()

    def power_w():
        return uint("nvmlDeviceGetPowerUsage") / 1000

    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    def utilization():
        u = Util()
        assert lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(u)) == 0
        return u.gpu

    out = {"device": {"name": text("nvmlDeviceGetName"), "uuid": text("nvmlDeviceGetUUID"),
                      "vbios": text("nvmlDeviceGetVbiosVersion", 32),
                      "driver": text("nvmlSystemGetDriverVersion", 80, device=False),
                      "power_limit_w": uint("nvmlDeviceGetPowerManagementLimit") / 1000,
                      "temperature_c": uint("nvmlDeviceGetTemperature", 0)}}

    # 1. update cadence: poll both readings in a tight loop for 4 s
    def cadence(read, seconds=4.0):
        changes, last, calls = [], read(), 0
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            value = read()
            calls += 1
            if value != last:
                changes.append((time.perf_counter(), value - last))
                last = value
        gaps = [b[0] - a[0] for a, b in zip(changes, changes[1:])]
        return {"reads": calls, "changes": len(changes), "read_us": seconds / calls * 1e6,
                "median_gap_ms": statistics.median(gaps) * 1e3 if gaps else None,
                "min_gap_ms": min(gaps) * 1e3 if gaps else None, "max_gap_ms": max(gaps) * 1e3 if gaps else None,
                "median_step": statistics.median(d for _, d in changes) if changes else None}

    out["cadence_idle_no_context"] = {"energy_counter": cadence(energy_mj), "power": cadence(power_w)}

    def window(seconds):
        """(mean watts from the counter, seconds, mean utilization %) over a window."""
        t0, e0, samples = time.perf_counter(), energy_mj(), []
        end = t0 + seconds
        while time.perf_counter() < end:
            time.sleep(0.1)
            samples.append(utilization())
        t1, e1 = time.perf_counter(), energy_mj()
        return {"watts": (e1 - e0) / 1e3 / (t1 - t0), "seconds": t1 - t0, "util_mean": statistics.mean(samples),
                "power_reading_w": power_w()}

    time.sleep(3)
    out["idle_no_context"] = window(8)

    child = subprocess.Popen([sys.executable, "-c", CHILD], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert child.stdout.read(1) == b"R"

    def ask(line):
        child.stdin.write((line + "\n").encode())
        child.stdin.flush()
        return child.stdout.readline().decode().strip()

    time.sleep(3)
    out["idle_context"] = window(8)
    out["cadence_idle_context"] = {"energy_counter": cadence(energy_mj, 3.0)}
    os.kill(child.pid, signal.SIGSTOP)
    time.sleep(1)
    out["idle_context_stopped"] = window(8)
    os.kill(child.pid, signal.SIGCONT)

    # 3. reference: 10 s of FP32 matmul, bracketed by idle
    ask("matmul 2")  # warm clocks and cuBLAS
    time.sleep(3)
    before = window(6)
    t0, e0 = time.perf_counter(), energy_mj()
    count = int(ask("matmul 10"))
    t1, e1 = time.perf_counter(), energy_mj()
    time.sleep(3)
    after = window(6)
    idle_w = (before["watts"] + after["watts"]) / 2
    flops = 2 * 4096 ** 3 * count
    net_j = (e1 - e0) / 1e3 - idle_w * (t1 - t0)
    out["reference"] = {"seconds": t1 - t0, "matmuls": count, "tflops_per_s": flops / 1e12 / (t1 - t0),
                        "active_w": (e1 - e0) / 1e3 / (t1 - t0), "idle_before_w": before["watts"],
                        "idle_after_w": after["watts"], "net_j_per_tflop": net_j / (flops / 1e12)}

    # 4. isolated bursts of 8 matmuls (~60 ms) read one at a time, vs the same bursts in a long window
    per_matmul_j = net_j / count
    bursts = []
    for _ in range(30):
        time.sleep(random.uniform(0.9, 1.7))
        idle_rate = idle_w
        t0, e0 = time.perf_counter(), energy_mj()
        ask("burst 8")
        t1, e1 = time.perf_counter(), energy_mj()
        time.sleep(0.25)
        t2, e2 = time.perf_counter(), energy_mj()
        bursts.append({"ms": (t1 - t0) * 1e3,
                       "net_mj_edges": (e1 - e0) - idle_rate * (t1 - t0) * 1e3,
                       "net_mj_with_tail": (e2 - e0) - idle_rate * (t2 - t0) * 1e3})
    expected = 8 * per_matmul_j * 1e3
    out["bursts"] = {"expected_mj_from_reference": expected, "rows": bursts,
                     "edges_mj": {"median": statistics.median(b["net_mj_edges"] for b in bursts),
                                  "min": min(b["net_mj_edges"] for b in bursts),
                                  "max": max(b["net_mj_edges"] for b in bursts)},
                     "tail_mj": {"median": statistics.median(b["net_mj_with_tail"] for b in bursts),
                                 "min": min(b["net_mj_with_tail"] for b in bursts),
                                 "max": max(b["net_mj_with_tail"] for b in bursts),
                                 "mean": statistics.mean(b["net_mj_with_tail"] for b in bursts)}}
    ask("quit")
    child.wait(timeout=30)
    out["device"]["temperature_end_c"] = uint("nvmlDeviceGetTemperature", 0)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with modal.enable_output(), app.run():
        result = probe.remote()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "bursts"}, indent=1))
    print(json.dumps({k: v for k, v in result["bursts"].items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
