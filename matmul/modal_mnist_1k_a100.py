"""Modal A100 leg of the MNIST-small practice run (Yaroslav, Sep-10).

Compiles the dally_ir_to_cuda-generated kernel on an A100, checks
bit-exactness against the expected bytes already validated by dally-eval,
then runs the a100-grid-energy-report measurement protocol from inside the
compiled binary (NVML nvmlDeviceGetTotalEnergyConsumption counter,
paired loaded-idle baseline, 5 trials, reps auto-sized to ~5 s per trial).

Usage:
    modal run modal_mnist_1k_a100.py

Inputs (all produced by mnist3_1k_demo.py + dally_ir_to_cuda.py):
    mnist3_1k_inputs.bin, mnist3_1k_expected.bin, generated .cu source
Output: printed JSON with bit-exactness result, audit, and the
synthetic-to-measured energy ratio.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP_NAME = "mnist-1k-dally-a100"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("nvidia-ml-py==12.560.30")
    .apt_install("gcc", "g++")
)

app = modal.App(APP_NAME, image=image)


def read_or_die(name: str) -> bytes:
    p = HERE / name
    if not p.exists():
        raise SystemExit(f"missing {p} - run mnist3_1k_demo.py and "
                         f"dally_ir_to_cuda.py first")
    return p.read_bytes()


@app.function(gpu="A100-40GB", timeout=1200)
def measure(kernel_src: bytes, inputs: bytes, expected: bytes,
            n_instances: int, static_cost: int,
            synthetic_j_per_inference: float) -> dict:
    import subprocess

    w = Path("/root/work")
    w.mkdir(exist_ok=True)
    (w / "kernel.cu").write_bytes(kernel_src)
    (w / "inputs.bin").write_bytes(inputs)
    (w / "expected.bin").write_bytes(expected)

    cc = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_80", "-DWITH_NVML",
         str(w / "kernel.cu"), "-o", str(w / "kernel")],
        capture_output=True, text=True, timeout=600)
    if cc.returncode != 0:
        return {"stage": "compile", "ok": False, "stderr": cc.stderr[-4000:]}

    run = subprocess.run(
        [str(w / "kernel"), str(w / "inputs.bin"), str(w / "expected.bin"),
         str(n_instances), "--energy", "0", "5"],
        capture_output=True, text=True, timeout=900)
    out = run.stdout
    audit = None
    m = re.search(r"AUDIT_JSON:(\{.*\})", out, re.S)
    if m:
        audit = json.loads(m.group(1))
    hw = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        hw = {
            "gpu_name": pynvml.nvmlDeviceGetName(h),
            "gpu_memory_bytes": pynvml.nvmlDeviceGetMemoryInfo(h).total,
            "power_limit_W": pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
            "driver_version": pynvml.nvmlSystemGetDriverVersion(),
            "cuda_version": pynvml.nvmlSystemGetCudaDriverVersion(),
        }
    except Exception as e:  # hardware facts are supporting telemetry only
        hw = {"error": repr(e)}
    result = {
        "stage": "measure",
        "ok": run.returncode == 0 and "BIT-EXACT" in out,
        "bit_exact": "BIT-EXACT" in out,
        "kernel_stdout": out[-4000:],
        "kernel_stderr": run.stderr[-2000:],
        "audit": audit,
        "hardware": hw,
        "static_cost": static_cost,
        "synthetic_j_per_inference": synthetic_j_per_inference,
    }
    if audit:
        meas = audit["idle_adjusted_J_per_batch_call"]
        synth_batch = synthetic_j_per_inference * n_instances
        result["synthetic_j_per_batch"] = synth_batch
        result["ratio_synthetic_to_measured"] = synth_batch / meas
    return result


@app.local_entrypoint()
def main() -> None:
    import subprocess, sys

    sys.path.insert(0, str(HERE))
    import dally_ir_to_cuda

    ir_text = (HERE / "mnist3_1k_ir.txt").read_text()
    in_cells, out_cells, ops = dally_ir_to_cuda.parse_ir(ir_text)
    cost = (sum(dally_ir_to_cuda.rc(o[2]) + (dally_ir_to_cuda.rc(o[3]) if o[3] != o[2] else 0)
                for o in ops if o[0] != "set")
            + sum(dally_ir_to_cuda.rc(c) for c in out_cells))
    n = 1000
    # regenerate the kernel source from the checked-in IR so source and IR
    # cannot drift
    gen = subprocess.run(
        [sys.executable, str(HERE / "dally_ir_to_cuda.py"),
         str(HERE / "mnist3_1k_ir.txt"), "--out", "/tmp/mnist3_1k_kernel.cu"],
        capture_output=True, text=True)
    if gen.returncode != 0:
        raise SystemExit(f"transpile failed: {gen.stderr}")
    res = measure.remote(
        Path("/tmp/mnist3_1k_kernel.cu").read_bytes(),
        read_or_die("mnist3_1k_inputs.bin"),
        read_or_die("mnist3_1k_expected.bin"),
        n, cost, cost * 1e-15,
    )
    print(json.dumps(res, indent=2))
    out = HERE / "mnist3_1k_a100_results.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"written: {out}")
