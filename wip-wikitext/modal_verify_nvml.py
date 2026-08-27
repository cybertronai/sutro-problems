"""Probe NVML on Modal — fast binary check, then optional stress.

The critical leaderboard question is binary: does Modal expose
nvmlDeviceGetTotalEnergyConsumption? Line 7 of verify_nvml.py warns
"e.g. Modal does not". We test this with a slim image (no PyTorch base,
no torch dep) so the first-build pull is ~100 MB instead of ~5 GB.

The host NVIDIA driver is mounted into the container by Modal at
gpu-attach time regardless of base image, so nvidia-ml-py alone is
enough to call nvmlDeviceGetTotalEnergyConsumption.

Two functions:
  - probe_counter():  ~100 MB image, no torch. Answers the binary
                      question. Run this first.
  - full_verify():    pytorch base + verify_nvml.py stress workload.
                      Only worth running if probe_counter passes.

Usage:
    modal run modal_verify_nvml.py                   # probe only
    modal run modal_verify_nvml.py::full_verify_app  # full stress
"""
from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

# ---- Fast probe: does the energy counter exist on Modal? ------------------
probe_image = modal.Image.debian_slim().pip_install("nvidia-ml-py==12.560.30")

app = modal.App("wikitext-nvml-probe", image=probe_image)


@app.function(gpu="A100-40GB", timeout=120)
def probe_counter() -> dict:
    """Bare-minimum check: NVML init, GPU name, energy counter call."""
    import time

    out: dict = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        out["nvml_available"] = True
    except Exception as e:
        return {"nvml_available": False, "error": repr(e)}

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    out["gpu_name"] = name.decode() if isinstance(name, bytes) else name

    try:
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        out["energy_counter_supported"] = True
        out["e0_mJ"] = e0
        time.sleep(2.0)
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        out["e1_mJ"] = e1
        out["delta_mJ_2s"] = e1 - e0
        out["implied_avg_W"] = (e1 - e0) / 1000.0 / 2.0
        out["monotonic"] = e1 >= e0
    except Exception as e:
        out["energy_counter_supported"] = False
        out["error"] = repr(e)

    try:
        out["idle_W_sample"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except Exception as e:
        out["idle_W_sample_error"] = repr(e)

    return out


@app.local_entrypoint()
def main() -> None:
    import json
    print(json.dumps(probe_counter.remote(), indent=2))


# ---- Full stress: only worth running if probe passes ----------------------
full_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime", add_python=None
    )
    .pip_install("nvidia-ml-py==12.560.30")
    .add_local_file(str(HERE / "verify_nvml.py"), "/workspace/verify_nvml.py")
)

full_verify_app = modal.App("wikitext-nvml-full", image=full_image)


@full_verify_app.function(gpu="A100-40GB", timeout=300)
def full_verify() -> dict:
    import json
    import subprocess
    r = subprocess.run(
        ["python3", "/workspace/verify_nvml.py"],
        capture_output=True, text=True, cwd="/workspace",
    )
    print(r.stdout)
    if r.stderr:
        print("--- stderr ---", r.stderr)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "{}"
    try:
        summary = json.loads(last)
    except json.JSONDecodeError:
        summary = {"_parse_error": last}
    return {"exit_code": r.returncode, "summary": summary}


@full_verify_app.local_entrypoint()
def full_main() -> None:
    import json
    print(json.dumps(full_verify.remote(), indent=2))
