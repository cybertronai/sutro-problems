"""NVML energy-counter verification for a Lambda A100 (or any Volta+ GPU).

Confirms three things:

1. ``nvmlDeviceGetTotalEnergyConsumption`` is **exposed** (not all
   virtualized hosts surface it; e.g. Modal does not).
2. The counter is **monotonic** across a short idle-then-stress run.
3. Energy attributable to a known stress workload is in the **expected
   ballpark** for the card (200–400 W average on an A100).

Run on a Lambda On-Demand A100 80GB instance after installing
``nvidia-ml-py`` and ``torch``::

    pip install nvidia-ml-py torch
    python3 verify_nvml.py

Exit code 0 if all three checks pass. Output is human-readable; the
final line is a JSON summary suitable for CI/grep.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass


@dataclass
class Result:
    nvml_available: bool
    energy_counter_supported: bool
    monotonic: bool
    idle_watts: float | None
    stress_watts_avg: float | None
    stress_energy_joules: float | None
    stress_duration_s: float | None
    gpu_name: str | None
    notes: list[str]


def _import_nvml():
    """Import + init NVML. Raises on failure."""
    import pynvml  # type: ignore[import-not-found]
    pynvml.nvmlInit()
    return pynvml


def _idle_power(pynvml, handle, *, seconds: float = 3.0) -> float:
    """Sample power over ``seconds`` of idle and return the mean Watts."""
    samples = []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        # nvmlDeviceGetPowerUsage returns milliwatts.
        samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        time.sleep(0.05)
    return sum(samples) / max(1, len(samples))


def _stress_workload(seconds: float = 30.0) -> None:
    """Sustained GPU compute workload — repeated large matmuls in fp16.

    Aims to keep tensor cores busy at near-TDP for the requested time
    so the energy delta is large vs idle and NVML noise.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; cannot stress GPU")
    dev = torch.device("cuda")
    n = 8192
    a = torch.randn(n, n, dtype=torch.float16, device=dev)
    b = torch.randn(n, n, dtype=torch.float16, device=dev)
    torch.cuda.synchronize()
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        c = a @ b
        a = c
    torch.cuda.synchronize()


def main() -> int:
    notes: list[str] = []
    res = Result(
        nvml_available=False,
        energy_counter_supported=False,
        monotonic=False,
        idle_watts=None,
        stress_watts_avg=None,
        stress_energy_joules=None,
        stress_duration_s=None,
        gpu_name=None,
        notes=notes,
    )

    try:
        nvml = _import_nvml()
    except Exception as e:
        notes.append(f"NVML import failed: {e!r}")
        print(json.dumps(asdict(res)))
        return 2
    res.nvml_available = True

    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    name = nvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode()
    res.gpu_name = name
    print(f"GPU: {name}")

    try:
        e0 = nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        res.energy_counter_supported = True
    except Exception as e:
        notes.append(f"nvmlDeviceGetTotalEnergyConsumption unsupported: {e!r}")
        print(json.dumps(asdict(res)))
        return 3

    print("sampling idle power for 3s ...")
    res.idle_watts = _idle_power(nvml, handle, seconds=3.0)
    print(f"  idle: {res.idle_watts:.1f} W")

    print("running 30s stress workload ...")
    t0 = time.monotonic()
    try:
        _stress_workload(seconds=30.0)
    except Exception as e:
        notes.append(f"stress workload failed: {e!r}")
        print(json.dumps(asdict(res)))
        return 4
    duration = time.monotonic() - t0
    e1 = nvml.nvmlDeviceGetTotalEnergyConsumption(handle)

    res.monotonic = (e1 >= e0)
    e_run_j = (e1 - e0) / 1000.0
    res.stress_energy_joules = e_run_j
    res.stress_duration_s = duration
    res.stress_watts_avg = e_run_j / max(0.001, duration)

    print(f"  duration:       {duration:.1f} s")
    print(f"  energy delta:   {e_run_j:,.1f} J")
    print(f"  avg power:      {res.stress_watts_avg:.1f} W")
    print(f"  monotonic:      {res.monotonic}")

    plausible = (
        res.monotonic
        and res.stress_watts_avg is not None
        and res.stress_watts_avg > 100  # well above any plausible A100 idle
        and res.stress_watts_avg < 700  # well below any plausible A100 ceiling
    )
    if not plausible:
        notes.append("stress power outside expected 100-700 W range; "
                     "investigate before trusting energy numbers")

    print("---")
    print(json.dumps(asdict(res)))
    return 0 if plausible else 5


if __name__ == "__main__":
    sys.exit(main())
