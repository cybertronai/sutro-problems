# Energy measurement harness: RFF-ridge on MNIST-medium

This directory holds the NVML harness and the measurement commands. It is the
part of this submission that has **not** been run on an A100, and it is the
single thing we are asking the organizers for.

## What is measured

One complete training-and-prediction cycle of [../rff_ridge.py](../rff_ridge.py)
for one draw: the 18-point per-draw hyperparameter grid plus the refit on all
train rows plus the test projection. Data is resident on the device, execution is
warm and repeated so that the active window is at least `--min-active-seconds`.

Excluded from the window and reported separately: dataset loading, host↔device
transfer, allocation, CUDA graph capture. This is **not** cold end-to-end energy
and **not** an external wall-socket measurement.

## Method

The method is the one used by the upstream PCA-QDA audit
(`submissions/medium-pca-qda-20260915/energy-audit/independent_measure.py`),
which is what the README asks for:

- `pynvml`, device 0. The cumulative energy counter
  `nvmlDeviceGetTotalEnergyConsumption` is stamped at interval **edges only**.
- A monitor thread samples `nvmlDeviceGetPowerUsage`, SM and memory clocks,
  temperature, pstate and `nvmlDeviceGetUtilizationRates` every 50 ms.
- Every active window is bracketed by `idle-before` and `idle-after` windows.
- Two independent readouts per interval: `counter` (`E_end − E_begin`) and
  `integrated_power` (trapezoid over sampled power).
- Idle-adjusted energy per task:
  `net J/task = [active_J − mean(idle_before_W, idle_after_W) × active_seconds] / repeats`.
- Sham controls: idle-only windows with zero repeats, published as the
  subtraction noise. Negative values are retained, never clipped.

## Commands

```bash
# environment
python -m pip install "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
python -m pip install pynvml numpy

# hardware and driver
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

cd mnist/submissions/mnist-medium-rff-ridge-20261006

# 1. idle baseline: 40 one-second windows, no CUDA context in this process
python energy/nvml_energy.py --data-root <draws-dir> --idle-baseline \
    --out energy/results/idle_baseline.json

# 2. the method: 3 active rounds, 10 s idle windows, 1 sham per round
python energy/nvml_energy.py --data-root <draws-dir> \
    --D 2448 --device cuda --dtype float32 --chunk 2048 \
    --gram-dtype float64 --f32-products \
    --rounds 3 --shams 1 --idle-seconds 10 \
    --out energy/results/energy-rff-cuda-f32p-chunk2048-D2448.json
```

`<draws-dir>` is a directory containing `data-<seed>/medium.npz` for each seed,
produced by the official generator:

```bash
python -m mnist.code.data --profile medium-error-targets-v1 \
    --output <draws-dir>/data-20261001 --seed 20261001
```

Optional variants, same flags:

| what to measure | flags |
|---|---|
| best accuracy basis | `--D 12000 --dtype float64 --chunk 0` |
| float64 GPU reference | `--D 2448 --dtype float64 --chunk 0` |
| blockwise Gram in float64 | `--D 2448 --dtype float64 --chunk 2048` |

**Use `--dtype float32 --gram-dtype float64 --f32-products` for the reported
configuration.** A pure float32 Gram fails: Cholesky reports a non-positive-definite
leading minor. See the note in ../README.md.

## Output

Each run writes a JSON document containing the hardware and software
identification, both interval families with their raw 50 ms samples, the counter
stamps, the idle-adjusted per-task values for all rounds, the sham noise, peak
allocated device memory, and the SHA-256 of both `nvml_energy.py` and
`rff_ridge.py` as executed.

Read these fields:

| field | meaning |
|---|---|
| `counter_idle_adjusted_mj_per_task_median` | energy per draw, counter readout |
| `integrated_power_idle_adjusted_mj_per_task_median` | energy per draw, sampled power |
| `gpu_ms_per_task_median` | device time per draw |
| `peak_cuda_allocated_bytes` | peak allocated device memory |
| `idle_baseline_w` | idle power bracketing the active windows |
| `comparisons[].sham.*` | subtraction noise for that round |

Report the median over rounds together with the sham noise; if the sham is
comparable to the result, the number is below this method's resolution on that
machine and should be reported as such rather than as a small positive value.

## Reference measurement on other hardware

A full sweep was run on an **NVIDIA GeForce RTX 2060** (12,288 MiB, driver
617.14), which is **not** an A100. The numbers are recorded in
[../README.md](../README.md) as a relative frontier for choosing between variants.
They are not comparable to the A100 rows in the main README and must not be
copied there.