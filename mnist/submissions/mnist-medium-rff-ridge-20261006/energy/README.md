# Energy harness — non-record research diagnostic

The requested A100 measurement was completed on 8 October 2026. The unchanged
[nvml_energy.py](nvml_energy.py) remains an exploratory measurement harness, with
known reporting defects. The [reviewer summary](../reviewer-evidence.json) records
the exact source hashes, all three round summaries and the corrected
interpretation; no new run is requested.

## Completed A100 scope

One existing MNIST-medium draw 20261001, D=2448, seed 42, float32 products with
float64 Gram accumulation, chunk 2048. Hardware was an A100-SXM4-40GB, not the
A100-80GB official harness. Each of three rounds repeated the complete
eighteen-point per-draw hyperparameter selection, refit and prediction 31 times
after warmup. These are three measurement rounds on the same draw, not three
independent qualification runs or eleven fresh draws.

Counter above-idle energy median: 19,854.014291898562 mJ/task.
Sampled-power median: 19,900.217386794706 mJ/task. CUDA-event time
median: 193.52409904233872 ms/task; wall time median: 193.51906867741957 ms/task.
Peak allocated CUDA memory: 220,134,912 bytes. Existing-draw prediction was
9684/10000 and matched the author’s prediction hash/selection. Label/index
hashes matched, but regenerated image-array/NPZ hashes did not; original arrays
are absent, so bitwise input reproduction is not established.

## Actual measurement boundary

Each `fn()` calls the full [fit_predict_torch](../rff_ridge.py) learner with host
arrays. It generates the random projection on the CPU and creates/copies inputs,
projection and working tensors on the requested device in every invocation.
Those operations occur inside the repeated fit window; they are **not excluded**.
Dataset file loading and initial warmup are outside the active interval.

The source docstring and saved JSON `scope` field incorrectly call inputs
device-resident and exclude transfer/allocation. Those historical bytes are
preserved for audit; this document corrects their interpretation. The method
uses ordinary PyTorch operations here, without CUDA graph capture. GPU board
energy includes board idle while the host prepares work, but excludes host
CPU/RAM energy; it is not wall-socket or whole-job cost.

## Counter and sampled-power calculation

The harness uses `pynvml` device 0. NVML cumulative energy is read at interval
edges; a monitoring thread samples power/clocks/temperature/utilization about
every 50 ms. Each active interval is bracketed by idle intervals. For either
readout:

```text
idle_W = (idle_before_W + idle_after_W) / 2
net_mJ_per_task = 1000 * (active_J - idle_W * active_seconds) / repeats
```

The cumulative counter and trapezoidal sampled-power integral are two readouts
of the same GPU sensors, not independent calibration measurements. All signed
values are retained. Three-round medians are descriptive measurements, without
a confidence interval or a fresh-draw generalization claim.

## Legacy fields and known defects

| Field or claim | Correct interpretation |
|---|---|
| `counter_idle_adjusted_mj_per_task_median` | Median counter above-idle energy for repeated full learner calls |
| `integrated_power_idle_adjusted_mj_per_task_median` | Same calculation using sampled-power integral |
| `gpu_ms_per_task_median` | Median CUDA-event interval per repeated call |
| `peak_cuda_allocated_bytes` | Peak allocated CUDA bytes from the separate measured fit |
| `idle_baseline_w` | Summary of bracketing idle-window readings; not a separate error bound |
| `comparisons[]` where `kind == "sham"`, `*_noise_mj_per_nominal_task` | Gross idle-only Joules×1000/nominal repeats; despite the name, **no baseline subtraction** |
| `intervals[].mean_util_percent` | Bug: average of accumulated historical samples, not interval-only samples; unsuitable as that interval’s utilization |

For an interpretable A100 sham residual the reviewer separately calculated
`1000*(sham_J-idle_W*sham_seconds)/nominal_repeats`, using the same round’s
bracketing baseline as its active interval. Counter residuals for rounds 1–3
were -1216.870552,-481.274963 and-249.915157 mJ/nominal task; full precision and
both readouts are in [../reviewer-evidence.json](../reviewer-evidence.json).
Neither these residuals nor the legacy gross shams are uncertainty intervals.

The original RTX claim that GPU shams were 4–7 times smaller than the active
result is contradicted by its saved gross-sham fields. CPU-only fit telemetry
measures the idle GPU, not CPU energy. Treat both historical claims accordingly.
Also, `run_protocol.py` ignores NumPy `--dtype`, and a directly requested
NumPy-float32 design promotes to float64 through projection arrays. That issue
does not change the explicitly recorded PyTorch configuration of this A100 run.

## Reproduction commands

These commands describe the frozen harness; they do not repair its known field
semantics or promise an identical input image hash across platforms. New runs
should record their own environment, inputs and interval results.

```bash
python -m pip install "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
python -m pip install pynvml numpy

# From the repository root, generate an existing research draw.
python -m mnist.code.data --profile medium-error-targets-v1 \
    --output <draws-dir>/data-20261001 --seed 20261001

cd mnist/submissions/mnist-medium-rff-ridge-20261006

# Optional idle-only baseline, separate from the bracketing active windows.
python energy/nvml_energy.py --data-root <draws-dir> --idle-baseline \
    --out energy/results/idle_baseline.json

# Full repeated fit; three rounds on one draw, not qualification.
python energy/nvml_energy.py --data-root <draws-dir> \
    --D 2448 --device cuda --dtype float32 --chunk 2048 \
    --gram-dtype float64 --f32-products \
    --rounds 3 --shams 1 --idle-seconds 10 \
    --out energy/results/energy-rff-cuda-f32p-chunk 2048-D=2448.json
```

Original RTX 2060 records remain unchanged in
[../evidence/energy_rtx2060.json](../evidence/energy_rtx2060.json). They do not
establish a record-table comparison with the added one-configuration A100
diagnostic or with the canonical MNIST-A100 benchmark.
