# GPU energy measurement protocol

The protocol behind the [KernelBot MNIST energy prototype](../kernelbot/README.md),
built from the two audits that settled the PCA-QDA discrepancy
([PCA-QDA energy audit](../submissions/medium-pca-qda-20260915/energy-audit/README.md),
[small-QDA energy benchmark](../submissions/small-qda-20260915/README.md)).
[`code/energy.py`](../code/energy.py) needs only NumPy. It drives any
telemetry and workload objects and recomputes every reported number from a
saved record:

```sh
python -m mnist.code.energy analyze RECORD.json
python -m unittest mnist.code.tests.test_energy -v   # simulated GPU, no hardware
```

## What is measured

**GPU board energy above idle, per complete training-and-prediction call**, from
warm repeated execution. One call is usually far shorter than the power
sensor's resolution, so each active window repeats it N times (N sized for
about 20 s):

```text
net mJ/call = (active J − mean(idle-before W, idle-after W) × active s) × 1000 / N − harness mJ/call
```

Energy is read two ways from the same sensors: the trapezoidal integral of
power polled every 50 ms (headline) and NVML's cumulative energy counter
(cross-check). Neither is an external wattmeter. Host CPU energy is out of scope.

Default sequence, each active window bracketed by 3 s of settling and 10 s of
measured idle:

1. **Reference**: 4096 × 4096 FP32 matmul for 10 s, TF32 off, output checked
   exactly. The run **aborts here** if telemetry fails the reference gate.
2. **Task** × 3, with an **idle-only sham** and (optionally) a **harness control**
   after the first. The control repeats the harness's per-call work without the
   learner; its net energy is subtracted.
3. Optional **duty-cycle diagnostic** windows.
4. **Reference** again, to catch drift or throttling.

An optional `quiesce` object keeps an untrusted task's process frozen except
during calibration, its own windows and verification.

## Gates

A record passes only if every gate passes. Thresholds live in the recorded,
hashed plan.

| Gate | Passes when | Why |
| --- | --- | --- |
| `reference` | Matmul net energy is 6–11 J per 10¹² FLOPs and throughput 15–23 TFLOP/s on both meters, in both reference windows (A100 only; other GPUs are `unchecked` and do not pass) | Two healthy A100-SXM4-40GB hosts measured **8.49 and 8.36 J/TFLOP at 19 TFLOP/s** despite 320 W vs 400 W power limits and 38 W vs 58 W idle. The faulty host behind the original 3.8 mJ PCA-QDA claim read ~1.4 W above idle under dense matmul (≈0.07 J/TFLOP). A "≥ 20 W above idle" check would pass a sensor at half scale. The band is provisional, and derived from 40GB boards. |
| `meters_agree` | Active-window mean power from counter and integral within 3% | Independent arithmetic over one sensor; disagreement means sampling or counter trouble. |
| `sham_residual` | Idle-only sham's apparent net energy per nominal call ≤ 5% of task net energy | Bounds baseline-subtraction error at this task's scale. |
| `round_spread` | (max − min) / median of task rounds ≤ 10% | Repeatability within the run. |
| `telemetry` | No sampler errors; no sample gap > 0.5 s | Samples cover every window. |
| `exclusive_gpu` | No unexpected compute process at any check | Another context's work would be billed to the task. |
| `task_verified` | The task's `verify()` passes before and after | The measured computation is the correct one. |

## Why the sampler is a separate process

The existing audits poll NVML from a thread inside the measured process. That
thread runs only when the GIL is free. In the simulated duty-cycle test (50%
busy, CPU-bound host loop) such a thread landed on load 25% of the time, so the
power integral read **0.63×** the true energy while the counter read 1.00×.
Replay loops that release the GIL inside CUDA calls avoid most of this, which is
why those audits' meters agreed, but a learner with Python between kernels
would be under-measured. The protocol samples from a spawned process;
`perf_counter` is system-wide, so timestamps share the parent's clock.

## Open measurement questions

- **Cadence dependence.** Back-to-back calls keep the board in its highest
  performance state. The duty-cycle diagnostic reports net energy per call when
  bursts are separated by pauses; if it rises as duty falls, the competition
  should fix the cadence explicitly.
- **Sensor timing.** NVML power on A100-class boards is reportedly an average over
  only part of each update period. Saturated 20 s windows average this out; a
  duty-cycle burst period near 0.1 s is the place to look for aliasing.
- **External calibration.** Both meters share the board's sensors. Only a wall or
  shunt measurement can check absolute scale; the reference gate checks
  consistency across boards.
