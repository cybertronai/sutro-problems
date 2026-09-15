# Corrected A100 energy and W&B rerun

AI authored, 2026-09-15. This audit corrects the energy claim in submission #83.
Accuracy and runtime reproduce; the original **3.815 mJ above idle does not**.
The corrected submission value is **173.793 mJ above idle** for one complete
10,000-training / 10,000-prediction task.

## Measured results

All entries are medians on NVIDIA A100-SXM4-40GB. The published measurement is
preserved as historical evidence. The primary corrected value uses the separate
sampled-power integration path; the counter provides a cross-check.

| Method | Above idle (mJ/task) | Runtime (ms/task) |
| --- | ---: | ---: |
| Original published result | 3.815 | 3.334 |
| Fresh original protocol: counter, 5 s idle | 165.564 | 3.383 |
| Independent harness: counter, 10 s idle | 174.878 | 3.335 |
| Independent harness: power integration, 10 s idle | **173.793** | **3.335** |

All **110,000 predictions match**, with **105,130 correct (95.5727%)**. Original
graph replay checks, scratch poisoning and training-label rotation checks pass.
The independent protocol measures draw 0 with 6,000 / 12,000 / 6,000 replays,
interleaving two 20-second idle-only controls. Validation covers all eleven
frozen draws; energy is not averaged over eleven different datasets.

Power is sampled approximately every 50 ms and integrated by trapezoidal rule
over approximately 20- or 40-second active windows, then divided by replay count:

```text
net J/task = [active integral - mean(before-idle W, after-idle W) × active seconds] / replays
```

Subtracting paired idle power gives approximately **174 mJ above idle**,
**45.6 times** the archived result. The
original archive implies approximately 41 W active; the cause of that difference
has not been established. A prior separate matching-model audit also measured
approximately 176 mJ above idle by power integration; the tracked run
here is the source of the corrected submission value.

Idle windows include a short post-work power tail. An exploratory calculation
that drops their first three seconds gives **180.111 mJ** net; it is a baseline
sensitivity check, not the primary result or a confidence interval. Idle-only
controls contain zero replays; their net noise is expressed per 6,000 *nominal*
tasks. Negative values are retained. Raw timestamps, samples and all rounds are
included for independent recalculation.

## Scope and instrumentation

- The unchanged learner performs the full feature transform, basis fitting,
  projection, QDA training and 10,000 predictions in each CUDA graph replay.
  Inputs are GPU-resident. Transfers, resizing, allocation, initialization and
  capture are excluded; host dispatch gaps are included. This is warm repeated
  execution, not cold end-to-end energy.
- Both NVML readouts share GPU sensors. They measure board energy, not host CPU,
  cooling, power-supply losses or external wall-plug energy.
- The GPU model and PyTorch 2.5.1+cu124 match the publication; this is a different
  physical board. Driver 580.95.05 differs from published 580.105.08. TF32 is off.
  Full board identity, clocks, temperatures and software versions are archived.
- Modal reports the worker's GPU context as PID 1. Before measuring, the runner
  checks that no GPU compute process exists, creates a context, and verifies that
  allocating and releasing 128 MiB changes the sole reported process's memory
  accordingly. Only the original self-PID filter is instrumented to accept that
  demonstrated alias. Both guards continue rejecting other reported PIDs.
  No learner or measurement arithmetic is changed. Earlier attempts that did not
  recognize the alias aborted before energy measurement; they contributed no
  energy results to this run.
- W&B initializes on the **local controller** before dispatch. GPU samples are
  buffered and logged after measurement, avoiding SDK activity in the timed GPU
  process. W&B controller system statistics are disabled; `telemetry/*` contains
  the measured GPU data with an elapsed-seconds axis.
- The measured scripts and inputs are also archived in W&B. The checked-in
  tracker additionally accepts `WANDB_ENTITY` and `WANDB_PROJECT` for reruns,
  limits the displayed energy metrics and plot to above-idle energy, and rechecks
  all saved comparison arithmetic against raw intervals.
  These controller-only changes do not alter measurement. The preparation helper
  verifies learner source and exact input hashes before a new allocation.

## Recalculate saved evidence without a GPU

From this directory, using Python 3.11 and `uv`:

```sh
uv run --python 3.11 --with-requirements requirements-controller.txt python analyze.py --results results-sxm40
uv run --python 3.11 --with-requirements requirements-controller.txt python verify_wandb.py
```

The first command recomputes counter arithmetic and sampled-power integrals,
checks validation and process-monitor status, and regenerates summary and plot.
The second reads W&B and checks finished status, summary values and artifacts;
it requires access to the linked project. Neither allocates a GPU.

## Rerun on a matching GPU

Authenticate the Modal CLI (`modal token new`) and W&B (`wandb login`) in an
environment containing the pinned controller requirements. From the repository
root, prepare a new directory using the pinned W&B input artifact:

```sh
uv run --python 3.11 --with-requirements mnist/submissions/medium-pca-qda-20260915/energy-audit/requirements-controller.txt \
  python mnist/submissions/medium-pca-qda-20260915/energy-audit/prepare_rerun.py /tmp/pca-qda-energy-rerun
cd /tmp/pca-qda-energy-rerun
# Set these to a W&B destination you can write to.
export WANDB_ENTITY=your-wandb-entity
export WANDB_PROJECT=sutro-mnist-tiers
uv run --python 3.11 --with-requirements requirements-controller.txt python run_wandb.py
uv run --python 3.11 --with-requirements requirements-controller.txt python verify_wandb.py
```

The preparation command downloads only the exact frozen inputs from the versioned
W&B artifact and checks all eleven payload hashes. If those inputs are already
available locally, append `--payloads /path/to/payloads` to preparation to avoid
the download. A fresh output directory is required; archived evidence is not
overwritten. The new run URL is printed and saved in `wandb-run.json`.

The runner allocates one A100 through Modal and requires the exact
`NVIDIA A100-SXM4-40GB` model. A different model or another reported compute
process aborts the run. The remote image is pinned by digest and requires PyTorch
2.5.1+cu124. The measured workload takes approximately six minutes, plus image
startup and artifact transfer. The remote app shuts down after completion.

## Evidence

The original `../results/gpu_results.json` is unchanged. `plan-wandb.json` records
the hashes of the scripts used in the completed measurement, before the added
rerun destination options. `payload-manifest.json` is the exact measured input
manifest; the payload binaries live in the versioned W&B artifact instead of Git.

- [Completed W&B run](https://wandb.ai/yaroslavvb/sutro-mnist-tiers/runs/rha1uwss)
- [Power trace and comparison](energy-audit-sxm40.png)
- [Recomputed summary](summary-sxm40.json)
- [Original-protocol raw measurements](results-sxm40/original.json)
- [Independent samples, intervals and controls](results-sxm40/independent.json)
- [Execution and PID ownership checks](results-sxm40/execution.json)
- [W&B completion verification](wandb-verification.json)
