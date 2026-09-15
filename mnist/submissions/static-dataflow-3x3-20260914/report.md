# MNIST-small: static dataflow energy demo (3x3)

Submission date: September 14, 2026. The implementation was previously merged
in [PR #58](https://github.com/cybertronai/sutro-problems/pull/58) and
[PR #69](https://github.com/cybertronai/sutro-problems/pull/69); this package
records it as a historical MNIST-small submission and adds the README row.

**Historical specification: 600/600 examples and single-core static scoring.**

Upstream main now requires 1,000/1,000 examples, 67% mean accuracy over
eleven draws, and spatial-computer grid scoring. This demo was measured under
the previous specification and is NOT a current qualifying submission. Its
movement score uses the matmul competition's single-core read-cost scorer
(`ceil(sqrt(addr))` per read), not the spatial-computer model, so the grid
columns in the README row are em dashes. Accuracy is one fixed draw, not an
eleven-dataset mean.

## Results

| Metric | Value |
|---|---:|
| Accuracy, float model (600/600) | 53.0% |
| Accuracy, int8 IR (same predictions) | 53.0% |
| IR ops | 280 (90 `mul`, 90 `add`, 100 `set`) |
| Static movement score | 4,625 distance units |
| On-chip movement energy | 4.625 pJ per inference (1 fJ per unit) |
| A100 energy per 1,000-example batch call | 0.108 mJ idle-adjusted (0.41 mJ raw) |
| A100 time per 1,000-example batch call | 6.24 µs launch cadence |
| Measured / model energy per inference | 108 nJ / 4.625 pJ = 23,400x |

Costs use two significant figures in the README row; exact values are in
`matmul/mnist3_1k_a100_results.json` and in the trial table below.

## What the demo does

`matmul/mnist3_demo.py` (self-contained, merged in PR #58):

1. Downsamples 28x28 MNIST to 3x3 by block averaging over unequal blocks
   (`[0:9]`, `[9:19]`, `[19:28]`), then zero-centers to signed 6-bit features.
2. Trains a 10-class softmax-linear classifier in float64 numpy (60 epochs,
   lr 0.12, seed 0) on 600 training images.
3. Quantizes weights to int8 under a single per-tensor scale. Quantization
   costs nothing here: float and int8 predictions give the same 53.0%.
4. Emits the trained inference as a straight-line Dally IR program
   (`matmul/mnist3_ir.txt`): inputs at cells 1-9, per-class `mul`/`add`
   dot-product chains, weights and biases as `set` immediates, 10 logit
   outputs. 280 ops total.
5. Scores movement with the matmul read-cost scorer: reading address `a` as
   an operand costs `ceil(sqrt(a))`, outputs are charged likewise, `set` is
   free. Total: 4,625 distance units.
6. Converts to energy with the calibration 1 fJ per byte per unit of charged
   distance: 4,625 fJ = 4.625 pJ per inference.

The emitted IR was executed through the dally-eval Rust engine on live
instances and is bit-exact against the numpy model modulo 256 (the model's
8-bit cell width). Argmax is taken in the pre-wrap integer domain, where the
chosen input scale keeps every |logit| below 128.

A local reproduction on 2026-09-14 re-printed 53.0% / 53.0%, 280 ops, static
cost 4,625, and a byte-identical IR file.

## A100 NVML audit

The 1,000/1,000 practice run (PR #69) transpiled the same 280-op IR to CUDA
(`matmul/dally_ir_to_cuda.py`) and measured it through Modal
(`matmul/modal_mnist_1k_a100.py`).

- Hardware: NVIDIA A100-SXM4-40GB, driver 580.95.05, CUDA runtime 13.0,
  power limit 400 W. Container `nvidia/cuda:12.4.1-devel-ubuntu22.04`,
  `nvcc -O3 -arch=sm_80`, nvidia-ml-py 12.560.30.
- Bit-exactness was checked before timing: all 1,000 instances × 10 outputs
  match the expected fixture.
- Protocol: NVML `nvmlDeviceGetTotalEnergyConsumption` counter. Per trial, a
  5 s loaded-idle baseline, then ~5 s of back-to-back 1,000-example batch
  calls (801,444 reps, auto-sized from the measured 6.24 µs per launch). Five
  trials. Idle-adjusted energy = counter delta − paired idle power × interval.
- Result: raw 0.411 mJ per batch call, idle baseline 61.7 W, idle-adjusted
  0.108 mJ per batch call (SD across trials 0.009 mJ), i.e. 108 nJ per
  inference.

| Trial | Raw mJ/call | Idle W | Adjusted mJ/call |
|---:|---:|---:|---:|
| 0 | 0.408 | 57.8 | 0.125 |
| 1 | 0.410 | 63.5 | 0.100 |
| 2 | 0.408 | 61.9 | 0.104 |
| 3 | 0.410 | 63.2 | 0.101 |
| 4 | 0.416 | 61.9 | 0.111 |

The measured-to-model energy ratio is 2.34e4. At this workload size a batch
call lasts 6.24 µs and is launch-bound, so both measured time and measured
energy sit on the kernel-launch overhead floor. The 23,400x gap is a
statement about workload size versus launch overhead, not a calibration of
the 1 fJ/unit figure.

## Scope notes

- The A100 columns measure batch inference of the transpiled kernel only.
  Training (softmax regression on 600 examples) ran in numpy on CPU, outside
  the measured GPU window. Rows such as Panel-cached MLP measure a complete
  train-and-predict task.
- Accuracy is one fixed dataset: the first 600 train and 600 test images of
  the official MNIST splits, not the historical competition draw and not an
  eleven-draw mean. Chance is 10%.
- On-chip movement only: 1 fJ/unit is an on-chip wire-energy figure.
  Off-chip DRAM/HBM streaming is not priced; a memory-tier edge cost is
  needed before larger tiers mean anything.
- No spatial-computer placement, hop count, scratch size, or time-to-score
  was measured; the grid columns are unmeasured, per the historical-section
  convention.

## Implementation

All code is merged on `main` under `matmul/`:

- `matmul/mnist3_demo.py` — Tier 1 demo: download, downsample, train,
  quantize, emit IR, static score (PR #58).
- `matmul/mnist3_ir.txt` — the 280-op IR as emitted.
- `matmul/mnist3_1k_demo.py` — 1,000/1,000 practice-run builder (PR #69).
- `matmul/dally_ir_to_cuda.py` — straight-line IR to CUDA C++ transpiler.
- `matmul/modal_mnist_1k_a100.py` — Modal A100 harness (NVML audit).
- `matmul/mnist3_1k_a100_results.json` — raw audit output with exact values.
