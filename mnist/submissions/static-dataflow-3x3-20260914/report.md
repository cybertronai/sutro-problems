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

**Correction (2026-09-15).** The review of the original commit (c47a203) on
[PR #78](https://github.com/cybertronai/sutro-problems/pull/78) found that
this report mixed numbers from two implementations: it presented the
600-example prototype's unwrapped 53.0% as "int8 IR" accuracy while pairing
it with A100 costs measured on a different, rescaled 1,000-example model.
The results below now report the 1,000-example int8-safe model that the A100
audit actually measured, and the 600-example prototype is documented
separately with its real bytecode accuracy. The README row features the
1,000-example model only.

## Results

Reported implementation: the 1,000-example int8-safe model
(`matmul/mnist3_1k_ir.txt`).

| Metric | Value |
|---|---:|
| Accuracy, IR bytecode (1,000/1,000) | 48.4% (484/1,000) |
| IR ops | 280 (90 `mul`, 90 `add`, 100 `set`) |
| Static movement score | 4,625 distance units |
| On-chip movement energy | 4.625 pJ per inference (1 fJ per unit) |
| A100 energy per 1,000-example batch call | 0.108 mJ idle-adjusted (0.41 mJ raw) |
| A100 time per 1,000-example batch call | 6.24 µs launch cadence |
| Measured / model energy per inference | 108 nJ / 4.625 pJ = 23,400x |

Costs use two significant figures in the README row; exact values are in
`matmul/mnist3_1k_a100_results.json` and in the trial table below.

## The 600-example prototype and its logit overflow

The first implementation (PR #58, `matmul/mnist3_demo.py`) trained the same
softmax-linear classifier on the first 600 train images and quantized
weights to the full int8 range under a single per-tensor scale. In
unwrapped integer arithmetic its predictions matched the float model:
318/600 = 53.0%.

The IR's cells are bytes: `add` and `mul` wrap mod 256, and the 10 output
cells hold the logits' residues mod 256, read as signed int8. Because the
dot-product chains are linear, wrapping each intermediate still yields
exactly `logit mod 256` at the output, so the IR is bit-exact against the
numpy model modulo 256. But residues do not preserve order. The prototype's
inputs span ±31 and its weights span ±127, and its unwrapped logits reach an
observed maximum of 4,556. A logit of 4,556 wraps to 204, read as signed
int8 that is −52; any pair of logits whose difference crosses a multiple of
256 can collapse or invert under the wrap. The original claim that the
chosen input scale kept every |logit| below 128 was wrong.

Measured on the retained `matmul/mnist3_ir.txt`, executed with the mod-256
byte semantics (`ref_execute` in `matmul/mnist3_1k_demo.py`) on the first
600 official test images:

| Check | Result |
|---|---:|
| Unwrapped integer classifier | 318/600 = 53.0% |
| Maximum absolute unwrapped logit | 4,556 |
| IR bytecode, outputs mod 256 read as signed int8 | 62/600 = 10.33% |

10.33% is chance level for 10 classes. The 53.0% is a property of the
unwrapped Python arithmetic, not of the bytecode as executed, so the
prototype is omitted from the README table.

## The 1,000-example int8-safe fix

The practice-run builder (PR #69, `matmul/mnist3_1k_demo.py`) scales the
problem to 1,000 train / 1,000 test images and removes the wrap hazard at
the source: `quantize_weights_int8safe` shrinks the quantized weights
(scale × 0.9 per step) until every train-set logit fits |logit| ≤ 120, with
the test batch asserted ≤ 127. The weights land near ±5, the mod-256 wrap
never fires, and argmax over the wrapped output bytes equals argmax over
the unwrapped integers.

The resulting `matmul/mnist3_1k_ir.txt` keeps the same straight-line
280-op structure (9 input cells, per-class `mul`/`add` chains, weights and
biases as `set` immediates, 10 logit outputs) and the same static movement
score of 4,625 units, because `set` immediates are free in the scorer and
the cell layout is unchanged. On the first 1,000 official test images its
outputs, read as signed int8, score 484/1,000 = 48.4%, verified two ways:
direct execution of the IR under mod-256 byte semantics, and the retained
`matmul/mnist3_1k_expected.bin` fixture. Locally the unwrapped logits peak
at |logit| = 116 and the wrapped and unwrapped argmaxes agree on all 1,000
examples, confirming the wrap is inert.

These checks were rerun locally on 2026-09-15 against the retained IR files,
fixtures, and official MNIST labels, and match the reviewer's numbers.

## What the demo does

The pipeline is shared by both builders (`matmul/mnist3_demo.py` for the
600-example prototype, `matmul/mnist3_1k_demo.py` for the featured
int8-safe model):

1. Downsamples 28x28 MNIST to 3x3 by block averaging over unequal blocks
   (`[0:9]`, `[9:19]`, `[19:28]`), then zero-centers to signed 6-bit
   features (±31).
2. Trains a 10-class softmax-linear classifier in float64 numpy (60 epochs,
   lr 0.12, seed 0).
3. Quantizes weights to int8: full range under one per-tensor scale
   (prototype), or rescaled by `quantize_weights_int8safe` until every
   train-set logit fits in signed int8 (featured model).
4. Emits the trained inference as a straight-line Dally IR program: inputs
   at cells 1-9, per-class `mul`/`add` dot-product chains, weights and
   biases as `set` immediates, 10 logit outputs. 280 ops.
5. Scores movement with the matmul read-cost scorer: reading address `a` as
   an operand costs `ceil(sqrt(a))`, outputs are charged likewise, `set` is
   free. Total: 4,625 distance units.
6. Converts to energy with the calibration 1 fJ per byte per unit of
   charged distance: 4,625 fJ = 4.625 pJ per inference.

The emitted IR was executed through the dally-eval Rust engine on live
instances and is bit-exact against the Python reference modulo 256 (the
model's 8-bit cell width). For the int8-safe model the wrap never fires, so
the byte outputs equal the unwrapped logits and argmax over the output
bytes is the classifier's real argmax; for the prototype, see the overflow
section above.

## A100 NVML audit

The audit measured the 1,000-example int8-safe model only: the practice
run (PR #69) transpiled `matmul/mnist3_1k_ir.txt` to CUDA
(`matmul/dally_ir_to_cuda.py`) and measured it through Modal
(`matmul/modal_mnist_1k_a100.py`). The 600-example prototype was never run
on the GPU; no A100 number in this report or in the README row belongs to
it.

- Hardware: NVIDIA A100-SXM4-40GB, driver 580.95.05, CUDA runtime 13.0,
  power limit 400 W. Container `nvidia/cuda:12.4.1-devel-ubuntu22.04`,
  `nvcc -O3 -arch=sm_80`, nvidia-ml-py 12.560.30.
- Bit-exactness was checked before timing: all 1,000 instances × 10 outputs
  match `matmul/mnist3_1k_expected.bin`.
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

- The A100 columns measure batch inference of the transpiled int8-safe
  1,000-example kernel only. Training (softmax regression) ran in numpy on
  CPU, outside the measured GPU window. Rows such as Panel-cached MLP
  measure a complete train-and-predict task.
- Accuracy is one fixed dataset: the first 1,000 test images of the official
  MNIST split, not the historical competition draw and not an eleven-draw
  mean. Chance is 10%. The 600-example prototype's numbers (53.0% unwrapped,
  10.33% bytecode) are recorded above and are not featured in the README
  table.
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
- `matmul/mnist3_ir.txt` — the prototype's 280-op IR as emitted.
- `matmul/mnist3_1k_demo.py` — 1,000/1,000 practice-run builder with
  `quantize_weights_int8safe` (PR #69).
- `matmul/mnist3_1k_ir.txt` — the featured int8-safe 280-op IR as emitted.
- `matmul/mnist3_1k_inputs.bin`, `matmul/mnist3_1k_expected.bin`,
  `matmul/mnist3_1k_fixture.bin` — retained inputs, expected outputs, and
  dally-eval fixture for reproduction.
- `matmul/dally_ir_to_cuda.py` — straight-line IR to CUDA C++ transpiler.
- `matmul/modal_mnist_1k_a100.py` — Modal A100 harness (NVML audit).
- `matmul/mnist3_1k_a100_results.json` — raw audit output with exact values.
