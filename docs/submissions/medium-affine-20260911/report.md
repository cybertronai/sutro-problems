# MNIST-medium: an attempt at 98% accuracy

> **Accuracy scope:** These are historical results on one fixed dataset. The current small/medium rule requires **mean ± sample SD over 11 independently resampled datasets**; that aggregate has not been measured here. Threshold checks below describe the fixed dataset. Training-seed repeats do not supply across-dataset SD. [Current accuracy protocol](https://github.com/cybertronai/sutro-problems/blob/main/mnist/instructions.md#accuracy-over-11-random-datasets).

The fixed network achieves **5,755/6,000 correct (96%)** on the canonical
MNIST-medium split. It **does not meet your 98% goal** and **does not meet the
repository's 98.14% requirement**. The attempt is 125 correct predictions short of your 98% goal and 134 short of the repository threshold.
This report records the attempted solution and its measured performance.

**Frozen algorithm:** 81 → 512 ReLU → 10; 200 epochs; learning rate 0.1; batch size 30; seed 101. It learns from all 6,000 supplied training
examples and predicts all 6,000 test examples. Configuration selection uses
only a training-data validation split; the final seed and configuration were
frozen before the test labels were opened.

Contributors: Yaroslav Bulatov (task and requirements), Codex (implementation,
experiments, measurements, and reporting). No W&B run was created; evidence is
preserved in the files linked below. Publication is a submission attempt, not
benchmark acceptance.

[TOC]

## Results in the same units

Execution times use **ms**, energies **mJ**, modeled area **mm²**, and scoring
time **s**. Human values have two significant figures; exact prediction counts,
thresholds, and raw measurements remain unrounded in the evidence. In these
units, energy in mJ equals power in W multiplied by time in ms.

| Execution | Accuracy | Time (ms) | Energy (mJ) | Area (mm²) | Time to score (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dally v4, complete learning task | 96% (5,755/6,000) | 93,000 | 200 | 0.63 | 2.7 |
| A100, complete learning task; three-trial mean | 96% (5,755/6,000) | 4,700 | 150,000 | — | — |

Every measured A100 task resets parameters, prepares inputs and targets, trains
the full frozen epoch budget, and predicts all test labels. The comparison
covers the same learning algorithm. The physical energy accounting differs:
the theoretical model prices scratch accesses, while NVML measures GPU-board
energy after subtracting idle power. Modeled area counts occupied scratch cells. The ratio of A100 to modeled time is
0.051; the energy ratio is 790.

## Train-only selection and the target

The predeclared grid contains 12 training runs: widths 128, 256, and 512; learning
rates 0.01, 0.03, 0.1, and 0.2. Each finite run is examined at epochs 25, 50, 100,
200, and 300. A fixed PCG64 permutation with seed 20260913 partitions only the
training arrays into 4,800 fitting examples and 1,200 validation examples. The
split is not stratified. All search runs initialize with seed 11.

The selection rule first seeks 1,176/1,200 validation predictions (98%), choosing
the least `epochs × width` work among candidates that meet it. If none meets it,
the rule maximizes validation correct, breaking ties by work, width, then rate.
One configuration and final seed 101 are frozen; it is refitted from scratch on
all 6,000 training examples. Predictions are saved before separate evaluation.
There is no further configuration or seed search after test evaluation.

Best observed checkpoint within each predeclared width/rate run:

| Width | Learning rate | Epochs | Validation accuracy | Fitting accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.01 | 300 | 95% (1143/1200) | 98% (4706/4800) |
| 128 | 0.03 | 100 | 95% (1142/1200) | 98% (4699/4800) |
| 128 | 0.1 | 100 | 95% (1141/1200) | 99% (4748/4800) |
| 128 | 0.2 | 100 | 95% (1135/1200) | 99% (4758/4800) |
| 256 | 0.01 | 300 | 95% (1142/1200) | 99% (4737/4800) |
| 256 | 0.03 | 300 | 96% (1147/1200) | 100% (4787/4800) |
| 256 | 0.1 | 300 | 95% (1139/1200) | 100% (4796/4800) |
| 256 | 0.2 | 50 | 95% (1144/1200) | 99% (4759/4800) |
| 512 | 0.01 | 200 | 96% (1148/1200) | 99% (4746/4800) |
| 512 | 0.03 | 300 | 96% (1148/1200) | 100% (4798/4800) |
| 512 | 0.1 | 200 | 96% (1151/1200) | 100% (4798/4800) |
| 512 | 0.2 | 100 | 95% (1138/1200) | 99% (4768/4800) |

The selected validation result is **1151/1,200**.
The complete checkpoint records, nonfinite stops if any, split indices, source
hashes, selection rule, and freeze timestamps are retained in the raw evidence.
One validation example changes accuracy by about 0.083 percentage points; tiny
ranking differences are not strong generalization evidence.

This is a bounded attempt using an algorithm that can be exactly represented
by the current compact IL. It does not establish that 98% is impossible for
MNIST-medium. The historical 98.14% result used a deeper CNN, a different
10,000/10,000 dataset, AdamW, cross-entropy, and other modeling choices. Its
weights are not reused: some examples in that historical training set occur in
the current test split. Only algorithm ideas and published summary results
inform the choice of model family.

## Exact learning algorithm

Input pixels are transformed as `FP32(FP32(x × 4) − 0.5)`. The first layer uses
ReLU and the second is linear. The loss is half squared error against one-hot
targets, averaged over each batch. Batches contain 30 contiguous examples in
supplied order, without shuffling, giving 200 updates per epoch.

PCG64 seed 101 generates input weights uniformly within `[-1/9, 1/9]` and output
weights within `[-1/sqrt(width), 1/sqrt(width)]`; biases start at zero. The initial
parameter words are data-independent program constants. Generating these
constants is outside model/GPU execution; writing/resetting them is included.

For a batch, compute `hidden = ReLU(x @ W1 + b1)`, `output = hidden @ W2 + b2`,
`delta2 = output − target`, and `delta1 = (delta2 @ old_W2.T) * (hidden > 0)`.
Sum gradients in ascending batch-row order and subtract
`FP32(learning_rate / 30) × gradient` from every parameter. All backpropagation
uses pre-update weights. Products and additions round separately to FP32 with
no fusion. Reduction indices increase monotonically. Inference chooses the
first class on an exact maximum-score tie.

## Compact program and modeled area

The model is single-core-with-tape, v4, pinned at
`26abcca402de647381d31286d42dfbb7a001763d`. The builder emits fixed loops, affine addresses,
and priced primitive instructions. It uses the unchanged compact scorer from
the prior study; no free matrix operation or learner-specific cost certificate
is introduced.

The program receives **978,000 input words**, executes
**247,121,890,842 primitive instructions** with
**736,336,410,842 charged operand accesses**, and sends 6,000
predictions. Fixed half-diamond placement determines each scratch access cost.
Tape operations retain their zero-access-cost convention at the pinned revision.

The medium builder streams test queries after training: receive one image,
normalize it, run inference, send a prediction, then reuse the query buffer.
It allocates **630,235 scratch words**,
reported as **0.63 mm²** at one square
micrometre per word. This is occupied cell area, excluding tape and instruction
storage, not a bounding rectangle or A100 allocation. Streaming keeps the
program within the existing scorer's allocation and initialization-proof guards.

The canonical program hash is `09258c87abec502d08622e0e800f1865ef1bc54e15d89eed238dade08ac0e828`. The serialized IL
has 46,819 syntax nodes and 46,757
primitive leaves. Exact affine address histograms aggregate all dynamic reads
and writes. Source aliases and unchosen select operands retain their charges.

The reported static scoring time is the median of
3 complete calls, including schema,
bounds and definite-initialization validation, placement, histograms, exact
integer sums, and canonical hashing. Program construction, JSON loading, numeric training, accuracy
evaluation, and file output are excluded. Scoring runs on Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz
with 16 logical processors and
64 GiB RAM; Python
3.11.13 and NumPy 2.4.6.
Native integer units convert as
`time_ticks_0_2_ps / 5e9` ms, `energy_fj / 1e12` mJ, and
`area_um2_occupied_cells / 1e6` mm².

## Numerical and scoring verification

The CPU learner's matrix operations are checked against explicit ordered FP32
reductions for the medium shapes, including transposed gradients and row sums.
A complete training epoch is compared in both implementations. The entire
final GPU run is checked against all CPU parameter words, every output-score
word, and every prediction, before and after timing.

The generalized builder also passes four tests: unchanged small-program
regression, independently expanded toy semantics including 81-feature inputs,
per-address cost agreement, and allocation/guard behavior. These establish
different properties. The full medium program is not expanded and interpreted
instruction by instruction, and a second slow ordered CPU replay of every
training epoch is not performed.

The allowed-input-only CLI check removes test labels and source indices, then
refits the model and reproduces the saved outputs. Independent verification
records the exact scope, source fingerprints, matrix checks, and timings.
The primary CPU refit took **41,000 ms**,
separate from the static scoring time and A100 measurements.

## A100 implementation and measurements

Hardware: **NVIDIA A100-SXM4-40GB**. Software: Python 3.11.15,
PyTorch 2.5.1+cu124, Triton 3.1.0, NumPy 2.2.6,
CUDA 12.4, NVIDIA driver 580.95.05,
NVML 13.580.95.05, and `nvidia-ml-py` 12.560.30.
Container: `ghcr.io/ab-10/wikitext-bench@sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f`.

The manual Triton implementation preserves ascending FP32 reductions and
disables fusion. Large reduction loops remain runtime loops to keep compilation
bounded. It is a port of the fixed algorithm, not an automatic IL-to-PTX compiler.

Three CUDA graphs hold initialization, one complete training epoch, and
inference. Each measured task runs initialization once, the epoch graph
200 times, and inference once. A graph replay is therefore not
an entire task; the trial counters record complete tasks separately. All
parameter reset and training are repeated. The GPU materializes all output
scores for verification, and that extra work is included in its measurements.

Independent CPU parameters, scores, and predictions are host-only verification
oracles. They are not copied to the GPU or consulted by learner kernels. The
GPU learner receives only the three allowed data arrays and seed-only initial
literals. Changed-query and bounded changed-training-label checks exercise
input dependence; their exact scope is retained in the GPU evidence.

Each of the three trials executes three complete tasks, about 14,000 ms of
active work. Each 3,000 ms idle sample follows a 3,000 ms settling gap, before
and after the active window. CUDA events and wall time measure complete repeated
tasks. NVML cumulative
board energy is differenced over each active interval, then adjusted by the
mean of bracketing idle power measurements. Per-task mJ is
`1000 × (active_J − mean_idle_W × active_seconds) / complete_tasks`.
The subtraction uses the NVML interval; no negative value is clipped.

| Trial | Accuracy | Complete tasks | Time (ms/task) | Raw energy (mJ/task) | Paired-idle energy (mJ/task) | Before-only / after-only energy (mJ/task) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 96% (5,755/6,000) | 3 | 4,700 | 430,000 | 150,000 | 150,000 / 150,000 |
| 2 | 96% (5,755/6,000) | 3 | 4,700 | 430,000 | 160,000 | 160,000 / 150,000 |
| 3 | 96% (5,755/6,000) | 3 | 4,700 | 430,000 | 160,000 | 160,000 / 160,000 |

Mean task time is **4,700 ms** (sample SD
3.4 ms).
Mean idle-adjusted board energy is **150,000 mJ** (sample SD
1,800 mJ).
The raw counter readings, idle measurements, graph/task counts, trial durations,
hardware telemetry, and both one-sided baseline alternatives are retained.

Transfers, allocations, compilation, graph capture, verification, cold start,
and host CPU energy are excluded. Repeated GPU-resident tasks may benefit from
cache residency. These measurements describe this implementation and boundary;
they are not a claim of optimal A100 performance or calibrated per-kernel energy.

## Provenance and reproduction

Dataset: `competition-v2`, canonical seed 20260910, 6,000/6,000 disjoint examples
from the original MNIST training pool, area-downsampled to 9 × 9. Allowed array
content SHA-256 values, C-order little-endian:

- `train_images`: `3c2c9d7cc3103e05c6e72e8b39ad7ae8f93fcbea8b23ac5f43ffacd9a9c4653b`.
- `train_labels`: `91633f2c00fe471cb130fb6831c3d7dca5899bd926a1c899d873826dfd62e617`.
- `test_images`: `f121936f4b6b0c2eebef773444fc45e72832d1c84948a50969a909467d6b1003`.

Predictions: `fba860ddadbb7007536ed08736f6e834588f01cec812fad375e55bf58fbd1133`.
Output scores: `dba18d2265ce7ecc376a03088342ec7da971534c2fbfd66df4e96f24ca5d7c3a`.
GPU source: `1f981a617cf49680f281a9015b159c85d3b444cb0883ab52d56773470bee3cb5`.
Further manifest, parameter, source, and PTX hashes are preserved in the evidence.


The submission branch is `codex/mnist-medium-submission`. After it is merged,
the same commands work on `main`. Run from the repository root:

```sh
git clone --branch codex/mnist-medium-submission https://github.com/cybertronai/sutro-problems.git
cd sutro-problems
python3.11 -m venv .venv-medium
.venv-medium/bin/python -m pip install -r mnist/submissions/medium-affine-20260911/requirements.txt
.venv-medium/bin/python -m mnist.code.data --output mnist/data --seed 20260910
S=mnist/submissions/medium-affine-20260911
.venv-medium/bin/python "$S/learner.py" --data mnist/data/medium.npz --config "$S/config.json" --output /tmp/mnist-medium-reproduction
.venv-medium/bin/python -m mnist.code.evaluate --tier medium --predictions /tmp/mnist-medium-reproduction/predictions.npy --data-dir mnist/data --output /tmp/mnist-medium-reproduction/accuracy.json
.venv-medium/bin/python "$S/score.py" --config "$S/config.json" --output /tmp/mnist-medium-reproduction
.venv-medium/bin/python "$S/verify.py" --data mnist/data/medium.npz --config "$S/config.json" --artifacts /tmp/mnist-medium-reproduction --output /tmp/mnist-medium-reproduction/verification.json
```

The evaluator reports the repository threshold; the verifier reports both the
requested and repository thresholds. Valid predictions below a threshold still
produce a successful evaluator command with `meets_accuracy_target=false`.
This flag checks classification accuracy, not complete benchmark acceptance.

The learner imports numerical routines from the checked-in
`mnist/experiments/accuracy-il-20260911/accuracy_study.py`. The model builder uses
the unchanged `il.py` in that directory. Neither dependency loads saved weights
or predictions. Canonical input array hashes are checked against
`mnist/doc/dataset_manifest.json`. The learner reads only `train_images`,
`train_labels`, and `test_images`; separate evaluation and verification code
may read test labels after predictions are fixed.

Verification compares a fresh complete fit with the retained parameter, score,
and prediction bits, including a learner CLI run whose NPZ contains only the
three allowed members. It independently checks ordered reductions on multiple
shapes, one complete training epoch, and final inference. It does not replay
every epoch through the independent ordered reference or numerically expand the
full affine program. Small expanded-program tests and an exact regression to
the previous small program are recorded in `model-ir-validation.json`.

The optional search reproduction below is substantially more work than
reproducing the selected learner. It writes to a separate directory and refuses
to overwrite existing search evidence. It reads training data only and does
not replace the frozen submission configuration:

```sh
.venv-medium/bin/python "$S/search.py" --data mnist/data/medium.npz --output /tmp/mnist-medium-search
```

## Reproduce the A100 measurements

With configured Modal credentials and `uv` installed, first generate the CPU
artifacts above, then run:

```sh
uvx --with numpy==2.2.6 modal==1.5.5 run "$S/gpu_benchmark.py" --data mnist/data/medium.npz --config "$S/config.json" --reference /tmp/mnist-medium-reproduction --output /tmp/mnist-medium-gpu
.venv-medium/bin/python -m mnist.code.evaluate --tier medium --predictions /tmp/mnist-medium-gpu/gpu_predictions.npy --data-dir mnist/data --output /tmp/mnist-medium-gpu/accuracy.json
```

The reference directory must contain `parameters.npz` with `w1`, `b1`, `w2`,
and `b2`, plus `output-scores.npy` and `predictions.npy`. These are host-only
verification oracles: no learned oracle weights, scores, or predictions are
copied to the GPU or supplied to its learner kernels. Full parameter, score,
and prediction bits are checked against the CPU artifacts before and after
measurement. The remote container pins NumPy 2.2.6; the local CPU and scoring
environment pins NumPy 2.4.6.

Each complete task replays three captured graph types: initialization once,
the one-epoch graph E times, and inference once, for E + 2 graph replays.
Initialization transforms training and test pixels, creates targets, and resets
parameters. Every task trains afresh. Three trials measure repeated,
GPU-resident complete tasks. Transfers, allocations, JIT compilation, capture,
verification, and host energy are excluded. The manually implemented GPU
kernels follow the learner equations; they are not generated by an automatic
IL-to-PTX compiler.

## Model scoring and report generation

The proposed `sutro-affine-v4/0.1` representation encodes bounded loops and
affine addresses, then counts every expanded v4 primitive and memory access.
Training data remain resident while test queries are received, normalized,
classified, and emitted one at a time. This keeps the searched models below
the unchanged scorer's one-million-word memory guard. The GPU uses its own
parallel buffer layout; the model's occupied scratch area is not GPU memory
usage or die area.

The scorer validates the restricted program, proves initialized reads under
its supported rules, constructs placement and per-address histograms, and
sums exact costs. It does not execute the billions of primitive instructions
numerically. Its timing excludes program construction, JSON loading, numerical
training, accuracy checks, and file output.

Human-readable results use two significant figures: execution time in ms,
energy in mJ, occupied scratch-cell area in mm², and time to score in s.
Exact measurements and the model's native ps/fJ/µm² quantities remain in JSON.
In these display units, `E_mJ = P_W × t_ms`.

To rebuild the report and Pages files from retained evidence:

```sh
.venv-medium/bin/python -m pip install -r "$S/requirements-report.txt"
.venv-medium/bin/python "$S/write_report.py"
.venv-medium/bin/python "$S/build_pages.py"
```

Report generation does not rerun training or GPU trials. The readable session
is a timestamped snapshot of visible user and assistant messages; hidden
reasoning, system instructions, and tool payloads are omitted.

- [Standalone report](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-affine-20260911/)
- [Separate ambiguities and problems](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-affine-20260911/ambiguities.html)
- [Human-readable session](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-affine-20260911/session.html)


## Download the evidence

- [Frozen configuration](config.json), [predeclared search plan](predeclared_plan.json), [validation results](validation_results.json), [split indices](validation_split.json).
- [Learner](learner.py), [CPU results](cpu_results.json), [predictions](predictions.npy), [parameters](parameters.npz), [output scores](output-scores.npy).
- [Official accuracy evaluation](accuracy.json), [requested 98% goal status](goal-status.json).
- [Compact program](program.il.json), [builder](model_ir.py), [score wrapper](score.py), [model scores](model-score.json), [independent count audit](model-score-audit.json), [builder validation](model-ir-validation.json).
- [Independent verifier](verify.py), [verification evidence](verification.json).
- [GPU source](gpu_benchmark.py), [GPU results](gpu_results.json), [measurement notes](gpu-notes.md).
- [Separate ambiguities and problems](ambiguities.html), [human-readable session](session.html).
