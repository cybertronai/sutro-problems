# MNIST-small: a submission for the 60% target

> **Accuracy scope:** These are historical results on one fixed dataset. The current small/medium rule requires **mean ± sample SD over 11 independently resampled datasets**; that aggregate has not been measured here. Threshold checks below describe the fixed dataset. Training-seed repeats do not supply across-dataset SD. [Current accuracy protocol](https://github.com/cybertronai/sutro-problems/blob/main/mnist/instructions.md#accuracy-over-11-random-datasets).

**374/600 correct (62%)** on the canonical 600-training / 600-test, 3 × 3
MNIST-small dataset. The fixed 32-unit network exceeds the per-draw diagnostic threshold
of **360/600 (60%)** by 14 correct predictions. Both the model scores and the
A100 measurements cover fresh training and all 600 predictions.

Contributors: Yaroslav Bulatov (task and benchmark requirements) and Codex
(implementation, experiments, verification, and reporting). No W&B run was
created for this submission; raw evidence is supplied below.

This is a submission attempt. The compact intermediate language and the
declared model conventions still require benchmark review. This particular
configuration was chosen for submission after the feasibility study's test
results were known; the configuration and seed themselves were fixed before
that study's test evaluation. See the separate ambiguity report for the
selection boundary.

[TOC]

## Comparable measurements

Human displays use two significant figures. **Execution time is in ms, energy
in mJ, area in mm², and time to score in s.** Exact measurements and primitive
counts remain in the linked JSON. The same fixed model and predictions appear
in both rows; area applies to the theoretical scratch-cell model.

| Execution | Accuracy | Time (ms) | Energy (mJ) | Area (mm²) | Time to score (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dally v4, complete compact program | 62% (374/600) | 93 | 0.11 | 0.020 | 0.081 |
| A100, complete task; mean of three trials | 62% (374/600) | 71 | 2000 | — | — |

The A100 time is 0.76 times the modeled serial time. Its
idle-adjusted board energy is 18000 times the model energy.
These ratios compare the same learning task under different hardware and
accounting assumptions; they do not calibrate the theoretical model to silicon.
GPU energy includes the entire board's incremental energy, whereas the model
charges scratch accesses under its declared tape and arithmetic conventions.

## Fixed learning algorithm

The network is `9 → 32 ReLU → 10`, with 650 scalar FP32 parameters. Inputs use
`FP32(FP32(x * 4) - 0.5)`. Training uses squared error against one-hot labels,
300 epochs, contiguous minibatches of 30, no shuffle, and learning rate 0.2.
There are exactly 20 minibatches per epoch and 6,000 updates per task.

Seed 101 initializes PCG64. Input weights use a uniform distribution on
`[-1/3, 1/3]`; output weights use `[-1/sqrt(32), 1/sqrt(32)]`; biases start at
zero. The 650 initialization words are data-independent program constants.
Generating those constants is outside the task; writing/resetting every
parameter is included in both model and GPU execution.

For a batch, compute the hidden activations and linear outputs, then
`delta2 = output - target` and
`delta1 = (delta2 @ old_W2.T) * (hidden > 0)`. Sum gradients in increasing batch
index and subtract `FP32(0.2 / 30) * gradient` from each parameter. Backpropagation
uses the old output weights. Every multiplication and addition rounds separately
to FP32, with ascending-index reductions and no fused multiply-add. The final
prediction is the first maximum among the ten output scores.

The [feasibility study](../accuracy-il-20260911/) froze this configuration as
the best training-validation choice at a 300-epoch budget. Its three predeclared
seeds scored 374/600, 371/600, and 377/600. This submission uses the first seed,
101, without further tuning or selection among seeds. Choosing this budget for
the new target happened after those study results were public.

## Complete-task model and compact scoring

The program uses the single-core-with-tape model and v4 specification at commit
`26abcca402de647381d31286d42dfbb7a001763d`. It receives 11,400 tape words (all training
pixels and labels, then all test pixels), performs input and target preparation,
sets initial parameters and work buffers, executes every training update, and
sends 600 predicted labels. Tape reads and writes have the pinned model's zero
access charge; ordinary scratch operands retain all their charges.

The compact program has 835 syntax nodes, including
773 primitive leaves. It represents
**618,842,313 primitive instructions** and
**1,833,513,513 charged operand accesses**. Each fixed loop
contributes its exact multiplicity. Affine address histograms retain every
source read and destination write, including repeated or aliased operands.
There is no fitted training cost or free matrix operation.

Scratch storage occupies 20,290 words in
13 non-overlapping regions. At one square micrometre per word, the reported
area is **0.020 mm²**. This is occupied cell area, not the placement's
enclosing rectangle, die area, or GPU allocation. Dividing µm² by 10⁶ changes
only the presentation units.

The five complete static scoring calls have median
**0.081 s**. This includes schema, bounds and
definite-initialization checks, placement, exact address histograms, integer
cost sums, and canonical hashing. It excludes JSON loading, numerical execution,
accuracy checking, and output files. CPU: Python 3.11.13,
NumPy 2.4.6, `macOS-26.6.2-x86_64-i386-64bit`.

The exact model uses integer 0.2-ps ticks and integer fJ internally. Conversion
to the display is `ticks / 5e9` ms, `fJ / 1e12` mJ, and `µm² / 1e6` mm².
The [language specification](../accuracy-il-20260911/il.html) explains the cost
equations, fixed half-diamond placement, initialization proof, and restrictions.

## Independent numerical verification

All 300 epochs were replayed using explicit, ordered FP32 reductions. All four
parameter arrays, every one of the 6,000 output scores, and all 600 predictions
match bit for bit. Parameter hashes and predictions also match the frozen study;
that study did not save its full output-score matrix.

Running the actual learner CLI with only `train_images`, `train_labels`, and
`test_images` in the input archive produces identical outputs. Changing every
training label to `(label + 1) % 10` changes learned parameters and 589 predictions.
The separate evaluator confirms the exact per-draw threshold, independent of rounded
display percentages. Test labels are never learner inputs.

| Separate host work | Accuracy checked | Time (ms) | Scope |
| --- | ---: | ---: | --- |
| CPU learner | 62% (374/600) | 660 | Fresh normalization, initialization, training, prediction |
| Ordered 300-epoch replay | 62% (374/600) | 3400 | Independent reduction implementation |
| Complete CPU verification suite | 62% (374/600) | 5700 | Replay, hashes, allowed-input CLI, mutation, static rescore, accuracy |

The full 618-million-instruction program was not numerically interpreted after
expansion. The earlier IL tests execute small expanded MLPs and check lowering
semantics and address counts; the full ordered replay checks this learner's
numerical result. The distinction is recorded in [verification.json](verification.json).

## A100 implementation and measurement

The measured device was **NVIDIA A100-SXM4-40GB**, with
108 streaming multiprocessors, compute capability
8.0, and board power limit 400 W.
It ran through Modal using a pinned container. Software: Python
3.11.15, PyTorch 2.5.1+cu124, Triton 3.1.0,
NumPy 2.2.6, CUDA runtime 12.4, NVIDIA driver
580.95.05, NVML 13.580.95.05, and
`nvidia-ml-py` 12.560.30.

Container: `ghcr.io/ab-10/wikitext-bench@sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f`.

Eight Triton kernels implement input normalization, parameter/target
initialization, hidden forward, output/error, hidden error, parameter updates,
and the two inference stages. Each minibatch runs four kernels; every graph
replay includes all 24,000 minibatch kernel launches plus preparation and
inference. The graph is replayed for a complete fresh learning task each time.
GPU memory is reused, but learned parameters are reset and recomputed.

The manual GPU implementation is checked against the ordered CPU calculation
for all 650 parameter words, all 6,000 output-score words, and all 600 predictions
before and after timing. Changed queries and changed training labels also pass
independent reference checks. The exported PTX contains no FP32 fused multiply-add
or flush-to-zero instruction. This establishes agreement for the tested data;
it is not a proof for every possible input.

Each trial targets ten seconds of active graph execution. Idle power is measured
for three seconds before and after the active window, with a three-second
settling gap before each idle sample. CUDA events and host wall time both cover
the repeated graph replays. A consistency assertion catches unit mismatches. Calibration selected 122
complete tasks per trial; actual active intervals were about 8,700 ms, shorter
than the calibration target after execution warmed up.

NVML cumulative board energy is sampled at active-window boundaries. If the
paired idle powers are `P_before` and `P_after`, the per-task adjusted energy is
`1000 * (E_active_J - (P_before + P_after)/2 * active_seconds) / tasks`, in mJ.
The subtraction uses the NVML counter's active interval, not the CUDA duration.
No negative energy is clipped. Means below are unweighted across three trials.

| Trial | Accuracy | Complete tasks | Time (ms/task) | Idle before / after (W) | Raw energy (mJ/task) | Paired-idle energy (mJ/task) | Before-only / after-only energy (mJ/task) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 62% (374/600) | 122 | 71 | 70 / 70 | 6900 | 1900 | 1900 / 2000 |
| 2 | 62% (374/600) | 122 | 71 | 69 / 69 | 6900 | 2000 | 2000 / 2000 |
| 3 | 62% (374/600) | 122 | 71 | 67 / 69 | 7000 | 2100 | 2200 / 2100 |

Mean CUDA time is **71 ms**, with sample standard deviation
**0.094 ms**.
Mean host wall time is **71 ms**.
Mean idle-adjusted energy is **2000 mJ**, with sample standard deviation
**97 mJ**.
Mean raw board energy is **6900 mJ**.
The separate baseline alternatives show sensitivity to the idle-power choice.
The third trial has a 2.3 W difference between its paired idle means; the
one-sided baseline energy envelope across trials is about 1,900–2,200 mJ/task.

Host/device transfers, allocations, JIT compilation, graph capture, validation,
cold start, and host CPU energy are excluded. These are GPU-resident steady-state
complete-task measurements. Repeated execution may benefit from cache residency.
The first run had a timing conversion bug and is excluded from every metric
above; the corrected second run supplies all results. The measurement history
retains that failure and its correction.

## Data and evidence fingerprints

The dataset is `competition-v2`, canonical seed 20260910. The 600 training and
600 test examples are disjoint subsets of the original MNIST training pool,
downsampled to 3 × 3. The manifest file SHA-256 is
`296359c9046a8045aefdb5bc7c471c5abc2c4c5c546d6c56ca53b1299d278054`.
Allowed array SHA-256 values (C-order, little-endian content):

- `train_images`: `1d123f4ff4c0fa6975ee07f5e647971713284515c4e10b88b943a257fdcb7ca5`.
- `train_labels`: `948baa48cdcf75b8b0187a9a5f2a1dcec108998dbfd0081cccc396bfb14321aa`.
- `test_images`: `9c0793497fb028e7f6ecea5819377502d4e25990aea59578db9ba6f52ba69a09`.

Prediction content SHA-256: `007b9462848073e7342050a67263c7efef0bdcf6c815eff1e352ba475c16b828`.
Output-score content SHA-256: `8520c9d95bed5047fff72217e518ead34e06fe3a16c6425c615885f68e0ea09a`.
Canonical IL hash: `9ba3446348eb559666e89a9466fa3475636cf1ad465fbb03a471d0b4fad20b2f`.
GPU source SHA-256: `9e85ccd67c25ddac738a7d65e67c04212e7564e4718a5d3a26d7875eccc20eb4`.

Scoring source fingerprints:

- `score.py`: `2a35978282dbfc30ce36b0202f1407971c83108ec37cc4cd049f4d265488429b`.
- `mlp_il.py`: `df55755126f06df297cdee092cbf3aa94fe4bd0d18b864ffd2e99515aedf54a1`.
- `il.py`: `24bd0b890c433029199a6b63d08e1a6730f7789856006351b8097885c17fd7bc`.

Additional source, parameter, and PTX fingerprints are recorded in the raw
evidence. The program-file byte hash differs from the canonical IL hash because
the canonical hash normalizes JSON formatting.

## Reproduction


The submission branch is `codex/mnist-small-60-submission`. After it is merged,
the same commands work on `main`. Run from the repository root:

```sh
git clone --branch codex/mnist-small-60-submission https://github.com/cybertronai/sutro-problems.git
cd sutro-problems
python3.11 -m venv .venv-mlp60
.venv-mlp60/bin/python -m pip install -r mnist/submissions/mlp60-affine-20260911/requirements.txt
.venv-mlp60/bin/python -m mnist.code.data --output mnist/data --seed 20260910
.venv-mlp60/bin/python mnist/submissions/mlp60-affine-20260911/learner.py --output /tmp/mlp60-reproduction
.venv-mlp60/bin/python -m mnist.code.evaluate --tier small --predictions /tmp/mlp60-reproduction/predictions.npy --data-dir mnist/data --output /tmp/mlp60-reproduction/accuracy.json
.venv-mlp60/bin/python mnist/submissions/mlp60-affine-20260911/score.py --output /tmp/mlp60-reproduction
.venv-mlp60/bin/python mnist/submissions/mlp60-affine-20260911/verify.py --artifacts /tmp/mlp60-reproduction --output /tmp/mlp60-reproduction/verification.json
```

The learner and generator import the checked-in feasibility study implementation
under `mnist/experiments/accuracy-il-20260911/`. They do not import saved weights
or predictions. Input array content hashes are checked against the canonical
manifest. The evaluator alone opens test labels after predictions are fixed.
Verification also runs the learner with an archive containing only the three
allowed input members.

With configured Modal credentials and `uv` installed, reproduce the complete
task on an A100-40GB:

```sh
uvx --with numpy==2.2.6 modal==1.5.5 run mnist/submissions/mlp60-affine-20260911/gpu_benchmark.py --data mnist/data/small.npz --output /tmp/mlp60-gpu
.venv-mlp60/bin/python -m mnist.code.evaluate --tier small --predictions /tmp/mlp60-gpu/gpu_predictions.npy --data-dir mnist/data --output /tmp/mlp60-gpu/accuracy.json
```

The GPU command runs three trials after numerical verification. Each graph
replay resets parameters, prepares inputs and targets, trains all 300 epochs,
and predicts all 600 test labels. Host/device transfers, JIT, graph capture,
allocation, and validation are excluded from steady-state GPU measurements.
The remote container pins NumPy 2.2.6; the local CPU and scoring environment
used NumPy 2.4.6. Parameter, score, and prediction bits match across them.

To regenerate the report from the saved evidence:

```sh
.venv-mlp60/bin/python -m pip install -r mnist/submissions/mlp60-affine-20260911/requirements-report.txt
.venv-mlp60/bin/python mnist/submissions/mlp60-affine-20260911/write_report.py
.venv-mlp60/bin/python mnist/submissions/mlp60-affine-20260911/build_pages.py
```

The report build uses checked-in measurements, never reruns GPU trials. The
readable session is a timestamped snapshot of visible user and assistant
messages; hidden reasoning, system instructions, and tool payloads are omitted.

- [Standalone report](https://cybertronai.github.io/sutro-problems/docs/submissions/mlp60-affine-20260911/)
- [Separate ambiguities and problems](https://cybertronai.github.io/sutro-problems/docs/submissions/mlp60-affine-20260911/ambiguities.html)
- [Human-readable session](https://cybertronai.github.io/sutro-problems/docs/submissions/mlp60-affine-20260911/session.html)


## Download the evidence

- [CPU learner](learner.py), [CPU results](cpu_results.json), [predictions](predictions.npy), [scores](output-scores.npy), [parameters](parameters.npz), [accuracy](accuracy.json).
- [Compact program](program.il.json), [score driver](score.py), [exact model score](model-score.json).
- [Independent verifier](verify.py), [verification evidence](verification.json).
- [GPU benchmark](gpu_benchmark.py), [GPU results and trial telemetry](gpu_results.json), [GPU verification](gpu_validation.json), [GPU predictions](gpu_predictions.npy), [measurement history](gpu_run_history.txt).
- [Separate ambiguities and problems](ambiguities.html), [human-readable session export](session.html).
