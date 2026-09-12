# MNIST-medium: 512-unit MLP, 96% target

This submission trains a fresh 81 → 512 → 10 ReLU network for each of 11
independent datasets, using 10,000 training and 10,000 test images at 9 × 9
pixels. Its requested target is at least **96% mean accuracy**, requiring
105,600 correct predictions among 110,000. It belongs in the README's existing
5% error band; the requested 96% target is checked
separately against exact counts.

**96.41% ± 0.13 percentage points** (mean ± sample standard deviation),
with **106,047 / 110,000 correct**. The unrounded mean is
96.40636363636364%, so this meets the requested 96% target and the 5% error
band. All eleven draws are included.

Mean A100 costs are **2.9 × 10⁵ mJ** and **9.3 × 10³ ms** per complete
training-and-prediction task. The conservative current-grid schedule costs
**4.4 × 10² mJ** and **3.7 × 10⁶ ms**. These are distinct hardware measurements
and theoretical model scores, respectively. Exact A100 means are
286,537.79874552164 mJ and 9,294.189630681818 ms.

| Draw | Dataset seed | Learner seed | Correct / total | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 00 | 2026091200 | 101 | 9,652 / 10,000 | 96.52% |
| 01 | 2026091201 | 102 | 9,642 / 10,000 | 96.42% |
| 02 | 2026091202 | 103 | 9,613 / 10,000 | 96.13% |
| 03 | 2026091203 | 104 | 9,626 / 10,000 | 96.26% |
| 04 | 2026091204 | 105 | 9,643 / 10,000 | 96.43% |
| 05 | 2026091205 | 106 | 9,650 / 10,000 | 96.50% |
| 06 | 2026091206 | 107 | 9,630 / 10,000 | 96.30% |
| 07 | 2026091207 | 108 | 9,646 / 10,000 | 96.46% |
| 08 | 2026091208 | 109 | 9,659 / 10,000 | 96.59% |
| 09 | 2026091209 | 110 | 9,642 / 10,000 | 96.42% |
| 10 | 2026091210 | 111 | 9,644 / 10,000 | 96.44% |

Contributors: Yaroslav Bulatov (benchmark requirements), Codex (implementation,
measurements, and verification). No W&B run was created.

## Learning and data protocol

Each dataset uses a new NumPy PCG64 permutation of the original 60,000 MNIST
training rows. The first 10,000 rows train the learner and the next 10,000 are
test inputs. They are disjoint within each draw; independent draws may overlap.
Dataset seeds are 2026091200 through 2026091210; corresponding learner seeds
are 101 through 111. The original official test split is not used.

Pixels are converted to float32, divided by 255, resized from 28 × 28 to 9 × 9
with the repository's exact separable box-area averaging, and clipped to
[0, 1]. The learner transforms them to `4*x - 0.5`. All array, source-file,
and dataset-index checksums are retained. Index arrays are reproducible from
the recorded seeds; raw datasets and checkpoints are not committed.

The fixed network has 47,114 FP32 parameter words. It uses 200 epochs,
contiguous cyclic minibatches of 25, and squared-error SGD at learning rate
0.1. Initial weights are PCG64 uniform samples within ±1/√81 and ±1/√512,
with zero biases. Every reduction uses ascending FP32 multiplication followed
by a separate addition; no fused multiply-add or flush-to-zero is allowed.
Gradients for all four parameter arrays use the same pre-update weights.
Argmax ties select the lowest digit. Every invocation rebuilds one-hot targets,
resets parameters from seed-only constants, performs all 80,000 training
minibatches, and predicts every test image. No learned state crosses draws.

The initial 512-unit, 200-epoch procedure comes from the historical affine MLP;
the minibatch changes from 30 to 25 to divide 10,000 exactly. A separate
8,000/2,000 diagnostic using dataset seed 2026091299 and learner seed 991
returned 1,927/2,000 correct. Its original written plan included a 400-epoch
fallback, which was never run. Before any final-draw evaluation, that fallback
was disabled and the initial 200-epoch procedure retained. The diagnostic may
overlap future test rows and is **not** qualification evidence or a learner
input. The original plan and explicit protocol amendment are preserved; no
width, learning rate, checkpoint, or seed was changed following the diagnostic.
All 11 final prediction arrays were frozen before final test labels were
compared with them.

## A100 measurements

Measurements used NVIDIA A100-SXM4-40GB GPUs, PyTorch 2.5.1+cu124, Triton 3.1.0,
CUDA 12.4, and NVIDIA driver 580.95.05. The Triton implementation adapts the
historical ordered-FP32 kernels to the current dataset sizes. Three CUDA graphs store initialization, one epoch, and
inference. A complete invocation replays initialization once, the epoch graph
200 times, and inference once. Each measurement repeats complete fresh
invocations; cached trained weights or saved predictions are never inputs.

Each draw has one NVML trial, repeating full invocations for at least 10
seconds based on calibration. The trial is bracketed by two three-second idle
samples, each after a three-second settling gap. Idle-adjusted joules equal
active cumulative-counter joules minus the average of the two idle powers
multiplied by the active interval. CUDA events measure task time. Summary
costs are arithmetic means across all 11 draws; raw counter readings, idle
powers, durations, telemetry, GPU identities, and software versions remain in
each result file.

Measurements include normalization, target construction, initial parameter
reset, every training update, all inference scores and labels, and graph
replay overhead. They exclude dataset preparation/resizing, host/device
transfers, allocations, JIT compilation, graph capture, verification, and
cold start. Energy is for the GPU board, excluding host CPU energy. These are
GPU-resident task costs, not application startup costs. The GPU stores all
queries and a 10,000 × 512 inference buffer, while the grid schedule streams
queries; both execute the same numerical learner. Neither implementation is
claimed optimal.

## Grid model

The shared scorer targets `spatial-computer` revision
`01a0bd5e0d2564825b0f53dd766f763c82dbc7c0`, using the pitch-128 grid and v4
instruction set. It counts an explicitly **globally serialized** legal schedule:
one MLP compute processor at `(125, 0)`, one staging word at each of the 250
bottom tape ports, and an injective legal placement across 314 memory tiles.
At most one instruction/access progresses at a time, avoiding contention.
There are 250 instruction-issuing processors over the complete schedule;
maximum concurrency is one. The most populated tile uses 12,288 scratch words.

Input tape order is all training pixels, all training labels, then all test
pixels. Word `k` enters bottom port `k % 250`, is staged and copied to its
declared scratch address. After training, one query is received, normalized,
classified, and emitted at a time. Output word `q` uses port `q % 250`.
Scratch initialization, seed-only parameter literals, local scratch accesses,
interprocessor traffic, and tape I/O are charged. The fixed schedule has
437,180,228,679,904 word-node hops, yielding **4.4 × 10² mJ** and
**3.7 × 10⁶ ms** (about 62 minutes) for the complete task. These theoretical
figures describe this conservative schedule, with the same costs for every
learner seed; they are not A100 measurements or an optimized parallel runtime.

Peak allocated scratch is **3,973,260 bytes** (993,315 words, including 250
staging words). The scorer took **5.1 seconds** on the host, including schema,
address and initialization validation, placement, affine histograms, integer
cost sums and hashing; it excludes program generation, file I/O, and numerical
execution. The generator/scorer retains the complete affine program and exact
component counts. Five small numerical/event-expansion tests verify its
arithmetic, schedule, and hop/time sums. It does not numerically expand the
413,755,929,672 instructions of the complete program.

## Verification

On every draw, an independent ordered CPU implementation checks two full
training minibatches with changed labels, all 100,000 final inference scores,
and all 10,000 predictions. Random-query mutation checks establish that outputs
depend on the supplied query images. Full task repetitions preserve every
parameter and score bit. Compiled PTX is checked for prohibited FP32 FMA and
flush-to-zero instructions; its hashes are recorded.

An additional independent CPU replay of draw 00 executes all 200 epochs and
matches **all 47,114 final parameters, all 100,000 scores, and all 10,000
predictions bitwise** with the A100 result. This took 86.7 seconds. Full
independent CPU training was checked for one draw; the bounded training and
complete inference checks above apply to all eleven.

`verify.py` independently regenerates every dataset from the canonical IDX
sources, checks all array and index hashes, recomputes all 110,000 prediction
comparisons and the sample standard deviation, and rederives all NVML energy
adjustments from the raw counters.

## Reproduce

Run these commands from the repository root. A100 execution requires configured
Modal credentials. The source image is pinned by digest and installs
NumPy 2.2.6 and `nvidia-ml-py` 12.560.30; hardware/software details are captured
per draw. Prediction runs refuse to overwrite an existing final freeze.

```sh
S=mnist/submissions/medium96-grid-20260912
uv run --with numpy==2.2.6 python "$S/verify.py"
uv run --with numpy==2.2.6 python "$S/cpu_reference.py" --draw 0 --output /tmp/medium96-cpu-check.json
mkdir -p /tmp/medium96-reproduction
cp "$S/config.json" "$S/protocol.json" /tmp/medium96-reproduction/
uvx --with numpy==2.2.6 modal==1.5.5 run "$S/run.py" --phase fit --output /tmp/medium96-reproduction
uvx --with numpy==2.2.6 modal==1.5.5 run "$S/run.py" --phase evaluate --output /tmp/medium96-reproduction
```

`cpu_reference.py` needs the committed result files for its bitwise comparisons;
it learns from the three permitted input arrays and does not read test labels.
The optional `--phase validate --output /tmp/medium96-diagnostic` reproduces the
fixed 200-epoch diagnostic. It does not select a different procedure.

Reproduction source and evidence:

- [Frozen protocol](protocol.json), [configuration](config.json), and [original diagnostic plan](validation_plan.json)
- [Learner and A100 measurement](learner.py), [draw preparation/evaluation](run.py), and [CPU reference](cpu_reference.py)
- [Shared grid source and schedule](../grid-mlp-scoring-20260912/), [exact grid score](../grid-mlp-scoring-20260912/medium96/grid-score.json), and [full CPU comparison](cpu-verification-draw00.json)
- [Accuracy and costs](accuracy.json), [frozen prediction manifest](prediction_manifest.json), and [verification](verification.json)
