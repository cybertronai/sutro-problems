# MNIST-medium: a fully scored 4% error submission

**The frozen three-ConvNet learner achieved 2.2% ± 0.1 pp mean error
(97.8% ± 0.1 pp accuracy)** across eleven independent draws, each with
10,000 training and 10,000 held-out query examples at 9×9 resolution. It clears
the requested **4% error** threshold. This package includes the complete v4
program generator, an exact aggregate scorer, actual A100 training and prediction
measurements, reproducible evidence, and a visible session export.

Contributors: Yaroslav Bulatov (task and evaluation requirements), Codex (implementation,
experiments, scoring and report). Prepared 11 September 2026. Acceptance remains
subject to benchmark review; the separate ambiguity report describes the declared
numerical, instruction-supply, development-data and measurement conventions.

## Results on a common scale

| Accuracy | Mean error | Model time (ms) | Model energy (mJ) | Area (mm²) | Time to score (s) | A100 time (ms) | A100 energy (mJ) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 97.8% ± 0.1 pp | 2.2% ± 0.1 pp | 4,900,000 | 12,000 | 8.9 | 11 | 13,000 | 780,000 |

Costs cover **one complete task**: fresh initialization and training of all three
members on 10,000 examples, prediction of all 10,000 queries, and ensemble output.
They are not per-image inference costs and are not multiplied by eleven. Accuracy
uses all eleven draws; A100 measurements use the predeclared first draw. Model
costs are input-independent for these fixed dimensions, epochs and seeds, so the
same exact score applies to every draw. A100 values are medians of three trials.
The energy column subtracts paired idle-board energy. All displayed costs use two
significant figures; accuracy and SD use one decimal. JSON retains full precision.

The A100 is about 370 times faster and uses about 65
times the model's energy under these boundaries. This comparison puts both sets
of numbers in the same units; it does not equate a modeled single core with an
entire physical GPU board.

## Accuracy and the five targets

There were **107,534 correct predictions out of 110,000**.
The inclusive 4% threshold requires **105,600 correct**, giving
a margin of **1,934**. The unrounded aggregate is used for the
decision. The five requested error levels are 10%, 8%, 6%, 4%, and 2%, corresponding
to 90%, 92%, 94%, 96%, and 98% accuracy. This is the 4% attempt; it also exceeds
the three looser levels, but misses the 2% level by 266 correct predictions.
No further target-specific tuning was performed after these results were opened.

| Draw | Dataset seed | Correct / total | Accuracy | Error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 20261101 | 9,781 / 10,000 | 97.8% | 2.2% |
| 2 | 20261102 | 9,780 / 10,000 | 97.8% | 2.2% |
| 3 | 20261103 | 9,776 / 10,000 | 97.8% | 2.2% |
| 4 | 20261104 | 9,783 / 10,000 | 97.8% | 2.2% |
| 5 | 20261105 | 9,762 / 10,000 | 97.6% | 2.4% |
| 6 | 20261106 | 9,789 / 10,000 | 97.9% | 2.1% |
| 7 | 20261107 | 9,761 / 10,000 | 97.6% | 2.4% |
| 8 | 20261108 | 9,786 / 10,000 | 97.9% | 2.1% |
| 9 | 20261109 | 9,772 / 10,000 | 97.7% | 2.3% |
| 10 | 20261110 | 9,766 / 10,000 | 97.7% | 2.3% |
| 11 | 20261111 | 9,778 / 10,000 | 97.8% | 2.2% |

The original 60,000 MNIST training rows form the source pool. Each dataset seed
creates a PCG64 permutation through `SeedSequence(seed).spawn(2)[0]`; positions
0–9,999 are training and 10,000–19,999 are queries. Sampling is without replacement
inside each draw. Independent draws may overlap. The official MNIST test split
is not used. Float32 pixels are divided by 255, then resized by separable box-area
averaging with fractional boundary overlap, and clipped to [0,1].

Learner seeds are fixed at **11, 22, 33** on every dataset. Weights, velocities and
all learned state are reset for each member and draw. The learner receives only
training images, training labels and query images. All eleven prediction archives
were saved and globally hashed before query-label arrays were extracted. The
evaluator rechecks the frozen sources, inputs, member logits and ordered ensemble
before scoring. SD is the sample SD of the eleven dataset accuracies (`ddof=1`),
in percentage points. It is also the SD of error, not a standard error or a
confidence interval.

## Algorithm and selection

Each member has three padded 3×3 convolutions with 32 output channels, each
followed by ReLU. There is no pooling or convolution bias. Flattened 32×9×9
activations feed a biased 128-unit ReLU head and a biased 10-class output layer.
Each member has **351,914 parameters**. Batch normalization
and dropout are disabled. Pixel normalization is `4*x - 0.5`.

Each member trains for **8 epochs**, with batch size 128 and a final batch of 16
(79 updates per epoch). SGD uses learning rate 0.03, momentum 0.9, and coupled
weight decay 0.0001. Weights use the published seed-only CPU PyTorch initialization
recipe (constructor draws followed by He-uniform reinitialization), biases and
initial velocities are zero. No learned weights are supplied to the learner.

Training uses fixed-seed mild affine augmentation independently with probability
0.5 per image: rotation
within ±8°, scale 0.94–1.06, and translation within ±0.35 pixel. Four-neighbor
bilinear interpolation has a fixed operation order and zero outside the image.
Permutation and affine geometry are generated from separate PCG64 streams; all
epoch hashes are retained. Compilation embeds only seed-dependent indices and
coefficients; pixel reads and interpolation remain charged runtime operations.

The output gradient uses a declared approximate softmax: subtract the largest
logit, clamp to [−16,0], form `1 + x/1024`, square ten times, sum the ten classes
in ascending order, divide, subtract the one-hot label, and multiply by the
pre-rounded FP32 reciprocal of batch size.
All these operations are present in the v4 program. Ensemble logits are added in
seed order starting from positive zero; ascending strict argmax chooses the
smallest class in a tie.

Development compared ReLU ConvNets with different widths/depths and losses, then
BN and dropout variants, on a prior 4,800/1,200 training/validation split. The final
ordered implementation reached **97.2% validation accuracy (1,166/1,200)** at epoch
8, the earliest common epoch with at least 97% validation accuracy. After the
request changed to 4% error, that shorter schedule was frozen before the new
formal draws. No formal query results selected the architecture, epoch or seeds.
Development rows can overlap later random queries; this open-data limitation is
disclosed separately. Fresh formal weights were trained on each full 10,000-row
training subset. Native-library search scores are development evidence, not the
accuracy claimed for this submission.

## Compact language and exact model scoring

The representation is **`sutro-literal-affine-v4/0.2`**: finite counted loops,
affine scratch addresses, and literal tables describing a static scalar v4
program. Its leaves are only `set`, `recv`, `send`, `copy`, `add`, `sub`, `mul`,
`div`, `cmp`, and `select`. Loops and tables compress the description; they are
not new machine operations with invented costs. `ir_core.expand()` can stream
every scalar instruction. The full program has **2,791,385,568,603 instructions**
but only **563 static leaf nodes**. Materializing the full stream
is unnecessary for scoring.

The scorer forms exact per-address read/write multiplicities, using finite
affine convolutions and seed-map histograms. It checks address bounds, definite
initialization, literal/source identity and integer overflow limits. Duplicate
operands count separately; `select` reads all three operands. It then sums the
model's geometry costs with integer arithmetic: **5,561,894,756,272 charged reads**
and **2,791,383,928,603 charged writes**. Three independent scoring calls agree
exactly; their median wall time is reported. Timing includes validation, schedule
generation, histograms, costs and canonical hashing, and excludes initial file
loading, IR construction, numerical training and output serialization.

The model is pinned to simplified-dally-model commit
`26abcca402de647381d31286d42dfbb7a001763d`, single core with tape and v4 instructions.
One 32-bit scratch word occupies one 1 µm² cell. The processor is at (0,0).
Cells are filled by increasing Manhattan distance, with ascending x within each
half-diamond shell. At distance `h`, each read/write costs `max(50,2h)` fJ;
read time is `max(50,0.8h)` ps and write time `max(50,0.4h)` ps. Accesses block
serially; arithmetic and instruction supply are uncharged. `recv` and `send`
are free under this model, including their scratch effects. Time accumulates in
integer 0.2 ps ticks; energy in integer fJ. Display conversion divides by 10⁹
ps/ms, 10¹² fJ/mJ, and 10⁶ µm²/mm².

The tape receives all training pixels, training labels, then query pixels once,
and sends the final predicted labels. Raw queries are retained once. One maximum
batch workspace and one parameter/gradient/velocity allocation are reused across
epochs, query batches and sequential members. Every occupied cell is explicitly
initialized and charged. Occupied scratch area is **8,853,627 cells**;
it excludes processor, tape, program storage and interconnect area.

## Translation to A100 and numerical evidence

The algorithm lowers to Triton kernels and retained PTX. All numerical primitives
use ordered FP32 round-to-nearest-even arithmetic. Multiplication and addition
remain separate; fusion is disabled. Division uses explicit `div.rn.f32`.
Convolution, dense and gradient reductions preserve their declared scalar order.
Parallelism is across independent outputs. The GPU may keep temporaries in
registers and use a different physical memory layout from the modeled machine.

Verification includes explicit small-program instruction execution against an
independent CPU reference, exact per-address histograms and geometry prices,
convolution finite-difference/autograd geometry checks, partial minibatches,
multiple epochs, ensembles, and GPU comparisons of every parameter, velocity,
score and prediction. The final no-BN A100 test checks **50 arrays bit for bit**
across two complete tiny-network cases, including state reset after deliberate
mutation. These tests use the frozen learner source. The production A100 benchmark
reproduces all three formal draw-1 learned-state and logit hashes, and every timed
invocation reproduces the complete state fingerprint. Full MNIST execution is on
A100; the trillions of scalar v4 instructions are scored statically, not executed
individually for all 110,000 queries.

The original benchmark retained Trainer kernels before the final whole-query
ensemble argmax was compiled. That last shape variant is therefore separately
regenerated from the same source and pinned compiler, with PTX and tie-case
verification retained as supplementary evidence. Original-run PTX checks apply
to every captured kernel; the supplementary file is explicitly a later
regeneration. No FMA, FTZ or tensor-core instructions are used in the declared
numerical lowering.

## A100 measurements

Hardware: **NVIDIA A100-SXM4-40GB**, 108 SMs,
compute capability 8.0, 400 W power limit,
MIG disabled. CUDA and NVML identify the same UUID; the raw record retains it.
Software: Python 3.11.15, PyTorch 2.5.1+cu124,
Triton 3.1.0, CUDA 12.4, driver
580.95.05, `nvidia-ml-py` 12.560.30.
The container image is pinned by digest in `modal_run.py`.

| Trial | Complete tasks | CUDA time/task (ms) | Wall time/task (ms) | Gross energy/task (mJ) | Idle-adjusted energy/task (mJ) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 13,000 | 13,000 | 1,700,000 | 790,000 |
| 2 | 1 | 13,000 | 13,000 | 1,700,000 | 780,000 |
| 3 | 1 | 13,000 | 13,000 | 1,700,000 | 770,000 |

The measured boundary starts with GPU-resident raw inputs and compiled graphs.
Every invocation resets all learned state, performs all 24 member-epochs,
normalizes/predicts every query, adds logits and emits final predictions.
Host-to-device input transfers, allocation, seed-only program compilation, JIT,
graph capture, fingerprint checking and cold startup are outside the timed region.
This is complete learning plus prediction under a GPU-resident boundary.

Each trial has a 3,000 ms settle and 3,000 ms idle-counter interval before and
after activity. NVML cumulative energy counters are sampled with timestamps;
the average of the two idle powers is subtracted over the same active-counter
duration. Counter values are already mJ. CUDA events and synchronized host wall
times agree. Using only the before or after baseline instead would give
770,000–800,000 mJ per task across these trials.
Gross energy is retained so this baseline sensitivity is visible. NVML measures
the whole GPU board and excludes the host CPU; caches and baseline drift remain
measurement limitations.

## Reproduction

Use Python 3.11 with NumPy, `uv`, and a configured Modal account with A100 access.
From a checkout containing this submission, install the local dependency and
download the two canonical training IDX files without building query labels:

```bash
git clone --branch codex/mnist-medium-four-percent-scored \
  https://github.com/cybertronai/sutro-problems.git sutro-medium-reproduction
cd sutro-medium-reproduction
python3 -m venv /tmp/mnist-reproduce-env
/tmp/mnist-reproduce-env/bin/pip install numpy==2.4.6
/tmp/mnist-reproduce-env/bin/python - <<'PY'
from pathlib import Path
from mnist.code.data import SOURCES, download_source
for key in ('train_images', 'train_labels'):
    download_source(Path('/tmp/mnist-raw'), *SOURCES[key])
PY
/tmp/mnist-reproduce-env/bin/python \
  mnist/submissions/medium-convnet-v4-20260911/reproduce.py \
  --raw /tmp/mnist-raw --output /tmp/mnist-medium-reproduction --run
```

The reproduction helper makes a fresh source-only tree, prepares eleven input
archives, runs the frozen learner, freezes every prediction, evaluates accuracy,
regenerates seed-only constants on the pinned CPU image, runs three exact scores,
and measures the first draw on A100. It refuses to overwrite existing evidence.
Omit `--run` to prepare the isolated inputs and print the remaining commands.
Dataset archives are regenerated rather than published; source indices and raw
file checksums are retained in every draw manifest. Exact GPU predictions should
reproduce under the pinned numerical environment; runtime and energy will vary.

To recompute only the theoretical costs from the published constants:

```bash
cd mnist/submissions/medium-convnet-v4-20260911
python ir_score.py --config config.json --constants constants/constants.json \
  --output /tmp/mnist-medium-score --repeats 3
python audit.py --skip-input-files --output /tmp/mnist-medium-audit.json
python ir_test_core.py
python ir_test_conv.py
python ir_test_bn.py
python ir_test_model.py
uvx modal==1.5.5 run ordered_backend/check_submission.py \
  --output /tmp/mnist-medium-gpu-validation
```

The retained scoring host used an Intel Core i9-9880H at 2.30 GHz, 64 GiB RAM,
macOS 26.6.2, Python 3.11.13, and NumPy 2.4.6.
The audit checks evidence and arithmetic without opening raw query labels.
The command above checks published evidence without local input archives;
omit `--skip-input-files` after regenerating them to check every input byte too.
Full testing takes longer than the reported scoring interval.

## Evidence and references

- [Accuracy and all eleven counts](accuracy.json), [frozen protocol](protocol.json), [global prediction manifest](prediction_manifest.json).
- [Exact model score](model-score.json), [compact program](program.il.json), [scorer](ir_core.py), [compiler](ir_model.py), [seed-only constants](constants/constants.json).
- [Raw A100 trials and counters](benchmark/results.json), [executed learner provenance](benchmark/results/draw-00.json), [configuration](config.json), [selection record](selection.json).
- [Independent evidence audit](audit.json), [complete numerical compiler checks](ir-model-validation.json), [current-source A100 checks](ordered_backend/submission-validation/results.json), [supplementary argmax checks](supplemental-argmax/results.json).
- [Reproduction helper](reproduce.py), [A100 runner](modal_run.py), [measurement code](measure_gpu.py), [ordered backend](ordered_backend/README.md).
- [Separate ambiguities and problems](ambiguities.html), [human-readable session](session.html), [machine-readable visible session](session.json).
- [Pinned single-core model](https://github.com/cybertronai/simplified-dally-model/tree/26abcca402de647381d31286d42dfbb7a001763d/models/single-core-with-tape), [pinned v4 instructions](https://github.com/cybertronai/simplified-dally-model/tree/26abcca402de647381d31286d42dfbb7a001763d/instruction-sets/v4).
