# PCA-QDA: MNIST-medium, 5% error target

September 15, 2026. Contributors: [@jurajselep](https://github.com/jurajselep)

PCA-QDA achieves **105,130 / 110,000 correct (95.57% ± 0.17 pp)**,
passing the 5% error target of at least 104,500 correct. Costs cover training
and prediction on one 10,000-training / 10,000-test dataset.

| Accuracy | A100 energy above idle (mJ) | A100 time (ms) | Grid energy (mJ) | Grid time (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 95.57% ± 0.17 pp | 174 | 3.3 | 0.19 | 2.0 × 10³ |

A100 energy is **above idle**. The original 3.8 mJ result did not reproduce. A100 costs are measured; grid costs are theoretical. Exact values and evidence
links are in [submission.json](submission.json).

## Data and learner

Each dataset uses `Generator(PCG64(seed)).permutation(60000)` over the original
MNIST training set: the first 10,000 indices train and the next 10,000 test.
Pixels are divided by 255 in FP32 and area-resized from 28×28 to 9×9.
`run.resize_recorded` fixes the accumulation order to reproduce all 22 archived
input hashes across the verified CPU environments. Checksums and complete
indices are in the [draw manifest](evidence/accuracy/draw_manifest.json).

The fixed learner is retrained independently on each dataset:

1. Apply six Newton square-root updates, starting at `x + 1e-6`.
2. Learn a 40-dimensional basis from the first 2,000 training images, using
   three covariance subspace iterations and sequential Gram-Schmidt with
   maximum-absolute-entry column scaling. The initial basis uses `PCG64(0)`.
3. Project all images and fit a Gaussian per class, using biased covariance,
   0.05 trace shrinkage, and Gauss-Jordan inversion without pivoting.
4. Predict the maximum quadratic discriminant score, including class priors
   and log-determinants; ties choose the first class.

[reference.py](reference.py) defines the ordered FP32 arithmetic and logarithm
approximation. [protocol.json](protocol.json) records all hyperparameters.
Test labels are used only for evaluation; no learned state passes between draws.

## Accuracy and verification

| Draw | Dataset seed | Correct / 10,000 |
| ---: | ---: | ---: |
| 0 | 20261501 | 9,544 |
| 1 | 20261502 | 9,572 |
| 2 | 20261503 | 9,572 |
| 3 | 20261504 | 9,544 |
| 4 | 20261505 | 9,538 |
| 5 | 20261506 | 9,555 |
| 6 | 20261507 | 9,540 |
| 7 | 20261508 | 9,540 |
| 8 | 20261509 | 9,577 |
| 9 | 20261510 | 9,571 |
| 10 | 20261511 | 9,577 |

The exact mean is `105130 / 110000`; sample standard deviation is
`0.16511153255244929` percentage points (`ddof=1`). All eleven draws are included.

- [CPU verification](results/cpu_verification.json): raw MNIST checksums,
  indices, input hashes, all parameter/score hashes, and frozen predictions match.
- [Grid verification](results/grid_verification.json): the reduced C executor
  matches every Python-executor memory region bitwise; all eleven full C runs
  reproduce the five learned parameter arrays and all 110,000 labels.
- [A100 rerun verification](energy-audit/results-sxm40/original.json): all 110,000 labels match;
  graph replay, poisoned-intermediate, and training-label rotation checks pass.

**Provenance limitation:** the original protocol creation timestamp (13:10 UTC)
follows the evaluation freeze (11:34:43 UTC). Historical pre-evaluation selection
is therefore unconfirmed. Original evidence is preserved; these reruns verify
published draws, not a new held-out evaluation.

## Grid costs

The [program](grid/program.spatial.json.gz) uses the unchanged
[shared scorer](../grid-mlp-scoring-20260912/README.md) for
[spatial-computer pitch 128 / ISA v4, revision 01a0bd5](https://github.com/cybertronai/simplified-dally-model/blob/01a0bd5e0d2564825b0f53dd766f763c82dbc7c0/models/spatial-computer/README.md).

| Quantity | Value |
| --- | ---: |
| Energy / word-node hops | 186,854,673,900 fJ / hops |
| Runtime | 2,047,716,289 cycles (1 ns/cycle) |
| Peak scratch, including tape staging | 5,125,276 bytes |
| Memory tiles / words per tile limit | 334 / 12,288 |
| Instruction-issuing processors / simultaneous instructions | 250 / 1 |
| Host scoring time | 2.8 s |

Arithmetic executes on `P(125,0)`, with scratch assigned nearest-first to legal
cells. The serialized schedule completes each instruction and access before the
next. Input tapes contain 810,000 training pixel words, 10,000 integer labels,
and 810,000 test pixel words; output is 10,000 integer labels. Tapes use 250
bottom ports, with 6,520 input and 40 output words per port. Floats use FP32
words; labels use raw integer words. Test labels and learned parameters are
absent from the input tape and program literals.

A local access at distance `d` costs `max(50, 2*d)` fJ and one cycle; an access
over `L` mesh links adds `256*L` fJ, taking `2*L+1` cycles for reads or `L+2`
for writes. Tape operations cost 128 fJ and two cycles. Requests route
horizontally then vertically; read responses retrace the route. All legal cell
distances are at least 32, so energy in fJ equals word-node hops.
The [score](grid/grid-score.json) records placement, access counts, and costs.

The score includes initialization, tape/staging traffic, feature transformation,
basis fitting, class fitting, and prediction. Sampling, resizing, and division
by 255 prepare the workload and are excluded. Host scoring time covers validation,
placement, cost calculation, and hashing, excluding generation, I/O, and numerical
execution. Costs describe the specified serialized schedule.

## A100 measurements

The [W&B-tracked rerun](https://wandb.ai/yaroslavvb/sutro-mnist-tiers/runs/rha1uwss)
on NVIDIA A100-SXM4-40GB measured **173.793 mJ above idle** and **3.335 ms** per training-and-prediction task using independent
sampled-power integration. Its three active rounds used 6,000 / 12,000 / 6,000
replays and ten-second idle windows before and after each block. All 110,000
predictions match the frozen results (105,130 correct).

| Rerun method | Above idle (mJ) |
| --- | ---: |
| Original protocol: counter, five-second paired idle | 165.564 |
| Independent harness: counter, ten-second paired idle | 174.878 |
| Independent harness: sampled-power integral, ten-second paired idle | **173.793** |

The [original archived measurement](results/gpu_results.json) reported 3.815 mJ
above idle. Its arithmetic is internally consistent, but
its approximately 41 W active power did not reproduce: the tracked rerun
measured approximately 117 W. The reason remains unresolved. The new run used
the same GPU model and PyTorch/CUDA versions on a different physical board,
with driver 580.95.05 rather than 580.105.08.

See the [audit](energy-audit/README.md) for raw records, baseline sensitivity,
container PID instrumentation, and runnable W&B/Modal reproduction instructions.

Each replay refits the basis and class models and predicts all 10,000 queries.
Inputs are device-resident; transfers, allocation, graph capture, and area-resize
preprocessing are excluded. The feature transform is included. GPU reduction
order, classical Gram-Schmidt, and `torch.log` differ from the ordered CPU
reference; all labels agree, while intermediate values need not be bitwise equal.

NVML counter deltas and independently integrated sampled power agree closely.
Both share the GPU's sensors; neither is an external wall-plug measurement.
The corrected estimate subtracts mean paired idle power from the active-window
integral. Signed estimates and idle-only controls are retained. Post-work power
tails affect the idle baseline: trimming the first three seconds of idle gives
180.111 mJ in an exploratory sensitivity check. This is not a confidence interval
or the primary table value. Historical raw measurements remain unchanged.

## Reproduce

For the corrected energy result, use the [W&B/Modal rerun instructions](energy-audit/README.md#rerun-on-a-matching-gpu). The commands below reproduce the original learner, grid execution, and standalone GPU protocol.

Use Python 3.11. From the repository root:

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r mnist/submissions/medium-pca-qda-20260915/requirements.txt
.venv/bin/python -m mnist.code.data --output mnist/data
. .venv/bin/activate
cd mnist/submissions/medium-pca-qda-20260915
python verify.py --output generated/cpu_verification.json
python spatial_program.py
python run.py gpu-payloads
```

To regenerate predictions before scoring, use a fresh evidence directory:

```sh
python run.py prepare --evidence-dir generated/reproduction
python run.py freeze --evidence-dir generated/reproduction
python run.py score --evidence-dir generated/reproduction
```

For the reduced Python/C cross-check and all eleven full grid executions, use
GCC on x86-64. Separate FP32 operations and disabled flush-to-zero preserve the
reference arithmetic:

```sh
mkdir -p generated/compiled
python spatial_program.py --emit-c generated/compiled/reduced.c --n-train 300 --n-test 120 --n-basis 100
python spatial_program.py --emit-c generated/compiled/full.c
for variant in reduced full; do
    gcc -std=c11 -O0 -ffp-contract=off -fno-fast-math -frounding-math \
        -fexcess-precision=standard -fno-strict-aliasing -march=x86-64 \
        -msse2 -mfpmath=sse generated/compiled/"$variant".c -lm \
        -o generated/compiled/"$variant"
done
python validate_reduced.py 300 120 100 \
    --compiled-executable generated/compiled/reduced \
    --full-executable generated/compiled/full --payload-dir generated/payloads \
    --output generated/grid_verification.json
```

Copy this submission and `generated/payloads/` to an idle A100. In a separate
Python 3.11 environment, from the submission directory:

```sh
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-gpu.txt
python gpu_benchmark.py generated/payloads generated/gpu_benchmark.json 3000 5
```

This verifies all eleven draws, then measures draw 0. Add `--verify-only` to skip
timing. Generated data and executables are ignored by Git.
