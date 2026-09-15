# Panel-Cached H32 MLP: Revised MNIST-Small

Local rerun of the historical panel policy, not an architecture search.
The learner was frozen before evaluation. No historical submission was changed.
Contributor: [SecurityQQ](https://github.com/SecurityQQ), with AI-assisted implementation and verification.
Run timestamps are 2026-09-15 UTC; the directory retains its original local-date name.

## Result

**7382/11,000 = 67.10909% +/- 1.63123 pp**
(sample standard deviation across 11 independently drawn datasets).
The exact 7,370 threshold is met by 12 correct predictions.

| Implementation | Accuracy | A100 energy | A100 time | Grid energy | Grid time |
| --- | --- | --- | --- | --- | --- |
| Revised panel, this run | 67.11% +/- 1.63 pp | 3.2e+03 mJ | 1.2e+02 ms | 0.22 mJ | 3.2e+03 ms |
| Archived H32 | 67.08% +/- 1.54 pp | 3.3e3 mJ | 1.3e2 ms | 0.24 mJ | 3.1e3 ms |

Exact panel A100 means: **3162.520774729 mJ / 117.994522923 ms**.
Exact panel grid costs: **0.223152979758 mJ / 3151.348671 ms**.
Against the archived H32 measurements this is 4.57% lower A100 energy,
12.08% lower A100 latency, 8.16% lower grid energy,
and 2.53% higher grid latency.
Separate A100 runs are not a paired hardware comparison; idle-baseline and clock uncertainty remain.
The three extra correct predictions are **not evidence of accuracy superiority**.

## Data and Arithmetic

For each seed below, use `Generator(PCG64(seed)).permutation(60000)`;
the first 1,000 official MNIST training examples are training data, the next 1,000
are disjoint test data. Raw-source checksums and sample indices match the archived H32 draws.
Preprocessing is FP32 division by 255 and the repository's separable 3x3 area resize.
Inputs retained under `private/` contain only training images, training labels and test images.
Only the evaluator uses test labels, after the complete prediction manifest is frozen.

Architecture: 9-32-10 ReLU, squared-error gradients, 300 epochs, learning rate 0.2,
fresh learner seed 101 on every draw, fixed sample order, minibatch 25.
The historical implementation used minibatch 30 and 600/600; those assumptions
were adapted consistently to 25 and 1,000/1,000. Inference groups also use 25.
The historical selected energy panel policy was retained without a new search:
right panels of width 4 for hidden, backward, W1 gradients and hidden inference;
left staging across 10 outputs for output, W2 gradients and output inference.
One 225-word input cache is reused across products. Reduction order is ascending K,
starting from FP32 +0, separately rounded multiply and add, without fusion.

| Dataset seed | Correct/total | Accuracy |
| --- | --- | --- |
| 20261201 | 683/1000 | 68.3% |
| 20261202 | 684/1000 | 68.4% |
| 20261203 | 676/1000 | 67.6% |
| 20261204 | 661/1000 | 66.1% |
| 20261205 | 656/1000 | 65.6% |
| 20261206 | 680/1000 | 68.0% |
| 20261207 | 654/1000 | 65.4% |
| 20261208 | 686/1000 | 68.6% |
| 20261209 | 670/1000 | 67.0% |
| 20261210 | 692/1000 | 69.2% |
| 20261211 | 640/1000 | 64.0% |

### Archived-Baseline Reproducibility Caveat

All 11 resized training/test image hashes differ from the archived H32 manifest,
although raw sources, train/test indices, labels and preprocessing-source hash match.
The resize implementation uses NumPy `matmul`, whose floating-point results can vary
with platform/BLAS. This is not a byte-identical recreation of the archived resized inputs.
The frozen local input archives and hashes identify the exact data actually evaluated and timed.

Separately, on draw 0, the archived reference's fast `einsum` training differs from
explicit ordered FP32 after the first epoch on this machine. Its local fast path
reproduces the archived draw-0 predictions, while the ordered path differs at 13 predictions.
Across all 11 local fast reruns, 107 predictions differ from the archived predictions.
This rules out attributing the accuracy difference solely to panel reuse.
No data or arithmetic policy was changed after seeing these results.
All 11 panel parameter and score bits match an independent full ordered reference rerun.

## Measured A100

Device: `NVIDIA A100-SXM4-40GB`. The sole measured policy is `energy`.
Three trials, 86 complete invocations per trial, roughly 10 seconds active each.
Every replay resets 650 parameters, normalizes 1,000 training and 1,000 test images,
constructs training targets, executes 12,000 minibatches and produces 1,000 predictions.
CUDA graph inspection confirmed exactly **48,004 kernel nodes**.
All 650 parameter words, 10,000 score words and 1,000 predictions agree with CPU bits
in eager execution, graph replay, before/after timing, and changed-query/changed-label checks.
Generated PTX was checked for fused floating-point multiply-add instructions; none were found.

Energy is NVML cumulative board energy minus the average pre/post idle power times
the active counter interval, divided by replay count. Idle measurement uses a 3-second
settle followed by 3 seconds of observation before and after each active interval.
Trial energy sample SD is 59.608 mJ;
time sample SD is 0.144664 ms.
Gross mean board energy is 10680.868 mJ per invocation.
The third trial has a noticeable idle-power shift; raw counters and one-sided estimates
are retained rather than discarding the trial.

Excluded: original-image resize, transfers, allocation, JIT, graph capture, validation,
startup and CPU energy. Included: device normalization, parameter initialization,
training and prediction. GPU SIMD/register broadcasts implement the reuse policy;
this is not a physical simulation of grid memory distances. GPU inference retains
1,000 hidden rows, while the grid reuses minibatch scratch.

## Modeled Spatial Grid

Model revision: `01a0bd5e0d2564825b0f53dd766f763c82dbc7c0`; spatial-computer pitch 128 / ISA v4.
This uses the existing research scorer's **globally serialized legal schedule**, not
a claim of parallel speed or an official submission acceptance decision.
The panel's own region layout, operations, staging copies and access counts are scored.
Only the historical program's machine header changes in the spatial adapter.

| Metric | Exact result |
| --- | --- |
| Word-node hops (1 fJ each) | 223,152,979,758 |
| Cycles at 1 GHz | 3,151,348,671 |
| Executed instructions | 1,074,169,999 |
| Peak allocated scratch, including tape stages | 128,424 bytes |
| Instruction-issuing processors | 250 |
| Maximum simultaneous instructions/accesses | 1 |
| Host time to score | 0.066908708 s |

Compute runs on P(125,0). Program words occupy nearest legal cells in nearest tiles;
one stage word is reserved in every bottom-row tile. Local coordinates in the central
64x64 processor region are excluded from scratch placement. Each tile has at most
12,288 scratch words. `d2_first` layout and reusable panel/operand/accumulator/cache
regions are explicit in the generated program.

Input tape order is all training pixels, raw uint32 training labels, then test pixels.
Logical word k uses bottom port k modulo 250. Each input is received into its stage
then copied to its program destination; output reverses that staging. FP32 payloads
are raw uint32 words and argmax uses first-class tie breaking.
Normal instructions read every source and write their destination, including copy,
initialization and all panel/cache accesses. Every blocking access completes before
the next starts. Remote traffic routes horizontally then vertically and responses
retrace the route, so there is no contention hidden by parallelism assumptions.

For Manhattan tile distance L and within-tile distance d, scratch-access hop cost is
`256*L + max(50,2*d)`. Local accesses take one cycle; remote reads take `2*L+1`,
remote writes `L+2`. Bottom-port tape cost is `64+max(50,2*d)` hops and two cycles,
plus the separately charged stage/program copy accesses. Exact integer histograms
sum these costs over the full program. All scratch initialization, literal setup,
input/output traffic, learner transform, targets, training and inference are included;
3x3 image resize is excluded, as in the A100 input boundary.
Host score time includes schema, address and initialization validation, physical
placement, histograms, exact costs and hashing, but excludes IR generation, file I/O
and numerical execution. Placement and program hashes are recorded.

## Verification and Reproduction

Eight grid tests pass, including the original scorer tests and exhaustive tiny
panel execution with selected/asymmetric panels and mutated data. Expanded individual
wire events, every address histogram, parameter bits, scores and predictions agree.
Tiny programs exercise partial panels. Production checks validate all addresses,
initialization and allocation before scoring; the billion-instruction full grid
program is statically counted, not exhaustively numerically interpreted.
Training checks compare panel and ordered reference parameter bits after every epoch
for each of the 11 full draws. `audit.py` independently repeats full ordered learning,
checks all draw score/parameter/prediction hashes, verifies GPU evidence and recomputes
NVML arithmetic and grid costs.

From the repository root, audit existing evidence without paid GPU work:

```sh
uv run --with numpy==2.2.6 python -B mnist/submissions/mlp-panels-revised-20260914/test_grid.py
uv run --with numpy==2.2.6 python -B mnist/submissions/mlp-panels-revised-20260914/test_reproduce.py
uv run --with numpy==2.2.6 python -B mnist/submissions/mlp-panels-revised-20260914/reproduce.py audit
```

Use `reproduce.py` rather than invoking the frozen `run.py` directly. Upstream added
a medium-dataset profile after this run, changing the shared helper's file hash.
`frozen_data.py` is a byte-identical copy of the helper actually used, checked against
the original protocol hash. The wrapper pins dependency resolution without modifying
the frozen learner, protocol, indices, predictions or measured hardware evidence.
Auditing reuses the committed resized inputs; regeneration on a different BLAS/platform
is not guaranteed to reproduce those input bits.

Fresh studies must use a new sibling directory. Copy the orchestration scripts
`adapt_sources.py`, `grid_score.py`, `test_grid.py`, `prepare_gpu.py`, `audit.py`,
`build_report.py`, `reproduce.py`, `test_reproduce.py`, and `frozen_data.py` there,
but no protocol, data or generated evidence.
`adapt_sources.py` derives the source using audited numeric-token substitutions,
records upstream hashes and refuses to regenerate once a protocol is frozen.
Run these phases in order from the repository root, replacing `<fresh>`:

```sh
python3 -B mnist/submissions/<fresh>/adapt_sources.py
uv run --with numpy==2.2.6 python -B mnist/submissions/<fresh>/test_grid.py
uv run --with numpy==2.2.6 python -B mnist/submissions/<fresh>/reproduce.py prepare --raw mnist/data/raw
uv run --with numpy==2.2.6 python -B mnist/submissions/<fresh>/reproduce.py fit
uv run --with numpy==2.2.6 python -B mnist/submissions/<fresh>/reproduce.py evaluate --raw mnist/data/raw
uv run --with numpy==2.2.6 python -B mnist/submissions/<fresh>/grid_score.py --epochs 300 --n-train 1000 --n-test 1000
uv run --with numpy==2.2.6 python -B mnist/submissions/<fresh>/reproduce.py gpu-reference
MODAL_PROFILE=vargapowercouple uvx --with numpy==2.2.6 modal==1.5.5 run --profile vargapowercouple mnist/submissions/<fresh>/gpu_benchmark.py --output mnist/submissions/<fresh>/gpu_measured
uv run --with numpy==2.2.6 python -B mnist/submissions/<fresh>/reproduce.py audit
uv run --with numpy==2.2.6 python -B mnist/submissions/<fresh>/reproduce.py report
```

The Modal command incurs paid A100 usage and explicitly requires the authorized profile.
Exact inputs and measured artifacts are retained for this run. This report supplies
pull-request evidence, not a claim of benchmark acceptance or merge. No W&B run was created.

## Evidence

The merge review also verified this program against main's shared scorer after
the Adam extension in `39da2cc`. All panel grid metrics are unchanged, and all
eight grid tests pass. `audit.py` accepts those three exact shared-source hashes
as compatible versions and records when they are used; all other source hashes
must still match the original evidence. The learner, inputs, predictions, GPU
measurements, and grid score retain their original hashes. The verification,
test results, and artifact manifest record the completed merge review.

- `protocol.json`, `source_adaptation.json`: predeclared configuration and source lineage.
- `draw_manifest.json`, `private/draw-*.npz`: indices, hashes and allowlisted learner inputs.
- `prediction_manifest.json`, `predictions/`, `accuracy.json`: frozen predictions and all counts.
- `program.spatial.json`, `grid-score.json`, `grid-test-results.json`: placement/schedule and costs.
- `gpu_measured/`: raw telemetry, validation hashes, generated kernels, PTX and GPU output arrays.
- `verification.json`, `metrics.json`: independent audit and machine-readable summary.
- Modal run: https://modal.com/apps/vargapowercouple/main/ap-g1CHhA6223NpkTeQAp3blG
