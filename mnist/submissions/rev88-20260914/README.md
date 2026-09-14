# MNIST-medium: reversible MLP, 12% error target

Submission date: September 14, 2026.

**89.2800% ± 0.3591 pp** mean accuracy ± sample standard deviation over all eleven draws, with **98,208 / 110,000 correct**. The unrounded mean error is 10.72%. This **meets** the current inclusive 12% error band, which requires at least 88% mean accuracy or 96,800 correct. Qualification uses exact counts, not the rounded display.

Draw-zero A100 measurements give **1.1e+03 mJ** idle-adjusted energy and **38 ms** per complete fresh GPU-resident training-and-prediction task. These are means of three repeated-measurement trials on one draw, not energy averaged over the eleven accuracy draws.

| A100 energy (mJ/task) | A100 CUDA time (ms/task) | Grid energy (mJ) | Grid time (ms) | Grid scratch (bytes) | Time to score (s) |
|---:|---:|---:|---:|---:|---:|
| 1.1e+03 | 38 | — | — | — | — |

## Accuracy evidence

| Draw | Dataset seed | Learner seed | Correct / total | Accuracy |
|---:|---:|---:|---:|---:|
| 00 | 2026091400 | 11 | 8,887 / 10,000 | 88.87% |
| 01 | 2026091401 | 11 | 8,916 / 10,000 | 89.16% |
| 02 | 2026091402 | 11 | 8,947 / 10,000 | 89.47% |
| 03 | 2026091403 | 11 | 8,904 / 10,000 | 89.04% |
| 04 | 2026091404 | 11 | 8,992 / 10,000 | 89.92% |
| 05 | 2026091405 | 11 | 8,979 / 10,000 | 89.79% |
| 06 | 2026091406 | 11 | 8,911 / 10,000 | 89.11% |
| 07 | 2026091407 | 11 | 8,922 / 10,000 | 89.22% |
| 08 | 2026091408 | 11 | 8,950 / 10,000 | 89.50% |
| 09 | 2026091409 | 11 | 8,877 / 10,000 | 88.77% |
| 10 | 2026091410 | 11 | 8,923 / 10,000 | 89.23% |

Each draw is a separate PCG64 permutation of the official 60,000 MNIST training rows. The first 10,000 rows train the learner and the next 10,000 provide evaluation inputs; the two sets are disjoint within that draw. Independently sampled draws can overlap one another. The official test split is unused. Mean and sample SD (ddof=1) describe these eleven datasets; no confidence interval or independent-image inference is claimed.

The procedure was chosen using only an 8,000/2,000 split inside draw 0's training portion. All 16 candidates were evaluated before the qualification predictions were scored. Candidates reaching 89% validation accuracy were ordered by fewest epochs, then shallowest depth, then smallest accuracy surplus over 89%; the selected candidate scored 1,780/2,000. The frozen procedure was then trained freshly on all eleven complete 10,000-example training draws. Selection examples can occur in another draw's evaluation set because the draws share the same source population; selection is not additional qualification evidence.

## Reversible learner and memory

The only model inputs are the 81 supplied image pixels. Pixels are converted to float32, divided by 255, resized from 28×28 to 9×9 with exact fractional box-area averaging, clipped to [0,1], then transformed inside each measured invocation to `4*x - 0.5`. Appending one zero gives an injective 82-coordinate lift, split into two 41-coordinate halves.

The core has 2 additive coupling blocks, each with two bias-free 41×41 linear branches applied after ReLU, with residual scale 0.5. A learned 82→10 linear readout with bias produces class scores. The core is reversible; the classifier is not. The model has **7,554 FP32 parameters** (30,216 payload bytes).

Training uses **2 epochs**, batch size **128**, mean cross-entropy loss, SGD learning rate **0.1**, momentum **0.9**, and weight decay **0.0**. There is no augmentation or ensemble. Each epoch uses a seed-only PCG64 permutation; the final 16-example minibatch is included. Branch weights start at normal standard deviation `0.05/sqrt(41)`; the head uses PyTorch's standard Linear initialization. Learner seed 11 is fixed, but weights, gradients and momentum are reset before every draw and timed invocation.

Backward uses a custom autograd function that saves the core endpoint and parameter references, reconstructs each previous state by subtraction, and creates at most one branch's ordinary autograd graph at a time. Thus the number of saved core activation states is one. The referenced parameter tensors are not copied activation snapshots. First-order differentiation is supported; this is not a claim of bit-identical stored/reconstructed gradients or constant total GPU memory.

The retained core endpoint contains 82 FP32 values per example: **328 bytes/example**, or **41,984 bytes (41.00 KiB)** for batch 128. Four resident parameter groups—weights, gradients, momentum and the seed-initialization copy—contain 120,864 payload bytes. These are exact tensor-size calculations. Raw/normalized training and query arrays, resident permutations, classifier tensors, temporary branch gradients and CUDA graph memory are additional costs.

Measured CUDA tensor-allocation peaks over preparation plus the first full invocation span **82,509,824–283,836,416 bytes** across eleven draws. Entry baselines on reused workers range from 0 to 201,326,592 bytes. Subtracting each draw's own entry baseline gives **82,509,824–82,509,824 added-peak bytes** (78.69–78.69 MiB); the raw variation comes entirely from preexisting resident allocations, not model/data-dependent growth. The cause of those preexisting allocations was not isolated in this submission. This scope includes CUDA graph pools and library workspaces, and is sampled before output serialization and synthetic CUDA diagnostics. It excludes driver/context allocations; it is not a pure activation measurement, a cache-working-set measurement, or energy's preparation-excluded scope. Grid scratch remains unmeasured.

CPU FP64/FP32 checks compare reconstructed outputs, all parameter gradients, real-input gradients and inverse recovery against ordinary autograd at both depths. A separate synthetic CUDA check compares eager gradients/SGD state with the captured implementation, verifies exact replay resets and prediction equality, and checks endpoint/parameter-reference retention. These diagnostics run outside energy timing. TF32 is disabled; this PyTorch implementation does not claim the older ordered-kernel arithmetic contract.

## A100 measurement method

Hardware: **NVIDIA A100-SXM4-40GB**, driver 580.95.05, power limit 400.0 W. Runtime: PyTorch 2.5.1+cu124, CUDA 12.4, NumPy 2.2.6, Python 3.11.15, nvidia-ml-py 12.560.30. The CUDA runtime's PCI bus ID selects the matching NVML handle; the report verifies both IDs identify the same physical device. GPU identity, clocks, temperature, power samples and raw cumulative NVML counter readings are retained in draw 0's result.

A CUDA-event calibration selected **104 complete invocations per trial**, targeting approximately five seconds of active measurement (capped at 1,000 repetitions). Each of three trials has a three-second idle measurement before and after its active interval; each idle sample follows a three-second settling interval. Paired idle power is the mean of the two idle samples. Idle-adjusted energy is `active joules - paired_idle_watts * active_seconds`, divided by complete invocations and converted to mJ. CUDA events measure execution time; wall time and unadjusted energy are separately retained.

| Trial | Fresh invocations | Idle-adjusted mJ/task | Unadjusted mJ/task | CUDA ms/task | Active seconds |
|---:|---:|---:|---:|---:|---:|
| 1 | 104 | 1.0e+03 | 3.5e+03 | 38 | 3.9 |
| 2 | 104 | 1.1e+03 | 3.5e+03 | 38 | 4.0 |
| 3 | 104 | 1.2e+03 | 3.5e+03 | 38 | 3.9 |

Exact three-trial means are 1101.24116693707 mJ/task (sample SD 72.6534989423127 mJ) and 37.9422388321314 ms/task (sample SD 0.145172720988073 ms). Raw precision is retained for reproduction, not a claim of corresponding measurement precision.

Every invocation resets initial weights, zeroes gradients and momentum, normalizes all raw training/query pixels, copies the resident epoch permutations device-to-device, performs every training update, and predicts all 10,000 query labels. Repeated runs are not inference from cached trained weights. Predictions and final state hashes are checked before/after all measurements and match qualification draw 0.

The GPU-resident scope excludes source download/resizing, CPU permutation generation, host/device transfer, allocation, JIT compilation, graph capture, cold start and verification. Energy measures the GPU board through NVML, excluding host CPU energy. Full application energy and end-to-end wall time are unavailable. This submission does not include a spatial-grid placement, word-node-hop model, theoretical grid energy/runtime or host time-to-score; those columns remain unmeasured.

## Reproduction and provenance

Install `modal` and `numpy` locally and authenticate Modal. The runner pins its container by digest and installs NumPy 2.2.6 and nvidia-ml-py 12.560.30. Use a fresh output directory so the retained qualification files are never overwritten. From the repository root:

```bash
mkdir -p /tmp/rev88-reproduction
cp mnist/submissions/rev88-20260914/{learner.py,energy.py,data_reference.py,run.py,build_report.py} /tmp/rev88-reproduction/
modal run /tmp/rev88-reproduction/run.py --phase validate --raw-dir "$PWD/mnist/data/raw"
modal run /tmp/rev88-reproduction/run.py --phase fit --raw-dir "$PWD/mnist/data/raw"
modal run /tmp/rev88-reproduction/run.py --phase evaluate --raw-dir "$PWD/mnist/data/raw"
python /tmp/rev88-reproduction/build_report.py --raw-dir "$PWD/mnist/data/raw" --rules-file "$PWD/mnist/README.md"
```

The `validate` stage uses no qualification-test labels and freezes the selected procedure before `fit`. Source/version hashes, raw MNIST checksums, all sampling indices' hashes, input hashes, seed-initialization hashes, epoch permutation hashes, final parameter/momentum hashes and prediction hashes are retained. The report builder verifies every frozen output before loading evaluation labels, independently replays sampling and preprocessing, rescores all eleven retained prediction arrays and retained training-only validation predictions, and recomputes energy arithmetic. Earlier development validation archives are not qualification evidence. No learned checkpoint or raw dataset is required in the submission directory.

Contributors: Yaroslav Bulatov (requirements), Codex (implementation, measurements and report). No W&B run was created. Reproduction uses at most four ephemeral A100 workers as configured in the runner; execution and shutdown are recorded by the submission coordinator.

## Evidence

[Current challenge rules](../../README.md) · [Machine-readable submission](submission.json) · [Accuracy counts](accuracy.json) · [Frozen protocol](protocol.json) · [Configuration](config.json)

[Training-only selection plan](validation-plan.json) · [Selection results](validation.json) · [Draw manifest](draw-manifest.json) · [Prediction freeze](prediction-manifest.json) · [Draw-zero NVML and CUDA validation](results/draw-00.json)

[Learner](learner.py) · [NVML measurement](energy.py) · [Runner](run.py) · [Canonical data routines](data_reference.py) · [Report/check generator](build_report.py)

[Independent verification](verification.json) · [Independent verifier](verify.py) · [Execution and shutdown record](execution.json)
