# PCANet nine-block + mixture: MNIST-original

PCANet with **K=100 PCA dimensions and k=8 mixture components per class**
achieves **9,906 / 10,000 correct (99.06%)** on official MNIST, meeting the
1% error target. Costs include fresh training on all 60,000 training images
and prediction on all 10,000 test images.

September 20, 2026. Contributor: [@jurajselep](https://github.com/jurajselep).
Prepared with AI assistance. The primary A100 measurement is **30,000 mJ above
idle and 290 ms**, with two further A100 hosts reproducing the predictions.
Grid costs are unavailable.

## A100 measurements on three boards

The unchanged **K=100** learner was measured on two separate Vast.ai machines.
Both produce **9,906 / 10,000 correct (99.06%)** and exactly the same 10,000
predictions as the initial Modal run. Model sources, dataset bytes, Python,
PyTorch, CUDA, cuDNN and NumPy versions match across all three boards.

| Host | Board power limit (W) | A100 energy above idle (mJ) | A100 time (ms) | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Vast.ai UK | 270 | 29,000 | 300 | 99.06% |
| Vast.ai Slovenia | 400 | 30,000 | 290 | 99.06% |
| Initial Modal comparison | 400 | 30,000 | 290 | 99.06% |

The cost columns display two significant figures. Exact counter medians are
**28,777.896732 mJ / 297.256503 ms** for UK and
**30,351.582470 mJ / 288.310343 ms** for Slovenia. Sampled-power medians are
28,291.346544 mJ and 30,130.742502 mJ, respectively: 1.69% and 0.73% below
counter estimates. The initial Modal result lies between these two counter
energy medians. Each host independently ran four windows of 60 complete
training-and-prediction tasks with the same paired-idle protocol.

These are distinct physical boards and Vast.ai machine/host IDs. UK uses
NVIDIA driver 570.211.01; Slovenia uses 570.133.20; Modal uses 580.95.05.
The host-configured power limits were retained. Driver, board, clocks, thermal
state and hosting conditions vary, so the energy difference cannot be assigned
to the power limit alone. Values remain per-host medians.

Each Vast.ai evidence set passes **322 offline checks**; the prior Modal set
passes 506. The cross-host check confirms source/data identity and byte-identical
predictions. The same one-prediction difference versus the PyTorch feature
reference remains on all three boards. Sensor checks, signed idle controls,
all rounds and raw power traces are preserved. Both Vast.ai rentals were
**destroyed after evidence retrieval**, confirmed by the final instance query.

- [UK raw evidence](evidence/vast-uk/) and [verification](evidence/vast-uk/verification.json)
- [Slovenia raw evidence](evidence/vast-slovenia/) and [verification](evidence/vast-slovenia/verification.json)
- [Cross-host comparison](evidence/vast-comparison.json)
- [Provisioning and teardown evidence](evidence/vast-provisioning.json)
- [Vast.ai reproduction instructions](vast/README.md) and [frozen run plan](vast/protocol.json)

## Initial Modal A100 result

| Configuration | Correct / 10,000 | Accuracy | A100 energy above idle (mJ) | A100 time (ms) | Grid energy / time |
| --- | ---: | ---: | ---: | ---: | --- |
| **K=100, k=8 — submitted model** | **9,906** | **99.06%** | **30,000** | **290** | — / — |
| K=80, k=8 — historical comparison | 9,906 | 99.06% | 27,000 | 270 | — / — |

Costs are displayed at two significant figures. For K=100, the exact medians are
**29,709.265735 mJ above idle and 287.581293 ms**; the separate power-integral
estimate is **28,955.648679 mJ**, 2.54% below the counter estimate. Gross energy
is 49,985.225 mJ. Four counter estimates span 29.103–29.761 J; this range is
not a confidence interval. K=80 measures 26,893.633381 mJ and 268.761428 ms.
These are fresh results from one A100-SXM4-40GB, not the historical two-host
measurements.

The run uses Python 3.11.10, PyTorch 2.5.1+cu124, CUDA 12.4, cuDNN 9.1.0,
NumPy 2.1.2 and driver 580.95.05, with a 400 W board limit. Both configurations
peak at 1,582,200,320 PyTorch allocated bytes and 2,615,148,544 reserved bytes.
Board identity and full device output are retained with the measurements.

**Validation passes with a numerical-equivalence limitation:** all three fresh
runs and the checked measurement outputs agree exactly. The PyTorch feature
reference scores 9,907 / 10,000 and differs on one prediction for each K.
Across 3,538,944 feature values from 1,536 images, 5,976 differ, with maximum
absolute difference 0.008417938. The feature paths are therefore **not bitwise
equivalent**. Different convolution arithmetic can affect zero thresholds;
this run does not isolate the cause. The reported 99.06% belongs to the
fused CUDA implementation included here.

Changing the training labels changes 9,999 predictions and reduces K=100
accuracy to 9 / 10,000. The dense-load sensor check reads 394 W versus 70 W
idle. The 15-second idle-only control is −7.901 J by counter and −1.088 J by
sampled power; signed residuals are retained, not clamped. Setup exclusions
and historical test-set tuning remain limitations of the claim.

[Raw measurements and predictions](evidence/rerun/),
[independent 506-check verification](evidence/verification.json), and
[completed GPU job](https://modal.com/apps/jurajselep/main/ap-sbUnakIg9FEhrpRekTR0ZR).


## Historical result attribution

An earlier working summary combined results from different configurations. The historical
**24.1 / 26.6 J and 99.06%** measurements used **K=80**, established by the
benchmark calling `task_cuda` with its default `K=80`. Historical **99.08%**
for K=100 is a retained count of **9,908 / 10,000**, without saved predictions.

The quoted **~3,542 mJ** is **3,343.457 mJ for the K=80 head plus 198.538 mJ
for a four-block front end**. That sum does not describe the nine-block model.
The historical K=100 head alone is reported as **5,513.176 mJ**. The front-end
grid program was also explicitly flagged as numerically unvalidated.
**No complete grid energy or runtime is claimed here.**

See [the attribution audit](evidence/historical_audit.json), its unchanged
[historical records](evidence/historical/), and
[source extraction evidence](evidence/source-extraction.json).

## Learner and numerical implementation

- Preserve official IDX row order. Convert pixels to FP32 and divide by 255;
  use all 28×28 pixels, with no resizing or augmentation.
- Learn 8 first-stage and 5 second-stage 7×7 PCA filters. The first bank uses
  the first 255 training images. The second uses the first 2,000 first-stage
  planes from the first 250 images. Each filter covariance is capped at the
  first 200,000 centered patches, with FP64 covariance/eigendecomposition.
- Threshold second-stage responses at zero into 5-bit codes. Form nine
  overlapping 14×14 histograms at stride 7 for each first-stage filter:
  **8 × 9 × 32 = 2,304 features**. Normalize by the full histogram sum per
  image, then take the square root.
- Center features using the training mean; fit K=100 PCA by FP32 covariance
  and eigendecomposition. Fit eight full-covariance mixture components per
  class, with eight k-means passes and eight EM updates followed by a final
  parameter fit. Class-specific initial seeds are 900 through 909.
- Mixture covariance blending is 0.5, class covariance shrinkage 0.05, and
  ridge 0.0001. Include class priors and choose the largest class score.

[model.py](model.py) exposes `train_predict(x, y, q, K=100)`. It accepts training
labels but **does not accept test labels**. Every call refits both filter banks,
the feature PCA and all mixtures. It caches only the compiled CUDA extension.

[cuda_kernel.py](cuda_kernel.py) fuses second-stage convolution, code packing
and histogram construction, avoiding a materialized second-stage response
tensor. Kernel arithmetic is unchanged from the imported implementation.
The wrapper adds argument checks, a device guard, current-stream dispatch and
a launch-error check. The standalone extraction removes unrelated learners,
hardcoded paths, and import-time measurement side effects.

Both rerun backends share the optimized FP32 PCA and mixture head; `reference`
uses the original PyTorch feature extractor. Matmul and cuDNN allow TF32.
This comparison checks the fused feature path, not an independent
implementation of the entire learning algorithm. Exact agreement, or any
observed disagreement, is reported in the saved validation evidence.

The complete hyperparameters, import revision and frozen rerun choices
are in [protocol.json](protocol.json). Historical choices used test-set
feedback; repeating the official test split does **not** provide a new held-out
evaluation. The fresh rerun did not tune hyperparameters against its results.

## Measurement and validation scope

One task includes learning both filter banks, all 70,000 feature extractions,
feature-PCA fitting, mixture training and 10,000 predictions. Normalized inputs
are already device-resident. Tensor allocation and host dispatch are included;
download, normalization, transfer, CUDA compilation, warmup and label scoring
are excluded. This is warm repeated training-and-prediction board energy.

Four active windows each execute 60 fresh tasks, bracketed by 10 seconds of
idle before and after. For each window:

```text
net J/task = [active J - mean(before-idle W, after-idle W) × active seconds] / 60
```

The main estimate is the median NVML cumulative-counter result. A separate
20 ms sampled-power trace is integrated with interpolated interval endpoints.
Both methods use the same board sensors. Gross energy, all round results,
raw counter endpoints, power samples and an idle-only control are retained.
They are not wall-plug or whole-system measurements. A dense-matmul sensor check
and GPU process monitoring precede and accompany measurement. A controlled
allocation/release identifies the container's NVML PID alias.

Validation checks canonical dataset hashes, three fresh prediction runs per K,
the PyTorch feature path, features on 1,536 images including a nondefault CUDA
stream, training-label perturbation, and the final predictions from each
measurement window. Predictions from every intermediate timed repetition are
not retained. Peak PyTorch allocated/reserved device bytes are reported; this
is GPU memory accounting, not grid scratch memory.

## Reproduce

On an otherwise idle Vast.ai A100-SXM4-40GB container, use the measured image
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`, which includes Python 3.11,
CUDA development tools and a C++ compiler. From this directory:

```sh
python vast/run_host.py --output generated/new-rerun --raw raw
```

The wrapper checks the environment, installs any missing pinned dependencies,
runs K=100 validation, saves the execution log and status, then verifies the
complete evidence bundle. It was exercised on both recorded Vast.ai hosts.
It does not create cloud instances. See [the host instructions](vast/README.md)
for allocation requirements.

The four canonical MNIST gzip files download automatically and are verified
before use. Supply `--raw /path/to/raw` to reuse a cache. Output directories must
be new. The host wrapper reruns only K=100. The low-level `validate.py` also accepts
`--dimensions 100 80` for the retained historical comparison.

Alternatively, with an authenticated Modal account, allocate one temporary
A100 and collect all output locally:

```sh
python -m pip install -r requirements-controller.txt
modal run modal_rerun.py --output generated/new-modal-rerun
```

Verify all three retained evidence sets without a GPU, including exact
accuracy against independently downloaded canonical MNIST:

```sh
python -m pip install numpy==2.1.2
python -c "from data import official; official('raw')"
python vast/compare.py --raw raw
```

To check one new run, use
`python verify_results.py --results generated/new-rerun --raw raw`.
The comparison reports 506 checks for the Modal evidence and 322 for each
Vast.ai host, along with source/data hashes, physical-board identity and
prediction agreement.

The source archive includes only this model, its reproduction/verification
tools and relevant evidence. It excludes datasets, trained caches, unrelated
experiments and credentials.

Run `sha256sum -c SHA256SUMS` from this directory to check all retained source
and evidence files. The [MNIST-original table](../../README.md#mnist-original--1-test-error-target)
uses the Modal K=100 measurement; all three host results remain separate above.
