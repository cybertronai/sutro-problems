# MNIST-small: a reproducible 1NN submission attempt

**308 / 600 correct · 51% accuracy · current 50% target met.**

This fixed 1-nearest-neighbor algorithm learns by memorizing the 600 supplied training examples, then labels each of the 600 test images with the label of its nearest training image. It uses the nine supplied pixel values directly. No neural-network training, extra data, pretrained weights or hyperparameter search is involved. This is a baseline attempt, with no claim of optimal accuracy, time or energy.

The submission was merged in [PR #64](https://github.com/cybertronai/sutro-problems/pull/64). [Submission source](https://github.com/cybertronai/sutro-problems/tree/main/mnist/submissions/1nn-v4-20260911). Its model scores depend on the explicit numeric, tape and area conventions below. They are calculated by a submission-owned interpreter, because the linked specification does not provide an authoritative executable scorer. The A100 measurements are actual NVML and CUDA-event observations.

Contributors: Codex (implementation, experiments and report), with independent scorer, GPU and rules-review agents; requested by `yaroslavvb`. Run date: September 10, 2026 Pacific / September 11 UTC. No W&B runs were created for this attempt. Before publication, main was rechecked at `1ede666`: the README now explicitly calls the model scores theoretical and allows an A100 ISA of choice. Dataset, target and v4 requirements are unchanged. The A100 source is a hand-written equivalent Triton implementation; it is not mechanically generated from the v4 text.

[TOC]

## Results and metric boundaries

All task runtimes use **milliseconds (ms)** and all energies use **millijoules (mJ)**, so the theoretical and measured results share the same units. **Time to score uses seconds (s)** because it measures the host scoring process. Each task includes 600 training examples and 600 test predictions. Per-task runtime and energy values in the comparison and trial tables are rounded to two significant figures; exact totals are retained in the linked measurement files, and the measured energy's baseline sensitivity is reported separately.

| Performance per complete task | Theoretical model | Measured A100, mean |
| --- | ---: | ---: |
| **Time (ms)** | **1.7** | **0.0069** |
| **Energy (mJ)** | **0.0019** | **0.52** |

The model sums charged scratch accesses and excludes tape I/O. The A100 time is CUDA-event steady-state graph throughput including the memorization copy; its energy is idle-adjusted GPU board energy from NVML. The shared units make the numerical scales directly comparable; the measurement boundaries remain as documented here.

| Supporting metric | Result | Meaning |
| --- | ---: | --- |
| Accuracy | **308/600 = 51%** | Canonical fixed small test split |
| Area, occupied-cell convention | **6.0 × 10³ µm²** | Peak 6,014 allocated 32-bit scratch words, 24,056 bytes |
| Time to score | **35 s** | Host runtime of one full generator/interpreter/accounting run on Intel Core i9-9880H, 2.30 GHz |

Measurement-window durations also use milliseconds. Unit conversions: **1 ms = 1000 µs = 10⁹ ps** and **1 mJ = 10¹² fJ**. Energy and time measure different quantities, related by **E(mJ) = P(W) × t(ms)**. The cost model retains its exact native ps/fJ accounting internally.

Both modeled and GPU task scopes include learning/memorization and all 600 predictions. Dataset preparation, the one training-only validation check, and host evaluation are outside those task scopes. The GPU additionally writes nearest-row indices and distances for verification; those writes are included in its measured runtime and energy. The CPU reference's 11 ms host runtime is supplementary and is **not** the model's Time or Time to score.

**The threshold margin is eight examples.** This establishes a pass on this particular fixed split. There is no measurement of generalization across alternative dataset samples, and no post-test algorithm or hyperparameter changes were made. Detailed rule gaps and experimental limitations are in the [separate ambiguities and problems report](ambiguities.html).

## Dataset and selection record

The task is `competition-v2`, seed **20260910**: 600 training and 600 test images at **3 × 3**, from disjoint subsets of the original 60,000-example MNIST training pool. The small training rows are positions `[0:600]` and test rows `[6000:6600]` in the documented PCG64 permutation. Pixels are normalized float32 box-area averages. This is the current 600/600 task, distinct from the historical 1,000/1,000 experiment. See the [pinned task instructions](https://github.com/cybertronai/sutro-problems/blob/e70f9c9e1db65b62d9256f7b1f9b668cf4c48909/mnist/instructions.md).

The original MNIST gzip checksums were verified during fresh dataset regeneration. All six regenerated array-content hashes, shapes and dtypes match the canonical manifest; the train/test source-index overlap is zero. The learner decodes only `train_images`, `train_labels`, and `test_images`. Test labels are consulted separately by the official accuracy evaluator. An independent CLI check removed the test-label and source-index members entirely and reproduced every prediction. The NPZ file checksum is bookkeeping after prediction and does not affect learned labels.

| Identifier | SHA-256 |
| --- | --- |
| Canonical small NPZ used here | `300384c0bcaf8c380d6ef2ff5013656b861d720f5c99f55cf93fbe1dbbfa046e` |
| Train images, C-order little-endian | `1d123f4ff4c0fa6975ee07f5e647971713284515c4e10b88b943a257fdcb7ca5` |
| Train labels, C-order little-endian int64 | `948baa48cdcf75b8b0187a9a5f2a1dcec108998dbfd0081cccc396bfb14321aa` |
| Test images, C-order little-endian | `9c0793497fb028e7f6ecea5819377502d4e25990aea59578db9ba6f52ba69a09` |
| 600 predicted labels, little-endian int64 | `4c6b5f9b1c8a77f04388904ca86e67a6f4ec99b7e00847d6df728bd4fcb48579` |

The NPZ identity happens to match the checked-in container hash in this environment; array hashes are the portable identity independent of ZIP metadata. Float32 resizing reduction order may still differ across numerical libraries, so reproduction must check the array hashes rather than assume equality.

Before test scoring, the single fixed candidate was checked on a training-only 480/120 split using PCG64 seed 20260911: **63/120 = 53%** (rounded). The full 600 training examples were then used to produce frozen test predictions. No choices of k, weighting, scaling or distance metric were searched. Exact training/validation row indices are retained in `cpu_results.json`.

## Algorithm and numerical convention

For each query q and training row i, initialize d to FP32 zero. For feature j from 0 through 8, calculate `t = float32(q[j] - x[i,j])`, then `s = float32(t*t)`, then `d = float32(d+s)`. Choose the smallest distance; equal distances keep the lowest index in the supplied training array (0–599). Output that row's training label. Squared distance suffices, so no square root or division is needed.

Every subtraction, multiplication and addition rounds separately to IEEE binary32, round-to-nearest ties-to-even. Fused multiply-add is disabled in Triton and checked in generated PTX. The independent NumPy implementation and scalar model interpreter agree on all predictions. GPU validation additionally confirms every winning index and bit pattern of every winning distance.

## Explicit tape program and placement

The [pinned v4 ISA](https://github.com/cybertronai/simplified-dally-model/blob/26abcca402de647381d31286d42dfbb7a001763d/instruction-sets/v4/README.md) defines 32-bit words and tape operations but does not fix floating-point arithmetic. This submission interprets distance `sub`, `mul`, `add` and `cmp lt` as FP32 operations; `copy` and `select` move raw words; comparison produces raw integer 0 or 1. Its only `set` literal is integer zero, whose raw bits are also FP32 +0.

The proposed input tape contains **11,400 little-endian 32-bit words**, in this fixed order:

1. 5,400 FP32 training pixels: training row order, then C-order pixels.
2. 600 training labels, represented as unsigned 32-bit integers 0–9 (lossless from int64).
3. 5,400 FP32 test pixels in their supplied row and pixel order.

The output tape is 600 unsigned 32-bit predicted labels in test-row order. These serialization choices need organizer agreement for comparisons across submissions. The input tape is consumed exactly once using `recv`; each output is produced using `send`. All training data enter scratch through `recv`. Test pixels are streamed nine at a time; test labels and source indices are absent from the input tape.

Scratch addresses 1–9 hold the current query, 10–14 hold five temporaries, 15–5414 hold training pixels, and 5415–6014 hold training labels. Addresses fill Manhattan half-diamond shells: increasing h, then increasing x in `-(h-1)..h-1`, with y = h − |x|. The hottest fourteen words are closest to the processor. The maximum distance is **78 hops**. Every allocated location is initialized before a source read.

The occupied-cell convention gives **6,014 µm²**. The enclosing grid-cell rectangle is **12,012 µm²**, illustrating why occupied cells and full physical footprint must be distinguished. Processor, instruction storage, tapes and routing area have no supplied area model and are excluded. The source generator emits a fully straight-line v4 program; its Python loops generate instructions and introduce no unpriced machine loops or branches.

## Model score derivation and verification

The following formulas use the specification’s **exact native units, ps and fJ**, rather than the report’s ms/mJ display units. The [single-core-with-tape cost model](https://github.com/cybertronai/simplified-dally-model/blob/26abcca402de647381d31286d42dfbb7a001763d/models/single-core-with-tape/README.md) charges each non-tape scratch read at distance h `max(50, 2h)` fJ and `max(50, 0.8h)` ps. Each scratch write costs `max(50, 2h)` fJ and `max(50, 0.4h)` ps. Both reads of `mul t,t,t` count, and `select` reads its condition and both alternatives. Accesses are blocking and are summed. `recv` and `send`, including their scratch accesses, cost zero by definition.

| Instruction | Count |
| --- | ---: |
| recv / send | 11,400 / 600 |
| set | 360,000 |
| sub / mul / add | 3,240,000 each |
| copy | 1,200 |
| cmp lt | 359,400 |
| select | 718,800 |
| **Total** | **11,171,400** |

There are **22,316,400 charged reads** and **11,159,400 charged writes**, totaling **33,475,800 accesses**. Each persistent training word is read 600 times: 3,600,000 persistent reads. The remaining 29,875,800 hot accesses are at the 50-unit floors. An independent count formula sums the exact placement costs of the persistent reads and the hot-access floors. It agrees exactly with the full interpreter's **0.0019 mJ** and **1.7 ms** (rounded here; exact equality was checked). Time is accumulated as integer 0.2 ps ticks to avoid rounding in score summation.

Time to score starts immediately before machine construction and includes instruction generation, all scratch-state checks, FP32 execution and integer cost accounting. It excludes dataset loading, placement generation, independent verification, result writing and optional text-IR emission. Host: Intel Core i9-9880H (8 physical / 16 logical CPUs), macOS 26.6.2, Python 3.11.13, NumPy 2.4.6; one Python interpreter, with possible concurrent work on the host. It is a measured single-run host duration, not a stable hardware-independent score.

The emitted 11,171,400-instruction text was also replayed through the parser and interpreter. The compressed 11.1 MB trace is generated on demand rather than duplicated in Git; its expanded SHA-256 is `f6f657cc159160050e73d1df8ae169b313b795f3d1b1a45a72ad4f757383ef2b`. The generator, a sample, placement/access CSV, and replay result are published. Self-tests cover the specification's exact native-unit 450 fJ/300 ps example, distant free tape I/O, read-before-write rejection, source/destination aliasing and nearest-neighbor ties.

## Measured A100 runtime and NVML energy

Hardware: **NVIDIA A100-SXM4-40GB**, 39.5 GiB visible memory, compute capability 8.0, 108 SMs, power limit 400 W. Driver 580.95.05; CUDA runtime 12.4; PyTorch 2.5.1+cu124; Triton 3.1.0; NumPy 2.2.6; NVML 13.580.95.05; `nvidia-ml-py` 12.560.30. The pinned base image digest and complete version/hardware record are saved in `gpu_results.json`.

One Triton kernel copies all training pixels and labels into preallocated model memory. A second uses one program per query, compares all 600 candidates, accumulates the nine FP32 squared differences in feature order, and reduces by distance then lowest training index. The graph contains 128 complete task invocations, each with its own memorization copy. All buffers are reused; input and working data remain on the GPU.

After compilation, correctness checks, warmup and graph capture, three active trials repeatedly execute this graph. CUDA events measure device elapsed time; synchronized host wall throughput is also retained. NVML's `nvmlDeviceGetTotalEnergyConsumption` supplies cumulative millijoule counters around the active window and a 2000 ms idle interval on each side. Convert the cumulative-counter result to millijoules per task:

```text
P_idle_W = (P_idle_before_W + P_idle_after_W) / 2
E_task_mJ = ((counter_after_mJ - counter_before_mJ)
             - P_idle_W * active_ms)
            / number_of_complete_task_invocations
```

Counter-read timestamps use the midpoint of the host call, and query latencies are retained. Negative adjusted values are not silently clipped. Reported aggregate values are the means over the three trials.

| Trial | Full tasks | Active NVML window (ms) | Raw board energy (mJ) | Idle before / after (W) | CUDA time/task (ms) | Adjusted energy/task (mJ) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 340,480 | 2400 | 3.6 × 10⁵ | 81 / 83 | 0.0070 | 0.48 |
| 2 | 340,480 | 2400 | 3.6 × 10⁵ | 64 / 80 | 0.0070 | 0.55 |
| 3 | 340,480 | 2400 | 3.5 × 10⁵ | 64 / 82 | 0.0069 | 0.53 |

Mean host wall throughput is **0.0069 ms/task**. Across trials, CUDA-event time ranges **0.0069–0.0070 ms**, and idle-adjusted energy ranges **0.48–0.55 mJ/task**. Using either the before-only or after-only idle baseline across these trials yields **0.47–0.60 mJ/task**. This is baseline sensitivity, not a confidence interval. Idle power drifts materially; energy deserves fewer significant figures than the raw counters provide.

The active trial target was 3000 ms; the actual windows in the table are authoritative. Observed throughput differed from calibration, producing shorter active windows; the cause was not measured. No GPU clock or power limit was changed by the benchmark. These are warm, repeated, cache-resident throughput measurements. Host/device transfers, allocation, compilation, graph capture, validation, startup, host energy and facility overhead are excluded. NVML measures GPU board energy, not individual instructions. Comparing its energy directly with the model's tape-excluded scratch score does not establish physical prediction accuracy of the model.

GPU outputs agree with all **600** canonical CPU predictions, all **600** nearest indices and all **600** winning FP32 distance bit patterns. Additional checks use fresh random queries and changed training labels to verify runtime input dependence. Changing training labels changes all 600 corresponding predicted labels. The saved PTX has no FP32 FMA.

Three preliminary remote launches failed before energy trials: a missing C compiler, a Triton language alias scoping requirement, and unavailable Modal package metadata. These were environment/reporting fixes; none changed the algorithm or selected among measured trials. Failure excerpts and successful raw logs are retained in `gpu_run_history.txt`.

## Reproduction

From a clone of the submission branch (or after it is merged), use Python 3.11 or newer. Modal GPU execution requires an already configured Modal account and incurs the account's normal compute charges.

```bash
python3 -m venv mnist/.venv
mnist/.venv/bin/python -m pip install -r mnist/submissions/1nn-v4-20260911/requirements.txt
mnist/.venv/bin/python -m mnist.code.data --output mnist/data --seed 20260910
S=mnist/submissions/1nn-v4-20260911
mnist/.venv/bin/python "$S/learner.py" --data mnist/data/small.npz --output /tmp/mnist-1nn-cpu
mnist/.venv/bin/python -m mnist.code.evaluate --tier small --data-dir mnist/data \
  --predictions /tmp/mnist-1nn-cpu/predictions.npy --output /tmp/mnist-1nn-cpu/accuracy.json
mnist/.venv/bin/python "$S/verify_learner.py" --data mnist/data/small.npz
mnist/.venv/bin/python "$S/score_v4.py" --self-test --data mnist/data/small.npz \
  --output /tmp/mnist-1nn-model --emit-ir /tmp/mnist-1nn-model/program.v4.gz
mnist/.venv/bin/python "$S/score_v4.py" --data mnist/data/small.npz \
  --replay-ir /tmp/mnist-1nn-model/program.v4.gz --output /tmp/mnist-1nn-replay
uvx --with numpy==2.2.6 modal==1.5.5 run "$S/gpu_benchmark.py" \
  --data mnist/data/small.npz --output /tmp/mnist-1nn-gpu
```

The generator's exact model scores and predictions should match; wall-clock score time and measured GPU energy/runtime vary with the host and GPU state. Canonical array hashes must match before claiming the same dataset. The full trace takes 220,146,130 bytes uncompressed (11.1 MB gzip); it can be generated without writing uncompressed text.

To render these saved results and export an available local session log:

```bash
mnist/.venv/bin/python -m pip install -r "$S/requirements-report.txt"
mnist/.venv/bin/python "$S/export_session.py" --input /path/to/rollout.jsonl --output "$S"
mnist/.venv/bin/python "$S/write_report.py"
mnist/.venv/bin/python "$S/build_pages.py" --output docs/submissions/1nn-v4-20260911
```

Session export is optional for reproducing the benchmark. It includes visible user/assistant messages through its stated cutoff. Internal reasoning, system/developer prompts and raw unfiltered tool logs are excluded. The [human-readable session export](session.html) and [separate audit](ambiguities.html) are independent documents.

## Accuracy by class

| Digit | Correct / total | Accuracy |
| --- | ---: | ---: |
| 0 | 51 / 65 | 78% |
| 1 | 46 / 67 | 69% |
| 2 | 30 / 63 | 48% |
| 3 | 27 / 58 | 47% |
| 4 | 38 / 56 | 68% |
| 5 | 13 / 49 | 27% |
| 6 | 34 / 67 | 51% |
| 7 | 20 / 54 | 37% |
| 8 | 16 / 61 | 26% |
| 9 | 33 / 60 | 55% |

The complete true-row/predicted-column confusion matrix is retained in the accuracy JSON.

## Artifacts

- [Source and reproduction overview](README.md), [fixed CPU learner](learner.py), [independent learner checks](verify_learner.py).
- [Full v4 generator/interpreter](score_v4.py), [model score](model-score.json), [replay score](replay-score.json), [placement and access counts](placement.csv), [scoring conventions](scoring-notes.md).
- [A100 Triton implementation](gpu_benchmark.py), [raw measurements and versions](gpu_results.json), [compiled PTX](gpu_predict.ptx), [run history](gpu_run_history.txt).
- [Accuracy and confusion matrix](accuracy.json), [CPU results and selection rows](cpu_results.json), [predictions](predictions.npy), [dataset manifest](dataset_manifest.json), [all-six-array verification](dataset_verification.json).
- [Conversation Markdown](session.md), [conversation JSON](session.json), [audit Markdown](ambiguities.md), [independent learner review](learner_review.md).
