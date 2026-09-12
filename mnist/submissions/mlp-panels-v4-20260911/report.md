# MNIST-small: energy-aware panel MLP

**Historical specification: 600/600 examples and single-core-with-tape scoring.**

Upstream main now requires 1,000/1,000 examples, 67% mean accuracy and spatial-computer grid scoring. This study was measured under the previous specification and is NOT a current qualifying submission. Its single-core numbers must not be placed in spatial-grid columns. The current-specification metrics remain unmeasured.

## Summary

A fixed 9-32-10 ReLU MLP is optimized through operand reuse, persistent rectangular panels and memory placement across all seven matrix products. Architecture, initialization, ordered FP32 arithmetic, learning rate, batch size and training budget are unchanged. The primary variant saves 14.14% modeled energy and 17.95% measured idle-adjusted A100 energy against the original MLP. It takes 1.31% more modeled time but 20.34% less A100 time.

| Accuracy | Time (ms) | Energy (mJ) | Area (mm2) | Time to score (s) | Time on A100 (ms) | Energy on A100 (mJ) |
|---|---:|---:|---:|---:|---:|---:|
| 65.0% +/- 2.1 pp | 94 | 0.098 | 0.021 | 0.031 | 57 | 1,800 |

Accuracy: **4292/6600**, above the historical requirement of 3960/6600, across eleven fresh predeclared datasets. SD is the sample standard deviation across datasets, not across training seeds. Costs use two significant figures here; exact values remain in JSON.

## Contributors and Prior Work

OpenCode: panel implementation, local search, CPU/IL verification, A100 experiments, submission packaging and documentation. SecurityQQ: research direction and submission. No legal name is inferred from the GitHub handle.

The original learner and measurement protocol are from the repository MLP60 submission (Yaroslav Bulatov and Codex). Scheduling ideas come from the matmul4x4 work of Juraj Selep and the matmul16x16 panel/capture work credited to Cosmin, sjbaebae and SecurityQQ/OpenCode. Their literal v0 scores and schedules are not presented as this submission's v4 measurements. No W&B run was created.

## Frozen Learner

- Tier: MNIST-small only,600 train and 600 test examples,3x3 images.
- Network:9 inputs,32 ReLU hidden units,10 output scores;650 FP32 parameters.
- Initialization: NumPy PCG64 seed 101, uniform bounds1/sqrt(fan-in), zero biases.
- Normalization: float32(x)*4-0.5. Loss: squared error against one-hot targets.
- SGD:300 epochs, batch 30, learning rate0.2, fixed cyclic supplied-row order, no shuffle.
- Fresh parameters for every draw/invocation; no pretrained weights, ensembles or cross-draw state.
- Every output reduction starts at+0 and accumulates in ascending K, with separate FP32 multiply/add.
- Predictions are argmax scores with first-class tie breaking.
- The learner reads only train_images, train_labels and test_images; never test_labels.

## Seven Matrix Products

| Product | Shape | Selected execution |
|---|---|---|
| X @ W1 |30x9 times 9x32|Cache minibatch X; retain 9x4 W1 panels across 30 rows|
| H @ W2 |30x32 times 32x10|Stage one H operand and reuse across ten classes|
| D2 @ old W2.T |30x10 times 10x32|Retain10x4 transposed-W2 panels across 30 rows|
| X.T @ D1 |9x30 times 30x32|Reuse X cache; retain 30x4 D1 panels across nine features|
| H.T @ D2 |32x30 times 30x10|Stage H values across ten classes; stream weight updates|
| Q @ W1 |30-query groups|Reuse query cache and 9x4 W1 panels|
| H_query @ W2 |30-query groups|Reuse hidden operands across ten classes|

All valid results are consumed once. The v4 instruction counts for multiply, add, subtract, compare and select equal the original baseline. The extra work is captures/copies and scratch initialization, traded against fewer expensive operand reads. Transposes are strided views, and no full gradient matrices are materialized.

Shared extra scratch is401 words:10 accumulators, one staged operand, a120-word maximum retained panel and a270-word minibatch/query cache. Total scratch is20,691 words (82,764 bytes), area0.020691 mm2 before rounding. All D1 values consume old W2 before weight updates. D2 storage is reused for inference scores only after training finishes.

## Model Scoring

The selected executable uses the repository's compact affine-v4 representation and the same pinned single-core-with-tape scorer as the original MLP submission. It includes all initialization, input normalization, target construction, training, output work, capture writes and placement shifts. Every expanded leaf is a v4 primitive. recv/send conventions and occupied-cell area follow the inherited scorer.

Energy per scratch access is max(50,2*Manhattan_distance) fJ. Read time is max(250,4*distance) ticks; write time is max(250,2*distance) ticks, each tick0.2 ps. Reads and writes count. Arithmetic operations themselves are not separately charged. Area covers peak initialized scratch only, excluding processor, routing, instruction storage and tape. Canonical600/600 input sizes are used for these costs.

Time to score is the recorded host median of five score(document) calls after a warmup. It includes validation, placement, address histograms, cost summation and hashing, but excludes generation, JSON I/O, search and numerical replay. Fresh reproduction updates timing samples without changing the exact model totals.

Search covered1284 initial distinct configurations and 826 joint-inference refinement evaluations, all local static costs. It tested asymmetric dimensions, staging sides, persistent panels, layouts and inference grouping. The primary objective was full-task energy, then time, then space. This was not a global optimality proof or accuracy search.

## Fresh Eleven-Draw Evaluation

The submission evaluation was performed AFTER freezing the final algorithm and BEFORE examining these eleven test results. It uses new seeds 2026091301 through 2026091311, not the earlier exploratory draw set. The plan stores source/config hashes and a timestamp. All eleven predictions were written and hashed before a separate scoring phase opened test-label slices. No tuning or retraining followed that scoring.

Sampling follows the repository generator: independently permute the original 60,000 training examples for each seed; take600 training examples at offset 0 and 600 test examples at offset 6000. Sampling is without replacement within a draw, train/test indices are disjoint, and different draws may overlap. Training randomness is fixed to seed 101. Pixel preprocessing uses float32 division by 255 and repository box-area resize to3x3, then clipping to[0,1]. Manifests record actual input bytes and source indices.

| Dataset seed | Learner seed | Correct / total | Accuracy |
|---|---:|---:|---:|
| 2026091301 |101| 384/600 | 64.0% |
| 2026091302 |101| 375/600 | 62.5% |
| 2026091303 |101| 384/600 | 64.0% |
| 2026091304 |101| 394/600 | 65.7% |
| 2026091305 |101| 402/600 | 67.0% |
| 2026091306 |101| 402/600 | 67.0% |
| 2026091307 |101| 399/600 | 66.5% |
| 2026091308 |101| 408/600 | 68.0% |
| 2026091309 |101| 381/600 | 63.5% |
| 2026091310 |101| 368/600 | 61.3% |
| 2026091311 |101| 395/600 | 65.8% |

The primary CPU panel learner and the original CPU learner matched parameters, scores and predictions bit-for-bit on every fresh draw. The reported variation is dataset variation, not training-seed variation. All eleven draws are included.

## A100 Protocol and Results

Primary measurements used one NVIDIA A100-SXM4-40GB. The original eight Triton kernel definitions are unchanged. All variants ran freshly in one container with three cyclic rounds: baseline/energy/no_slowdown, energy/no_slowdown/baseline, no_slowdown/baseline/energy. Each occupied each trial position once.

Every graph replay resets the model, normalizes inputs, constructs targets, trains all 300 epochs and writes 600 predictions plus 6000 scores. CUDA runtime graph enumeration confirmed24,004 kernel nodes per variant: four per minibatch plus four setup/inference launches. No extra per-minibatch prefetch launch is hidden.

Time is measured by CUDA events and synchronized wall time. Each active interval targets 10 seconds. Before and after it, allow3 seconds settling and measure idle consumption for3 seconds. Adjusted energy equals the NVML cumulative counter difference minus mean paired idle power times the actual NVML interval, divided by graph replays. Raw counters, temperatures, clocks, throttling and one-sided idle corrections are saved.

| Variant | CUDA time (ms) | Adjusted energy (mJ) | Gross energy (mJ) | CUDA change | Adjusted-energy change |
|---|---:|---:|---:|---:|---:|
| baseline | 72 | 2,200 | 7,500 | +0.00% | +0.00% |
| energy | 57 | 1,800 | 6,000 | -20.34% | -17.95% |
| no_slowdown | 83 | 2,500 | 8,600 | +15.63% | +14.15% |

The energy profile wins time and both energy measures in all three matched rounds. The name no_slowdown refers only to its model-selection constraint; it is slower on A100 and is retained as a negative comparison, not a second headline submission.

GPU kernels use SIMD/register broadcasts for panel reuse. The X cache is real270-word cross-kernel storage, filled inside a single-CTA hidden kernel with a barrier. The energy update kernel computes disjoint W1/W2/bias outputs with no repeated valid results. Candidate CTA grids differ; original baseline grids and arithmetic flags are retained. GPU parallel inference materializes all 600 hidden rows between two launches, unlike the serial model's30-row reuse. Dally distances do not map to GPU virtual addresses, and model area is not GPU memory.

GPU validation covers all 650parameter words,6000 scores and 600 predictions after eager execution, captured replay, each timed round, changed queries and changed training labels before/after timing, and canonical restoration. All match the independent ordered-FP32 CPU reference. PTX has no floating FMA/MAD; SASS was not separately disassembled in this experiment.

Excluded from steady-state metrics: host/device transfers, allocation, JIT, graph capture, validation, startup and CPU energy. These can still incur cloud charges. Board-energy telemetry and idle subtraction have uncertainty; three trials on one device are not a population confidence interval.

## Hardware and Software

```json
{
  "hardware": {
    "name": "NVIDIA A100-SXM4-40GB",
    "uuid": "GPU-dbec2e2a-fce5-d0b5-46fe-b90a6fe0cc97",
    "memory_bytes": 42405855232,
    "sm_count": 108,
    "compute_capability": "8.0",
    "power_limit_w": 400.0,
    "mig_mode": [
      0,
      0
    ],
    "processes": [
      {
        "pid": 1,
        "used_memory_bytes": 979369984
      }
    ]
  },
  "versions": {
    "python": "3.11.15",
    "platform": "Linux-4.19.0-gvisor-x86_64-with-glibc2.41",
    "torch": "2.5.1+cu124",
    "triton": "3.1.0",
    "numpy": "2.2.6",
    "cuda": "12.4",
    "driver": "580.95.05",
    "nvml": "13.580.95.05",
    "image": "ghcr.io/ab-10/wikitext-bench@sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f"
  }
}
```

CPU scoring/evaluation used Python3.14.7, NumPy2.4.1 and macOS15.5 ARM64. The canonical GPU input arrays are recovered from the already tracked input tape and checked against the canonical manifest. This avoids silently accepting platform-dependent resize-bit differences. The tape contains no test labels. Fresh draws retain their own manifests; exact resize-byte reproduction across other BLAS/CPU environments is not guaranteed.

## Verification and Limitations

Actual expanded-IL tests cover full minibatches, multiple batches/epochs, inference group tails and changed inputs/labels. Every parameter, score, prediction and per-address read/write count matches reference execution. A full300-epoch CPU run matches the original after each epoch. The approximately641million-instruction whole IL is scored statically, not fully interpreted. This evidence is not a claim of full instruction-by-instruction execution or official acceptance of the prototype affine representation.

Two initial GPU attempts failed before timing due to Triton compilation restrictions. A preliminary completed run used a shared buffer that misaligned D2 relative to the original allocation. It is preserved but not primary. The final run restored separate aligned buffers and repeated all three variants. The two completed runs used different GPU allocations, so their absolute differences do not isolate alignment effects. The headline uses only the final aligned run, not a pooled or best-of selection.

## Reproduction

The package imports only its own files and existing tracked repository modules. No untracked research directory, generated archive, saved model or external workspace is required by the learner or model generator. Commands are listed in README.md. Data/model caches are generated locally and excluded from the proposed PR; small prediction artifacts and exact measurement evidence are included.

## Artifacts

- [Reproduction instructions](README.md)
- [Requirement checklist and caveats](REVIEW_CHECKLIST.md)
- [Submission verification results](evidence/submission_verification.json)
- [Full-precision headline metrics](metrics.json)
- [Fresh11 accuracy and per-draw counts](evidence/accuracy/accuracy.json)
- [Predeclared evaluation plan](evidence/accuracy/plan.json)
- [Frozen prediction manifest](evidence/accuracy/predictions_frozen.json)
- [Exact theoretical scores](costs.json)
- [Executable optimized IL](optimized.il.json)
- [Primary A100 raw counters and validation](evidence/gpu/results.json)
- [Measured GPU source snapshot](evidence/gpu/runner.py)
- [GPU run history](evidence/gpu_run_history.json)
- [Canonical input provenance](evidence/input_provenance.json)
- [Primary Modal run](https://modal.com/apps/vargapowercouple/main/ap-5pImGSZ90Y9PNI1vVQf9TO)
