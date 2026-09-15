# Reversible MNIST: reducing library workspace and measuring cache traffic

September 14, 2026. Follow-up to the frozen 89.28% MNIST-medium submission.

Setting `CUBLAS_WORKSPACE_CONFIG=:16:8` before starting the process reduces the measured fresh preparation-plus-task peak from **78.6875 MiB to 14.9375 MiB**, an **81.02% reduction**, while preserving all **110,000 class predictions** across the eleven qualification draws. This is a library-workspace change: the model, two-epoch training procedure, supplied images, labels, initialization and shuffles are unchanged. It puts the tracked allocation below the A100's 40 MiB L2 capacity. The preferred setting's complete-task profile still measures **5.124 MB of HBM traffic per task** and a **95.45% L2 sector hit rate**; it does not stay entirely in cache. **A material energy improvement has not been demonstrated.**

The current runtime entry uses the qualified `:16:8` default: **89.28% ± 0.36 percentage points**, approximately **1,000 mJ/task** and **39 ms/task** at two significant digits. The unrounded energy mean is 1,018.42 mJ/task. The original submission's source, protocol and measurement records remain historical evidence; the new runtime record references them rather than replacing them.

## What changed and what stayed equal

The preferred runtime setting is `:16:8`; `:4096:8` is the explicit baseline, and `:0:0` is a diagnostic comparison. Every configuration runs in a fresh subprocess with the environment set before importing PyTorch or creating a CUDA context. NVIDIA documents `:16:8` and `:4096:8` as cuBLAS workspace configurations with different memory/performance tradeoffs; the measured allocations below are from this particular PyTorch/CUDA workload, not a general allocation formula.

The unchanged model has two additive reversible coupling blocks over an 82-coordinate input, with two 41-coordinate branches and a ten-class head: **7,554 parameters**. Backward reconstructs earlier core states from the final endpoint and builds one local branch graph at a time. The saved core endpoint is 41 KiB for a batch of 128; the classifier is not reversible. Each fresh task resets weights, gradients and momentum, normalizes all train/query inputs, trains for two epochs with batch size 128, learning rate 0.1 and momentum 0.9, and predicts all 10,000 queries. There is no checkpoint reuse, ensemble or augmentation.

The eleven dataset seeds are 2026091400 through 2026091410; learner seed 11 produces fresh initial weights in every task. Each draw uses disjoint 10,000-example training and 10,000-example evaluation subsets of the same 60,000-example MNIST population. Draws overlap one another. The workspace follow-up uses the already frozen predictions for comparison and does not read evaluation labels or select a new learning procedure.

All eleven `:16:8` prediction arrays match the original arrays byte for byte. Consequently the original verified score remains **98,208/110,000 = 89.28% accuracy**, with **0.359082 percentage-point sample SD** across draws and **10.72% error**, passing the 12% error requirement. Inputs, initial parameters and epoch permutation hashes also match. **Final parameter, momentum, score and aggregate state hashes differ from the baseline under the smaller workspaces.** Identical class decisions therefore do not imply bit-identical floating-point training, and hash differences alone do not quantify numerical error. Fresh repeated runs within each individual workspace configuration are byte-repeatable. The explicit `:4096:8` baseline reproduces the original final-state hashes.

The preferred runtime also passed **212 independent CUDA numerical comparisons** against ordinary eager autograd and SGD on synthetic cases covering both candidate depths. These compare gradients, parameter updates, momentum and captured outputs; the largest recorded absolute difference was **1.91 × 10⁻⁶** under tolerances `atol=2e-5, rtol=3e-3`. Its API and fresh-subprocess CLI produced identical predictions, scores and final parameter/momentum hashes, repeated API runs were exact, and forbidden evaluation-label input and late CUDA initialization were rejected. This validates the implementation under `:16:8`; it does not bound the difference between workspace configurations on full MNIST.

## Measured allocated memory

All measurements begin with zero PyTorch allocated and reserved bytes in a fresh process. The peak covers task preparation, CUDA graph capture and one complete fresh reset/train/predict invocation, before output diagnostics or profiling warmup. It includes model/data storage, graph pools and library allocations tracked by the CUDA allocator; it excludes the driver/context and is not a count of retained neural activations.

| cuBLAS workspace | Peak allocated (MiB) | Live allocated after task (MiB) | Peak reserved (MiB) | Unique named tensor storage (bytes) |
|---|---:|---:|---:|---:|
| `:4096:8` baseline | 78.6875 | 77.3281 | 92 | 13,960,864 |
| **`:16:8` preferred** | **14.9375** | **13.5781** | **28** | **13,960,864** |
| `:0:0` diagnostic | 14.6875 | 13.3281 | 28 | 13,960,864 |

The preferred setting saves exactly **63.75 MiB** at the measured peak; disabling workspace saves a further 0.25 MiB. All eleven preferred-setting qualification runs have the same **15,663,104-byte** peak. The named inventory counts actual underlying tensor storages once, including parameters, gradients, seed copies, momentum, raw and normalized inputs, training labels, permutations, logits and predictions. Its total is unchanged. Live allocator bytes without a named tensor reference fall from 67,123,552 to 276,832; these bytes are reported separately rather than presented as hidden activations.

Allocated, reserved, capacity and traffic answer different questions. Reserved memory is the allocator's pool capacity. A 14.94 MiB allocation can fit within a nominal 40 MiB capacity budget, but cache lines can still be fetched, evicted or written back. Conversely, a 78.69 MiB allocation need not cause 78.69 MiB of traffic on each task: allocated library workspace is not necessarily all accessed. The following hardware counters test actual traffic during the selected range.

## Nsight Compute hardware traffic

All four profiles use **draw zero** and passed independent counter, source, prediction, scope and storage verification. Each retained range required **two profiler passes**. MB and GB below are decimal (10⁶ and 10⁹ bytes); MiB in the allocation table is binary (2²⁰ bytes). L2 request traffic is the sum of read and write request sectors multiplied by 32 bytes. Complete-task values are normalized per fresh task; training-only values describe one two-epoch sequence.

| Profile scope | Workspace | Actual A100 model | HBM read (MB) | HBM write (MB) | HBM total (MB) | L2 request traffic (GB) | L2 sector hit rate |
|---|---|---|---:|---:|---:|---:|---:|
| Complete task | `:4096:8` | SXM4-80GB | 3.081 | 1.833 | 4.914 | 1.423 | 95.49% |
| **Complete task** | **`:16:8`** | **SXM4-40GB** | **3.256** | **1.869** | **5.124** | **1.312** | **95.45%** |
| Training only | `:4096:8` | SXM4-40GB | 2.266 | 1.897 | 4.163 | 1.246 | 95.29% |
| Training only | `:16:8` | SXM4-40GB | 2.036 | 1.596 | 3.632 | 1.130 | 97.58% |

Every profile reports an actual 41,943,040-byte L2 cache. The full-task baseline ran on an **80GB A100**, while the preferred-setting full-task profile ran on a **40GB A100**. This hardware difference prevents a controlled attribution of their HBM difference to workspace size alone. The two training-only profiles both used the 40GB model and observed lower HBM traffic with the smaller workspace, but each is a single two-pass range measurement. No repeat-run uncertainty is available for these counters.

The direct conclusion is that the task is served heavily by cache while still generating HBM reads and writes. The **1.312 GB of L2 requests** and **5.124 MB of HBM traffic** are counts at different interfaces; their ratio is **not** the L2 hit rate. The independently reported hit-rate metric is **95.45%**. The larger baseline allocation also has a high measured hit rate, showing why allocation exceeding L2 capacity does not by itself establish cache thrashing. These results do not demonstrate an HBM or energy win for the complete task.

The frozen profiler plan uses Nsight Compute 2025.1.1 with `--replay-mode app-range --cache-control none --clock-control none` and CUDA Profiler Start/Stop markers. There is a preparation/first-run check and two further complete warmups before the marked range. Application-range replay measures the whole range and reruns the application when multiple collection passes are needed. Cache flushing and clock locking by the profiler are disabled; these results describe that warmed, uncontrolled-clock measurement context.

Two scopes are kept separate:

- **Complete task:** sixteen fresh reset/normalize/train/predict invocations inside the range. Reported byte counters are divided by sixteen. Every invocation performs the original two training epochs from freshly reset weights and momentum.
- **Training only:** reset and normalization occur once before the range; one original two-epoch sequence, including both resident epoch-order copies, occurs inside it. Prediction occurs after profiler stop solely to compare outputs. This is 158 SGD updates, including the short final batch in each epoch; it does not lengthen training to manufacture a larger profile.

Both scopes synchronize before profiler stop and exclude output serialization, CPU reconstruction validation, allocation, capture, transfers and source preprocessing. The counters are `dram__bytes_read.sum`, `dram__bytes_write.sum`, `lts__t_sectors_op_read.sum`, `lts__t_sectors_op_write.sum`, and `lts__t_sector_hit_rate.pct`. DRAM read/write bytes describe HBM traffic. L2 read/write request sectors are converted at 32 bytes per sector and reported separately; they are not HBM bytes. The L2 sector hit percentage is the profiler's aggregate metric, not a guarantee that every load hits cache.

## Energy and runtime: no demonstrated energy win

These are separate, unprofiled measurements of complete GPU-resident fresh tasks on draw zero. Each configuration has three paired-idle NVML trials on an A100-SXM4-40GB. Values are mean ± sample SD across the three trials.

| cuBLAS workspace | Idle-adjusted GPU energy (mJ/task) | CUDA time (ms/task) | Fresh invocations per trial |
|---|---:|---:|---:|
| `:4096:8` | 1,002.98 ± 55.63 | 37.910 ± 0.161 | 104 |
| `:16:8` | 1,018.42 ± 47.32 | 38.859 ± 0.154 | 108 |
| `:0:0` | 994.92 ± 5.53 | 37.375 ± 0.041 | 124 |

The preferred setting's mean energy is 1.54% higher and mean CUDA time 2.51% higher than this run's baseline. Each configuration used a different physical board, with only three trials per configuration; these measurements do not establish a meaningful energy difference. The clear benefit is allocation reduction. These figures do not replace the original submission's separately measured 1,101.24 mJ/task result.

Each trial brackets roughly 3.9–4.6 seconds of repeated fresh tasks with a three-second idle measurement, preceded by three seconds of settling, on each side. Energy is the NVML cumulative board-energy difference minus the mean bracketing idle power multiplied by active duration, divided by the number of tasks. The CUDA device's PCI identity selects the NVML handle. Predictions and final state are checked before and after measurements within each configuration. GPU-resident energy excludes CPU preprocessing, transfer, allocation, graph capture, compilation, cold start and verification; host energy and complete application energy remain unmeasured.

## Reproduction

Run in a fresh copy so the retained plans and evidence are never overwritten. From the repository root, with the existing local Python/Modal environment and raw MNIST directory available:

```bash
CACHE_COPY=$(mktemp -d /tmp/rev88-cache-reproduction.XXXXXX)
cp -R mnist/submissions/rev88-20260914 "$CACHE_COPY/submission"
rm -f "$CACHE_COPY/submission/cache-audit/"profile-*.gz
modal run "$CACHE_COPY/submission/cache-audit/run.py" --phase compare --raw-dir "$PWD/mnist/data/raw"
modal run "$CACHE_COPY/submission/cache-audit/run.py" --phase qualify --raw-dir "$PWD/mnist/data/raw"
modal run "$CACHE_COPY/submission/cache-audit/run.py" --phase profile --raw-dir "$PWD/mnist/data/raw"
modal run "$CACHE_COPY/submission/cache-audit/validate_runtime.py"
python "$CACHE_COPY/submission/cache-audit/analyze.py"
python "$CACHE_COPY/submission/cache-audit/build_submission.py"
```

The removal command clears only copied compressed profile archives, preventing old compressed evidence from conflicting with new uncompressed outputs. The originals are untouched. The runner verifies the four original frozen source hashes before execution and launches each fixture in a fresh subprocess with its workspace setting. Its image and Nsight Compute package checksum are retained in the plan. Each comparison, qualification or profiling app allows at most three GPU workers; the overlapping workflow was bounded by nine, and the separate runtime validation allows one. All seven task apps were verified stopped, with zero active workers. The original learner, energy function, accuracy results and qualification protocol are preserved unchanged. Retained development attempts are distinguished from the successful final evidence.

`cache_runtime.py` is the current standalone entry point and also exports `prepare_task` and `train_predict`. It selects `:16:8` before importing the frozen learner, rejects import after CUDA initialization, and guards against changing the setting afterward. In the fresh copied directory, a separate CUDA process can run it as follows; the supplied NPZ must contain exactly `train_images`, `train_labels` and `test_images`, with no evaluation labels:

```bash
python "$CACHE_COPY/submission/cache-audit/cache_runtime.py" \
  --input /path/to/allowed.npz \
  --config "$CACHE_COPY/submission/config.json" \
  --output "$CACHE_COPY/runtime-result.json"
```

It emits adjacent JSON metadata and `.predictions.npy` files. The one-worker runtime validation can be repeated in the same copied submission with `modal run "$CACHE_COPY/submission/cache-audit/validate_runtime.py"`. `build_submission.py` generates the current machine-readable entry from verified evidence. Large raw profiler JSON and binary Ncu reports are retained with lossless gzip compression; the analyzer reads compressed evidence directly.

## Primary documentation and evidence

NVIDIA's [cuBLAS 12.4.1 reproducibility documentation](https://docs.nvidia.com/cuda/archive/12.4.1/cublas/index.html#results-reproducibility) describes the workspace configurations. The [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) explains application-range replay, cache/clock control and memory-sector metrics. PyTorch documents the [peak allocated-memory metric](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html). These references explain the mechanisms; all numeric experiment results come from the retained local evidence.

[Current submission record](submission.json) · [Submission builder](build_submission.py) · [Independent summary](summary.json) · [Independent analyzer](analyze.py) · [Execution and shutdown](execution.json)

[Workspace plan](plan.json) · [Profiler plan/amendment](profile-plan.json) · [Fixture](fixture.py) · [Runner](run.py) · [Current runtime entry point](cache_runtime.py) · [Runtime validator](validate_runtime.py) · [Passed CUDA/API/CLI validation](cuda-validation.json)

[Baseline measurement](baseline.json) · [Small-workspace measurement](small-workspace.json) · [Zero-workspace measurement](zero-workspace.json) · [First qualification draw](qualify-00.json) · [Last qualification draw](qualify-10.json) · [Original submission](../README.md) · [Original verified accuracy](../accuracy.json)

[Full-task baseline counters](profile-baseline-task.json.gz) · [Full-task small-workspace counters](profile-small-task.json.gz) · [Training baseline counters](profile-baseline-training.json.gz) · [Training small-workspace counters](profile-small-training.json.gz)

[Full-task baseline Ncu report](profile-baseline-task-profile.ncu-rep.gz) · [Full-task small-workspace Ncu report](profile-small-task-profile.ncu-rep.gz) · [Training baseline Ncu report](profile-baseline-training-profile.ncu-rep.gz) · [Training small-workspace Ncu report](profile-small-training-profile.ncu-rep.gz)
