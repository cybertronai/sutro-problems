# A100 implementation and measurements

The fixed H32 / 300-epoch / learning-rate-0.2 / seed-101 learner produced the
same **374/600 (62%)** predictions on an A100-40GB as the independent CPU
submission reference. The measured median complete-task time is **71 ms**,
with **2,000 mJ** of idle-adjusted NVML energy per complete task.

## What executes on every timed task

Every CUDA-graph replay starts from the raw GPU-resident training images,
training labels, test images, and fixed seed-dependent initialization literals.
It transforms all train/test pixels as `x * 4 - 0.5`, constructs all one-hot
targets, resets all 650 parameters, executes all 300 training epochs, and
predicts all 600 test labels. It does not reuse a trained model from an earlier
task. The initial literals depend only on the frozen architecture and seed.

Each minibatch has four kernels: hidden activations; output errors; hidden
errors using the old output weights; and independent parameter updates from
saved activations/errors. The last kernel streams each gradient in ascending
batch-row order. A task has 6,000 minibatches and 24,004 kernel launches in its
captured graph, including input preparation and inference. This is a direct
implementation of the fixed algorithm; further launch/occupancy optimization
has not been attempted.

FP32 multiply and add are separately rounded, reductions follow the CPU
reference's order, and inference chooses the first class on an exact tie.
Fusion is disabled. The eight exported PTX files contain no FP32 FMA or
flush-to-zero instructions. Inference also materializes all 6,000 output
scores, so that extra output work is included in the measured task.

## Measurement boundaries

Reported execution covers GPU-resident, steady-state CUDA-graph throughput.
Host/device transfer, allocations, JIT compilation, graph capture, validation,
and container startup are outside the interval. Initial parameter reset, all
training, input transformation, one-hot construction, and inference are
inside it. CPU reference validation is separate, and host CPU energy is not
part of NVML's whole-GPU-board energy measurement.

Three trials each replayed the complete task 122 times. Calibration targeted
10,000 ms of active execution; the actual windows were approximately 8,700 ms.
CUDA-event and host-wall times agree closely. Each active window was bracketed
by two 3,000 ms idle measurements, with a 3,000 ms settling gap before each
idle sample. Adjusted energy is the active cumulative-counter delta minus
the average of the paired idle powers multiplied by the active interval.

| Quantity | Accuracy | Result |
| --- | ---: | ---: |
| Median complete-task A100 time | 62% (374/600) | 71 ms |
| Median idle-adjusted energy | 62% (374/600) | 2,000 mJ |
| Trial range, idle-adjusted energy | 62% (374/600) | 1,900–2,100 mJ |
| Before-only / after-only idle sensitivity envelope | 62% (374/600) | 1,900–2,200 mJ |
| Median unadjusted board energy | 62% (374/600) | 6,900 mJ |

The third trial's paired idle means differ by 2.3 W. The raw trial and
baseline-sensitivity results are retained rather than implying exact
per-kernel energy metering. These are measurements of this port and this
GPU-resident execution boundary, not a claim of an optimal A100 implementation.

## Independent verification and run history

All 650 final parameter words, all 6,000 output-score words, and all 600
predictions match the independently generated CPU submission artifacts.
The same checks pass after all three timed trials have completed. Separate changed
query and changed training-label tests also match an ordered-FP32 CPU
reference. Only `train_images`, `train_labels`, and `test_images` are loaded
and transmitted; the GPU benchmark never reads `test_labels`.

The first attempt passed numerical checks but had a CUDA timing-conversion
error, detected because CUDA and wall times disagreed. It is preserved and
explicitly excluded from the primary results. The corrected attempt added a
CUDA/wall unit-consistency assertion, longer active windows, settled idle
samples, canonical input-hash checks, and full post-timing numerical checks.
The algorithm, hyperparameters, and seed remained fixed. Only the corrected
attempt supplies submission metrics.

## Reproduce

From the repository root, with a configured Modal account:

```bash
uvx --with numpy==2.2.6 modal==1.5.5 run \
  mnist/submissions/mlp60-affine-20260911/gpu_benchmark.py \
  --data mnist/data/small.npz --output /tmp/mnist-mlp60-gpu
```

The pinned container image and full hardware/software metadata are recorded
in the result. Reproduction writes a fresh result, all eight PTX files, and
the 600 prediction labels into the requested directory.

[Benchmark source](gpu_benchmark.py) · [Primary raw results](gpu_results.json) ·
[Independent verification](gpu_validation.json) · [Run history](gpu_run_history.txt)
