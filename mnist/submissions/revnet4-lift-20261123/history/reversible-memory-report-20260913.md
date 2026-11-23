# Actual reconstruction backward: memory and accuracy

## Accuracy with actual reconstruction backward

The selected four-block lossless ReLU-lift RevNet was trained again using reconstruction backward and the same original ordered FP32 procedure: eight epochs, batch 128, SGD .03 / momentum .9 / decay .0001, and three fresh members (seeds 11/22/33). All eleven new prediction sets were frozen before scoring. Inputs, initialization and augmentation/shuffle hashes match the earlier stored-network runs exactly.

| Implementation | Accuracy, mean ± sample SD | Paired gap vs CNN | Paired gap vs stored | Mean draw time |
|---|---:|---:|---:|---:|
| Original CNN | 97.7036% ± 0.2076 pp | +0.0000 pp | +0.2991 pp | 13.1 s |
| Selected RevNet, stored activations | 97.4045% ± 0.2272 pp | -0.2991 pp | +0.0000 pp | 37.2 s |
| Same RevNet, actual reconstruction | 97.4236% ± 0.2454 pp | -0.2800 pp | +0.0191 pp | 35.5 s |

Actual reconstruction scores **97.42%**, a **0.28 percentage-point** gap from the original CNN. The paired-gap sample SD is 0.1788 pp. Its mean is +0.0191 pp relative to stored reversible training, with 322 changed predictions out of 110,000. The gap is within the previously chosen descriptive 0.3 pp band; this is not a statistical equivalence result.

These are the same eleven confirmation draws (seeds 20261112–20261122), each with 10,000 training and 10,000 disjoint evaluation examples from MNIST's 60,000-image training population. Draws overlap one another. The CNN and stored-RevNet columns reuse the exact frozen confirmation predictions and are independently rescored here; this backward implementation check is not another independent confirmation or an untouched external test set. Subtractive FP32 reconstruction can change backward rounding, so accuracy is measured rather than assumed identical.

The RevNet has 351,905 parameters and 1,848,647 nominal forward MACs per image per member, versus 351,914 and 1,849,376 for the CNN. Draw timings cover three members' GPU invocation, including reset, training and prediction; they exclude preparation, compilation and diagnostics. They describe these implementations on the A100, not an architecture-independent compute cost.

[Ordered accuracy summary](ordered-summary.json) · [Per-draw results](ordered-results.csv) · [Independent verification](ordered-verification.json) · [Prediction freeze](ordered-prediction-freeze.json)

## Whole training procedure: selected network memory

One fresh seed-11 member of each control was rerun on confirmation draw 0 (10,000 training + 10,000 evaluation images), including all eight epochs, data, query tensors, precomputed augmentation schedules, initialization copies and CUDA graph preparation. Final logits, parameters and momentum match that exact member's frozen accuracy run byte for byte. Evaluation labels were not supplied.

| Implementation | End-to-end allocated peak MiB | Invocation peak MiB | Schedules MiB | Training workspace MiB | Query workspace MiB | Invoke seconds |
|---|---:|---:|---:|---:|---:|---:|
| Original CNN | 251.81 | 251.81 | 198.06 | 28.12 | 13.06 | 4.42 |
| Selected RevNet, stored | 342.31 | 342.31 | 198.06 | 99.22 | 32.88 | 12.42 |
| Selected RevNet, reconstructed | 242.28 | 242.28 | 198.06 | 22.30 | 9.78 | 11.81 |

For the complete procedure, reconstruction reduces the selected RevNet's peak by 29.2% relative to its stored-DAG implementation and by 3.8% relative to the original CNN. The large, shared precomputed schedules remain resident in both implementations, so full-procedure savings are smaller than the isolated deep-network workspace ratio. The peak is measured for one member at a time; the accuracy ensemble trains its three members sequentially, so these peaks should not be multiplied by three. Preparation and invocation peaks precede serialization and diagnostics; driver/context memory outside the CUDA tensor allocator is excluded.

[Whole-procedure profiles](full-memory-profiles.json) · [Independent member/output verification](full-memory-verification.json) · [Frozen whole-procedure protocol](full-memory-protocol.json)

## Actual reconstruction with the original ordered backend

At batch 128 and 32 coupling blocks, the original stored-DAG implementation uses **556.48 MiB** peak versus **27.16 MiB** for reconstruction. Reconstruction's actual activation/workspace storage is exactly **19.823 MiB** at every depth; its raw peak still grows from 25.20 to 27.16 MiB as parameter state grows.

This is the original ordered FP32 backend, including forward, backward and SGD. The comparator is the earlier stored-DAG implementation, which preallocates forward activations and per-node backward/padding workspaces. The new implementation reconstructs activations and reuses branch scratch. Its savings therefore include both reconstruction and workspace reuse; the ratio is not a universal advantage over all conventional implementations.

![Ordered backend memory, batch 128](ordered-memory-scaling-b128.png)

| Blocks | Stored raw peak MiB | Reconstructed raw peak MiB | Stored workspace MiB | Reconstructed workspace MiB |
|---:|---:|---:|---:|---:|
| 4 | 94.26 | 25.20 | 88.197 | 19.823 |
| 8 | 160.29 | 25.48 | 153.947 | 19.823 |
| 16 | 292.35 | 26.04 | 285.447 | 19.823 |
| 32 | 556.48 | 27.16 | 548.447 | 19.823 |

For the selected four-block architecture (α=.25), this single-batch ordered profile uses **94.26 MiB stored / 25.20 MiB reconstructed**; the original three-convolution CNN uses 30.37 MiB. The frozen CNN profile incorrectly marks the input-image gradient as computed; the original CNN skips that gradient. Both matched reversible implementations compute it, so their comparison is unaffected.

These profiles include initialization copies, parameters, parameter gradients, momentum, boundary caches and reusable scratch. They use one synthetic batch, one warmup and three synchronized updates. Full datasets, precomputed augmentation schedules and CUDA graph capture are excluded here and measured separately during the accuracy run. Steady-step excess above the live baseline is zero because the kernels use preallocated buffers; raw total peaks and the explicit workspace inventory remain the relevant measurements. Workspace bytes include padding and backward scratch, not only forward-retained activations.

[Ordered profile verification](ordered-profile-verification.json) · [Raw ordered profiles](ordered-profiles.json) · [Ordered numerical validation](ordered-validation.json)

## PyTorch cross-check with ordinary tensor lifetimes

The following independent PyTorch comparison isolates stored versus reconstructed autograd with ordinary tensor lifetimes. It avoids attributing all of the ordered implementation's workspace-reuse savings to reversibility alone.

At batch 128 and 32 coupling blocks, actual CUDA peak is **123.45 MiB stored versus 89.83 MiB reconstructed**. Excess above resident baseline is 53.90 versus 20.29 MiB. The reconstructed full-model forward activation retention remains exactly 2.599 MiB from 4 through 32 blocks, while its total peak changes from 87.86 to 89.83 MiB.

These measurements execute the custom reconstruction backward through a full training update. The matched stored/reconstructed pair has identical topology, initial weights, training inputs, optimizer and resident gradient/momentum buffers. The ordinary CNN is a secondary depth comparison: it uses full-width 32-channel convolutions, so it has more parameters than the coupling branches at the same number of convolutions.

## Batch 128: primary memory comparison

![Primary memory scaling](memory-scaling-b128.png)

| Blocks | Stored total MiB | Reconstructed total MiB | Stored excess MiB | Reconstructed excess MiB | Stored retained MiB | Reconstructed retained MiB | Reconstruction / stored time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 86.54 | 87.86 | 18.47 | 19.79 | 7.661 | 2.599 | 1.26× |
| 8 | 91.81 | 88.15 | 23.53 | 19.86 | 12.724 | 2.599 | 1.26× |
| 16 | 102.36 | 88.71 | 33.65 | 20.01 | 22.849 | 2.599 | 1.21× |
| 32 | 123.45 | 89.83 | 53.90 | 20.29 | 43.099 | 2.599 | 1.32× |

Blocks 4/8/16/32 contain 8/16/32/64 coupling-branch convolutions, respectively. Both reversible implementations retain the original input in the first channel of a learned lossless ReLU lift. The dense classifier head is unchanged. The injective backbone is invertible on its image; the ten-class readout is not invertible.

## Selected architecture and original CNN anchors

| Model | Parameters | Total peak MiB | Excess peak MiB | Forward retained MiB | Median step ms |
|---|---:|---:|---:|---:|---:|
| Selected 4-block RevNet, α=.25, stored | 351,905 | 86.54 | 18.47 | 7.661 | 6.23 |
| Same selected RevNet, reconstructed | 351,905 | 87.86 | 19.79 | 2.599 | 8.03 |
| Original three-convolution CNN | 351,914 | 83.44 | 15.37 | 3.904 | 3.87 |

The shallow PyTorch reconstruction anchor uses 4.425 MiB more peak memory than the original CNN and 1.327 MiB more than its matched stored counterpart. These anchors are memory controls using standard PyTorch cross entropy; their timings and memory should not be substituted for the original ordered training procedure. The earlier 97.4045% fresh-draw result trained the reversible architecture while storing its activation DAG. Actual reconstruction accuracy is measured separately with the original ordered kernels, eight epochs and three fresh members.

## What is being measured

Each profile uses the first 128 or 512 training examples from the frozen original MNIST draw. No evaluation labels are used. After two warmup updates initialize momentum and parameter gradients, five synchronized forward + cross entropy + backward + SGD updates are measured, with gradient buffers retained. These five timings are repeated steps in one run, not five independent datasets. Tensor-saving hooks are disabled during the peak measurements and audited separately on the same profiling batch.

Total peak uses the CUDA tensor allocator's `max_memory_allocated`, reset immediately before each full update. It includes model state, input data, activations, temporary parameter gradients and library workspaces. The excess subtracts that update's resident allocated baseline; it is not a pure activation count. Reserved cache memory and `nvidia-smi` process memory are different metrics. The actual run is pinned to PyTorch 2.5.1; current documentation is cited only to explain these metrics. [Peak allocated bytes](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_allocated.html), [CUDA memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html).

Forward-retained activations count unique saved floating-point storage, excluding references to parameter storage. The reconstruction core saves exactly one final B×32×9×9 feature state and references to the weights, then recovers prior states and recomputes each branch during backward. This exact retention is constant with depth. Parameters, parameter gradients, momentum, temporary gradient returns and workspaces still consume memory, so total training memory is not constant. Three resident FP32 copies of parameter storage alone grow from 4.027 MiB at four blocks to 5.504 MiB at 32 blocks.

The independent audit rechecked 1,555 numerical comparisons, including logits, input and parameter gradients, and three SGD updates of parameters and momentum against stored backpropagation. An FP64 CPU control uses three small blocks; FP32 CUDA controls cover four through 32 blocks. Subtractive FP32 reconstruction has rounding error: validation uses explicit normwise tolerances and does not claim bit-identical gradients. Hidden tensor attributes and persistent activation buffers are checked; higher-order gradients are unsupported.

The first profiling attempt was preserved after its cleanup gate exposed 64 MiB of persistent cuBLAS workspace. Every revised profile begins and ends at zero normalized allocated bytes, finds zero retained Python CUDA tensor storages after model deletion, and verifies that the dedicated cuBLAS workspace-clear call releases exactly 64 MiB. The revised instrumentation clears those caches only before model construction and after all measurements. No architecture or training setting was changed. The charts and tables include the full 64 MiB in raw total peaks; they do not treat library workspace as an activation saving.

Across the measured depths at batch 128, each additional block adds 1.318359 MiB to stored raw peak versus 0.070313 MiB to reconstructed raw peak. Exact forward-retained activation storage adds 1.265625 MiB per block when stored and zero when reconstructed. At batch 512 the corresponding raw-peak slopes are 5.115234 and 0.070313 MiB per block, with retained-activation slopes 5.062500 and zero. The shared 64 MiB library workspace changes intercepts, not these slopes.

## Batch 512: secondary stress comparison

![Secondary memory scaling](memory-scaling-b512.png)

The larger batch is a memory-scaling check. Batch 128 remains the accuracy procedure and primary comparison.

## Ordered backend, batch 512

![Ordered backend memory, batch 512](ordered-memory-scaling-b512.png)

## Sources and retained evidence

The additive-coupling reconstruction method follows [Gomez et al., The Reversible Residual Network (2017)](https://arxiv.org/abs/1707.04585). The experiments here independently measure this small-image implementation; they do not inherit the paper's memory or accuracy results.

[Memory summary](summary.json) · [Per-profile CSV](profiles.csv) · [Independent verification](verification.json) · [Frozen profiles](profile-freeze.json) · [V2 protocol](memory-protocol-v2.json) · [Numerical validation](validation.json)

[Model and custom backward](models.py) · [Profiler](profile_gpu_v2.py) · [Audit and report](report.py) · [Original profiling attempt](memory-protocol.json) · [Architecture/accuracy study](../reversible-networks/README.md)

[Batch 128 PNG](memory-scaling-b128.png) · [Batch 128 SVG](memory-scaling-b128.svg) · [Batch 512 PNG](memory-scaling-b512.png) · [Batch 512 SVG](memory-scaling-b512.svg)
