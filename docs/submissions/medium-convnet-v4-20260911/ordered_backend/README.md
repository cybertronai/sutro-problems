# Ordered FP32 ConvNet backend

The backend implements a fresh bias-free 3×3 padded convolution/ReLU stack,
followed by a biased dense/ReLU/dense head. It supports one-hot squared-error
and approximate-softmax gradients, momentum SGD with coupled weight decay,
seeded affine augmentation, fixed permutation schedules, and complete inference.
No batch normalization, pooling, dropout, or pretrained state is used.

## Stable interface

Add this directory to Python's import path on the pinned Torch/Triton host.

```python
from network import Trainer
trainer = Trainer(train_images, train_labels, config, seed)
trainer.prepare(query_images)  # allocation, constants, compilation, graph capture
trainer.invoke()               # reset, every epoch, every query, predictions
result = trainer.outputs()     # host copies: parameters, velocities, scores, predictions
```

`train_images` and optional `query_images` are contiguous FP32 NCHW arrays with
one square channel; production uses 9×9 images. Labels contain the corresponding
training classes. The interface accepts no evaluation labels or learned weights.
`config` supplies `width`, `depth`, `head_width`, `epochs`, `batch_size`,
`learning_rate`, `loss`, and optionally `momentum` (0.9), `weight_decay` (0.0001),
and `augmentation` (`mild_affine`). Supported loss names are
`half_sum_mse_batch_mean`, `approx_softmax`, and `approx_softmax_gradient`;
the latter two select the same derivative. Other loss names fail explicitly.

For training-only validation, call `prepare()` without queries, then
`initialize()` and `train_epoch(epoch)` in one-based ascending epoch order.
`trainer.network.infer(raw_query_cuda_tensor)` returns FP32 scores without
updating parameters. `trainer.network.state()` returns host parameter arrays.
`trainer.initial_arrays` contains only fresh seed-derived initial literals;
export those arrays for the scalar compiler, never the trained state.
`trainer.schedule_manifests` contains each epoch's order, transform, index, and
coefficient hashes. Graph replay operations are asynchronous; synchronize
before wall timing or read outputs through the provided host-copy methods.

Every `invoke()` resets all parameters, velocities, and schedule position.
Queries use forward-only batches of at most 128 rows, with a shared output-score
buffer. A three-member ensemble must initialize its FP32 sum to positive zero,
add member scores in the declared ascending seed order, and apply strict
ascending-class argmax. Member parameters and workspaces may be reused
sequentially. The caller owns ensemble orchestration and evaluation.

## Numerical contract

- Convolution products accumulate from positive zero in input-channel,
  kernel-row, kernel-column order.
- Input gradients accumulate in output-channel, flipped-kernel-row,
  flipped-kernel-column order, reading `W[co,ci,2-ky,2-kx]`.
- Convolution weight gradients accumulate in sample, output-row, output-column
  order. Dense products and bias sums also use ascending scalar reductions.
- All gradients are complete before any parameter update. The update is
  `d = g + decay*w; v = momentum*v + d; w = w - lr*v`, rounding each operation
  separately to FP32. Initial velocity is positive zero.
- Approximate softmax subtracts the row maximum, clamps to [−16,0], forms
  `1 + x*FP32(1/1024)`, squares ten times, sums classes in ascending order from
  positive zero, performs rounded FP32 division, subtracts one-hot labels, and
  multiplies by `FP32(1/batch_size)`.
- Triton fusion is disabled. Division uses explicit `div.rn.f32` because this
  pinned Triton version lowers `tl.div_rn` to a flush-to-zero variant.
- ReLU uses `x > 0`; exact argmax ties choose the smallest class.

`schedule.py` is shared with scalar instruction expansion. Its initial arrays
reproduce CPU PyTorch constructor RNG draws followed by He-uniform weight
initialization and zero biases. Permutations and affine parameters use separate
PCG64 streams from `SeedSequence([seed,20261202]).spawn(2)`. These schedules are
part of the final declared algorithm and differ from the preliminary CUDA RNG
screen. Accuracy must be revalidated with this backend.

Each affine map contains four raw-image indices and four FP32 coefficient
literals. Invalid neighbours read a positive-zero sentinel. Interpolation uses
`(((p00*c00+p01*c01)+p10*c10)+p11*c11)`, then separate `*4` and `−0.5`.
The scalar compiler charges coefficient materialization, reads, and every
arithmetic operation. No image-dependent values become program constants.

## Preparation and measured scope

`prepare()` materializes every epoch's seed-only maps and orders on the GPU;
this can consume several GB for a large training schedule. It also allocates
workspaces, compiles kernels, and captures graphs. None of these setup costs is
included in the proposed GPU-resident task boundary. `invoke()` includes fresh
parameter and momentum initialization, every augmentation/read/normalization,
every SGD update, final query normalization, all output scores, and predictions.
The root submission's measurement helper owns warmup, repeated complete tasks,
CUDA/wall timing, NVML energy, source guards, and output fingerprints.

## Retained verification

`primitive-results-02` contains exact CPU/GPU primitive checks, independent
FP64 autograd gradient-geometry checks, representative B128/C32/9×9 feasibility
timings, and actual PTX. `network-results-03` retains the tested source snapshots
and complete tiny-network results: two epochs, partial minibatches, both losses,
131 queries through 128+3 batches, all parameters/velocities/scores/predictions,
affine maps, and reset after deliberate state mutation. Its PTX contains no
FP32 FMA, FTZ, or tensor-core instructions. The only later numerical-interface
change accepts the preliminary runner's equivalent loss-name alias and rejects
unknown names; no arithmetic kernel or schedule changed.

Earlier failed runs are retained separately: the initial primitive image lacked
a C compiler, and the first complete-network PTX audit found `div.rn.ftz.f32`.
They are superseded by the successful runs above. These are correctness and
feasibility experiments, not MNIST accuracy results or submission cost scores.
