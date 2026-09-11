# Higher-accuracy MNIST-small feasibility study

This is an exploratory follow-up to the 308/600 fixed 1NN submission. It uses
the canonical `competition-v2`, seed 20260910, 600-training/600-test dataset.
Only the supplied nine FP32 pixels and the 600 supplied training labels are
used for learning. There are no pretrained weights, extra labeled examples,
cross-tier data, augmentation, or W&B runs.

## Results

All final models use learning rate 0.2. The counts below list the fixed seeds
101, 102, and 103 in that order; each count is out of 600 test examples.
Percentages are rounded to two significant figures; target checks use exact counts.

| Hidden width | Epochs | Validation correct / 120 | Test correct by seed | Mean accuracy ± seed SD |
| ---: | ---: | ---: | :--- | ---: |
| 32 | 100 | 62 | 356, 365, 362 | 60% ± 0.76 pp |
| 32 | 300 | 69 | 374, 371, 377 | 62% ± 0.50 pp |
| 32 | 1,000 | 71 | 383, 405, 376 | 65% ± 2.5 pp |
| 128 | 3,000 | 71 | 384, 388, 385 | 64% ± 0.35 pp |
| 32 | 10,000 | 73 | 381, 389, 387 | 64% ± 0.69 pp |

The highest-validation configuration was width 32 with 10,000 epochs. The
highest observed test mean instead came from 1,000 epochs; that distinction
does not retroactively change which configuration validation selected.

| Accuracy target | Observed feasibility in the frozen study |
| ---: | :--- |
| 55% | All 15 refits pass. |
| 60% | 14 of 15 pass; width 32 with 300 epochs passes in all three seeds. |
| 65% | One of 15 passes: the predeclared width-32, 1,000-epoch seed 102 gives 405/600 (about 68%). Its two companion seeds fail the 65% target. |
| 70% | No tested refit passes. |
| 75% | No tested refit passes. |

The 60% target is supported across several configurations and seeds. The 65%
target is attainable in one observed run but has not been established as
reliable under the fixed refit protocol. This bounded search does not settle
whether 70% or 75% can be reached with a different algorithm. The small
120-example validation set and 39 inspected checkpoints also leave room for
validation selection noise.

## Selection protocol

1. `predeclared_plan.json` records widths 16/32/64, learning rates
   0.01/0.05/0.2, epoch checkpoints 100/300/1000, validation seed 11, and final
   seeds 101/102/103. A single fixed PCG64 permutation with seed 20260912
   reserves 120 of the 600 training rows for validation and uses the other
   480 for candidate training. The exact indices are saved.
2. The first 27 checkpoint configurations reached at most 71/120 validation
   correct. Before any test-label access or final refits, a declared extension
   added widths 32/64/128, learning rates 0.2/0.5, and epoch checkpoints
   3000/10000. The original validation results and shortlist are preserved as
   `stage1_*`; the rationale and second grid are in `extension_plan.json`.
   This is an adaptive, validation-led extension, not a claim that the complete
   two-stage grid was preregistered before observing validation results.
3. Across the combined 39 checkpoints, select the highest validation count
   at each epoch budget. Also include the cheapest candidate, by the declared
   `epochs × width` work proxy, meeting each validation target of
   55/60/65/70/75%. Ties prefer lower work, lower width, then lower learning
   rate. Deduplication produces five configurations in `shortlist.json`.
4. Refit every frozen configuration on all 600 supplied training examples
   using all three fixed final seeds. Save every prediction before the
   evaluator opens test labels. Neither the highest test seed nor a
   test-selected configuration is presented as an independently selected
   winning submission.
5. The separate evaluation phase opens test labels once the full prediction
   manifest has been frozen. Report each seed, mean and sample standard
   deviation, and the number of seeds meeting each target. Seed variation is
   not a confidence interval for generalization; all runs share the same test
   set of 600 examples. An unmet target is not proved impossible by this grid.

## Exact learning algorithm

The network is `9 → H → 10`, with one ReLU hidden layer. Pixels are transformed
once as two separately rounded FP32 operations, `x * 4 - 0.5`. Training labels
become ten-element one-hot targets. The loss is one half of the sum of the ten
squared output errors, averaged over a minibatch. Inference chooses the first
class with the highest output value; no softmax is needed.

All training uses fixed, contiguous minibatches of 30 in original row order,
repeated cyclically for the declared number of epochs. There is no shuffle,
momentum, learning-rate schedule, weight decay, or early stopping within a
refit. For each batch:

```
z  = X W1 + b1
h  = select(z > 0, z, 0)
o  = h W2 + b2
d2 = o - onehot(labels)
d1 = select(z > 0, d2 W2ᵀ, 0)
gW1 = Xᵀ d1; gb1 = sum_rows(d1)
gW2 = hᵀ d2; gb2 = sum_rows(d2)
W1 -= step*gW1; b1 -= step*gb1
W2 -= step*gW2; b2 -= step*gb2
```

`step` is the precomputed FP32 literal `float32(learning_rate / 30)`. All four
gradients use pre-update parameters. Matrix products and row sums start with
FP32 zero and reduce in ascending index order, using a separately rounded
FP32 multiply followed by an FP32 add. The derivative at zero is zero.

Row-major parameter arrays are `W1[9,H]`, `b1[H]`, `W2[H,10]`, and `b2[10]`.
Initialization uses PCG64(seed): draw `W1` uniformly on `[-1/3, 1/3]`, then
`W2` uniformly on `[-1/sqrt(H), 1/sqrt(H)]`, and cast to FP32; biases are zero.
These are architecture/seed-dependent literals, independent of all datasets.
Their raw bits can be emitted as `set` operations in the scored program. The
square root only produces the compile-time initialization bound; it is not an
unaccounted training instruction.

## Arithmetic verification and scoring boundary

The fast CPU reference uses unoptimized NumPy einsum with a contiguous
right-hand matrix. It does not call BLAS. A transposed right-hand matrix
originally selected a different reduction path; this was detected and fixed
before candidate search. The published checks compare the final implementation
with explicit NumPy multiply/add FP32 reductions on representative matrix
shapes and one complete 600-example training epoch. All final test-output
vectors also use a separate explicit-reduction check. In addition,
`verify_ordered_training.py` repeats the validation-best configuration's full
training for seed 101 with explicit reductions, comparing parameter bits at
checkpoints and all final parameter and prediction bits. It never accesses
test labels.

The reference training runtime is **not** Dally-model time or A100 time.
Model scoring must include loading and normalizing the dataset, one-hot
construction, parameter initialization, every training update, and inference.
The compact IL work in this directory represents the complete fixed-loop
program and statically aggregates its explicit v4 operations. It does not
change the learning algorithm or skip the energy/time cost of training.
These MLPs have no new measured A100 runtime or energy and are not yet complete
new benchmark submissions.

## Reproduction

From the repository root, using Python 3.11 and the recorded NumPy version, use
a fresh output directory:

```bash
S=mnist/experiments/accuracy-il-20260911
python -m pip install -r "$S/study-requirements.txt"
python "$S/accuracy_study.py" --phase search --output /tmp/mnist-accuracy-study
python "$S/accuracy_study.py" --phase extend --output /tmp/mnist-accuracy-study
python "$S/accuracy_study.py" --phase final --output /tmp/mnist-accuracy-study
python "$S/verify_ordered_training.py" --output /tmp/mnist-accuracy-study
python "$S/accuracy_study.py" --phase evaluate --output /tmp/mnist-accuracy-study
```

Both learner input reads and test-label evaluation verify canonical array
checksums from `mnist/doc/dataset_manifest.json`. The recorded results include
source checksums, exact predictions, initial and final parameter checksums,
the frozen shortlist checksum, software versions, and measured CPU reference
times. The `.npy` prediction files are optional convenience outputs; identical
integer predictions are preserved in the JSON manifest.
