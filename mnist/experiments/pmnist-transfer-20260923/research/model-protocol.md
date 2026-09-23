# AMLP Ladder implementation protocol

The candidate is AMLP `[2,2]`, the best fully supervised entry in Table 2 of
Pezeshki et al., ICML 2016. Its reported error is **0.569% ± 0.010 percentage
points standard error**, averaged over ten runs with 60,000 labeled MNIST
training examples. This is **99.431% average accuracy**, not a single model's
guaranteed accuracy. The numerical lead over the original Ladder result is
smaller than the reported uncertainty. The model here is a fresh PyTorch port;
it is not an authors' checkpoint or independently verified reproduction of that
error rate. [Paper, Table 2 and §4.3](https://proceedings.mlr.press/v48/pezeshki16.pdf)

## Fixed settings

| Setting | Value | Source |
|---|---|---|
| Encoder | input → 1000 → 500 → 250 → 250 → 250 → 10 | Original authors' reference architecture |
| Hidden activation | ReLU | Original reference implementation |
| Combinator | Independent nodewise 3 → 2 → 2 → 1 MLP | Pezeshki §4.2.3 / Table 2 |
| Combinator inputs | vertical, lateral, vertical × lateral | Pezeshki §4.2.3 |
| Combinator activation | Leaky ReLU, negative slope 0.1 | Pezeshki Eq. 24 |
| Gaussian corruption | Standard deviation 0.3 at input and every encoder layer | Supplement Table 4 |
| Reconstruction coefficient | Input: 2000; all six hidden/output levels: 0 | Supplement Table 4 |
| Combinator weight initialization | Normal, mean 0, standard deviation 0.025 | Supplement Table 5 |
| Learning rate | Adam 0.002; 100 constant epochs followed by 50 decay epochs | Pezeshki §4.3 |

The supplement explicitly calls η an initialization **standard deviation**, so
0.025 is not interpreted as a variance. Decoder hidden widths describe each
coordinate's small combinator MLP; they do not replace the large encoder.
[Supplement, Tables 4–5](https://proceedings.mlr.press/v48/pezeshki16-supp.pdf)

## Reference-code choices

The inspected baseline repository is `CuriousAI/ladder`, pinned at
`5a8daa1760535ec4aa25c20c531e1cc31c76d911`. No authors' AMLP-specific repository
was located. The original source fills in these otherwise underspecified
choices: independent encoder and decoder matrices use normal initialization
with standard deviation `1/sqrt(fan_in)`; normalization epsilon is `1e-10`;
hidden layers have a trainable shift but no scale; the top layer has both, in
the order `(normalized + beta) * gamma`. The decoder starts from normalized
noisy softmax output and uses normalized vertical linear projections.
Reconstruction MSE is averaged over examples and coordinates.
[Reference ladder.py](https://github.com/CuriousAI/ladder/blob/5a8daa1760535ec4aa25c20c531e1cc31c76d911/ladder.py)

The reference training stream pairs 100 labeled examples with 100 examples from
the training pool used without labels for reconstruction; the full-label
experiment still uses only the same 60,000 examples. Each stream is normalized
separately. The reference creates two Fuel `ShuffledScheme` instances without
explicit separate seeds. Fuel's identical default seeds can make their full-label
row orders coincide, while noise remains independent. Our runner deliberately
uses independently shuffled row orders for the two streams. This is a documented
sampling adaptation, not claimed literal reference-code parity.
`loss(x,y,x_unlabeled)` implements two separately corrupted streams. With omitted
`x_unlabeled`, both losses reuse one noisy batch; that cheaper option is another
protocol change. Input values are
in `[0,1]`, with no MNIST whitening or contrast normalization. The runner must
supply scaled floating-point inputs.
[Reference run.py](https://github.com/CuriousAI/ladder/blob/5a8daa1760535ec4aa25c20c531e1cc31c76d911/run.py)

The clean branch is computed without autograd to update approximate running
statistics. This is exact for the selected objective: every hidden clean-target
loss has coefficient zero, and the input target is the supplied training data.

## Clean evaluation and normalization

`calibrate_bn(train_x,batch_size=100)` makes one clean pass over shuffled training
data, accumulating the mean of batch means and the mean of unbiased within-batch
variances. Each deeper layer receives the preceding layer's batch-normalized
activations. This follows the **actual code path** of `FinalTestMonitoring` and
the `annotate_bn` updates. Despite a broader claim in the reference docstring,
that code does not compute a full-population variance including between-batch
variation. Training uses cumulative estimates for its first ten batches, then
momentum 0.1. Final test predictions use the fixed training-derived statistics;
test batch composition cannot change predictions. No noise is applied in
`forward`, and no decoder is used for evaluation.
[Reference nn.py](https://github.com/CuriousAI/ladder/blob/5a8daa1760535ec4aa25c20c531e1cc31c76d911/nn.py)

## Explicit adaptation and reproduction limits

- This implementation uses PyTorch rather than Theano/Blocks. It uses fresh
  seeds and PyTorch's random-number generators. It has no converted weights.
- Combinator biases start at zero. The paper specifies the Gaussian
  initialization of weight parameters but does not separately specify biases.
- PyTorch Adam uses β₁=0.9, β₂=0.999, ε=1e-8 and no weight decay. These are the
  documented Blocks defaults; the location of epsilon relative to the bias
  correction differs slightly between implementations.
- The paper's 100+50 epoch schedule is implemented with a zero-based epoch
  index. Epochs 0–99 use 0.002, epochs 100–149 use 0.002 down to 0.00004, and
  the rate reaches zero immediately after epoch 149. No test-set early stopping
  or checkpoint selection is part of the model.
- The schedule follows the paper rather than the public repository's apparent
  legacy scheduler bug: `LRDecay` reassigns its own Python scalar without clearly
  updating the optimizer's shared learning-rate variable. The public baseline
  implementation therefore cannot certify exact numerical reproduction of the
  later AMLP paper.
- Changing the input dimension to 81 adapts the first encoder and last decoder
  matrices plus the input combinator. It retains the same five hidden widths,
  ten classes, corruption, and loss coefficients. This tests method transfer
  by fitting a fresh model; it is not pretrained-weight transfer from 784 inputs.
- The model has no spatial operations or convolution. A fixed permutation can
  be absorbed into input-facing parameters. Tests verify equality of clean
  logits, full noiseless loss, and gradients after this reindexing. Random
  noise preserves the same distribution under a permutation; separate random
  training runs need not give bitwise-identical results.
- Truncated epoch counts, smaller training sets, larger batches, or omitted
  paired streams are pilot runs and must not be described as a reproduction
  of the paper's reported accuracy.

## Verification

Six focused CPU tests pass using Python 3.11 and PyTorch 2.2.2:

1. Vectorized combinators equal independent coordinate MLPs.
2. The full noiseless objective, input-facing gradients, and clean predictions
   agree after coordinate/parameter permutation.
3. Calibrated evaluation is deterministic, independent of batch composition,
   and does not modify model state.
4. Calibration matches independently computed reference-style moments.
5. Noisy outputs vary; the exact stated loss identity holds; all parameters
   receive finite gradients and an Adam update leaves the next loss finite.
6. Epoch-boundary learning rates match the documented schedule.

Run from the experiment directory: `python -m unittest -v test_model.py`.
