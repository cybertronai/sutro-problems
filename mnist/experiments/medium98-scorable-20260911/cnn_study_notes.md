# Training-only search for a fully scorable MNIST-medium learner

These experiments use only the canonical 6,000 training images and their labels. A fixed, stratified split (seed 20260914) assigns 4,800 rows to fitting and 1,200 rows to validation. No new-draw images or test labels have entered this search. Native PyTorch/cuDNN results screen candidate algorithms; they are not accuracy evidence for the later ordered-FP32 backend.

The first frozen screen tested eight plain ReLU convolutional networks with squared-error SGD and momentum. The best model reached 1,164/1,200 validation predictions correct. Four larger-rate/deeper configurations became nonfinite, and a fifth collapsed to near chance. Every failure is retained.

The second frozen screen replaced the update direction with an approximate softmax made from subtraction, comparisons, multiplication, division and ten repeated squarings. The expression is a declared update direction, not an assertion that it is the exact derivative of a log-likelihood. Six of eight initial runs completed; both learning-rate0.1 runs became nonfinite. The two best validation configurations were replicated at the predeclared seeds22 and33.

| Plain ReLU architecture | Seed11 | Seed22 | Seed33 | Three-seed validation ensemble |
|---|---:|---:|---:|---:|
| 64 channels,4 convolutions,rate0.03 | 1,171 | 1,173 | 1,170 | 1,172/1,200 |
| 32 channels,3 convolutions,rate0.03 | 1,171 | 1,169 | 1,171 | 1,174/1,200 |

Those native ensembles use their individual validation-selected checkpoints and the frozen FP32 raw-logit sum. Both fell below1,176/1,200 (98%), motivating a separately frozen four-configuration batch-normalized ReLU screen. Its initial results were1,173/1,200 for32channels/rate0.03,1,172 for32/rate0.1,1,170 for64/rate0.03 and1,163 for64/rate0.1. Its top-two seed replications are retained separately.

The pending ordered stage uses a different, explicit PCG64 augmentation/permutation schedule and ascending FP32 reductions with separate multiplication and addition. It must be validated on this training-only split before the final learner and eleven fresh dataset draws are frozen. Its common stopping epoch will be selected from a three-seed ensemble's validation history, with count, half-Brier diagnostic and earliest epoch as successive tie-breakers.

The validation set has influenced several bounded, sequential screens, so its accuracy is a development result and can be optimistic. The untouched, newly sampled eleven-draw evaluation is the later accuracy gate. Formal scoring, complete-task A100 measurements and an official submission remain dependent on that gate and exact-backend equivalence checks.
