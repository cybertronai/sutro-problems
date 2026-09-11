# Instructions for agents: MNIST

[Back to the MNIST problem](README.md)

## Submit a solution

Open a pull request adding your solution under `mnist/submissions/<name>/`.
Include the source or generator, instructions to reproduce training, prediction,
and scoring calculations, and a link to a standalone report naming the dataset
tier, contributors, test accuracy (`correct` / `total`), dataset checksum, and any
W&B runs.

Add a row to the matching tier's results table on the [MNIST page](README.md),
with a link to its standalone report. Preserve the MNIST page’s task outline,
dataset definitions, model, submission instructions, and competition diagram.
Keep individual submission entries concise: show **Accuracy** as a percentage
only, with **Time**, **Energy**, **Area**, **Time to score**,
**Time on A100**, and **Energy on A100**, giving units, metric definitions,
measurement commands, and hardware/software versions in the report. Put
per-submission `correct` / `total` counts, qualification notes, measurement
footnotes, and other submission-specific commentary in the individual report. Use an em dash for unmeasured values;
do not substitute estimates for measurements without labeling them.

The [competition sketch](doc/competition-overview.png) illustrates the
[Bill Dally single-core model with tape](https://github.com/cybertronai/simplified-dally-model/tree/main/models/single-core-with-tape),
using the [v4 instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets/v4).
Use its tape operations for dataset reads and writes. Compute **Time**, **Energy**,
and **Area** under this model; Area comes from peak memory use and is reported in **mm²**. **Time to score**
is the runtime of those calculations on your machine. Then implement the algorithm
on an A100 with an ISA or toolchain of your choice (for example,
[pyptx](https://github.com/patrick-toulme/pyptx) or Triton) and report
its runtime in **ms** and idle-adjusted energy in **mJ** measured via NVML.
Use **ms** and **mJ** for the theoretical scores too, **mm²** for Area, and **s** for Time to score.
Convert occupied-cell area from µm² by dividing by 10⁶; the cell/grid convention is unchanged.
Display measured values with two significant figures; exact counts and target thresholds are not rounded. Include the
background needed to reproduce these calculations in the standalone report.
The included evaluator checks classification accuracy only; it does not calculate
these scoring metrics.

**Task:** given training images, training labels, and test images, produce one
predicted digit label (0–9) for each test image meeting the tier-specific
[accuracy target](#accuracy-targets). The task includes learning from
the supplied training data and predicting the test labels; no model architecture
is prescribed.

## Accuracy targets

| Tier | Required accuracy | Minimum correct predictions | Evaluation basis |
| --- | ---: | ---: | --- |
| MNIST-small | 60% | 360 / 600 | Current 600/600 study: all three fixed seeds at 300 epochs and above exceeded 60% |
| MNIST-medium | 98.14% | 5,889 / 6,000 | Mean test accuracy from the neighboring “Build MNIST competition tiers” evaluation; historical 10,000/10,000 split |
| MNIST-large | 98% | 9,800 / 10,000 | User-selected requirement; the neighboring evaluation prepared this dataset but did not train a large baseline |

The medium target adopts the reported historical mean as a policy requirement
for the current 6,000/6,000 tier; it is not a measurement on that current split.
The reference run's three accuracies were 98.18%, 98.19%, and 98.05%.
[Reference results and W&B runs](#reference-results) preserve the original
protocol and evidence. The small target is supported by the
[current-split feasibility study](https://cybertronai.github.io/sutro-problems/docs/submissions/accuracy-il-20260911/).

Targets are inclusive and eligibility uses exact counts:
`required_correct = ceil(total * target_percent / 100)`.
In particular, 98.14% of 6,000 is 5,888.4, so medium requires 5,889 correct
predictions. A rounded display percentage does not establish a pass.
The evaluator reads the decimal target strings from
[accuracy_targets.json](doc/accuracy_targets.json).

The historical 1NN attempt scored 308/600: it met the former 50% small target,
but does not meet the current 60% target. Saved historical measurements and
session exports retain their original results and chronology.

## Datasets

| Problem | Image resolution | Training examples | Test examples | Accuracy requirement | Source |
| --- | --- | ---: | ---: | ---: | --- |
| MNIST-small | 3 × 3 | 600 | 600 | 60% | Disjoint random subsets of the original 60,000 MNIST training examples |
| MNIST-medium | 9 × 9 | 6,000 | 6,000 | 98.14% | Disjoint random subsets of the original 60,000 MNIST training examples |
| MNIST-large | 28 × 28 | 60,000 | 10,000 | 98% | Classic MNIST training and test splits, in full |

Small and medium are sampled **without replacement**, with no train/test overlap.
Both use the original training pool, including for their test examples. Large
uses the official 10,000-example test set. Pixels are grayscale and labels are
integers from 0 through 9.

Accuracy is the fraction of test labels predicted correctly. For energy-efficient
learning comparisons, the computation of interest includes training and
inference. The included evaluator measures accuracy; report model scores and
A100 measurements separately as described above.

## Reference results

The neighboring experiment, **“Build MNIST competition tiers”** (`mnist-20260910`),
provides the following reference numbers. **It used different small and medium
datasets:** 1,000/1,000 and 10,000/10,000 examples, with training examples from the
official training split and test examples from the official test split. These
are historical reference results, not measurements of the 600/600 and
6,000/6,000 problems above. The current small split has since been evaluated in the linked feasibility study; these historical numbers do not establish medium performance on the new split.

| Historical dataset | Selected model | Parameters | Final epochs | Test accuracy, mean ± seed SD |
| --- | --- | ---: | ---: | ---: |
| 3 × 3, 1,000 train / 1,000 test | CNN: two 32-channel conv layers, 128-unit head | 47,914 | 42 | **73.47% ± 0.70 percentage points** |
| 9 × 9, 10,000 train / 10,000 test | CNN: three 32-channel conv layers, 128-unit head | 352,106 | 50 | **98.14% ± 0.08 percentage points** |
| 28 × 28, 60,000 train / 10,000 test | Dataset prepared; baseline not trained | — | — | — |

Each selected CNN uses padded 3 × 3 convolutions, BatchNorm, GELU, and no pooling.
Both use AdamW with learning rate 0.001, weight decay 0.001, batch size 128, and
cosine decay; dropout is 0.1 for small and 0.2 for medium. No augmentation or
pretrained weights were used.

The experiment completed **54 A100 training runs**: per tier, 18 candidate
configurations, six finalist replications, and three final refits. Architectures
and stopping epochs were selected using a fixed stratified 20% validation split
within the training set. Final models were then refitted on all training examples
with seeds 101, 102, and 103 and evaluated once each. The table reports their
mean and sample standard deviation, not an ensemble or the best test seed. Seed
SD describes training variability on one fixed dataset, not a confidence interval.

| Seed | Historical small accuracy | W&B run | Historical medium accuracy | W&B run |
| ---: | ---: | --- | ---: | --- |
| 101 | 73.40% | [Small 101](https://wandb.ai/yaroslavvb/sutro-mnist-tiers/runs/i74bfmk3) | 98.18% | [Medium 101](https://wandb.ai/yaroslavvb/sutro-mnist-tiers/runs/b02t0xdi) |
| 102 | 74.20% | [Small 102](https://wandb.ai/yaroslavvb/sutro-mnist-tiers/runs/m9xpcern) | 98.19% | [Medium 102](https://wandb.ai/yaroslavvb/sutro-mnist-tiers/runs/79tnclhp) |
| 103 | 72.80% | [Small 103](https://wandb.ai/yaroslavvb/sutro-mnist-tiers/runs/tvie0dtp) | 98.05% | [Medium 103](https://wandb.ai/yaroslavvb/sutro-mnist-tiers/runs/xjmlhkwj) |

- [W&B training curves and all runs](https://wandb.ai/yaroslavvb/sutro-mnist-tiers/groups/mnist-20260910)
- [Historical report, tuning comparisons, and plots](results/mnist-20260910/report/report.md)
- [Historical dataset artifacts](https://wandb.ai/yaroslavvb/sutro-mnist-tiers/runs/jvrlu4vl)
- [Historical results and report artifacts](https://wandb.ai/yaroslavvb/sutro-mnist-tiers/runs/hjm4znt9)
- [Historical dataset checksum manifest](results/mnist-20260910/dataset_manifest.json)

## Prepare the problem datasets

From the repository root, with Python 3.11 or newer:

```bash
python3 -m venv mnist/.venv
mnist/.venv/bin/python -m pip install 'numpy>=1.26,<3'
mnist/.venv/bin/python -m mnist.code.data --output mnist/data --seed 20260910
mnist/.venv/bin/python -m unittest mnist.code.tests.test_data mnist.code.tests.test_evaluate -v
```

The default profile is `competition-v2`, implementing the table at the top of
this page. Preparation downloads the original gzip IDX files from the
[MNIST mirror](https://ossci-datasets.s3.amazonaws.com/mnist/) used by
[torchvision](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py)
and verifies their published MD5 checksums before parsing.

The canonical seed is **20260910**. Two PCG64 generators derived from
`numpy.random.SeedSequence(seed).spawn(2)` permute the official training and test
splits. From the training permutation, medium takes positions `[0:6000]` for
training and `[6000:12000]` for testing; small takes `[0:600]` and `[6000:6600]`.
Thus small is nested within medium on each side, with no training/test overlap
across these two tiers. Sampling is not stratified. Large uses the entire
training permutation and the entire official test permutation.

Pixels are converted to float32 and divided by 255. Downsampling uses separable
box-area averaging: output pixel `j` covers `[j*28/size, (j+1)*28/size)` in each
dimension, including fractional overlap at boundaries. Large keeps its original
resolution. The [generator](code/data.py) and [canonical manifest](doc/dataset_manifest.json)
specify source indices, shapes, dtypes, class counts, and array checksums.

To reproduce the historical datasets separately:

```bash
mnist/.venv/bin/python -m mnist.code.data \
  --profile reference-20260910 --output mnist/data-reference-20260910
```

That profile retains the historical sampling and preprocessing. Its manifest
explicitly identifies the profile; array content hashes can be compared with the
archived manifest. The existing W&B dataset artifacts contain this historical
profile, not the new problem datasets.

To regenerate the reference report, pass its historical data directory explicitly:

```bash
mnist/.venv/bin/python -m mnist.code.report --results mnist/results/mnist-20260910 \
  --data-dir mnist/data-reference-20260910 --output mnist/results/reference-report
```

Report generation requires Matplotlib and checks dataset hashes against the
archived experiment manifest before using images.

## Input and output contract

Preparation writes `small.npz`, `medium.npz`, `large.npz`, and `manifest.json`.
Load archives with `np.load(path, allow_pickle=False)`.

| NPZ key | Shape | Type | Purpose |
| --- | --- | --- | --- |
| `train_images` | `(N_train, 1, H, W)` | `float32` | Training pixels in `[0, 1]` |
| `train_labels` | `(N_train,)` | `int64` | Training digit labels |
| `test_images` | `(N_test, 1, H, W)` | `float32` | Test pixels in `[0, 1]` |
| `test_labels` | `(N_test,)` | `int64` | Ground truth for local scoring only |
| `train_indices`, `test_indices` | `(N_train,)`, `(N_test,)` | `int64` | Original row indices in the source split recorded in the manifest |

A learner receives `train_images`, `train_labels`, and `test_images`; it must not
read `test_labels`. Test labels are packaged for reproducible local scoring, so
this is an open benchmark, not a hidden-label evaluation service. Choose models,
hyperparameters, and stopping epochs using only the supplied training labels;
reserve validation examples from that training set. Unlabeled test images are
available to the learner. Pretrained models, extra labeled examples, and
cross-tier training must be reported separately from learning solely from the
prescribed training set.

Return exactly one integer label from 0 through 9 per test image, in the original
`test_images` row order. Save a one-dimensional `.npy` array, or a `.npz` archive
with key `predictions`. Probabilities, floating-point labels, booleans, incorrect
lengths, and two-dimensional arrays are rejected.

```python
import numpy as np

# predictions is a vector of integer digit labels, in test_images order.
np.save("submission.npy", predictions.astype(np.int64))
```

```bash
mnist/.venv/bin/python -m mnist.code.evaluate \
  --tier small --data-dir mnist/data --predictions submission.npy --output score.json
```

The evaluator reports accuracy as a fraction, integer `correct` and `total`, a
confusion matrix (rows = true classes; columns = predictions), and per-class
accuracy. It also reports `accuracy_target_percent`, `required_correct`, and
`meets_accuracy_target`, using exact integer/rational comparison. Valid input
still produces a successful CLI exit when it is below target; inspect
`meets_accuracy_target` for the classification requirement. This flag does not
certify the dataset, training protocol, model costs, or hardware measurements.
Energy is not measured by this command.

## Baseline code

The [model definitions](code/models.py) and [training runner](code/train.py) preserve the
18-configuration search used in the reference experiment. They also accept the
new problem datasets. Install [requirements.txt](code/requirements.txt) in a compatible
PyTorch environment and use a fresh output directory and W&B group for new runs:

```bash
python -m mnist.code.train --tier small --data-dir mnist/data --device cuda \
  --output mnist/results/competition-v2/small --group competition-v2
```

The [Modal runner](code/modal_train.py) can run small and medium on A100 GPUs with
configured Modal and W&B accounts. [Prediction](code/predict.py), [reporting](code/report.py),
and [artifact publishing](code/publish.py) helpers are included. Data archives and
model checkpoint binaries are excluded from Git. Historical JSON results,
plots, and the exact [executed source snapshot](results/mnist-20260910/source/)
are preserved alongside the reference report.
