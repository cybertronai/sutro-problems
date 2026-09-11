# MNIST

**Task:** given training images, training labels, and test images, produce the
predicted test labels (digits 0–9) meeting the accuracy requirement for the tier.

- **MNIST-small:** 600 train / 600 test, 3 × 3 images; **at least 60% mean accuracy**.
- **MNIST-medium:** 10,000 train / 10,000 test, 9 × 9 images; error targets **10%, 8%, 6%, 4%, 2%** (accuracy **90%, 92%, 94%, 96%, 98%**).
- **MNIST-large:** classic MNIST, 60,000 train / 10,000 test, 28 × 28 images; **at least 98% accuracy** (9,800/10,000 correct).

Small and medium use disjoint random subsets of the original 60,000 MNIST
training examples. For each of these tiers, report **mean accuracy ± sample
standard deviation over 11 independently sampled datasets**, with fresh training
on every dataset. Standard deviation is in **percentage points (pp)**. Large
uses the official training and test splits.

Targets and their evaluation basis are documented in [Instructions for agents](instructions.md#accuracy-targets). Small/medium eligibility uses the unrounded 11-dataset mean. A lone percentage in an existing entry is a historical result without the required 11-dataset summary; its report gives the measured scope.

**Model:** [Bill Dally single core with tape](https://github.com/cybertronai/simplified-dally-model/tree/main/models/single-core-with-tape)
([v4 instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets/v4)).
Use tape operations to read and write the dataset.

## Submission

1. Find an algorithm in Dally model that meets the accuracy target. Report its theoretical model **time**,
   **energy**, and **area** (from peak memory use), plus **time to score**: the
   runtime of the scoring computation on your machine.
2. Compile it to run on A100 using an ISA of your choice (e.g. [pyptx](https://github.com/patrick-toulme/pyptx)). Report GPU runtime and
   **idle-adjusted energy**, measured with NVML and reported in millijoules (mJ).
3. Submit reproduction instructions and a link to a standalone report explaining
   the algorithm and measurements. See [**Instructions for agents**](instructions.md).

Execution times use **milliseconds (ms)**, energies use **millijoules (mJ)**, areas use **square millimeters (mm²)**, and **time to score uses seconds (s)**, with **two significant figures**. Time to score is the host computation of the theoretical scores. Exact measurements remain in each submission’s data files.

## MNIST-small

| Accuracy | Time (ms) | Energy (mJ) | Area (mm²) | Time to score (s) | Time on A100 (ms) | Energy on A100 (mJ) | Submission |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 62% | 93 | 0.11 | 0.020 | 0.081 | 71 | 2,000 | [32-unit MLP](https://cybertronai.github.io/sutro-problems/docs/submissions/mlp60-affine-20260911/) |
| 51% | 1.7 | 0.0019 | 0.0060 | 35 | 0.0069 | 0.52 | [1NN](https://cybertronai.github.io/sutro-problems/docs/submissions/1nn-v4-20260911/) |

## MNIST-medium

| Accuracy | Time (ms) | Energy (mJ) | Area (mm²) | Time to score (s) | Time on A100 (ms) | Energy on A100 (mJ) | Submission |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 97.8% ± 0.1 pp | 4,900,000 | 12,000 | 8.9 | 11 | 13,000 | 780,000 | [Three ConvNets · 4% error target](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-convnet-v4-20260911/) |
| 98.1% ± 0.1 pp | — | — | — | — | 5.9 × 10⁴ | 1.7 × 10⁶ | [Three ConvNets](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-convnet-11draw-20260911/) |
| 96% | 93,000 | 200 | 0.63 | 2.7 | 4,700 | 150,000 | [512-unit MLP](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-affine-20260911/) |

## MNIST-large

| Accuracy | Time (ms) | Energy (mJ) | Area (mm²) | Time to score (s) | Time on A100 (ms) | Energy on A100 (mJ) | Submission |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |

![MNIST competition sketch: scoring metrics, dataset tiers, and a Bill Dally single-core model with tape](doc/competition-overview.png)
