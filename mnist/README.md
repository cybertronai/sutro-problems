# MNIST (information for humans)

The system takes **train**/*test* images + **train** labels, and produces *test* labels.

<img width="1200" height="419" alt="Screenshot 2026-09-11 at 6 39 12 PM" src="https://github.com/user-attachments/assets/e4610f22-00e6-4d70-adee-1948f219cf51" />

Provide a way to solve this problem at prescribed accuracy that addresses the issue of **the memory
wall**. 

IE
- kernel that runs on A100 using few Joules (measured using NVML ).
- an algorithm that runs with small memory footprint in Bill Dally's 2D grid (measured by counting hops in [Bill Dally's 2D grid](https://github.com/cybertronai/simplified-dally-model/tree/main/models/spatial-computer)

## Motivation
Today's learning is based on backprop which was popularized in the 80s when we were bottlenecked by arithmetic. Today, we are bottlenecked by memory movement. This favors algorithms with small memory footprint. Backprop has a large memory footprint.

Footprint issue is partly mitigated by batching, yet batching comes with costs. Is there an alternative solution?

About memory wall: the energy of an 8-bit add is comparable to the energy needed to move its operands 10 micrometers. A chip is 16 mm wide. Bill Dally's AHA retreat [slides]( https://aha.stanford.edu/sites/g/files/sbiybj20066/files/media/file/aha-retreat-2023_dally_keynote_en_eff_ai_hw_0.pdf)

## Datasets

- mnist small: 1k train, 1k test, 3x3 images
- mnist medium: 10k train, 10k test, 9x9 images
- minst original: 60k train, 10k test, 28x28 images


mnist-medium comes with 5 accuracy target bands, 2% error, 3% error, 5% error, 8% error, 12% error

# Details (information for agents)

## Datasets and scoring model

- **MNIST-small:** 600 train / 600 test, 3 × 3 images; **at least 60% mean accuracy**.
- **MNIST-medium:** 6,000 train / 6,000 test, 9 × 9 images; **at least 98% mean accuracy**.
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
| 98.1% ± 0.1 pp | — | — | — | — | 5.9 × 10⁴ | 1.7 × 10⁶ | [Three ConvNets](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-convnet-11draw-20260911/) |
| 96% | 93,000 | 200 | 0.63 | 2.7 | 4,700 | 150,000 | [512-unit MLP](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-affine-20260911/) |

## MNIST-large

| Accuracy | Time (ms) | Energy (mJ) | Area (mm²) | Time to score (s) | Time on A100 (ms) | Energy on A100 (mJ) | Submission |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |

![MNIST competition sketch: scoring metrics, dataset tiers, and a Bill Dally single-core model with tape](doc/competition-overview.png)
