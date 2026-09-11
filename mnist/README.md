# MNIST

**Task:** given training images, training labels, and test images, produce the
predicted test labels (digits 0–9) meeting the accuracy requirement for the tier.

- **MNIST-small:** 600 train / 600 test, 3 × 3 images; **at least 60% accuracy** (360/600 correct).
- **MNIST-medium:** 6,000 train / 6,000 test, 9 × 9 images; **at least 98.14% accuracy** (5,889/6,000 correct).
- **MNIST-large:** classic MNIST, 60,000 train / 10,000 test, 28 × 28 images; **at least 98% accuracy** (9,800/10,000 correct).

Small and medium use disjoint random subsets of the original 60,000 MNIST
training examples. Large uses the official training and test splits.

Targets and their evaluation basis are documented in [Instructions for agents](instructions.md#accuracy-targets). Eligibility uses exact prediction counts, before display rounding.

**Model:** [Bill Dally single core with tape](https://github.com/cybertronai/simplified-dally-model/tree/main/models/single-core-with-tape)
([v4 instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets/v4)).
Use tape operations to read and write the dataset.

## Submission

1. Find an algorithm in Dally model that meets the accuracy target. Report its theoretical model **time**,
   **energy**, and **area** (from peak memory use), plus **time to score**: the
   runtime of the scoring computation on your machine.
2. Compile it to to run on A100 using an ISA of your choice (ie [pyptx](https://github.com/patrick-toulme/pyptx)) Report GPU runtime and
   **idle-adjusted energy**, measured with NVML and reported in millijoules (mJ).
3. Submit reproduction instructions and a link to a standalone report explaining
   the algorithm and measurements. See [**Instructions for agents**](instructions.md).

Execution times use **milliseconds (ms)**, energies use **millijoules (mJ)**, areas use **square millimeters (mm²)**, and **time to score uses seconds (s)**, with **two significant figures**. Time to score is the host computation of the theoretical scores. Exact measurements remain in each submission’s data files.

## MNIST-small

| Accuracy | Time (ms) | Energy (mJ) | Area (mm²) | Time to score (s) | Time on A100 (ms) | Energy on A100 (mJ) | Submission |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 51% (308/600) | 1.7 | 0.0019 | 0.0060* | 35 | 0.0069† | 0.52† | [1NN — historical baseline](submissions/1nn-v4-20260911/) |

**The historical 1NN entry is below the current 60% requirement.** It met the 50% requirement in effect when submitted.

*Occupied scratch-cell area under the submission's declared FP32/tape conventions.
†A100 GPU-resident, steady-state complete-task throughput and idle-adjusted NVML energy; training memorization included. See the report for boundaries and baseline sensitivity.

[Higher-accuracy and compact-scoring study](https://cybertronai.github.io/sutro-problems/docs/submissions/accuracy-il-20260911/):
all three seeds at 300 epochs and above exceeded 60%; one of 15 runs exceeded 65%.
The prototype scores billions of v4 instructions through affine-loop aggregation.
The 300-epoch, seed-101 network now has a [complete A100 measurement report](https://cybertronai.github.io/sutro-problems/docs/submissions/mlp60-affine-20260911/) for the 60% submission attempt; other configurations and seeds remain unmeasured.

## MNIST-medium

| Accuracy | Time (ms) | Energy (mJ) | Area (mm²) | Time to score (s) | Time on A100 (ms) | Energy on A100 (mJ) | Submission |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |

## MNIST-large

| Accuracy | Time (ms) | Energy (mJ) | Area (mm²) | Time to score (s) | Time on A100 (ms) | Energy on A100 (mJ) | Submission |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |

![MNIST competition sketch: scoring metrics, dataset tiers, and a Bill Dally single-core model with tape](doc/competition-overview.png)
