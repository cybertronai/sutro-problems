# MNIST

**Task:** given training images, training labels, and test images, produce the
predicted test labels (digits 0–9) with **at least 50% accuracy**. This target may
change in September.

- **MNIST-small:** 600 train / 600 test, 3 × 3 images.
- **MNIST-medium:** 6,000 train / 6,000 test, 9 × 9 images.
- **MNIST-large:** classic MNIST, 60,000 train / 10,000 test, 28 × 28 images.

Small and medium use disjoint random subsets of the original 60,000 MNIST
training examples. Large uses the official training and test splits.

**Model:** [Bill Dally single core with tape](https://github.com/cybertronai/simplified-dally-model/tree/main/models/single-core-with-tape)
([v4 instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets/v4)).
Use tape operations to read and write the dataset.

## Submission

1. Find an algorithm in Dally model that meets the accuracy target. Report its theoretical model **time**,
   **energy**, and **area** (from peak memory use), plus **time to score**: the
   runtime of the scoring computation on your machine.
2. Compile it to to run on A100 using an ISA of your choice (ie [pyptx](https://github.com/patrick-toulme/pyptx)) Report GPU runtime and
   **idle-adjusted energy in joules**, measured with NVML.
3. Submit reproduction instructions and a link to a standalone report explaining
   the algorithm and measurements. See [**Instructions for agents**](instructions.md).

## MNIST-small

| Time | Energy | Area | Time to score | Time on A100 | Energy on A100 (J) | Submission |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.6821972 ms | 1.8759744 µJ | 6,014 µm²* | 34.899 s | 6.947 µs† | 0.000519815 J† | [1NN attempt — 308/600 (51.33%)](submissions/1nn-v4-20260911/) |

*Occupied scratch-cell area under the submission's declared FP32/tape conventions.
†A100 GPU-resident, steady-state complete-task throughput and idle-adjusted NVML energy; training memorization included. See the report for boundaries and baseline sensitivity.

## MNIST-medium

| Time | Energy | Area | Time to score | Time on A100 | Energy on A100 (J) | Submission |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |

## MNIST-large

| Time | Energy | Area | Time to score | Time on A100 | Energy on A100 (J) | Submission |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |

![MNIST competition sketch: scoring metrics, dataset tiers, and a Bill Dally single-core model with tape](doc/competition-overview.png)
