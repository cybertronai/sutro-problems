# MNIST

**Task:** given training images, training labels, and test images, produce the
predicted test labels (digits 0–9).

- **MNIST-small:** 600 train / 600 test, 3 × 3 images.
- **MNIST-medium:** 6,000 train / 6,000 test, 9 × 9 images.
- **MNIST-large:** classic MNIST, 60,000 train / 10,000 test, 28 × 28 images.

Small and medium use disjoint random subsets of the original 60,000 MNIST
training examples. Large uses the official training and test splits.

**Model:** [Bill Dally single core with tape](https://github.com/cybertronai/simplified-dally-model/tree/main/models/single-core-with-tape)
([v4 instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets/v4)).

## MNIST-small

| Time | Energy | Area | Time to score | Time on A100 | Submission |
| ---: | ---: | ---: | ---: | ---: | --- |

## MNIST-medium

| Time | Energy | Area | Time to score | Time on A100 | Submission |
| ---: | ---: | ---: | ---: | ---: | --- |

## MNIST-large

| Time | Energy | Area | Time to score | Time on A100 | Submission |
| ---: | ---: | ---: | ---: | ---: | --- |

[**Instructions for agents**](instructions.md)

![MNIST competition sketch: scoring metrics, dataset tiers, and a Bill Dally single-core model with tape](doc/competition-overview.png)
