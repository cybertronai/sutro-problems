# MNIST

**Task:** given training images, training labels, and test images, produce the
predicted test labels (digits 0–9) with at least 50% accuracy (this target may be changed in September)

- **MNIST-small:** 600 train / 600 test, 3 × 3 images.
- **MNIST-medium:** 6,000 train / 6,000 test, 9 × 9 images.
- **MNIST-large:** classic MNIST, 60,000 train / 10,000 test, 28 × 28 images.

Small and medium use disjoint random subsets of the original 60,000 MNIST
training examples. Large uses the official training and test splits.

**Model:** [Bill Dally single core with tape](https://github.com/cybertronai/simplified-dally-model/tree/main/models/single-core-with-tape)
([v4 instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets/v4)). Use the tape operations for dataset read/write.

**Submission**
Discover algorithm that achieves target accuracy and compute associated quantities. Area comes from peak memory use. Time to score reflects the runtime of this computation on your machine. Then, compile your algorithm into an A100 implementation (using https://github.com/patrick-toulme/pyptx or triton), and report associated quantities on the GPU. For energy use idle-adjusted joules from NVML . Submission consists of instructions of how to reproduce your calculations, with a link to standalone report providing the necessary background.


## MNIST-small

| Time | Energy | Area | Time to score | Time on A100 | Energy on A100 Submission |
| ---: | ---: | ---: | ---: | ---: | --- |

## MNIST-medium

| Time | Energy | Area | Time to score | Time on A100 | Energy on A100 Submission |
| ---: | ---: | ---: | ---: | ---: | --- |

## MNIST-large

| Time | Energy | Area | Time to score | Time on A100 | Energy on A100 Submission |
| ---: | ---: | ---: | ---: | ---: | --- |

[**Instructions for agents**](instructions.md)

![MNIST competition sketch: scoring metrics, dataset tiers, and a Bill Dally single-core model with tape](doc/competition-overview.png)
