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

## Task and datasets

Learn from the supplied training images and labels, then produce one digit label
(0–9) for each test image. The computation being compared includes both training
and prediction. Test labels are for evaluation only.

| Tier | Training examples | Test examples | Image size | Accuracy target |
| --- | ---: | ---: | --- | --- |
| MNIST-small | 1,000 | 1,000 | 3 × 3 | Not specified in the human section; report achieved accuracy |
| MNIST-medium | 10,000 | 10,000 | 9 × 9 | Five error bands, listed below |
| MNIST-original (MNIST-large) | 60,000 | 10,000 | 28 × 28 | Not specified in the human section; report achieved accuracy |

For small and medium, use disjoint random training and test subsets of the
original 60,000 MNIST training examples. Report **mean accuracy ± sample standard
deviation over 11 independently sampled datasets**, with fresh training on each
dataset. Record all dataset seeds, preprocessing, checksums, learner seeds, and
individual `correct / total` counts. Choose the learning procedure before
inspecting test results; do not transfer learned state between datasets.
Standard deviation is in **percentage points (pp)**. Original uses the complete
official training and test splits.

### MNIST-medium error bands

Error is the fraction of incorrect test predictions. Each band is an inclusive
maximum error, equivalent to the following minimum accuracy:

| Maximum mean error | Minimum mean accuracy | Minimum total correct across 11 datasets (110,000 test predictions) |
| ---: | ---: | ---: |
| 2% | 98% | 107,800 |
| 3% | 97% | 106,700 |
| 5% | 95% | 104,500 |
| 8% | 92% | 101,200 |
| 12% | 88% | 96,800 |

Label each medium result with the error band it targets and whether it meets
that band. Apply thresholds to the **unrounded 11-dataset mean**, calculated as
`sum(correct) / 110000`; a rounded display percentage does not establish a pass.
Report all 11 draws, including their mean and sample standard deviation, rather
than selecting favorable draws. For small and original, report accuracy without
claiming qualification against an unstated target.

## Efficiency measurements

Address the memory wall by reducing memory footprint and data movement. The two
implementation goals are:

- **A100:** provide a kernel or implementation that uses little energy. Use an
  ISA or toolchain of your choice, such as
  [pyptx](https://github.com/patrick-toulme/pyptx) or Triton. Report runtime and
  **idle-adjusted energy measured with NVML**, including the measurement commands
  and idle-baseline method.
- **Bill Dally's 2D grid:** use the
  [spatial-computer model](https://github.com/cybertronai/simplified-dally-model/tree/main/models/spatial-computer)
  and its specified instruction set. Declare processor and memory placement,
  data representation, tape layout, and execution schedule. Count data movement
  in **word-node hops**, including local accesses, interprocessor traffic, and
  tape I/O according to that model. Report **peak scratch-memory use**, active
  processors, hop counts, and the resulting model energy and elapsed time.

Identify the model revision and scoring method used. Report **time to score**:
the host runtime of computing the theoretical metrics. Account for the full
training-and-prediction computation, identifying any setup or preprocessing
excluded from a measurement. Keep theoretical scores distinct from measured
A100 costs; report an em dash for unmeasured values.

Display execution times in **ms**, energies in **mJ** (1 J = 1,000 mJ), and time
to score in **s**, using two significant figures for cost measurements. Report
memory in bytes or KiB and retain exact counts, accuracies, and measurements in
the accompanying data files. If reporting area, state its spatial-model
definition and derivation; do not reuse the old single-core area conversion.

## Submission

Open a pull request adding the source or generator, reproduction commands, and
a standalone report under `mnist/submissions/<name>/`. Include the tier, target
error band for medium, accuracy evidence, dataset and learner seeds, model
revision, memory layout, scoring calculations, hardware/software versions, A100
measurements, and any W&B runs. Identify which efficiency metrics have been
measured and which remain unavailable.

Add new results above the historical tables below, clearly labeled with the
current dataset sizes and, for medium, the error band. Keep individual entries
concise and link to the full report.

## Existing tooling

The [older agent instructions](instructions.md), default dataset generator, and
accuracy evaluator still describe the previous specification. Their 600/600 and
6,000/6,000 sizes, fixed 60%/98%/98% thresholds, and single-core scoring rules do
not define this problem. Update or configure reproduction code to match the
datasets, error bands, and spatial model above before claiming current results.
The generator's `reference-20260910` profile has the new counts but uses a
different train/test split protocol; matching counts alone is insufficient.

## Historical submissions

The entries below retain their original measurements. Small used 600 training
and 600 test images; medium used 6,000 of each. Reported Dally scores and areas
use the former **single-core-with-tape** model. These results do not establish
accuracy or spatial-computer costs for the revised datasets and error bands.
Their reports document the original evaluation scope, including whether an
entry used one dataset or 11 draws.

### MNIST-small (historical)

| Accuracy | Time (ms) | Energy (mJ) | Area (mm²) | Time to score (s) | Time on A100 (ms) | Energy on A100 (mJ) | Submission |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 62% | 93 | 0.11 | 0.020 | 0.081 | 71 | 2,000 | [32-unit MLP](https://cybertronai.github.io/sutro-problems/docs/submissions/mlp60-affine-20260911/) |
| 51% | 1.7 | 0.0019 | 0.0060 | 35 | 0.0069 | 0.52 | [1NN](https://cybertronai.github.io/sutro-problems/docs/submissions/1nn-v4-20260911/) |

### MNIST-medium (historical)

| Accuracy | Time (ms) | Energy (mJ) | Area (mm²) | Time to score (s) | Time on A100 (ms) | Energy on A100 (mJ) | Submission |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 98.1% ± 0.1 pp | — | — | — | — | 5.9 × 10⁴ | 1.7 × 10⁶ | [Three ConvNets](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-convnet-11draw-20260911/) |
| 96% | 93,000 | 200 | 0.63 | 2.7 | 4,700 | 150,000 | [512-unit MLP](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-affine-20260911/) |

### MNIST-large (historical)

| Accuracy | Time (ms) | Energy (mJ) | Area (mm²) | Time to score (s) | Time on A100 (ms) | Energy on A100 (mJ) | Submission |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |

Historical single-core-with-tape sketch:

![MNIST competition sketch: scoring metrics, dataset tiers, and a Bill Dally single-core model with tape](doc/competition-overview.png)
