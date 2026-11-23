# MNIST-medium: RevNet-4 with a lossless lift, 3% error target

Submission date: November 23, 2026. Tier: MNIST-medium (10,000 train / 10,000 test, 9 × 9). Declared target: **3% mean error**.

**97.3464% ± 0.1578 pp** mean accuracy ± sample standard deviation over eleven fresh draws, with **107,081 / 110,000 correct**. This **meets** the inclusive 3% band, which requires 106,700 correct. It also meets the 5%, 8% and 12% bands, and not the 2% band. The lowest draw scored 97.01%.

A reversible network: its backward pass rebuilds each block's inputs from the block's outputs instead of storing activations, so activation memory does not grow with depth. It is not an energy record. On the same A100 it uses 1.9× the energy and 2.7× the time of the matched non-reversible ensemble (below).

| A100 energy (mJ/task) | A100 CUDA time (ms/task) | Grid energy (mJ) | Grid time (ms) | Grid scratch (bytes) | Time to score (s) |
|---:|---:|---:|---:|---:|---:|
| 1.5 × 10⁶ | 35,000 | — | — | — | — |

## Accuracy evidence

| Draw | Dataset seed | Member seeds | Correct / total | Accuracy |
|---:|---:|---|---:|---:|
| 00 | 2026112300 | 11, 22, 33 | 9,752 / 10,000 | 97.52% |
| 01 | 2026112301 | 11, 22, 33 | 9,714 / 10,000 | 97.14% |
| 02 | 2026112302 | 11, 22, 33 | 9,731 / 10,000 | 97.31% |
| 03 | 2026112303 | 11, 22, 33 | 9,753 / 10,000 | 97.53% |
| 04 | 2026112304 | 11, 22, 33 | 9,736 / 10,000 | 97.36% |
| 05 | 2026112305 | 11, 22, 33 | 9,701 / 10,000 | 97.01% |
| 06 | 2026112306 | 11, 22, 33 | 9,743 / 10,000 | 97.43% |
| 07 | 2026112307 | 11, 22, 33 | 9,736 / 10,000 | 97.36% |
| 08 | 2026112308 | 11, 22, 33 | 9,728 / 10,000 | 97.28% |
| 09 | 2026112309 | 11, 22, 33 | 9,742 / 10,000 | 97.42% |
| 10 | 2026112310 | 11, 22, 33 | 9,745 / 10,000 | 97.45% |

The order of events is in the git history of this directory:

1. The [protocol](protocol.json) was committed and pushed before any draw existed. It fixes the eleven dataset seeds, the learner's source hashes, the target and the scoring rule.
2. Each draw was generated with the repository's generator, `mnist.code.data --profile medium-error-targets-v1 --seed <seed>`. Its [manifests](draws/) record the array hashes, source-index hashes and class counts.
3. The eleven fits ran without test labels. [prediction-freeze.json](prediction-freeze.json) hashes all eleven prediction files and fit records, and was committed before any label was read.
4. `mnist.code.evaluate --tier medium --error-target 3` then scored each draw ([scores](scores/)), and [accuracy.json](accuracy.json) aggregates `sum(correct) / 110000`.

Training randomness is held fixed across draws. Member seeds 11, 22 and 33 set each member's initialization and its augmentation and shuffle schedules, so the spread above is dataset variation. Within each draw, training and test rows are disjoint; draws overlap one another, as the rules allow. All eleven draws completed, with no divergence.

## Learner

The learner is the model selected by the 13 Sep 2026 reversible-network study, unchanged. Every file under [source/](source/) is byte-identical to the code that study ran, and `run.py` refuses to run if a hash differs from the protocol.

- **Input:** the supplied 9 × 9 pixels, normalized inside the learner as `4x − 0.5`.
- **Lossless lift:** a padded 3 × 3 convolution from 1 to 31 channels, then ReLU, concatenated after the normalized input. The input survives as channel 0, so the lift is injective.
- **Reversible core:** 4 additive coupling blocks on two 16-channel halves: `y1 = x1 + 0.25·F(x2)`, `y2 = x2 + 0.25·G(y1)`. Each branch is ReLU, then a padded 3 × 3 convolution from 16 to 16 channels with no bias.
- **Head:** 2,592 → 128 → 10 with ReLU. The head is not reversible.
- **Size:** 351,905 parameters and 1,848,647 nominal forward MACs per image, per member.
- **Training:** 8 epochs at batch 128, so each epoch is 79 minibatches and the last holds 16 examples. SGD with learning rate 0.03, momentum 0.9 and weight decay 1e-4, with mild affine augmentation and the original approximate-softmax gradient.
- **Kernels:** ordered FP32 Triton kernels replayed through CUDA graphs. Every compiled kernel's PTX is checked for FP32 FMA, flush-to-zero and tensor-core instructions, and none occur.
- **Backward:** working from the last block to the first, it recomputes each branch, subtracts it to recover the block's inputs, and reuses one branch workspace. No per-block activations are stored.
- **Ensemble:** the three members' logits are summed in FP32 in seed order, then argmax.

The kernels, schedules, recipe and head come from the original 3-conv CNN (97.76% on that study's draws). Only the convolutional body changed.

As a reconstruction check after training (draw 00, member 11), recovering the core input from the core output on eight images had a maximum absolute error of 7.2 × 10⁻⁷.

## Memory

Each of the 33 trained members peaked at 242.28 MiB of allocation, measured with `torch.cuda.max_memory_allocated()` before diagnostics. The breakdown for draw 00, member 11:

| Component | MiB |
|---|---:|
| Precomputed augmentation and shuffle schedules, all 8 epochs | 198.06 |
| Training workspaces, batches of 128 and 16 | 22.30 |
| Inference workspaces | 9.78 |
| Training and test images, labels | 6.26 |
| Parameters, gradients, momentum, initial copy (1.34 each) | 5.37 |
| Outputs | 0.46 |

Reversibility removes the depth dependence of the activation workspace, not most of this peak. In the 13 Sep study ([report](history/reversible-memory-report-20260913.md)), at batch 128 the reconstruction backward used 19.8 MiB of activation and workspace storage at 4, 8, 16 and 32 blocks. At 32 blocks the run peaked at 27.2 MiB, against 556.5 MiB for the same network storing its activations. Over the whole run, the peak was only 3.8% below the matched CNN's (242.3 against 251.8 MiB), because the schedules are 198 MiB of both.

## A100 measurement method

The measurement is on draw 00 in one Modal container: an NVIDIA A100-SXM4-40GB with a 400 W power limit and driver 610.57.04. [nvml_energy.py](nvml_energy.py) is `rev88-20260914/energy.py`, unchanged:

- NVML selects the device by its CUDA PCI bus ID.
- One warm-up task and one calibration task run first.
- Each of three trials brackets the active window with a 3 s settle and a 3 s idle reading on each side.
- Idle-adjusted energy is the active energy minus the mean paired idle power times the active duration.

One task resets and trains all three members, predicts all 10,000 test images, and forms the ensemble and argmax on the GPU. It excludes preparation (schedule upload, Triton compilation, CUDA-graph capture), host transfers and verification, as in the Ordered ConvNets and Rev88 measurements. The measured task's predictions are identical to the frozen draw-00 predictions.

The control is the matched non-reversible ensemble: the 3-conv CNN with the same kernels, schedules, recipe and member seeds. It was measured the same way, in the same container, after the reversible ensemble.

| Ensemble | Idle-adjusted mJ/task (3 trials) | Mean ± SD | CUDA ms/task | Active / idle W |
|---|---|---:|---:|---:|
| RevNet-4, this submission | 1,512,186; 1,449,142; 1,420,742 | 1,460,690 ± 46,803 | 35,418 | 109 / 68 |
| 3-conv CNN control | 756,365; 785,974; 775,798 | 772,712 ± 15,044 | 13,162 | 127 / 68 |

The control reproduces the historical Ordered ConvNets row (780,000 mJ, 13,000 ms). The reversible ensemble draws less power but runs 2.7× longer, so it uses 1.9× the energy. Recomputation is not the cause: in the 13 Sep study the same network storing its activations took 37.2 s per draw, against 35.5 s with reconstruction. The likelier cause is the architecture itself. At the same nominal MACs, each pass runs eight small branch convolutions and their elementwise kernels, where the CNN runs three convolutions.

Raw records: [evidence/energy.json](evidence/energy.json), [evidence/selftest.json](evidence/selftest.json), [results/](results/).

## Reproduction

Install `modal` and `numpy` locally and authenticate Modal. From the repository root:

```bash
python mnist/submissions/revnet4-lift-20261123/run.py prepare    # eleven draws into mnist/data/revnet4-lift-20261123/
modal run mnist/submissions/revnet4-lift-20261123/run.py --phase selftest
modal run mnist/submissions/revnet4-lift-20261123/run.py --phase fit       # freezes predictions
python mnist/submissions/revnet4-lift-20261123/run.py score
modal run mnist/submissions/revnet4-lift-20261123/run.py --phase energy
```

Every phase writes once and refuses to overwrite a different frozen file, so reproduce in a fresh copy of the directory. The container is pinned by digest (`ghcr.io/ab-10/wikitext-bench@sha256:95de8931…`), with torch 2.5.1+cu124, Triton 3.1.0, NumPy 2.2.6 and nvidia-ml-py 12.560.30. `fit` uses at most four ephemeral A100-40GB workers. This verification used 15 minutes of A100 time and cost $0.61 on Modal. No W&B run was created.

## Provenance and limits

- **Selection history.** The architecture was chosen on 13 Sep 2026. Among the input-preserving reversible candidates, it had the highest mean accuracy on eleven exploratory draws of the same 60,000-image pool ([selection rule](history/selection.json)). It was then confirmed on eleven further draws: 97.40% storing activations, and 97.42% with reconstruction, against 97.70% for the CNN ([confirmation protocol](history/confirmation-protocol.json), [reconstruction protocol](history/reconstruction-protocol.json)). Those draws used the study's own sampling, not `medium-error-targets-v1`. The eleven draws here are new, and their seeds were declared before they were generated.
- **Inherited labels.** The inherited CNN config still carries `target_percent: 96` and `error_targets` from the original 4% submission. Training does not read them; this submission's target is 3%.
- **Not a record.** The [512-filter CG pair](../medium-cg-pair-20260916/README.md) meets the stricter 2% band at 65,000 mJ and 260 ms.
- **Grid model not measured.**

Contributors: Yaroslav Bulatov (requirements); Codex (13 Sep 2026 architecture study and reconstruction backward); Claude Code (23 Nov 2026 packaging, Modal verification and this report).
