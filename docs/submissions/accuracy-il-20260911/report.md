# Higher accuracy, practical scoring

**MNIST-small · exploratory results · 11 September 2026**

**A 60% accuracy target looks feasible on this fixed dataset.** A 32-hidden-unit network trained for 300 epochs cleared it with all three predeclared seeds. A 65% target was reached by only one run; 70% and 75% were not reached. This finite search does not establish an upper limit on accuracy.

**Scoring the training algorithm is practical with a compact intermediate language.** The tested neural networks represent up to 24 billion primitive instructions, yet exact cost aggregation takes about 0.13 s. Numerical training and accuracy verification are separate. These MLPs have no measured A100 runtime or energy yet.

[TOC]

## One display convention

Execution times use **milliseconds (ms)**, energies use **millijoules (mJ)**, and **time to score uses seconds (s)**, with **two significant figures**. Prediction counts, model dimensions, and exact target thresholds retain their integer values. Raw JSON preserves full measurements and exact integer cost totals.

Energy and time use matching milli prefixes: **E(mJ) = P(W) × t(ms)**. At 1 W, their numerical values are equal. Model and measured execution share ms and mJ; host scoring work is shown separately in s. The raw scorer retains exact internal ps ticks and fJ counts, converted only for display.

## Higher accuracy results

Each count below is correct predictions out of the same 600 test examples, in seed order 101 / 102 / 103. The shortlist and all 15 prediction arrays were frozen before test evaluation. Means and sample standard deviations describe seed variation on this fixed test set, not population uncertainty.

| Configuration | Correct / 600, by seed | Mean accuracy (rounded) | Seed SD (percentage points) |
| --- | --- | ---: | ---: |
| H32 · 100 epochs | 356 / 365 / 362 | 60% | 0.76 |
| H32 · 300 epochs | 374 / 371 / 377 | 62% | 0.5 |
| H32 · 1,000 epochs | 383 / 405 / 376 | 65% | 2.5 |
| H32 · 10,000 epochs | 381 / 389 / 387 | 64% | 0.69 |
| H128 · 3,000 epochs | 384 / 388 / 385 | 64% | 0.35 |

The original 1NN baseline scored **308/600 (51%)**. All neural-network configurations use learning rate 0.2 and minibatches of 30. H denotes hidden-layer width.

| Target | Required correct / 600 | Runs meeting it | Interpretation |
| ---: | ---: | ---: | --- |
| 55% | 330 | 15/15 | Reached by every tested run. |
| 60% | 360 | 14/15 | Reached by every seed at 300 epochs and above. |
| 65% | 390 | 1/15 | One exploratory run; not reliable across seeds. |
| 70% | 420 | 0/15 | Not reached in this search. |
| 75% | 450 | 0/15 | Not reached in this search. |

**Target checks use exact counts, not rounded display percentages.** The single run above 65% was H32 / 1,000 epochs / seed 102, with 405/600 correct (about 68%). Its other two seeds scored 383/600 and 376/600. Choosing that seed after viewing the test result would need separate validation. The best training-validation configuration was H32 / 10,000 epochs; its three test results were 381/600, 389/600, and 387/600.

## Complete-task model costs

Every MLP score includes explicit scratch initialization, dataset tape operations, pixel transformation, one-hot target construction, initial weight writes, all training updates, inference, and output selection. Costs use the same pinned Dally v4 conventions as the 1NN baseline. Area is occupied scratch-cell area with the declared fixed placement.

| Configuration | Model time (ms) | Model energy (mJ) | Area (µm²) |
| --- | ---: | ---: | ---: |
| Original 1NN | 1.7 | 0.0019 | 6.0 × 10³ |
| H32 · 100 epochs | 31 | 0.038 | 2.0 × 10⁴ |
| H32 · 300 epochs | 93 | 0.11 | 2.0 × 10⁴ |
| H32 · 1,000 epochs | 310 | 0.38 | 2.0 × 10⁴ |
| H32 · 10,000 epochs | 3100 | 3.8 | 2.0 × 10⁴ |
| H128 · 3,000 epochs | 4000 | 6.1 | 2.8 × 10⁴ |

All three seeds of each configuration have identical model costs: only the seed-dependent literal bits differ. The learner uses separately rounded FP32 multiplication and addition with ascending reduction order. The cost model charges memory reads and writes; it is not a hardware power simulator.

## Cost-evaluation work

These timings include schema, address-bound and initialization checks, placement, exact access histograms, integer cost sums, and canonical program hashing. They exclude JSON loading, file output, numerical training, and accuracy verification. Each timing is the median of five complete scoring calls on the same host and Python environment.

| Configuration | Expanded instructions | Compact JSON bytes | Time to score (s) |
| --- | ---: | ---: | ---: |
| Original 1NN | 1.1 × 10⁷ | 8.8 × 10³ | 0.019 |
| H32 · 100 epochs | 2.1 × 10⁸ | 1.8 × 10⁵ | 0.079 |
| H32 · 300 epochs | 6.2 × 10⁸ | 1.8 × 10⁵ | 0.078 |
| H32 · 1,000 epochs | 2.1 × 10⁹ | 1.8 × 10⁵ | 0.079 |
| H32 · 10,000 epochs | 2.1 × 10¹⁰ | 1.8 × 10⁵ | 0.078 |
| H128 · 3,000 epochs | 2.4 × 10¹⁰ | 4.6 × 10⁵ | 0.13 |

At fixed width H32, increasing training from 100 to 10,000 epochs multiplies the training cost by 100 while the static scoring time stays nearly constant. The epoch loop changes repetition count, not accessed addresses. The compact representation does not forgive the repeated work: each occurrence contributes its full v4 cost.

The original 1NN interpreter took about **35 s** while also executing each FP32 instruction. That is a different workload from static cost evaluation. The new number is not an end-to-end verification speedup. A separate scaling stress test, without an accuracy claim, also scored a 41-billion-instruction program; raw measurements use a separately recorded Python/NumPy environment.

## Measured execution and comparison boundaries

For context, the CPU reference actually performed training and inference. The table gives the range across the three final seeds. Timing begins after input transformation, one-hot conversion, and parameter initialization; it excludes the independent ordered-reduction checks. Those operations are included in the theoretical IL costs above. CPU energy was not measured.

| Configuration | CPU reference training + inference time (ms) | A100 time / energy |
| --- | ---: | --- |
| H32 · 100 epochs | 210 | Not measured |
| H32 · 300 epochs | 620–630 | Not measured |
| H32 · 1,000 epochs | 2100 | Not measured |
| H32 · 10,000 epochs | 21000 | Not measured |
| H128 · 3,000 epochs | 11000 | Not measured |

The already measured **1NN** comparison remains:

| Quantity | Dally model | A100 measured |
| --- | ---: | ---: |
| Time (ms) | 1.7 | 0.0069 |
| Energy (mJ) | 0.0019 | 0.52 |

A100 values are GPU-resident steady-state complete-task throughput and idle-adjusted NVML energy, including training memorization. Host transfer, compilation, warm-up, and idle baseline selection have different boundaries. See the original submission for raw trials and baseline sensitivity. No A100 values have been extrapolated to the MLPs.

## The proposed intermediate language

`sutro-affine-v4/0.1` stores a fixed scratch layout, nested constant-bound loops, affine addresses, and ordinary v4 instructions. A dot product is a loop of `mul` and `add`; training is loops around explicitly represented forward, backward, and update operations. There is no free matrix-multiply operation or caller-supplied cost certificate.

For each primitive operand, the scorer computes the exact histogram of concrete addresses across its enclosing loops. It combines address progressions with discrete convolution. An epoch index absent from an address simply multiplies the count. Reads and writes remain separate, aliased operands are charged repeatedly, and `select` charges both candidate values. Per-address counts are then multiplied by the pinned distance-dependent costs.

The prototype deliberately restricts programs to fixed control flow and affine addressing. It checks all address bounds and proves sources initialized before use, including unchosen selections. Difficult initialization proofs are rejected when the proof budget is exhausted. Correctness, training/test separation, and accuracy still require an independent semantic evaluator.

## Verification evidence

- **Original 1NN:** the compact program expands byte-for-byte to the original 220 MB v4 trace. Exact model time, energy, opcode counts, and all 6,014 per-address read/write counts agree.
- **General scorer:** 11 tests cover independent enumeration, negative strides, overlapping addresses, aliases, selection, initialization, bounds, tape semantics, overflow, and a small multi-batch MLP.
- **MLP lowering:** two complete small training/inference programs were expanded and executed in the original interpreter with explicit comparison predicates. Every learned parameter bit, output prediction, instruction count, and model score matched the ordered reference.
- **Full training arithmetic:** the validation-best H32 / 10,000-epoch / seed-101 run was independently repeated with explicit ordered FP32 reductions. All final parameter bits and all 600 output score vectors matched. The comparison took about **140,000 ms**.
- **Every final run:** all 600 final score vectors matched explicit ordered reductions. Canonical dataset hashes, source hashes, prediction arrays, selection plans, and chronology are saved.

The full 24-billion-instruction MLP trace was not expanded and interpreted. The evidence combines independent scorer checks, small end-to-end lowering checks, source review, and a full ordered numerical training check. Official acceptance of the IL and a complete MLP A100 submission remain future work.

## Reproduce and inspect

Run from the repository root with Python 3.11. The accuracy study records NumPy 2.4.6; the separate legacy scaling file records its own environment. Use a fresh directory for accuracy reruns so the frozen published evidence is preserved.

```bash
S=mnist/experiments/accuracy-il-20260911
python -m pip install -r "$S/study-requirements.txt"
python -m unittest discover -s "$S" -p test_il.py -v
python "$S/validate_mlp_il.py" --output /tmp/mnist-mlp-validation.json
python "$S/score_study.py" --output /tmp/mnist-cost-rerun
python "$S/accuracy_study.py" --phase search --output /tmp/mnist-accuracy-rerun
python "$S/accuracy_study.py" --phase extend --output /tmp/mnist-accuracy-rerun
python "$S/accuracy_study.py" --phase final --output /tmp/mnist-accuracy-rerun
python "$S/verify_ordered_training.py" --output /tmp/mnist-accuracy-rerun
python "$S/accuracy_study.py" --phase evaluate --output /tmp/mnist-accuracy-rerun
```

The first scoring command uses the published frozen shortlist and emits regenerated IL programs plus repeated timing samples. Training and scoring need no external model weights. Reproduction of the validation protocol intentionally follows the recorded second-stage extension.

**Documents:** [Language specification](il.html) · [Study protocol](protocol.html) · [Ambiguities and remaining work](ambiguities.html) · [Visible session export](session.html).

**Exact evidence:** [Accuracy results](accuracy_results.json) · [Cost results](scoring_results.json) · [Frozen predictions](frozen_predictions.json) · [Full training check](ordered_training_verification.json) · [MLP lowering check](mlp_validation.json) · [IL checks](il-validation.json) · [Trace identity](il-expansion-validation.json) · [Scaling stress test](il-scaling.json).

**Source:** [Learner](accuracy_study.py) · [IL scorer](il.py) · [MLP lowering](mlp_il.py) · [Scoring driver](score_study.py) · [Repository directory](https://github.com/cybertronai/sutro-problems/tree/main/mnist/experiments/accuracy-il-20260911).

**Related:** [Original submission and A100 measurements](../1nn-v4-20260911/) · [MNIST task](https://github.com/cybertronai/sutro-problems/blob/main/mnist/README.md#mnist-small).
