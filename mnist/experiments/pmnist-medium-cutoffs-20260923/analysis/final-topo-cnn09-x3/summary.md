# Five levels of sample complexity
Permutation-invariant MNIST-medium (9x9, 81 permuted features) - candidate `topo-cnn09-x3`

Levels are `N_i = 1000 * 10 ** (i/4)`, i = 0..4, rounded to **1,000, 1,778, 3,162, 5,623, 10,000** training examples; every step multiplies the budget by **1.778279**.
All five levels are **measured** on 11 dataset draws (2026092001..2026092011) with 10,000 queries each.

## The difficulty ladder

| Level | Examples | Expected error | Accuracy | 95% CI | Pass fraction | Rounded 0.05 / 0.1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1,000 | 3.9991% | 96.0009% | [3.777, 4.222]% | 9/11 | 4.00% / 4.00% |
| 2 | 1,778 | 3.1445% | 96.8555% | [3.013, 3.276]% | 6/11 | 3.15% / 3.10% |
| 3 | 3,162 | 2.4127% | 97.5873% | [2.299, 2.526]% | 6/11 | 2.40% / 2.40% |
| 4 | 5,623 | 1.8964% | 98.1036% | [1.835, 1.958]% | 6/11 | 1.90% / 1.90% |
| 5 | 10,000 | 1.5455% | 98.4545% | [1.470, 1.621]% | 6/11 | 1.55% / 1.50% |

Expected error is the pooled mean over the draws, `100 * (sum(total) - sum(correct)) / sum(total)`, computed from integer counts. The 95% CI is a Student-t interval for the mean of the per-draw errors (10 df). Pass fraction counts draws at or below the expected-mean cutoff; a mean-achievement cutoff is not a per-run guarantee.

## Per level: dispersion and cost

| Level | Examples | Mean of draws | SD (pp) | Min draw | Max draw | Bootstrap 95% | Mean train s | Mean fit wall s | Truncated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1,000 | 3.9991% | 0.3313 | 3.50% | 4.63% | [3.822, 4.201]% | 11.4 | 18.3 | 0/11 |
| 2 | 1,778 | 3.1445% | 0.1961 | 2.84% | 3.45% | [3.033, 3.248]% | 18.5 | 25.8 | 0/11 |
| 3 | 3,162 | 2.4127% | 0.1692 | 2.16% | 2.66% | [2.319, 2.504]% | 28.1 | 35.0 | 0/11 |
| 4 | 5,623 | 1.8964% | 0.0916 | 1.75% | 2.10% | [1.849, 1.948]% | 49.6 | 56.5 | 0/11 |
| 5 | 10,000 | 1.5455% | 0.1123 | 1.37% | 1.67% | [1.483, 1.606]% | 96.6 | 103.9 | 0/11 |

## Is the geometric spacing consistent with a power law?

Two-anchor model through the endpoints: **error(N) = 3.9991 * (N / 1,000) ** (-0.4129) percent**. Because the three middle levels are measured here, this is a check, not a source of cutoffs.

| Level | Examples | Measured | Two-anchor prediction | Prediction - measured (pp) |
| --- | --- | --- | --- | --- |
| 1 | 1,000 | 3.9991% | 3.9991% | +0.0000 |
| 2 | 1,778 | 3.1445% | 3.1533% | +0.0087 |
| 3 | 3,162 | 2.4127% | 2.4861% | +0.0734 |
| 4 | 5,623 | 1.8964% | 1.9602% | +0.0638 |
| 5 | 10,000 | 1.5455% | 1.5455% | +0.0000 |

Largest interior deviation: **0.0734 pp**.

Three-parameter fit with a nonnegative floor: **e(N) = 0.3320 + 3.6789 * (N / 1,000) ** (-0.4876)**, RMSE 0.0214 pp, max |residual| 0.0338 pp on 2 residual degrees of freedom. Fitted asymptote of this recipe on this pool; not a Bayes-error estimate.

## Bootstrap over whole draws

2,000 resamples, seed 20260923, unit = one complete dataset draw, all five levels together (pairing preserved). Percentile 95% intervals.

Endpoint ratio error(1,000)/error(10,000) = **2.5876** [2.496, 2.686].

| Step | Sample multiplier | Error ratio | Bootstrap 95% |
| --- | --- | --- | --- |
| 1,000 -> 1,778 | 1.778279 | 1.2718 | [1.219, 1.346] |
| 1,778 -> 3,162 | 1.778279 | 1.3033 | [1.257, 1.345] |
| 3,162 -> 5,623 | 1.778279 | 1.2723 | [1.237, 1.307] |
| 5,623 -> 10,000 | 1.778279 | 1.2271 | [1.192, 1.262] |

## Paired comparison: `topo-cnn09-x3` vs the spatial CNN reference

**N = 10,000 (both measured).** Mean paired difference (ours minus spatial) **+0.0355 pp**, t 95% [-0.0055, 0.0764] pp, bootstrap 95% [0.0018, 0.0700] pp over 11 paired draws; ours is lower in 3/11. Both sides measured on the same dataset seeds and the same query slice.

**N = 1,000 (reference interpolated).** Mean paired difference (ours minus spatial) **+0.0240 pp**, t 95% [-0.1522, 0.2001] pp, bootstrap 95% [-0.0911, 0.1943] pp over 11 paired draws; ours is lower in 8/11. INTERPOLATION, NOT A MEASUREMENT: the spatial study has no 1,000-example row; its per-draw error at 1,000 is interpolated in log error / log N between 800 and 1600.

**N = 1,000 (reference isotonic-interpolated).** Mean paired difference (ours minus spatial) **+0.0240 pp**, t 95% [-0.1522, 0.2001] pp, bootstrap 95% [-0.0911, 0.1943] pp over 11 paired draws; ours is lower in 8/11. INTERPOLATION, NOT A MEASUREMENT: the spatial study has no 1,000-example row; its per-draw error at 1,000 is interpolated in log error / log N between 800 and 1600 after a nonincreasing (PAVA) projection of the draw's curve.


## Scope

- Errors are measured on the 10,000 held-out query images of each dataset draw, drawn from the official MNIST training split; the official test split is not used.
- Every mean comes from integer correct/total counts; no rounded percentage is averaged.
- The cutoffs are expected-mean thresholds for this frozen recipe. They are not minimum necessary sample counts, a convergence certificate, or a per-run pass guarantee.
- The bootstrap resamples whole dataset draws with all five levels together, so it reflects draw-level variability conditional on the fixed pool, the frozen recipe and the fixed feature permutation. It is not a prediction interval for a new run.
- The training prefixes are nested within a dataset seed, so the five levels of one draw are positively dependent; the paired bootstrap keeps that dependence.
- Rounded cutoff options use half-up rounding to 0.05 pp and 0.1 pp; rounding changes the implied spacing between levels.
