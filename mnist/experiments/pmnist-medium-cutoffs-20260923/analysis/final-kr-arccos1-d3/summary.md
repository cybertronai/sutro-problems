# Five levels of sample complexity
Permutation-invariant MNIST-medium (9x9, 81 permuted features) - candidate `kr-arccos1-d3`

Levels are `N_i = 1000 * 10 ** (i/4)`, i = 0..4, rounded to **1,000, 1,778, 3,162, 5,623, 10,000** training examples; every step multiplies the budget by **1.778279**.
All five levels are **measured** on 11 dataset draws (2026092001..2026092011) with 10,000 queries each.

## The difficulty ladder

| Level | Examples | Expected error | Accuracy | 95% CI | Pass fraction | Rounded 0.05 / 0.1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1,000 | 6.3245% | 93.6755% | [6.130, 6.519]% | 6/11 | 6.30% / 6.30% |
| 2 | 1,778 | 4.9464% | 95.0536% | [4.741, 5.152]% | 5/11 | 4.95% / 4.90% |
| 3 | 3,162 | 3.8173% | 96.1827% | [3.678, 3.957]% | 5/11 | 3.80% / 3.80% |
| 4 | 5,623 | 3.0755% | 96.9245% | [2.991, 3.160]% | 6/11 | 3.10% / 3.10% |
| 5 | 10,000 | 2.4682% | 97.5318% | [2.361, 2.575]% | 5/11 | 2.45% / 2.50% |

Expected error is the pooled mean over the draws, `100 * (sum(total) - sum(correct)) / sum(total)`, computed from integer counts. The 95% CI is a Student-t interval for the mean of the per-draw errors (10 df). Pass fraction counts draws at or below the expected-mean cutoff; a mean-achievement cutoff is not a per-run guarantee.

## Per level: dispersion and cost

| Level | Examples | Mean of draws | SD (pp) | Min draw | Max draw | Bootstrap 95% | Mean train s | Mean fit wall s | Truncated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1,000 | 6.3245% | 0.2894 | 5.72% | 6.76% | [6.145, 6.488]% | 0.6 | 7.7 | 0/11 |
| 2 | 1,778 | 4.9464% | 0.3061 | 4.43% | 5.38% | [4.771, 5.111]% | 1.8 | 16.0 | 0/11 |
| 3 | 3,162 | 3.8173% | 0.2075 | 3.43% | 4.09% | [3.702, 3.928]% | 4.7 | 33.4 | 0/11 |
| 4 | 5,623 | 3.0755% | 0.1253 | 2.88% | 3.28% | [3.007, 3.147]% | 14.5 | 60.6 | 0/11 |
| 5 | 10,000 | 2.4682% | 0.1594 | 2.07% | 2.62% | [2.376, 2.548]% | 55.2 | 124.3 | 0/11 |

## Is the geometric spacing consistent with a power law?

Two-anchor model through the endpoints: **error(N) = 6.3245 * (N / 1,000) ** (-0.4087) percent**. Because the three middle levels are measured here, this is a check, not a source of cutoffs.

| Level | Examples | Measured | Two-anchor prediction | Prediction - measured (pp) |
| --- | --- | --- | --- | --- |
| 1 | 1,000 | 6.3245% | 6.3245% | +0.0000 |
| 2 | 1,778 | 4.9464% | 4.9991% | +0.0528 |
| 3 | 3,162 | 3.8173% | 3.9511% | +0.1338 |
| 4 | 5,623 | 3.0755% | 3.1229% | +0.0474 |
| 5 | 10,000 | 2.4682% | 2.4682% | +0.0000 |

Largest interior deviation: **0.1338 pp**.

Three-parameter fit with a nonnegative floor: **e(N) = 0.7076 + 5.6255 * (N / 1,000) ** (-0.5046)**, RMSE 0.0229 pp, max |residual| 0.0372 pp on 2 residual degrees of freedom. Fitted asymptote of this recipe on this pool; not a Bayes-error estimate.

## Bootstrap over whole draws

2,000 resamples, seed 20260923, unit = one complete dataset draw, all five levels together (pairing preserved). Percentile 95% intervals.

Endpoint ratio error(1,000)/error(10,000) = **2.5624** [2.439, 2.694].

| Step | Sample multiplier | Error ratio | Bootstrap 95% |
| --- | --- | --- | --- |
| 1,000 -> 1,778 | 1.778279 | 1.2786 | [1.242, 1.325] |
| 1,778 -> 3,162 | 1.778279 | 1.2958 | [1.262, 1.326] |
| 3,162 -> 5,623 | 1.778279 | 1.2412 | [1.211, 1.274] |
| 5,623 -> 10,000 | 1.778279 | 1.2460 | [1.201, 1.294] |

## Paired comparison: `kr-arccos1-d3` vs the spatial CNN reference

**N = 10,000 (both measured).** Mean paired difference (ours minus spatial) **+0.9582 pp**, t 95% [0.8538, 1.0626] pp, bootstrap 95% [0.8709, 1.0437] pp over 11 paired draws; ours is lower in 0/11. Both sides measured on the same dataset seeds and the same query slice.

**N = 1,000 (reference interpolated).** Mean paired difference (ours minus spatial) **+2.3494 pp**, t 95% [2.1696, 2.5292] pp, bootstrap 95% [2.2023, 2.5036] pp over 11 paired draws; ours is lower in 0/11. INTERPOLATION, NOT A MEASUREMENT: the spatial study has no 1,000-example row; its per-draw error at 1,000 is interpolated in log error / log N between 800 and 1600.

**N = 1,000 (reference isotonic-interpolated).** Mean paired difference (ours minus spatial) **+2.3494 pp**, t 95% [2.1696, 2.5292] pp, bootstrap 95% [2.2023, 2.5036] pp over 11 paired draws; ours is lower in 0/11. INTERPOLATION, NOT A MEASUREMENT: the spatial study has no 1,000-example row; its per-draw error at 1,000 is interpolated in log error / log N between 800 and 1600 after a nonincreasing (PAVA) projection of the draw's curve.


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
