# Sparse Parity

**Author:** [@zh4ngx](https://github.com/zh4ngx)
**Date:** 2026-08-29
**Problem:** Mask sparse parity, n=32 — 20% and 40% accuracy targets
**Cost:** 4,991,107
**IR:** [`siswalk1_cap2_mask32.ir`](siswalk1_cap2_mask32.ir)
**Method:** `generate_sis_mask(1, 2)` (static information set + weight-ordered walk)

## Idea — RREF a fixed square subsystem, get the knobs for free

The full scan pays ~12.2M reads to RREF all 32 columns with dynamic
pivoting before its first step. This family instead fixes one
information set S (a random 18-column subset, seed 0) and RREFs the
square subsystem [X_S | y | X_F] — 33 columns wide, pivots statically
ordered on the S columns — which costs about a fifth as much. The
walk's starting point s0 reads directly off the y column, and each of
the 14 knob vectors is the solution of X_S v = X_f padded with e_f,
read straight off the augmented X_F columns of the same table. The
walk itself is the coefficient-weight-ordered walk of the weightscan
submissions (cap 2: 106 visits).

When X_S is rank-deficient (probability ~0.71 per uniform draw — no
suite-conditioning bonus; measured 28.5% full-rank rate on the dev
suite) the static pivoting degrades gracefully: some S columns get no
pivot, the walked space is no longer guaranteed to be the true
solution space, and the set contributes nothing (or, rarely, a wrong
weight-5 visitor, which scores 0 exactly like an abstention). Even so,
one set at cap 2 recovers 44.5% on the dev suite — well above the
28.5% invertibility rate, because deficient solves often leave the
secret inside the walked space anyway. Adding sets saturates near 76%;
the residual is a correlated hard class only the full dynamic RREF
handles, so the 80/100% bands stay with the scan family.

## Numbers

Dev suite: recovery 0.4453 at 4,991,107 reads (72,782 lines). Fresh
2,048-instance draws: 0.4536, 0.4580. Previous records at these
bands: 12,042,480 (ISD restarts, 20%) and 17,418,235 (weightscan2, 40%).

**Margin on the 40% band.** Recovery here is only a few points above the 40%
line, and fresh-draw spread is wider than the ±2 pp quoted for the scan
families — success is correlated through the information set, so per-suite
variance exceeds binomial. Over 15 independent adjudication draws: min 0.4048,
max 0.5063, mean 0.4527, sd 0.0283, none below 0.40, but the threshold sits
only ~1.8 sd under the mean, so an occasional draw can miss it. The 20% band
(+20 pp) and the cap-3 submission's 60% band are not close calls.

## Reproduce

```python
import mask_sparse_parity as mp

ir = mp.generate_sis_mask(1, 2)
res = mp.evaluate_mask(ir)          # → cost 4,991,107, recovery 0.4453
```
