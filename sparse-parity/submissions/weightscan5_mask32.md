# Sparse Parity

**Author:** [@zh4ngx](https://github.com/zh4ngx)
**Date:** 2026-08-28
**Problem:** Mask sparse parity, n=32 — 100% accuracy target
**Cost:** 28,169,066
**IR:** [`weightscan5_mask32.ir`](weightscan5_mask32.ir)
**Method:** `generate_scan(0, walk="weight", weight_cap=5)` (weight-ordered null-space scan)

## Idea — the full space in weight order, 4.7x fewer visits

The coefficient weight is structurally bounded by k = 5: it counts the
secret positions that land on free columns (s0 is zero on free
columns, and the secret has exactly k = 5 ones), so visiting every
coefficient vector of weight ≤ 5 — 3,473 of the 16,384 in the
solution space — provably covers every full-rank instance, not just
the measured ones. Measured: 100% recovery on the dev suite and on
both 2,048-instance fresh draws, at 28,169,066 reads (709,002 lines).
The reflected-Gray full walk needs all 16,384 visits for the same
coverage and costs 43,325,468.

Caveat, same as the Gray full walk: a rank-deficient training draw
(~2⁻¹⁴ — fewer than 18 pivots, so more than 14 free columns and the
circuit's 14 knob slots cannot represent the full null space) would be
missed. Weight ≥ 6 secrets are impossible for full-rank instances by
the argument above; `weight_cap=6` (13,399 visits, 38,142,569 reads,
also 1.0000 on both fresh draws) exists purely as insurance against
rank-deficient adjudication draws.

## Reproduce

```python
import mask_sparse_parity as mp

ir = mp.generate_scan(0, walk="weight", weight_cap=5)
res = mp.evaluate_mask(ir)          # → cost 28,169,066, recovery 1.0
```
