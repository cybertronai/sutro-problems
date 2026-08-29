# Sparse Parity

**Author:** [@zh4ngx](https://github.com/zh4ngx)
**Date:** 2026-08-28
**Problem:** Mask sparse parity, n=32 — 40% accuracy target
**Cost:** 17,418,235
**IR:** [`weightscan2_mask32.ir`](weightscan2_mask32.ir)
**Method:** `generate_scan(0, walk="weight", weight_cap=2)` (weight-ordered null-space scan)

## Idea — visit the null space in coefficient-weight order

Same circuit as the Gray scan (full-width GF(2) RREF, particular solution
s₀, null-space basis of G = 14 vectors), but the walk order changes:
instead of the reflected Gray code, coefficient vectors are visited in
increasing Hamming weight — s₀ itself, then all 14 single-basis flips,
then all 91 pairs — 106 visits total, each transition XORing the flipped
basis vectors into the running solution and capturing on weight k.

Why it wins: measured on the dev suite, the secret's coefficient vector
(relative to the RREF basis) has Hamming weight ≤ 2 for 56.5% of
instances (weight distribution: 6.1% / 13.9% / 36.5% / 33.3% / 9.2% /
1.0% for weights 0-5, mean 2.29, max 5). A weight-ordered walk finds
every weight-≤2 secret within 106 visits, where the reflected Gray walk
needs ~1,100 visits to reach the median secret. Capture logic, phases,
and addressing are unchanged from the Gray scan; only the flip schedule
differs (emitted by `_weight_order_flips`).

## Numbers

Dev suite: recovery 0.5654 at 17,418,235 reads (214,250 lines). Two
fresh 2,048-instance adjudication draws: 0.6304, 0.6338.

## Reproduce

```python
import mask_sparse_parity as mp

ir = mp.generate_scan(0, walk="weight", weight_cap=2)
res = mp.evaluate_mask(ir)          # → cost 17,418,235, recovery 0.5654
```
