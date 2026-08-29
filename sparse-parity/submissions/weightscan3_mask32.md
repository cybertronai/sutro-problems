# Sparse Parity

**Author:** [@zh4ngx](https://github.com/zh4ngx)
**Date:** 2026-08-28
**Problem:** Mask sparse parity, n=32 — 60% and 80% accuracy targets
**Cost:** 18,509,753
**IR:** [`weightscan3_mask32.ir`](weightscan3_mask32.ir)
**Method:** `generate_scan(0, walk="weight", weight_cap=3)` (weight-ordered null-space scan)

## Idea — weight ≤ 3 covers ~90% of instances

Identical construction to the cap-2 submission, extended one weight
class: s₀, 14 single flips, 91 pairs, 364 triples — 470 visits total.
The dev-suite coefficient-weight distribution puts 89.8% of secrets at
weight ≤ 3, so one IR serves both the 60% and the 80% band at
18,509,753 reads (265,674 lines), versus 23,676,539 (scan 4,095) and
30,226,172 (scan 8,191) for the reflected Gray walk.

## Numbers

Dev suite: recovery 0.8984. Two fresh 2,048-instance adjudication
draws: 0.9048, 0.8960.

## Reproduce

```python
import mask_sparse_parity as mp

ir = mp.generate_scan(0, walk="weight", weight_cap=3)
res = mp.evaluate_mask(ir)          # → cost 18,509,753, recovery 0.8984
```
