# Sparse Parity

**Author:** [@zh4ngx](https://github.com/zh4ngx)
**Date:** 2026-08-29
**Problem:** Mask sparse parity, n=32 — 60% accuracy target
**Cost:** 2,137,725
**IR:** [`siswalk1_cap3_mask32.ir`](siswalk1_cap3_mask32.ir)
**Method:** `generate_sis_mask(1, 3)` (static information set + weight-ordered walk)

## Idea — same family, one more weight class

Identical construction to the cap-2 submission with the walk extended
to weight ≤ 3 (470 visits), covering 89.8% of the coefficient-weight
distribution. One static information set, seed 0.

## Numbers

Dev suite: recovery 0.6348 at 2,732,251 reads (124,206 lines;
layout-optimized 2026-08-30 from 6,082,625 via
`renumber_addresses` and staged into the dead RREF address range
via `optimize_layout`, recovery bit-identical). Fresh 2,048-instance
draws post-staging: 0.6455, 0.6270. Previous record: 18,509,753
(weightscan3).

## Reproduce

```python
import mask_sparse_parity as mp

ir = mp.optimize_layout(mp.generate_sis_mask(1, 3))
res = mp.evaluate_mask(ir)          # → cost 2,137,725, recovery 0.6348
```
