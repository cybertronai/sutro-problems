# Sparse Parity

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-08-27
**Problem:** Mask sparse parity, n=32 — 40% accuracy target
**Cost:** 18,764,343
**IR:** `mask_sparse_parity.generate_scan(1023)` (regenerate; not checked in)
**Method:** `generate_scan(1023)` (Gray scan, 1,023-step walk)

## Idea

The [Gray scan](scan_full_mask32.md) circuit with the null-space walk capped
at s = 1,023 steps: 49.2% recovery at 18,764,343 reads. Clears the 40% band
with slack, but the shorter [511-step walk](scan511_mask32.md) already clears
it for 818,683 fewer reads, which makes that the band record.

## Reproduce

```python
import mask_sparse_parity as mp

ir = mp.generate_scan(1023)
mp.evaluate_mask(ir)          # → cost 18,764,343, recovery 0.4922
```
