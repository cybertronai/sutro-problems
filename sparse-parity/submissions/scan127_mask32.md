# Sparse Parity

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-08-27
**Problem:** Mask sparse parity, n=32 — 20% accuracy target
**Cost:** 17,331,683
**IR:** `mask_sparse_parity.generate_scan(127)` (regenerate; not checked in)
**Method:** `generate_scan(127)` (Gray scan, 127-step walk)

## Idea

The [Gray scan](scan_full_mask32.md) circuit with the null-space walk capped
at s = 127 steps: 24.9% recovery at 17,331,683 reads. The Gaussian
elimination + basis extraction dominate the cost (~17.1M reads at s = 0), so
short walks are nearly free — but that fixed cost is why this entry loses the
20% band to [ISD restarts](isd8_mask32.md) (12,042,480), which skip the
full-width solve entirely.

## Reproduce

```python
import mask_sparse_parity as mp

ir = mp.generate_scan(127)
mp.evaluate_mask(ir)          # → cost 17,331,683, recovery 0.2490
```
