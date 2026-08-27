# Sparse Parity

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-08-27
**Problem:** Mask sparse parity, n=32 — 80% accuracy target
**Cost:** 33,501,030
**IR:** `mask_sparse_parity.generate_scan(10239)` (regenerate; not checked in)
**Method:** `generate_scan(10239)` (Gray scan, 10,239-step walk)

## Idea

The [Gray scan](scan_full_mask32.md) circuit with the null-space walk capped
at s = 10,239 steps: 86.7% recovery at 33,501,030 reads — identical recovery
to the [8,191-step walk](scan8191_mask32.md) for 3.3M more reads. The dev
suite's remaining secrets sit in the final quarter of the Gray order, so
intermediate walk lengths are dead weight and the shorter walk holds the 80%
band.

## Reproduce

```python
import mask_sparse_parity as mp

ir = mp.generate_scan(10239)
mp.evaluate_mask(ir)          # → cost 33,501,030, recovery 0.8672
```
