# Sparse Parity

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-08-27
**Problem:** Mask sparse parity, n=32 — 60% accuracy target
**Cost:** 26,951,367
**IR:** `mask_sparse_parity.generate_scan(6143)` (regenerate; not checked in)
**Method:** `generate_scan(6143)` (Gray scan, 6,143-step walk)

## Idea

The [Gray scan](scan_full_mask32.md) circuit with the null-space walk capped
at s = 6,143 steps: 76.6% recovery at 26,951,367 reads. The extra 2,048 steps
over the [4,095-step walk](scan4095_mask32.md) buy only +2.3 pp of recovery
for 3.3M extra reads — the shorter walk holds the 60% band.

## Reproduce

```python
import mask_sparse_parity as mp

ir = mp.generate_scan(6143)
mp.evaluate_mask(ir)          # → cost 26,951,367, recovery 0.7656
```
