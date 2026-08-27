# Sparse Parity

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-08-26
**Problem:** Mask sparse parity, n=32 — 60% accuracy target
**Cost:** 23,676,539
**IR:** [`scan4095_mask32.ir`](scan4095_mask32.ir)
**Method:** `generate_scan(4095)` (Gray scan, 4,095-step walk)

## Idea

The [Gray scan](scan_full_mask32.md) circuit with the null-space walk capped
at s = 4,095 steps: 74.2% recovery at 23,676,539 reads and 593,160
instructions — the cheapest known circuit at or above the 60% band (it clears
it with a wide margin; no cheaper swept setting reaches 60%). Only the Gray
scan family reaches this band under the 2,000,000-instruction cap.

## Reproduce

```python
import mask_sparse_parity as mp

ir = mp.generate_scan(4095)
mp.evaluate_mask(ir)          # → cost 23,676,539, recovery 0.7422
```
