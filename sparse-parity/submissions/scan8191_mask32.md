# Sparse Parity

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-08-27
**Problem:** Mask sparse parity, n=32 — 80% accuracy target
**Cost:** 30,226,172
**IR:** [`scan8191_mask32.ir`](scan8191_mask32.ir)
**Method:** `generate_scan(8191)` (Gray scan, 8,191-step walk)

## Idea

The [Gray scan](scan_full_mask32.md) circuit with the null-space walk capped
at s = 8,191 steps — half the solution space: 86.7% recovery at 30,226,172
reads and 986,376 instructions, the cheapest known circuit at or above the
80% band. The walk hits a recovery plateau here: s = 10,239 and s = 12,287
measure the same 86.7%, so the remaining secrets hide in the last quarter of
the Gray order and only the [full walk](scan_full_mask32.md) collects them.

## Reproduce

```python
import mask_sparse_parity as mp

ir = mp.generate_scan(8191)
mp.evaluate_mask(ir)          # → cost 30,226,172, recovery 0.8672
```
