# Sparse Parity

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-08-27
**Problem:** Mask sparse parity, n=32 — 40% accuracy target
**Cost:** 17,945,660
**IR:** [`scan511_mask32.ir`](scan511_mask32.ir)
**Method:** `generate_scan(511)` (Gray scan, 511-step walk)

## Idea

The [Gray scan](scan_full_mask32.md) circuit with the null-space walk capped
at s = 511 steps. Because the secret tends to be visited early in the walk
(its Gray coefficient vector is low-weight), the first 511 of 16,383 steps
already capture 42.2% of secrets — at 17,945,660 reads and 249,096
instructions, this is the cheapest known circuit at or above the 40% band.
s = 511 is the smallest swept setting that clears 40%; ISD restarts plateau
near 36% (see [isd8_mask32](isd8_mask32.md)) and cannot compete here.

## Reproduce

```python
import mask_sparse_parity as mp

ir = mp.generate_scan(511)
mp.evaluate_mask(ir)          # → cost 17,945,660, recovery 0.4219
```
