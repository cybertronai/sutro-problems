# Sparse Parity — septwalk at weight_cap=3 (80% band)

**Date:** 2026-08-31
**Problem:** Mask sparse parity, n=32 — 80% accuracy band
**Cost:** 493,193
**IR:** [`septwalk_wcap3_mask32.ir`](septwalk_wcap3_mask32.ir)
**Generator:** [`septwalk.py`](septwalk.py) — `generate_staged(weight_cap=3)`

## septwalk was only exercised at weight_cap=5; a lower cap serves the 80% band cheaper

`septwalk_mask32.md` (this same generator, `weight_cap=5`) was built for
100% recovery and became that band's record. It was never checked at
lower `weight_cap`, but the family's own coverage argument scales down
cleanly: visiting every coefficient vector of weight ≤ `weight_cap` over
the 14 free columns is guaranteed to include the secret whenever
`|secret ∩ free cols| ≤ weight_cap` (unique identifiability — no poison
captures, same as the record). At `weight_cap=3` that already exceeds
the 80% line, at a small fraction of the weight-5 walk's cost, because
the visit count (and so the walk's share of total cost) drops sharply
with the cap: 2^14 possible coefficient vectors have C(14,0)+…+C(14,3) =
471 of weight ≤ 3, vs 6,947 (the revolving-door schedule) at weight ≤ 5.

No changes to `septwalk.py`, `_emit_rref`, or the two-phase staging —
this is the existing `generate_staged` entry point at a smaller
`weight_cap`, exactly as the other walk generators use different knob values.

## Numbers

Dev suite (1,024 instances): cost 493,193, recovery 0.8984
(46,391 lines). On ten fixed, hashed 2,048-instance suites it scores
18,483/20,480 overall, with per-suite recovery min 0.8848, mean 0.9025,
max 0.9277 — clearing the 80% line by 8.48 points in the minimum suite.

Suite keys, hashes, successes, and denominators are in
[`packed_records_audit.json`](packed_records_audit.json).

## Reproduce

```python
import sys; sys.path.insert(0, "submissions")
import mask_sparse_parity as mp
import septwalk

ir = septwalk.generate_staged(weight_cap=3)
res = mp.evaluate_mask(ir)             # → cost 493,193, recovery 0.8984
```
