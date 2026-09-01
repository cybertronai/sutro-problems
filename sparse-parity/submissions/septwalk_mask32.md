# Sparse Parity — septwalk (100% band)

**Date:** 2026-08-31
**Problem:** Mask sparse parity, n=32 — 100% accuracy target
**Cost:** 938,331 (recovery 1.0000 on the 1,024-instance dev suite and
20,480/20,480 on ten fixed, hashed 2,048-instance adjudication suites;
see [`packed_records_audit.json`](packed_records_audit.json))
**IR:** [`septwalk_mask32.ir`](septwalk_mask32.ir)
**Generator:** [`septwalk.py`](septwalk.py)

## Method — septet-packed RREF + row-coordinate walk

Same coverage argument as the weightscan5 record (visit every
coefficient vector of weight ≤ k = 5 over the 14 free columns; fails
only on rank-deficient draws, ~2⁻¹⁴), but the circuit is repacked:

* **Septet packing.** Rows of [X|y] (33 bits) live in 5 cells of 7 bits
  (values stay ≤ 127 so `div` is a clean unsigned shift).  GF(2) row
  ops cost 5 XORs instead of 33 masked selects; one Gauss-Jordan pivot
  is ~570 ops instead of ~3,200.  Dynamic pivoting identical to the
  record.
* **Row-coordinate walk.**  The walk state z is the pivot-part of the
  current solution (18 row-bits, 3 septets); the free part is the
  coefficient vector itself, known statically per visit, so
  weight(w) = popcount(z) + |a_t| is tested against a constant cell.
  A flip costs 3 XORs; popcount is a merged-tail SWAR (26 ops),
  specialized to a 3-op `z == 0` test for the 2,002 weight-5 visits
  and an 18-op power-of-two test for the 1,001 weight-4 visits.
* **Coefficient-only capture.**  By unique identifiability OK fires at
  most once, so `xor cap[j], OK` for the statically known j ∈ a_t
  rebuilds the secret's coefficient vector — no per-visit output
  selects.  The 32-bit mask is reconstructed once at the end
  (zz = ycol ⊕ Σ cap[j]·A[j], then per-column pivot/free select).
* **Revolving-door order** (2 flips per transition, 6,947 total vs
  8,959 lexicographic).
* **Two-phase staging** (`stage_septwalk_layout`): the walk re-addresses
  from 1 into the dead RREF range; 141 live cells cross the boundary
  through hazard-free two-round scratch copies.

## Reproduce

```python
import sys; sys.path.insert(0, "submissions")
import mask_sparse_parity as mp
import septwalk as at

ir = at.generate_staged()              # == submissions/septwalk_mask32.ir
res = mp.evaluate_mask(ir)             # → cost 938331, recovery 1.0
```
