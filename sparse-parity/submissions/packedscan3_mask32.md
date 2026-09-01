# Sparse Parity — packed-column scan, cap 3

**Author:** [@jurajselep](https://github.com/jurajselep)  
**Date:** 2026-09-01  
**Problem:** MASK32 sparse parity, 60% and 80% target  
**Cost:** 200,937  
**IR lines:** 23,325 (23,323 body operations)  
**IR:** [`packedscan3_mask32.ir`](packedscan3_mask32.ir)  
**Generator:** [`../packed_sparse_parity.py`](../packed_sparse_parity.py)

## Result

`generate_packed_scan(3)` recovers **89.8438%** of the deterministic
1,024-instance dev suite at static read cost **200,937**.

Two independent, final-sized deterministic validation suites (256 secrets × 8
repetitions, 2,048 instances each) produced **89.9902%** and
**88.8184%** recovery. Every miss at partial caps was the all-zero
mask; no non-secret mask was emitted.

## Construction

The 18 rows of each augmented `[X | y]` column are packed into three nonnegative
6-bit cells. Branchless Gaussian elimination therefore updates all rows with
three bytewise XORs rather than an 18×33 bit matrix. Earlier pivot columns are
not revisited: once a column is one-hot, a later unused pivot row is necessarily
zero there.

After elimination, the affine candidate state is only three packed pivot-row
cells. The coefficient vectors of weight at most `3` are visited in the
binary-reflected Gray order after filtering. Target pivot weights zero through
three use dedicated exact bit predicates; the remaining rare targets use a
packed popcount.

A two-phase SSA pass removes dead writes and reuses cells by exact liveness.
The elimination phase uses 594 slots, equal to the number of simultaneously
initialized inputs. A final frequency sort is the rearrangement-inequality
optimum for the emitted fixed access trace.

For the full cap-5 proof, exact walk lower bound, arbitrary-circuit lower-bound
status, and rank-deficient caveat, see
[`../doc/packed_scan_lower_bound.md`](../doc/packed_scan_lower_bound.md).

## Reproduce

```python
import mask_sparse_parity as mp
import packed_sparse_parity as packed

ir = packed.generate_packed_scan(3)
result = mp.evaluate_mask(ir)
assert len(ir.splitlines()) == 23325
assert result.cost == 200937
assert result.recovery == 0.8984375
```

SHA-256 of the stored IR (including the final newline): `55ee1193a7f5ac6e43eb675f873d02b01e6fd4c8b32e79800c935310bf6f3f6d`.
