# Sparse Parity — packed-column scan, cap 2

**Author:** [@jurajselep](https://github.com/jurajselep)  
**Date:** 2026-09-01  
**Problem:** MASK32 sparse parity, 40% target  
**Cost:** 151,943  
**IR lines:** 13,752 (13,750 body operations)  
**IR:** [`packedscan2_mask32.ir`](packedscan2_mask32.ir)  
**Generator:** [`../packed_sparse_parity.py`](../packed_sparse_parity.py)

## Result

`generate_packed_scan(2)` recovers **56.5430%** of the deterministic
1,024-instance dev suite at static read cost **151,943**.

Two independent, final-sized deterministic validation suites (256 secrets × 8
repetitions, 2,048 instances each) produced **62.1582%** and
**60.5469%** recovery. Every miss at partial caps was the all-zero
mask; no non-secret mask was emitted.

## Construction

The 18 rows of each augmented `[X | y]` column are packed into three nonnegative
6-bit cells. Branchless Gaussian elimination therefore updates all rows with
three bytewise XORs rather than an 18×33 bit matrix. Earlier pivot columns are
not revisited: once a column is one-hot, a later unused pivot row is necessarily
zero there.

After elimination, the affine candidate state is only three packed pivot-row
cells. The coefficient vectors of weight at most `2` are visited in the
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

ir = packed.generate_packed_scan(2)
result = mp.evaluate_mask(ir)
assert len(ir.splitlines()) == 13752
assert result.cost == 151943
assert result.recovery == 0.5654296875
```

SHA-256 of the stored IR (including the final newline): `f9204a3fd09d8d0dfc8dc4b4d1f177d8159ed615eb4ab04815f6342e26fb3db4`.
