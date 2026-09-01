# Sparse Parity — packed-column scan, cap 1

**Author:** [@jurajselep](https://github.com/jurajselep)  
**Date:** 2026-09-01  
**Problem:** MASK32 sparse parity, 20% target  
**Cost:** 135,348  
**IR lines:** 10,732 (10,730 body operations)  
**IR:** [`packedscan1_mask32.ir`](packedscan1_mask32.ir)  
**Generator:** [`../packed_sparse_parity.py`](../packed_sparse_parity.py)

## Result

`generate_packed_scan(1)` recovers **20.0195%** of the deterministic
1,024-instance dev suite at static read cost **135,348**.

Two independent, final-sized deterministic validation suites (256 secrets × 8
repetitions, 2,048 instances each) produced **25.3418%** and
**25.1953%** recovery. Every miss at partial caps was the all-zero
mask; no non-secret mask was emitted.

## Construction

The 18 rows of each augmented `[X | y]` column are packed into three nonnegative
6-bit cells. Branchless Gaussian elimination therefore updates all rows with
three bytewise XORs rather than an 18×33 bit matrix. Earlier pivot columns are
not revisited: once a column is one-hot, a later unused pivot row is necessarily
zero there.

After elimination, the affine candidate state is only three packed pivot-row
cells. The coefficient vectors of weight at most `1` are visited in the
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

ir = packed.generate_packed_scan(1)
result = mp.evaluate_mask(ir)
assert len(ir.splitlines()) == 10732
assert result.cost == 135348
assert result.recovery == 0.2001953125
```

SHA-256 of the stored IR (including the final newline): `4160c6f5787bb456f8470eba1e66f28d6c225b1b019111f86b0fcae249759509`.
