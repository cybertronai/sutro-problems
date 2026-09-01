# Sparse Parity — packed-column scan, cap 5

**Authors:** [@jurajselep](https://github.com/jurajselep), [@b0nce](https://github.com/b0nce)
**Date:** 2026-09-01  
**Problem:** MASK32 sparse parity, 100% target  
**Cost:** 392,666
**IR lines:** 61,070 (61,068 body operations)
**IR:** [`packedscan5_mask32.ir`](packedscan5_mask32.ir)  
**Generator:** [`../packed_sparse_parity.py`](../packed_sparse_parity.py)

## Result

`generate_packed_scan(5)` recovers **100.0000%** of the deterministic
1,024-instance dev suite at static read cost **392,666**.

Two independent, final-sized deterministic validation suites (256 secrets × 8
repetitions, 2,048 instances each) produced **100.0000%** and
**100.0000%** recovery. Every miss at partial caps was the all-zero
mask; no non-secret mask was emitted.

## Construction

The 18 rows of each augmented `[X | y]` column are packed into three nonnegative
6-bit cells. Branchless Gaussian elimination therefore updates all rows with
three bytewise XORs rather than an 18×33 bit matrix. Earlier pivot columns are
not revisited: once a column is one-hot, a later unused pivot row is necessarily
zero there.

After elimination, the affine candidate state is only three packed pivot-row
cells. The coefficient vectors of weight at most `5` are visited in the
binary-reflected Gray order after filtering. Target pivot weights zero through
three use dedicated exact bit predicates. Zero-weight capture uses an inverted
select directly, and weight-one capture uses packed sum-versus-XOR; the
remaining rare targets use a packed popcount.

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

ir = packed.generate_packed_scan(5)
result = mp.evaluate_mask(ir)
assert len(ir.splitlines()) == 61070
assert result.cost == 392666
assert result.recovery == 1.0
```

SHA-256 of the stored IR (including the final newline): `9fcc726e1d4635dcb9ba9ff5e4150d75f7dfc221755010b9c1054ef52c01219a`.
