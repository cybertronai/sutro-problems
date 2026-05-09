# Sparse parity — Gaussian elimination (medium)

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-05-08
**Problem:** sparse parity (n=8, k=3, 8 train / 64 test)
**Cost:** 473,046
**IR:** [`ge_medium.ir`](ge_medium.ir)
**Generator:** [`ge_medium.py`](ge_medium.py)
**Method:** `generate_ge_medium` (branchless GF(2) Gaussian elimination, v3 `cmp` + `select`, with brute-force fallback for rank-deficient instances)

## Algorithm

Identical structure to [`ge_small`](ge_small.md), generalized to `n=8`, `k=3`, `m=8`, `m_test=64`. The shared core is `_generate_ge(spec)` in [`ge_small.py`](ge_small.py); this submission is just a thin wrapper:

```python
from ge_small import _generate_ge
def generate_ge_medium(): return _generate_ge(MEDIUM)
```

The IR copies the training inputs into an in-memory `m × (n+1)` augmented matrix `[A | b]`, then performs branchless GF(2) row-reduction with full pivot tracking using v3 `cmp` + `select`. For each column it finds the first **unused** row with a 1 in that column, builds a pivot-row buffer via select chain (no indirect addressing), and eliminates the column from every other row. After RREF the secret indicator is read out via select chain over `pivot_row_for_col[c]`. A weight-fix step handles rank-`(n−1)` cases by flipping the free variable and propagating the off-diagonal RREF entries. For deeper rank deficiency, an `(A·s ≠ b) OR (weight ≠ k)` check selects between the GE result and a brute-force result (enumerates the `C(8, 3) = 56` weight-`k` candidates).

## Cost vs the try-each-candidate baseline

| baseline                              |   cost | ops    |
|---------------------------------------|-------:|-------:|
| `generate_baseline_medium` (v3, brute) | 816,251 | 16,457 |
| `generate_ge_medium` (v3, GE)          | **473,046** | **8,590** |

GE runs in `O(n³)` for decoding and `O(n)` per test row for prediction (total `O(n m_test)`). The brute-force-baseline is `O(C(n,k) · m)` for decoding and `O(C(n,k) · m_test)` for prediction. With `n=8`, `k=3`, `C(8,3)=56`, GE is roughly 2× cheaper.

## Robustness

Verified at 100 % over 500 fresh `score_medium` calls (each draws 8 random canonical seeds, see `_canonical_seeds`).
