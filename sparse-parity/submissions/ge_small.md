# Sparse parity — Gaussian elimination (small)

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-05-08
**Problem:** sparse parity (n=3, k=2, 4 train / 32 test)
**Cost:** 22,238
**IR:** [`ge_small.ir`](ge_small.ir)
**Generator:** [`ge_small.py`](ge_small.py)
**Method:** `generate_ge_small` (branchless GF(2) Gaussian elimination, v3 `cmp` + `select`, with brute-force fallback for rank-deficient instances)

## Algorithm

The IR copies the training inputs into an in-memory augmented matrix `[A | b]` of shape `m × (n+1)` and performs branchless GF(2) row-reduction with full pivot tracking.

For each column `c = 0…n−1`:

1. **Pivot search.** Iterate over every row `r = 0…m−1`. The first row with `M[r][c] = 1` and `used[r] = 0` becomes the pivot. The "first unused row" rule (rather than `r ≥ c`) is necessary because earlier columns may be free, leaving low-index rows still available as pivots for later columns.
2. **Pivot-row buffer.** Build `PR[j] = M[pivot_idx][j]` via a select chain, since v3 has no indirect addressing.
3. **Elimination.** For every other row `i ≠ pivot_idx` with `M[i][c] = 1`, XOR `PR` into row `i`. Both the "≠ pivot" check and the "M[i][c] == 1" gate are expressed via `cmp` + `select`.

After RREF the secret indicator falls out of the augmented column at each pivot row. A select chain reads `s[c] = M[pivot_for_col[c]][n]` (or 0 for free columns).

## Rank-deficient handling

Empirically across 1,000 identifiable training sets:

| rank(A) | frequency |
|---------|-----------|
| 3       | ~70 %     |
| 2       | ~29 %     |
| 1       | ~0.6 %    |

For `rank = n` GE alone gives the secret. For `rank = n − 1` (one free column) the IR computes the integer weight of the default solution and, if `weight(s_GE) ≠ k`, branchlessly flips the free variable *and* every pivot row whose RREF entry in the free column is 1. For the rare `rank ≤ n − 2` case (two free columns), neither single-flip nor `A·s = b` verification alone is sufficient: the affine solution space contains multiple consistent vectors of differing weights. The IR therefore also computes a brute-force result by enumerating the `C(3, 2) = 3` weight-`k` candidates, AND-reducing the per-row match indicator across training rows for each, and OR-combining the unique winner. A `(verify failed) OR (weight ≠ k)` predicate selects between the GE result and the brute-force result.

## Cost breakdown

| section                              | ops |
|--------------------------------------|----:|
| constants + augmented-matrix copy    | 28  |
| pivot search (3 cols × 4 rows)       | 84  |
| pivot-row buffer build (3 × 16)      | 48  |
| RREF elimination (3 cols × m loops)  | 252 |
| dynamic readout of s\_GE             | 36  |
| weight-fix corrections               | 99  |
| verify + weight check                | 35  |
| brute-force fallback (3 candidates)  | 39  |
| s\_brute composition + final select  | 9   |
| predictions (32 test rows)           | 192 |
| **total**                            | **822** |

(IR contains 811 instructions after `set` ops drop out of the simulation cost; `set` itself is free.) Most reads land at addresses 1…200 with average `⌈√addr⌉ ≈ 9`, giving a total cost of **22,238**.

## Robustness

Verified at 100 % over 2,000 fresh `score_small` calls (each draws 3 random canonical seeds covering all 3 possible secrets).
