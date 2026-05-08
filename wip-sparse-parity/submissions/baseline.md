# Sparse parity

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-05-07
**Problem:** sparse parity (n=3, k=2, 5 train / 5 test)
**Cost:** 1,269
**IR:** [`baseline.ir`](baseline.ir)
**Method:** `generate_baseline` (try-each-candidate, v2 `xor` + `and` + `or` + `set`)

Baseline. The IR mirrors the brute-force solver in pure v2 IR: it
tries each of the `C(3, 2) = 3` candidate 2-subsets `T = (t0, t1)`,
computes an indicator `ind_T = AND_i (1 XOR (y_train[i] XOR
X_train[i,t0] XOR X_train[i,t1]))` from the training data, and uses
the indicator to OR-combine the per-candidate test predictions.
Because the training set is constructed so that exactly one
candidate explains every label, exactly one `ind_T` is 1, and the
final OR cleanly selects that candidate's prediction.

Memory layout:

| region    | range  | role                                 |
|-----------|--------|--------------------------------------|
| pred      | 1..5   | output (5 predictions)               |
| X_train   | 6..20  | input (5 × 3, row-major)             |
| y_train   | 21..25 | input                                |
| X_test    | 26..40 | input (5 × 3, row-major)             |
| ONE       | 41     | constant 1, written by `set` (free)  |
| tmp       | 42     | scratch, reused                      |
| parity    | 43     | scratch, reused                      |
| matched_T | 44..58 | per-row match flag, 3 cands × 5 rows |
| ind_T     | 59..61 | per-candidate "explains all" flag    |
| predT     | 62     | scratch, reused per test row         |
| term_T    | 63..65 | per-candidate masked test prediction |

Op count: 1 `set` (free) + 97 binary ops = 98 instructions. Cost
breakdown: 97 binary ops × 2 reads/op + 5 output reads = 199 read
events, with addresses concentrated in `[1, 65]` (mean ⌈√addr⌉ ≈ 6).

Robustness: scored against three canonical seeds (one per possible
secret subset), and the cost is identical across all seeds — the
IR's read pattern is independent of input *values*.
