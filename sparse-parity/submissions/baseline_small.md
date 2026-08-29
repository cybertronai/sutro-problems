# Sparse parity

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-05-07
**Problem:** sparse parity (n=3, k=2, 4 train / 32 test)
**Cost:** 6,918
**IR:** [`baseline_small.ir`](baseline_small.ir)
**Method:** `generate_baseline_small` (try-each-candidate, v3 `xor` + `and` + `or` + `set`)

Baseline. The IR mirrors the brute-force solver in pure v3 IR: it
tries each of the `C(3, 2) = 3` candidate 2-subsets `T = (t0, t1)`,
computes an indicator `ind_T = AND_i (1 XOR (y_train[i] XOR
X_train[i,t0] XOR X_train[i,t1]))` from the training data, and uses
the indicator to OR-combine the per-candidate test predictions.
Because the training set is constructed so that exactly one
candidate explains every label, exactly one `ind_T` is 1, and the
final OR cleanly selects that candidate's prediction.

Memory layout (computed from `M_TRAIN=4`, `M_TEST=32`, `N_BITS=3`):

| region    | range    | role                                 |
|-----------|----------|--------------------------------------|
| pred      | 1..32    | output (32 predictions)              |
| X_train   | 33..44   | input (4 × 3, row-major)             |
| y_train   | 45..48   | input                                |
| X_test    | 49..144  | input (32 × 3, row-major)            |
| ONE       | 145      | constant 1, written by `set` (free)  |
| tmp       | 146      | scratch, reused                      |
| parity    | 147      | scratch, reused                      |
| matched_T | 148..159 | per-row match flag, 3 cands × 4 rows |
| ind_T     | 160..162 | per-candidate "explains all" flag    |
| predT     | 163      | scratch, reused per test row         |
| term_T    | 164..166 | per-candidate masked test prediction |

Op count: 1 `set` (free) + 301 binary ops = 302 instructions.

- Decoding (per candidate): 4 rows × 3 xors + 3 AND-reductions = 15 ops.
  Three candidates → 45 ops.
- Predictions (per test row): 3 candidates × (1 xor + 1 and) + 2 ORs = 8 ops.
  Thirty-two test rows → 256 ops.

Robustness: scored against three canonical seeds (one per possible
secret subset), and the cost is identical across all seeds — the
IR's read pattern is independent of input *values*.
