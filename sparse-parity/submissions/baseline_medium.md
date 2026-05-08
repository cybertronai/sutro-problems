# Sparse parity (medium)

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-05-07
**Problem:** sparse parity (n=8, k=3, 8 train / 64 test)
**Cost:** 816,251
**IR:** [`baseline_medium.ir`](baseline_medium.ir)
**Method:** `generate_baseline_medium` (try-each-candidate, v3 `xor` + `and` + `or` + `set`)

Same try-each-candidate algorithm as the small baseline, but over
`C(8, 3) = 56` candidate 3-subsets. For each candidate `T = (t0,t1,t2)`:

- compute `parity_T_i = y_train[i] XOR X_train[i,t0] XOR X_train[i,t1] XOR X_train[i,t2]`
- `matched_T_i = parity_T_i XOR 1`
- `ind_T = AND_i matched_T_i` (1 iff T explains every training row)

Each test row is then `OR_T (ind_T AND (X_test[j,t0] XOR X_test[j,t1] XOR X_test[j,t2]))`.

Op count: 1 `set` (free) + 16,456 binary ops = 16,457 instructions.

- Decoding (per candidate): 8 rows × 4 xors + 7 AND-reductions = 39 ops.
  Fifty-six candidates → 2,184 ops.
- Predictions (per test row): 56 candidates × (2 xors + 1 and) + 55 ORs = 223 ops.
  Sixty-four test rows → 14,272 ops.

Robustness: scored against 8 canonical seeds (8 distinct secrets out
of 56 possible), and the cost is identical across seeds (the IR's
read pattern is independent of input *values*).
