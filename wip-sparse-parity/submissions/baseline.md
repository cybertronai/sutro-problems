# Sparse parity

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-05-07
**Problem:** sparse parity (n=3, k=2, 5 train / 5 test)
**Cost:** 91
**IR:** [`baseline.ir`](baseline.ir)
**Method:** `generate_baseline` (naive, v2 `xor`)

Baseline. The Python brute-force solver enumerates the `C(3, 2) = 3`
candidate subsets and recovers the unique secret subset from the
training labels (free, just like matmul's algorithm choice is free);
the IR then predicts each test row with a single `xor` over the two
secret columns of `X_test`. Memory layout:

| region   | range | role                              |
|----------|-------|-----------------------------------|
| X_train  | 1..15 | input (row-major, 5 × 3, unread)  |
| y_train  | 16..20 | input (unread)                   |
| X_test   | 21..35 | input (row-major, 5 × 3)         |
| pred     | 36..40 | output (5 predictions)           |

Per-row cost: each `xor` reads two `X_test` cells (avg ~5–6 each), and
the exit reads each pred cell (avg ~7). 5 xors × ~11 + 5 outputs × ~7
= **91**.
