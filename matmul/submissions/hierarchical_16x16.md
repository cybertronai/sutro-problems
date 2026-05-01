# Matrix Multiplication

**Author:** [@sjbaebae](https://github.com/sjbaebae)
**Date:** 2026-04-30
**Problem:** 16×16 matmul
**Cost:** 80,217
**IR:** [`hierarchical_16x16.ir`](hierarchical_16x16.ir)
**Method:** `generate_hierarchical_16x16` (asymmetric reload)

## Idea

Two-level blocking with loop order `(i_block, k, j_block)` and `Tio = 4`.
A 64-cell sC accumulator (4×16 stripe of C) stays alive in cheap scratch
across the whole k-sweep of an i-block. Each i-block reads B four times
(once per j sub-block) but reads A only once.

That asymmetry is the key. Once `B_reload = 4×` and `A_reload = 1×`, the
right address layout is:

- B at the **cheap** end of bulk space (low addresses)
- A at the **expensive** tail (high addresses)

## Layout

| addrs       | region    | reads/cell |
|------------:|-----------|-----------:|
| 1           | sB        | very high  |
| 2           | tmp       | high       |
| 3..6        | sA (4)    | 1024       |
| 7..70       | sC (64)   | 64         |
| 71..326     | B bulk    | 4          |
| 327..582    | A bulk    | 1          |
| 583..838    | C bulk    | 1          |

Total bulk cost decomposes as `4·SumB_low + 1·SumA_high`, beating any
symmetric-reload scheme that would pay `r·(SumA + SumB)` for some single
reload count.

## Why it wins (133,783 → 80,217, −40%)

Two distinct moves: **(1)** pick a loop order that creates reload
asymmetry; **(2)** sort addresses by reads-per-cell descending. The
prior tiled baselines (`tiled_16x16`, `tiled_16x16_opt1`) had a
symmetric `(bi, bj, bk)` shape with no reload asymmetry to exploit, so
both A and B paid the same bulk cost. The hierarchical schedule pays
the asymmetry forward: B's 4× reload count is offset by living at
addresses that cost ~3× less than where A sits.
