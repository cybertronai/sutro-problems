# Matrix Multiplication

**Author:** [@sjbaebae](https://github.com/sjbaebae)
**Date:** 2026-04-30
**Problem:** 4×4 matmul
**Cost:** 800
**IR:** [`outer_product_4x4.ir`](outer_product_4x4.ir)
**Method:** `generate_outer_product_4x4` (size-1 sA scratch)

## Idea

Hold just **one** A element in scratch — `sA` at addr 1 — instead of caching
a whole row or tile. Loop nest is `(i, k, j)`: copy `A[i,k]` into sA once,
then sweep all four j's reading `A` from cost 1 instead of cost ⌈√(addr)⌉.

## Why it wins (1,316 → 800, −39%)

With sA pinned at the cheapest cell, the rest of the layout falls into a
read-count-descending order:

| addr  | cell  | reads | cost/read |
|------:|-------|------:|----------:|
| 1     | sA    | 64    | 1         |
| 2     | tmp   | 48    | 2         |
| 3..6  | sB    | 16    | 2         |
| 7..22 | C     | 16    | 3–5       |
| 23..38| A     | 4     | 5–7       |
| 39..54| B     | 4     | 7–8       |

Each multiply pays `cost(sA) + cost(sB) = 1 + 2 = 3` instead of the
baseline's `cost(A_bulk) + cost(B_bulk) = 5–8 + 7–8 ≈ 13`. The size-1
scratch is provably optimal among outer-product layouts: any larger sA
cache would push hotter cells (tmp, sB) to costlier addresses and lose
more than it saves on A reuse.
