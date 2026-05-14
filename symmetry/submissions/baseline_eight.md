# Symmetry — Eight-bit baseline

**Author:** baseline
**Date:** 2026-05-11
**Problem:** 8-bit palindrome
**Cost:** 29
**IR:** [`baseline_eight.ir`](baseline_eight.ir)
**Method:** `generate_baseline_eight` (4× `cmp eq` + 3× `and`)

## Idea

Same structure as the six-bit baseline, extended to 4 mirror pairs:

```
cmp 1, 1, 8, eq   # x[0] == x[7]?
cmp 2, 2, 7, eq   # x[1] == x[6]?
cmp 3, 3, 6, eq   # x[2] == x[5]?
cmp 4, 4, 5, eq   # x[3] == x[4]?
and 1, 2
and 1, 3
and 1, 4
```

Equality flags land at addrs 1–4; three `and` instructions accumulate
the conjunction at addr 1.

## Cost breakdown

| instruction        | reads              | cost |
|--------------------|--------------------|-----:|
| `cmp 1, 1, 8, eq`  | addr 1 + addr 8    |  1+3 |
| `cmp 2, 2, 7, eq`  | addr 2 + addr 7    |  2+3 |
| `cmp 3, 3, 6, eq`  | addr 3 + addr 6    |  2+3 |
| `cmp 4, 4, 5, eq`  | addr 4 + addr 5    |  2+3 |
| `and 1, 2`         | addr 1 + addr 2    |  1+2 |
| `and 1, 3`         | addr 1 + addr 3    |  1+2 |
| `and 1, 4`         | addr 1 + addr 4    |  1+2 |
| output read        | addr 1             |    1 |
| **total**          |                    | **29** |
