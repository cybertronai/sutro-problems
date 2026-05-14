# Symmetry — Six-bit baseline

**Author:** baseline
**Date:** 2026-05-11
**Problem:** 6-bit palindrome (canonical RHW1986 task)
**Cost:** 20
**IR:** [`baseline_six.ir`](baseline_six.ir)
**Method:** `generate_baseline_six` (3× `cmp eq` + 2× `and`)

## Idea

Test each mirror pair with `cmp dest, a, b, eq`, then AND the three
results together:

```
cmp 1, 1, 6, eq   # x[0] == x[5]?
cmp 2, 2, 5, eq   # x[1] == x[4]?
cmp 3, 3, 4, eq   # x[2] == x[3]?
and 1, 2
and 1, 3
```

Each `cmp` overwrites its first input address with 0 or 1, so the three
equality flags land at addrs 1, 2, 3 — the cheapest slots available.
The two `and` instructions accumulate the conjunction in-place at addr 1.

## Cost breakdown

| instruction        | reads              | cost |
|--------------------|--------------------|-----:|
| `cmp 1, 1, 6, eq`  | addr 1 + addr 6    |  1+3 |
| `cmp 2, 2, 5, eq`  | addr 2 + addr 5    |  2+3 |
| `cmp 3, 3, 4, eq`  | addr 3 + addr 4    |  2+2 |
| `and 1, 2`         | addr 1 + addr 2    |  1+2 |
| `and 1, 3`         | addr 1 + addr 3    |  1+2 |
| output read        | addr 1             |    1 |
| **total**          |                    | **20** |
