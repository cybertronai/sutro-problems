# Matrix Multiplication

**Author:** Matrix Multiplication
**Date:** 2026-04-29
**Problem:** 16×16 matmul
**Cost:** 133,783
**IR:** [`tiled_16x16.ir`](tiled_16x16.ir)
**Method:** `generate_tiled_16x16` (4×4 tiles)

## Tiled version

A 16×16 matmul is decomposed into a 4×4 grid of 4×4 tiles, evaluated as
sixteen `(bi, bj)` output blocks accumulated across four `bk` inner blocks.

Each `(bi, bj, bk)` step copies the relevant 4×4 A-tile and 4×4 B-tile from
the long-distance bulk into a small scratchpad region (the cheapest 48
addresses), runs the inner 4×4×4 multiply against a 4×4 sC accumulator, and
on the final `bk` copies sC out to the C bulk. Memory layout:

| region | range  | role                              |
|--------|--------|-----------------------------------|
| sA     | 1..16  | cached A tile (4×4)               |
| sB     | 17..32 | cached B tile (4×4)               |
| sC     | 33..48 | accumulator for current C tile    |
| tmp    | 49     | scratch for fused mul-add         |
| A bulk | 50..305  | input A (16×16)                 |
| B bulk | 306..561 | input B (16×16)                 |
| C bulk | 562..817 | output C (16×16)                |

The win over the naive baseline (340,704 → 133,783, ~61%) comes from
amortizing each long-distance bulk read across the inner-tile reuses
inside the scratchpad.
