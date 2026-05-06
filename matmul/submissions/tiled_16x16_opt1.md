# Tiled 16×16 with Optimal Scratchpad Layout

**Author:** Seth Stafford (@SethTS)
**Date:** 2026-04-30
**Problem:** 16×16 matmul
**Cost:** 110,743
**IR:** [`tiled_16x16_opt1.ir`](tiled_16x16_opt1.ir)
**Method:** `generate_tiled_16x16_opt1` (tmp@1)

## Approach

Starting from `generate_tiled_16x16` (cost 133,783), profile the read count for every scratchpad slot to find the optimal address assignment.

**Read counts:**

| Slot | Reads | How |
|------|------:|-----|
| `tmp` | 4,096 | read after every `mul` — once per inner loop iteration (nb³ × T³) |
| each `sA` cell | 256 | read T times per (bi,bj,bk) block × nb³ blocks |
| each `sB` cell | 256 | same |
| each `sC` cell | 256 | 15 accumulation reads + 1 writeback read per (bi,bj) × nb² |

`tmp` is 16× hotter than any other slot. Since `_cost(addr) = ⌈√addr⌉` is
monotone, the optimal assignment is greedy: place the hottest slot at the
lowest address. The 48 `sA`/`sB`/`sC` cells have identical read counts, so
their ordering among the remaining addresses does not matter.

## New Layout

| Region | Addresses | Change |
|--------|-----------|--------|
| `tmp`  | 1         | was 49 (cost 7 → 1) |
| `sA`   | 2–17      | was 1–16 |
| `sB`   | 18–33     | was 17–32 |
| `sC`   | 34–49     | was 33–48 |
| A/B/C bulk | 50–817 | unchanged |

## Result

| Method | Cost | Δ |
|--------|-----:|---|
| `generate_tiled_16x16` | 133,783 | baseline |
| `generate_tiled_16x16_opt1` | 110,743 | **-17.2%** |

## Optimality Note

This layout is provably optimal for this algorithm. Since all 48 non-tmp
slots have equal read counts, no further permutation of addresses 2–49 can
reduce cost. Further improvement requires algorithmic changes: eliminating
the `tmp` indirection (a fused mul-add instruction), reducing sC accumulation
reads, or a different tiling strategy.
