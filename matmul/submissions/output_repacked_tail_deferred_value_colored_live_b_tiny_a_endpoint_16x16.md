# Matrix Multiplication

**Author:** Codex and Cosmin<br>
**Date:** 2026-05-14<br>
**Problem:** 16x16 matmul<br>
**Cost:** 67,821<br>
**IR:** [`output_repacked_tail_deferred_value_colored_live_b_tiny_a_endpoint_16x16.ir`](output_repacked_tail_deferred_value_colored_live_b_tiny_a_endpoint_16x16.ir)<br>
**Method:** output deferral + tiny A-input staging mask + staged-reload endpoint lift + value-lifetime coloring

## Idea

This submission builds on the 67,911 AG trace.  The AG trace already combines
live-B evacuation, three output-write deferrals, and value-lifetime coloring.
Wave 13 found one more lifetime-shaping move: stage only eight early A inputs
from row group `bi=0`, `k=0..1`, all four local row lanes.  Wave 14 refined
that to a smaller five-cell mask: all four `k=0` lanes plus `k=1,ii=0`.
Wave 15 adds back exactly one cell, `k=1,ii=1`, and lifts the matching staged
reload endpoint left by five legal slots.  The six-cell mask keeps the Wave 14
copy savings mostly intact while recovering the useful AY endpoint lifetime.

For each selected A cell, the uncolored trace inserts a temporary copy after
the first-column load:

```text
copy temp,sA[ii]
```

Then the second-column reload reads that temporary instead of the original A
input.  This shortens the original A-input intervals while keeping the
B-friendly macro order.  After value coloring, the temporary high addresses
are packed back into the normal address range.

## Path to 67,821

| step | score | savings |
|------|------:|--------:|
| current-order output-repacked tail | 68,433 | |
| + 39 live-B evacuation moves | 68,041 | 392 |
| + value-lifetime address coloring | 67,927 | 114 |
| + three output-write deferrals before coloring | 67,911 | 16 |
| + stage `bi=0`, `k=0..1` A reloads before coloring | 67,834 | 77 |
| + refine to mask `k0ii0..3 + k1ii0` before coloring | **67,826** | 8 |
| + add `k1ii1` and lift its staged reload endpoint left five slots | **67,821** | 5 |

## Cost Breakdown By Address Tier

| tier | addrs | reads | cost |
|------|-------|------:|-----:|
| 1 | 1 | 5,058 | 5,058 |
| 2 | 2..4 | 5,001 | 10,002 |
| 3 | 5..9 | 2,351 | 7,053 |
| 4 | 10..16 | 847 | 3,388 |
| 5 | 17..25 | 1,089 | 5,445 |
| 6 | 26..36 | 1,329 | 7,974 |
| 7 | 37..49 | 328 | 2,296 |
| 8 | 50..64 | 120 | 960 |
| 9 | 65..81 | 124 | 1,116 |
| 10..16 | 82..256 | 785 | 10,304 |
| 17..26 | 257..676 | 677 | 14,225 |
| **total** | | **17,709** | **67,821** |

## Instruction Distribution

| instruction | count | paid reads |
|-------------|------:|-----------:|
| `mul` | 4,096 | 8,192 |
| `add` | 3,840 | 7,680 |
| `copy` | 1,581 | 1,581 |
| output exit | 256 | 256 |
| **total** | **9,517 ops** | **17,709** |

## Verification

```bash
python matmul/submissions/output_repacked_tail_deferred_value_colored_live_b_tiny_a_endpoint_16x16.py
```

Observed locally:

```text
output_repacked_tail_deferred_value_colored_live_b_tiny_a_endpoint_16x16.ir  cost=67,821
```

The promoted IR also passed exact read-count agreement and 1,000 random signed
true-matmul trials with values in `[-73, 73]` (seed `2026051504`).
