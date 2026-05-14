# Matrix Multiplication

**Author:** Codex and Cosmin
**Date:** 2026-05-13
**Problem:** 16x16 matmul
**Cost:** 67,911
**IR:** [`output_repacked_tail_deferred_value_colored_live_b_16x16.ir`](output_repacked_tail_deferred_value_colored_live_b_16x16.ir)
**Method:** `generate_output_repacked_tail_deferred_value_colored_live_b_16x16` (output deferral + live-B evacuation + value-lifetime coloring)

## Idea

This submission builds on the 67,927 value-colored live-B evacuation trace.
Agent AG found that a small set of safe output-write deferrals changes value
lifetimes in a way that improves the same max-chain value coloring.

The best branch defers three compatible output writes:

- `op4733` writing addr `69` by 44 legal positions;
- `op4735` writing addr `70` by 4 legal positions;
- `op8285` writing addr `165` by 44 legal positions.

The live-B evacuation model still selects 39 moves with a 392-point pre-coloring
saving, and the post-evac exact score remains 68,041.  The improvement comes
from better value-lifetime coloring after the deferrals, reducing the final
weighted read cost to 67,911.

## Path to 67,911

| step | score | savings |
|------|------:|--------:|
| current-order output-repacked tail | 68,433 | |
| + 39 live-B evacuation moves | 68,041 | 392 |
| + value-lifetime address coloring | 67,927 | 114 |
| + three output-write deferrals before coloring | **67,911** | 16 |

## Bound Context

Agent AA's fixed-trace value-prefix lower bound for the 68,041 post-evac trace
is 67,612.  This 67,911 trace changes a few output-write lifetimes before
coloring; it preserves the instruction counts but improves the coloring result
by 16 points over 67,927.

## Cost Breakdown By Address Tier

| tier | addrs | reads | cost |
|------|-------|------:|-----:|
| 1 | 1 | 5,057 | 5,057 |
| 2 | 2..4 | 4,993 | 9,986 |
| 3 | 5..9 | 2,349 | 7,047 |
| 4 | 10..16 | 847 | 3,388 |
| 5 | 17..25 | 1,089 | 5,445 |
| 6 | 26..36 | 1,329 | 7,974 |
| 7 | 37..49 | 328 | 2,296 |
| 8 | 50..64 | 120 | 960 |
| 9 | 65..81 | 124 | 1,116 |
| 10..16 | 82..256 | 785 | 10,304 |
| 17..26 | 257..676 | 682 | 14,338 |
| **total** | | **17,703** | **67,911** |

## Instruction Distribution

| instruction | count | paid reads |
|-------------|------:|-----------:|
| `mul` | 4,096 | 8,192 |
| `add` | 3,840 | 7,680 |
| `copy` | 1,575 | 1,575 |
| output exit | 256 | 256 |
| **total** | **9,511 ops** | **17,703** |

## Verification

```bash
python matmul/submissions/output_repacked_tail_deferred_value_colored_live_b_16x16.py
python matmul/experiments/random_true_matmul_check.py \
  matmul/submissions/output_repacked_tail_deferred_value_colored_live_b_16x16.ir \
  --trials 500 --seed 20260523 --min -73 --max 73
```

Observed locally:

```text
output_repacked_tail_deferred_value_colored_live_b_16x16.ir  cost=67,911
matmul/submissions/output_repacked_tail_deferred_value_colored_live_b_16x16.ir: cost=67,911 ok 500 random trials
```

Agent AG also validated each sub-67,927 candidate with 100 arbitrary signed
true-matmul trials, including this best IR.
