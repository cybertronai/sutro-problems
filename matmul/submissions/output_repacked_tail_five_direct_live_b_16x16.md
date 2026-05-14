# Matrix Multiplication

**Author:** Codex and Cosmin
**Date:** 2026-05-13
**Problem:** 16x16 matmul
**Cost:** 68,039
**IR:** [`output_repacked_tail_five_direct_live_b_16x16.ir`](output_repacked_tail_five_direct_live_b_16x16.ir)
**Method:** `generate_output_repacked_tail_five_direct_live_b_16x16` (five-direct output-repacked tail + live-B evacuation)

## Idea

This submission is a two-point improvement over
`output_repacked_tail_current_order_live_b_16x16`.  It keeps the same
current-order evacuation basin: 4,190 profitable live-B evacuation edges and
39 selected nonconflicting moves for a 392-point saving.

The improvement comes before evacuation.  A five-direct final tail writes four
direct outputs at `final_jb=7` plus one extra TMP direct output at
`tmp_jb=5`, making the base cost 68,431 instead of 68,433.  Applying the same
39 live-B evacuation moves lands at 68,039.

The scalar arithmetic remains ordinary 16x16 matmul.  The improvement is
entirely from lifetime and address placement.

## Path to 68,039

| step | score | savings |
|------|------:|--------:|
| five-direct current-order output-repacked tail | 68,431 | |
| + 39 live-B evacuation moves | **68,039** | 392 |

## Region Cost Breakdown

| region | addrs | reads | cost |
|--------|-------|------:|-----:|
| SB | 1 | 5,057 | 5,057 |
| TMP | 2 | 2,878 | 5,756 |
| sA | 3..7 | 4,223 | 10,617 |
| sC | 8..39 | 3,754 | 19,263 |
| B bulk | 40..295 | 1,185 | 14,249 |
| A bulk | 296..551 | 511 | 10,726 |
| C spill | 552..647 | 95 | 2,371 |
| **total** | | **17,703** | **68,039** |

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
python matmul/submissions/output_repacked_tail_five_direct_live_b_16x16.py
python matmul/experiments/random_true_matmul_check.py \
  matmul/submissions/output_repacked_tail_five_direct_live_b_16x16.ir \
  --trials 500 --seed 20260519 --min -59 --max 59
```

Observed locally:

```text
output_repacked_tail_five_direct_live_b_16x16.ir  cost=68,039
matmul/submissions/output_repacked_tail_five_direct_live_b_16x16.ir: cost=68,039 ok 500 random trials
```

The Agent AC scratch-placement search also exact-scored the IR and validated
sub-68,041 leaders with 200 arbitrary signed true-matmul trials.
