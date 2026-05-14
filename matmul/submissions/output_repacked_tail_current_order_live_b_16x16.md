# Matrix Multiplication

**Author:** Codex and Cosmin
**Date:** 2026-05-13
**Problem:** 16x16 matmul
**Cost:** 68,041
**IR:** [`output_repacked_tail_current_order_live_b_16x16.ir`](output_repacked_tail_current_order_live_b_16x16.ir)
**Method:** `generate_output_repacked_tail_current_order_live_b_16x16` (current-order output-repacked tail + live-B evacuation)

## Idea

This submission comes from a record-family evacuation sweep.  Instead of applying live-B evacuation to the 68,390 submission directly, it starts from a nearby current-order `scratch4/jb7` output-repacked tail variant with cost 68,433.

That slightly worse base exposes a much larger evacuation basin: 4,190 profitable live-B evacuation edges and 39 selected nonconflicting moves.  The 39 moves save 392 weighted cost units, landing at 68,041.

The scalar arithmetic remains ordinary 16x16 matmul.  The improvement is entirely from lifetime and address placement: still-live B values are copied into cheap dead output homes, the corresponding outputs are written into the B values' old homes, and later B reads are redirected to the cheaper former output homes.

## Path to 68,041

| step | score | savings |
|------|------:|--------:|
| current-order `scratch4/jb7` output-repacked tail | 68,433 | |
| + 39 live-B evacuation moves | **68,041** | 392 |

## Region cost breakdown

| region | addrs | reads | cost |
|--------|-------|------:|-----:|
| SB | 1 | 5,057 | 5,057 |
| TMP | 2 | 2,879 | 5,758 |
| sA | 3..6 | 4,100 | 10,248 |
| sC | 7..38 | 3,868 | 19,576 |
| B bulk | 39..294 | 1,191 | 14,269 |
| A bulk | 295..550 | 512 | 10,738 |
| C spill | 551..646 | 96 | 2,395 |
| **total** | | **17,703** | **68,041** |

## Instruction distribution

| instruction | count | paid reads |
|-------------|------:|-----------:|
| `mul` | 4,096 | 8,192 |
| `add` | 3,840 | 7,680 |
| `copy` | 1,575 | 1,575 |
| output exit | 256 | 256 |
| **total** | **9,511 ops** | **17,703** |

## Verification

```bash
python matmul/submissions/output_repacked_tail_current_order_live_b_16x16.py
python matmul/experiments/random_true_matmul_check.py \
  matmul/submissions/output_repacked_tail_current_order_live_b_16x16.ir \
  --trials 300 --seed 20260518 --min -53 --max 53
```

Observed locally:

```text
output_repacked_tail_current_order_live_b_16x16.ir  cost=68,041
matmul/submissions/output_repacked_tail_current_order_live_b_16x16.ir: cost=68,041 ok 300 random trials
```

The experiment sweep also validated the same IR with 100 required arbitrary signed trials plus an additional 1000 signed trials.
