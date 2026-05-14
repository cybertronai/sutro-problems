# Matrix Multiplication

**Author:** Codex and Cosmin
**Date:** 2026-05-13
**Problem:** 16x16 matmul
**Cost:** 68,341
**IR:** [`output_repacked_tail_live_b_16x16.ir`](output_repacked_tail_live_b_16x16.ir)
**Method:** `generate_output_repacked_tail_live_b_16x16` (output-repacked tail + live-B evacuation)

## Idea

This submission builds on [`output_repacked_tail_16x16`](output_repacked_tail_16x16.md), preserving its 68,390 liveness-ordered 4x8 schedule and five-output scratch tail.  It then applies a lifetime rewrite to 22 late outputs.

For each move, a still-live B input is copied from its original expensive address into a cheap output home that is dead immediately before the output's final write.  The output is then written into the B input's old home, and the remaining reads of that B input are redirected to the cheaper former output home.

This adds 22 copy reads, but redirects 66 later B reads and 22 output exit reads enough to save 49 weighted cost units.

## Path to 68,341

| step | score | savings |
|------|------:|--------:|
| `output_repacked_tail_16x16` | 68,390 | |
| + 22 live-B evacuation moves | **68,341** | 49 |

## Region cost breakdown

| region | addrs | reads | cost |
|--------|-------|------:|-----:|
| SB | 1 | 5,057 | 5,057 |
| TMP | 2 | 2,878 | 5,756 |
| sA | 3..6 | 4,102 | 10,254 |
| sC | 7..38 | 3,867 | 19,570 |
| B bulk | 39..294 | 1,142 | 14,206 |
| A bulk | 295..550 | 576 | 11,924 |
| C spill | 551..614 | 64 | 1,574 |
| **total** | | **17,686** | **68,341** |

## Instruction distribution

The arithmetic is still standard 16x16 matmul.  The live-B evacuation adds 22 copy instructions and does not change the scalar multiplication/addition DAG.

| instruction | count | paid reads |
|-------------|------:|-----------:|
| `mul` | 4,096 | 8,192 |
| `add` | 3,840 | 7,680 |
| `copy` | 1,558 | 1,558 |
| output exit | 256 | 256 |
| **total** | **9,494 ops** | **17,686** |

## Verification

```bash
python matmul/submissions/output_repacked_tail_live_b_16x16.py
python matmul/experiments/random_true_matmul_check.py \
  matmul/submissions/output_repacked_tail_live_b_16x16.ir \
  --trials 300 --seed 20260517 --min -47 --max 47
```

Observed locally:

```text
output_repacked_tail_live_b_16x16.ir  cost=68,341
matmul/submissions/output_repacked_tail_live_b_16x16.ir: cost=68,341 ok 300 random trials
```

The generator also asserts:

- 512 distinct positive input addresses.
- 256 distinct positive output addresses.
- 160 input addresses safely reused as output homes.
- Every output address has a generating write.
