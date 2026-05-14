# Matrix Multiplication

**Author:** Codex and Cosmin
**Date:** 2026-05-13
**Problem:** 16x16 matmul
**Cost:** 67,927
**IR:** [`output_repacked_tail_value_colored_live_b_16x16.ir`](output_repacked_tail_value_colored_live_b_16x16.ir)
**Method:** `generate_output_repacked_tail_value_colored_live_b_16x16` (current-order live-B evacuation + value-lifetime address coloring)

## Idea

This submission keeps the 68,041 current-order live-B evacuation operation DAG
but reassigns produced values to physical addresses by compatible lifetimes.
There are no added operations and no removed operations: the IR still performs
ordinary 16x16 scalar matmul with 4,096 `mul`, 3,840 `add`, and 1,575 `copy`
instructions.

The value-coloring pass treats each input or produced value as an interval from
definition to last read.  Non-overlapping intervals may share the same physical
address.  A greedy maximum-chain coloring gives the heaviest compatible value
chains the cheapest address tiers.  This reduces the weighted read cost from
68,041 to 67,927.

## Path to 67,927

| step | score | savings |
|------|------:|--------:|
| current-order output-repacked tail | 68,433 | |
| + 39 live-B evacuation moves | 68,041 | 392 |
| + value-lifetime address coloring | **67,927** | 114 |

Agent AC also found a five-direct tail base at 68,039 before coloring.  Applying
the same max-chain coloring to that trace ties this 67,927 score, so this
submission uses the simpler 68,041 trace as the coloring source.

## Bound Context

The address-reassignment analysis found:

| trace | pure renumber bound | value-prefix lower bound | best safe coloring |
|-------|--------------------:|-------------------------:|-------------------:|
| current-order base | 68,433 | 68,255 | 68,369 |
| post-evac trace | 68,041 | 67,612 | **67,927** |

Pure global renumbering is exhausted for both the base and post-evac traces.
For this fixed post-evac operation trace, address reassignment alone cannot go
below 67,612.

## Cost Breakdown By Address Tier

| tier | addrs | reads | cost |
|------|-------|------:|-----:|
| 1 | 1 | 5,057 | 5,057 |
| 2 | 2..4 | 4,995 | 9,990 |
| 3 | 5..9 | 2,347 | 7,041 |
| 4 | 10..16 | 847 | 3,388 |
| 5 | 17..25 | 1,089 | 5,445 |
| 6 | 26..36 | 1,329 | 7,974 |
| 7 | 37..49 | 328 | 2,296 |
| 8 | 50..64 | 120 | 960 |
| 9 | 65..81 | 124 | 1,116 |
| 10..16 | 82..256 | 784 | 10,291 |
| 17..26 | 257..676 | 683 | 14,369 |
| **total** | | **17,703** | **67,927** |

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
python matmul/submissions/output_repacked_tail_value_colored_live_b_16x16.py
python matmul/experiments/random_true_matmul_check.py \
  matmul/submissions/output_repacked_tail_value_colored_live_b_16x16.ir \
  --trials 500 --seed 20260520 --min -61 --max 61
```

Observed locally:

```text
output_repacked_tail_value_colored_live_b_16x16.ir  cost=67,927
matmul/submissions/output_repacked_tail_value_colored_live_b_16x16.ir: cost=67,927 ok 500 random trials
```

Agent AA also validated the same IR with 100 arbitrary signed true-matmul trials
using seed `2026051308` and range `[-53, 53]`.
