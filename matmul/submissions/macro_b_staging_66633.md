# Matrix Multiplication

**Author:** Codex and Cosmin<br>
**Date:** 2026-05-25<br>
**Problem:** 16x16 matmul<br>
**Cost:** 66,633<br>
**IR:** [`macro_b_staging_66633.ir`](macro_b_staging_66633.ir)<br>
**Method:** Macro B-staging with B[7,10..15] later-panel prestaging from address 1

## Summary

This IR computes ordinary `16x16` matrix multiplication. There are 256 `A`
inputs, 256 `B` inputs, and 256 final `C` outputs. The arithmetic is the
standard dot-product computation: 4,096 multiplies and 3,840 additions.

The optimization is about where intermediate values live. In this problem,
writes are free, but every read from address `x` costs `ceil(sqrt(x))`.
Reading address `1` costs `1`, reading addresses `2..4` costs `2`, and so on.
The best schedules therefore spend most of their effort keeping frequently
reused values in cheap addresses, or copying them out of cheap addresses at
the right moment so they can be reused later without paying a high reload
cost.

This submission stages six `B` values:

```text
B[7,10], B[7,11], B[7,12], B[7,13], B[7,14], B[7,15]
```

Each value passes through address `1` during its first use. The schedule copies
the value while it is still available at that cheapest address, stores the copy
in a temporary staging home, and then uses that staged copy in later panels.
The copy has an immediate cost, but it prevents repeated later reads from more
expensive homes.

Row `7` is not mathematically special. It is special in this schedule because
the late panel works through `B` rows `5`, `6`, and `7` over columns `8..15`.
Within that panel, the row-7 columns `10..15` are the exposed reuse gap: they
are available cheaply once, and then needed again later. Staging them turns
six long reuse paths into shorter lifetimes that are easier to pack into cheap
addresses.

This helps the full solution because every `B[k,j]` participates in many
outputs:

```text
A[0,k] * B[k,j]
A[1,k] * B[k,j]
...
A[15,k] * B[k,j]
```

A bad placement for one reused `B` value can therefore become many expensive
reads. Improving six reused `B` values is a small local change, but it reduces
both direct read cost and global address pressure.

## Parameter Selection

The final parameters were chosen by exact-scoring a small family of staging
choices. The two important choices were:

- Stage from address `1`, not from the original `B` input home.
- Stage all six row-7 target columns `10..15`.

The best tested variants were:

| staged B row-7 columns | source used for staging copy | score |
|------------------------|------------------------------|------:|
| `10, 11, 12, 13, 14, 15` | address `1` | **66,633** |
| `10, 12, 13, 14, 15` | address `1` | 66,634 |
| `12, 13, 14, 15` | address `1` | 66,640 |
| `10, 11` | address `1` | 66,660 |

The same column sets staged from their original source homes were worse than
staging from address `1`, because the staging copy itself became more
expensive. The winning choice is therefore exactly the case where the value is
captured while it is already sitting in the cheapest possible address.

## Reproduction Logic

The useful pattern is a value that is briefly available at a very cheap
address, is needed again later, and would otherwise be recovered from a more
expensive address. The reproduction recipe is:

1. Inspect the late macro panel that uses `B` rows `5`, `6`, and `7` over
   columns `8..15`.
2. Ignore values that already have persistent staging. In this schedule,
   `B[7,8]` and `B[7,9]` are already protected, while `B[7,10..15]` are not.
3. For each unprotected value, wait until its normal first load puts it in
   address `1`.
4. Immediately copy that address-`1` value into a fresh temporary. This pays
   one cheapest-possible read now.
5. For the rest of the schedule, use the temporary whenever that same `B`
   value is needed again.
6. Re-run value-lifetime coloring so the temporary lifetimes can be packed
   into cheap physical addresses.

The decision rule is simple: stage only when the value is captured from address
`1` and will be reused enough later to repay the extra copy. Staging from the
original source home is worse, because the staging copy itself becomes
expensive. Staging only two or four of the row-7 columns helps, but staging all
six columns `10..15` gives the best tested balance of copy cost, later-read
savings, and coloring pressure.

The raw edit allocates six fresh staging homes:

| B value | first-load op | original source | inserted staging copy |
|---------|--------------:|----------------:|-----------------------|
| `B[7,10]` | 5360 | 197 | `copy 725,1` |
| `B[7,11]` | 5369 | 72 | `copy 726,1` |
| `B[7,12]` | 5378 | 196 | `copy 727,1` |
| `B[7,13]` | 5387 | 195 | `copy 728,1` |
| `B[7,14]` | 5396 | 194 | `copy 729,1` |
| `B[7,15]` | 5405 | 193 | `copy 730,1` |

These raw staging homes are not meant to be final low addresses. They are
semantic placeholders. The final address-coloring pass sees their lifetimes,
chains them with non-overlapping values, and assigns the actual cheap physical
addresses used by [`macro_b_staging_66633.ir`](macro_b_staging_66633.ir).

## Cost Breakdown By Address Tier

| tier | addrs | reads | cost |
|------|-------|------:|-----:|
| 1 | 1 | 5,122 | 5,122 |
| 2 | 2..4 | 5,000 | 10,000 |
| 3 | 5..9 | 2,363 | 7,089 |
| 4 | 10..16 | 861 | 3,444 |
| 5 | 17..25 | 1,089 | 5,445 |
| 6 | 26..36 | 1,329 | 7,974 |
| 7 | 37..49 | 329 | 2,303 |
| 8 | 50..64 | 120 | 960 |
| 9 | 65..81 | 126 | 1,134 |
| 10..16 | 82..256 | 831 | 10,514 |
| 17..26 | 257..676 | 614 | 13,648 |
| **total** | | **17,784** | **66,633** |

## Instruction Distribution

| instruction | count | paid reads |
|-------------|------:|-----------:|
| `mul` | 4,096 | 8,192 |
| `add` | 3,840 | 7,680 |
| `copy` | 1,656 | 1,656 |
| output exit | 256 | 256 |
| **total** | **9,592 ops** | **17,784** |

## Verification

```bash
/Users/cosmin/miniconda/bin/python matmul/submissions/macro_b_staging_66633.py
/Users/cosmin/miniconda/bin/python matmul/experiments/random_true_matmul_check.py matmul/submissions/macro_b_staging_66633.ir --n 16 --trials 100 --seed 20260524 --min -31 --max 31
```

Observed locally:

```text
macro_b_staging_66633.ir  cost=66,633
matmul/submissions/macro_b_staging_66633.ir: cost=66,633 ok 100 random trials
```
