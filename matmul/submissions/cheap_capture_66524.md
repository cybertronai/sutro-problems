# Matrix Multiplication

**Author:** Codex and Cosmin<br>
**Date:** 2026-05-25<br>
**Problem:** 16x16 matmul<br>
**Cost:** 66,524<br>
**IR:** [`cheap_capture_66524.ir`](cheap_capture_66524.ir)<br>
**Method:** cheap capture of late B-block values from address 1

## Summary

This IR computes ordinary `16x16` matrix multiplication. There are 256 `A`
inputs, 256 `B` inputs, and 256 final `C` outputs. The arithmetic is the
standard dot-product computation: 4,096 multiplies and 3,840 additions.

The optimization is about read placement. In this problem, writes are free,
but every read from address `x` costs `ceil(sqrt(x))`. Reading address `1`
costs `1`, while reading addresses in the high hundreds costs more than `20`.
So a good schedule is not only about doing the right arithmetic; it is also
about arranging for reused values to be read from cheap addresses.

This submission stages 26 `B` values in the late block:

```text
B[8,8..15]
B[9,8..15]
B[10,8..15]
B[11,8..9]
```

Each of these values naturally passes through address `1` during its first
use. The schedule copies the value at that moment, while the read is as cheap
as possible, and stores the copy in a fresh semantic temporary. Later uses of
the same `B` value read through the staged copy instead of returning to the
more expensive original input home.

The key intuition is that this is a toll-booth trade. The schedule pays one
extra address-1 read now, then avoids several later expensive reads of the same
value. The copies also give value-lifetime coloring a better object to pack:
the original `B` input and the staged reuse copy no longer need to share one
long lifetime.

## Reproduction Logic

The useful pattern is a value that:

1. Is already loaded into address `1` for normal arithmetic.
2. Will be needed again in later panels.
3. Would otherwise be reloaded from a substantially more expensive address.

For every such value, immediately after the normal `copy 1,<B source>` load,
insert a fresh semantic copy:

```text
copy <fresh temporary>,1
```

Then redirect later reloads of that same `B` value through the fresh temporary.
After the raw trace is changed, rerun value-lifetime coloring so those semantic
temporaries are assigned real physical addresses.

The raw edit uses fresh homes `731..756`:

| B value | first-load op | original source | inserted staging copy |
|---------|--------------:|----------------:|-----------------------|
| `B[8,8]` | 5424 | 192 | `copy 731,1` |
| `B[8,9]` | 5433 | 191 | `copy 732,1` |
| `B[8,10]` | 5442 | 190 | `copy 733,1` |
| `B[8,11]` | 5451 | 189 | `copy 734,1` |
| `B[8,12]` | 5460 | 188 | `copy 735,1` |
| `B[8,13]` | 5469 | 187 | `copy 736,1` |
| `B[8,14]` | 5478 | 186 | `copy 737,1` |
| `B[8,15]` | 5487 | 185 | `copy 738,1` |
| `B[9,8]` | 5500 | 184 | `copy 739,1` |
| `B[9,9]` | 5509 | 183 | `copy 740,1` |
| `B[9,10]` | 5518 | 182 | `copy 741,1` |
| `B[9,11]` | 5527 | 181 | `copy 742,1` |
| `B[9,12]` | 5536 | 180 | `copy 743,1` |
| `B[9,13]` | 5545 | 179 | `copy 744,1` |
| `B[9,14]` | 5554 | 178 | `copy 745,1` |
| `B[9,15]` | 5563 | 177 | `copy 746,1` |
| `B[10,8]` | 5576 | 176 | `copy 747,1` |
| `B[10,9]` | 5585 | 175 | `copy 748,1` |
| `B[10,10]` | 5594 | 174 | `copy 749,1` |
| `B[10,11]` | 5603 | 173 | `copy 750,1` |
| `B[10,12]` | 5612 | 172 | `copy 751,1` |
| `B[10,13]` | 5621 | 171 | `copy 752,1` |
| `B[10,14]` | 5630 | 170 | `copy 753,1` |
| `B[10,15]` | 5639 | 169 | `copy 754,1` |
| `B[11,8]` | 5652 | 168 | `copy 755,1` |
| `B[11,9]` | 5661 | 167 | `copy 756,1` |

Those raw homes are semantic placeholders, not final physical addresses. The
final coloring pass packs them with other non-overlapping lifetimes and emits
the checked-in [`cheap_capture_66524.ir`](cheap_capture_66524.ir).

## Cost Breakdown By Address Tier

| tier | addrs | reads | cost |
|------|-------|------:|-----:|
| 1 | 1 | 5,148 | 5,148 |
| 2 | 2..4 | 5,000 | 10,000 |
| 3 | 5..9 | 2,363 | 7,089 |
| 4 | 10..16 | 861 | 3,444 |
| 5 | 17..25 | 1,089 | 5,445 |
| 6 | 26..36 | 1,329 | 7,974 |
| 7 | 37..49 | 329 | 2,303 |
| 8 | 50..64 | 120 | 960 |
| 9 | 65..81 | 126 | 1,134 |
| 10..16 | 82..256 | 857 | 10,821 |
| 17..26 | 257..676 | 588 | 12,206 |
| **total** | | **17,810** | **66,524** |

## Instruction Distribution

| instruction | count | paid reads |
|-------------|------:|-----------:|
| `mul` | 4,096 | 8,192 |
| `add` | 3,840 | 7,680 |
| `copy` | 1,682 | 1,682 |
| output exit | 256 | 256 |
| **total** | **9,618 ops** | **17,810** |

## Verification

```bash
/Users/cosmin/miniconda/bin/python matmul/submissions/cheap_capture_66524.py
/Users/cosmin/miniconda/bin/python matmul/experiments/random_true_matmul_check.py matmul/submissions/cheap_capture_66524.ir --n 16 --trials 100 --seed 20260525 --min -31 --max 31
```

Observed locally:

```text
cheap_capture_66524.ir  cost=66,524
matmul/submissions/cheap_capture_66524.ir: cost=66,524 ok 100 random trials
```
