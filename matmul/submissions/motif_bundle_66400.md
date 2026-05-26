# Matrix Multiplication

**Author:** Codex and Cosmin<br>
**Date:** 2026-05-26<br>
**Problem:** 16x16 matmul<br>
**Cost:** 66,400<br>
**IR:** [`motif_bundle_66400.ir`](motif_bundle_66400.ir)<br>
**Method:** late copy-schedule motif bundle plus value-lifetime coloring

## Summary

This IR computes ordinary `16x16` matrix multiplication: 256 `A` inputs,
256 `B` inputs, and 256 `C` outputs. It uses the standard arithmetic shape:
4,096 multiplies and 3,840 additions.

The score is dominated by reads. A read from address `x` costs
`ceil(sqrt(x))`; writes and arithmetic themselves are free. The useful
optimization is therefore to make repeatedly read values spend their expensive
future reads in cheap physical addresses.

The submission uses a raw trace that has already separated short-lived
semantic values from long-lived matrix inputs. The final pass then recolors the
semantic values by lifetime so non-overlapping hot values can share low
physical addresses. The improvement here comes from a late repeated
copy-schedule motif: a small group of copies feeding the late `B` panel is
moved slightly later in a regular pattern. That shortens the window where those
copies occupy cheap addresses, so the coloring pass can pack nearby hot reads
with less interference.

The important intuition is not that one individual copy is valuable. The win
comes from making several neighboring copy lifetimes line up with the actual
reuse window of the late panel. Each move is tiny, but together they remove
enough address pressure that the same arithmetic trace recolors to cost
66,400.

## Reproduction Logic

The checked-in raw trace is semantic: addresses above the live scratch band are
temporary value names, not final physical homes. Reproduction is:

1. Load [`motif_bundle_66400.raw.ir`](motif_bundle_66400.raw.ir).
2. Build value lifetimes from the raw trace.
3. Assign non-overlapping lifetimes to cheap physical addresses with the shared
   value-lifetime coloring helper.
4. Emit [`motif_bundle_66400.ir`](motif_bundle_66400.ir) and verify it with
   `score_16x16`.

The score-improving motif that produced the 66,400 family shifted this
repeated late-copy family:

| source op | destination op |
|----------:|---------------:|
| 5964 | 5985 |
| 5967 | 5995 |
| 5970 | 6005 |
| 5975 | 6015 |
| 5978 | 6025 |
| 5981 | 6035 |
| 5985 | 5987 |
| 5995 | 5997 |
| 6005 | 6007 |
| 6015 | 6017 |
| 6025 | 6027 |
| 6035 | 6037 |

Those shifts preserve the arithmetic dependencies. Their purpose is only to
change when short `B`-panel helper values become live, so the final coloring
has an easier packing problem.

The checked-in representative also applies one score-neutral neighboring
shift, op `5808` before op `5804`, from the same motif-bundle search. It keeps
the exact cost at 66,400.

## Cost Breakdown By Address Tier

| tier | addrs | reads | cost |
|------|-------|------:|-----:|
| 1 | 1 | 5,163 | 5,163 |
| 2 | 2..4 | 5,000 | 10,000 |
| 3 | 5..9 | 2,357 | 7,071 |
| 4 | 10..16 | 855 | 3,420 |
| 5 | 17..25 | 1,089 | 5,445 |
| 6 | 26..36 | 1,329 | 7,974 |
| 7 | 37..49 | 326 | 2,282 |
| 8 | 50..64 | 105 | 840 |
| 9 | 65..81 | 119 | 1,071 |
| 10..16 | 82..256 | 835 | 10,554 |
| 17..26 | 257..676 | 610 | 12,580 |
| **total** | | **17,788** | **66,400** |

## Instruction Distribution

| instruction | count | paid reads |
|-------------|------:|-----------:|
| `mul` | 4,096 | 8,192 |
| `add` | 3,840 | 7,680 |
| `copy` | 1,660 | 1,660 |
| output exit | 256 | 256 |
| **total** | **9,596 ops** | **17,788** |

## Verification

```bash
/Users/cosmin/miniconda/bin/python matmul/submissions/motif_bundle_66400.py
/Users/cosmin/miniconda/bin/python matmul/experiments/random_true_matmul_check.py matmul/submissions/motif_bundle_66400.ir --n 16 --trials 100 --seed 20260526 --min -31 --max 31
```

Observed locally:

```text
motif_bundle_66400.ir  cost=66,400
matmul/submissions/motif_bundle_66400.ir: cost=66,400 ok 100 random trials
```
