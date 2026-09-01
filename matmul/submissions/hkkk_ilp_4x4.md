# 4×4 matmul — 689 to 683 via HKKK order + exact allocation

**Date:** 2026-09-01
**Cost:** 683 (verified with `matmul.score_4x4`)
**IR:** [`hkkk_ilp_4x4.ir`](hkkk_ilp_4x4.ir)
**Generator:** [`hkkk_ilp_4x4.py`](hkkk_ilp_4x4.py) (trace builder + exact ILP
address assignment; `main` emits the 683 IR; requires numpy + scipy)
**SHA-256:** `562af9a9b848ca325499a63525b5e2d614339632365ebb712c651202a627ab3c`

## Method

This improves the current 689 record by 6 read-cost units (0.87%).

- **HKKK op order.** Row 0 is computed j-outer reading A from its cheap,
  early-freed home cells;
  rows 1–3 are computed k-outer with *just-in-time* staging —
  `copy s, A[i][k]` immediately before its k-step, so each staged value is
  read 4× at address 1 and dies. Products are consumed by an add
  immediately after their mul, so they too land on addresses 1–2.
  124 ops: 64 mul, 48 add, 12 copy; 252 reads including 16 exit reads.
- **Dependency-safe output deferral.** Delaying the final add for `O[2,1]`
  by six operation slots, to just after row 3's first multiply, leaves the
  arithmetic DAG unchanged but shortens a congested overlap. The generator
  solves both schedules and certifies that the address-allocation objective
  drops from 686 to 683.
- **Exact liveness-aware ILP address assignment** for the fixed operation
  order. Searching addresses 1–156 is complete because the trace has 156
  values; rank-compressing any assignment that used a larger address label
  preserves conflicts and cannot increase cost. The generator requires an
  optimal solver status and zero MIP gap before it emits an IR; a time-limit
  incumbent is rejected rather than mislabeled as exact.

## Scope of the certificate

The 64-mul scheme needs at least 240 essential reads (128 mul-operand + 96
add + 16 exit), before the 12 staging-copy reads. All 16 B values remain live
across the output rows, creating most of the address pressure. This is a
useful structural lower-bound heuristic. The exact result applies only to the
submitted operation schedule; it is not a claim of global optimality over
other arithmetic DAGs or operation orders.

Explored but non-winning directions included 47–49-multiply bilinear schemes,
global k-outer B clustering, split row/column phases, B mid-life migration,
and permutation/annealing searches. These observations motivate the submitted
schedule but are not used as an optimality certificate.

## Reproduce

```python
import matmul
matmul.score_4x4(open('submissions/hkkk_ilp_4x4.ir').read())   # → 683

# Rebuilds the IR and certifies base=686, deferred=683 before writing it.
exec(open('submissions/hkkk_ilp_4x4.py').read())
```
