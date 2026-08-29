# Sparse Parity

**Author:** [@zh4ngx](https://github.com/zh4ngx)
**Date:** 2026-08-28
**Problem:** Mask sparse parity, n=32 — ISD family note (no band change)
**Method:** `generate_isd_mask(T, subset_seed=0)` (randomized information sets)

## Idea — the 36.2% plateau was the rotation schedule, not the problem

The published ISD family draws its information sets from a fixed
rotation: restart t uses columns `[(7t + j) mod 32]`. Since
gcd(7, 32) = 1 the rotation has period exactly 32, so restart 33
re-deals restart 1's subset — every T > 32 restart is a duplicate, and
recovery freezes at exactly 36.2%. The freeze is a property of the
schedule, not of the instances.

`subset_seed` (added to `generate_isd_mask` / `generate_isd`) switches
to independent uniform 18-column subsets, drawn once at generation
time from a seeded RNG; the emitted circuit stays deterministic. Dev
suite, same costs as the rotation (1.505M reads per restart):

| T | rotation | random (seed 0) |
| -: | -------: | --------------: |
| 1 | 5.8% | 6.0% |
| 4 | 17.2% | 9.8% |
| 8 | 22.5% | 22.5% |
| 16 | 29.8% | 40.9% |
| 32 | 36.2% | 67.2% |

Random restarts do not stack as 1 − (1 − p)^T with the T = 1 rate
(that would predict ~86% at T = 32): per-instance success is
correlated through the training matrix, so the family saturates
slower than independent coin flips. Two honest notes: the rotation is
actually *better* for T ≤ 6 (its stride-7 subsets tile the columns
with less overlap than random draws), and at T = 7 random restarts
cost 10.5M reads for ~19.2-20.5% recovery on fresh suites — not a
reliable 20%-band record, which stays with the rotation's T = 8 at
12,042,480. The randomized family's value is reach: 67.2% at 48.2M
reads where the rotation caps at 36.2% — though the weight-ordered
scan (see the weightscan submissions) dominates both ISD variants
above ~40% recovery at every cost.

## Reproduce

```python
import mask_sparse_parity as mp

ir = mp.generate_isd_mask(32, subset_seed=0)
res = mp.evaluate_mask(ir)          # → cost 48,167,636, recovery 0.6719
```
