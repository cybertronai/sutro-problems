# Matrix Multiplication

**Author:** [@zh4ngx](https://github.com/zh4ngx)
**Date:** 2026-05-05
**Problem:** 16x16 matmul
**Cost:** 68,452
**IR:** [`colmajor_fused_16x16.ir`](colmajor_fused_16x16.ir)
**Method:** `generate_colmajor_fused_16x16` (column-major super-block order + fused final copy-out)

## Idea

This builds on [@sjbaebae](https://github.com/sjbaebae)'s
`dead_input_outputs_packed_16x16` submission. That submission uses the
same `(Tio=4, Tjo=8, Tii=4, Tji=1)` blocking family, dead-input output
reuse, and B packing.

The extra step here is to fuse each non-final super-block's last
accumulation with its copy-out. In the previous schedule, a completed
non-final output is first accumulated into `sC`, then copied to its
final output address:

```text
mul TMP, sA, sB
add sC, TMP
copy OUT, sC
```

At `k=15`, that `sC` value is not needed again. The next super-block
initializes `sC` from scratch at `k=0`. So the final update can write
directly to the output address:

```text
mul TMP, sA, sB
add OUT, sC, TMP
```

For the redirected `ii=3` lane, the same idea writes the final add
from `sC` and `sB` directly to `OUT`.

## Why column-major order

The row-major `70,053` schedule has one block with no dead input cells
available, so its outputs spill to fresh C cells. Super-block reordering
does not remove the spill count, but column-major order places the final
output homes so the fused-copy schedule is cheaper overall.

The final super-block still keeps its outputs directly in `sC@7..38`.
All other super-blocks write their completed outputs directly to their
dead-input or C-spill homes on the last accumulation.

## Path to 68,452

| step                                      | score  | savings |
|-------------------------------------------|-------:|--------:|
| dead-input output reuse + B packing       | 70,053 |         |
| + column-major super-block order          | 69,824 | 229     |
| + fused final accumulation/copy-out       | 68,452 | 1,372   |

## Verification

```bash
python3 matmul/submissions/colmajor_fused_16x16.py
python3 /home/andy/sutro/SutroAna/lower_bound.py matmul/submissions/colmajor_fused_16x16.ir --no-remap
nix run nixpkgs#python313Packages.pytest -- -q matmul/test_matmul.py
```

Observed locally:

```text
colmajor_fused_16x16.ir  cost=68,452
current_cost : 68,452
lower_bound  : 68,452
gap          : 0  (0.0%)
29 passed
```
