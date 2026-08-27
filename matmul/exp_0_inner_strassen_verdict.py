"""Lane 0 verdict: Inner-2x2-Strassen and B-cell aliasing — both lose to 69,697.

EXPERIMENT 1: 2x2-Strassen at the inner kernel
==============================================

Implementation: ``exp_0_inner_strassen2x2.py``.

Approach: take the 16x16 matmul, split it into 8x8x8 grid of 2x2 sub-block
products, and compute each 2x2 product via Strassen's 7 multiplications.
Maintain a 16x16 sC accumulator for the final write-back.

Result:
  raw     = 222,490
  renamed = 204,570

That is **2.94x worse** than the 69,697 baseline.

Why it loses (matches the brief's a-priori prediction):

Per 2x2 panel: 7 muls + 4 derived-A ops + 4 derived-B ops + 4 sC updates
(the simple `add sC, tmp` pattern). The Strassen savings is 1 mul × 2
reads = -2 reads per panel; the cost of derived-operand generation is
~10 add/sub ops × 2 reads = +20 reads per panel. Net: +18 reads per
panel × 512 panels = +9216 extra reads. With even modest read costs,
this swamps the cost.

The reason is structural: Strassen's bilinear identity reduces *bilinear*
multiplications, but the identity itself uses linear combinations of
inputs that cost reads on the same scratchpad. Under cost(addr)=⌈√addr⌉,
extra scratchpad reads at hot addresses cost the same as extra mul
reads. Hence we trade 2 mul reads for ~20 derived-op reads — a 10x
cost increase.

EXPERIMENT 2 (analysis only): B-cell aliasing with C output
==========================================================

Cost breakdown of the 69,697 IR (``exp_0_cost_breakdown.py``):
  scratch (1-38):   38 cells, 15872 reads, cost = 41,456 (59.5%)
  A inputs (39-294): 256 cells, 1184 reads, cost = 15,108 (21.7%)
  B inputs (295-550): 256 cells, 512 reads,  cost = 10,738 (15.4%)
  C bulk (551+):     96 cells,  96 reads,    cost = 2,395  (3.4%)

The 96 unaliased C outputs (bi=0, bj < 3) cost only 2,395 total — a
maximum 3.4% potential savings even with perfect B-cell aliasing.

Loop-order analysis: in the current `for bi: for bj: for bk` order, B
cells stay live until bi=1 (the last bi), making them dead AFTER all
unaliased C outputs (which are written during bi=0) are produced. So
B-cell aliasing for unaliased C outputs is **impossible** with this loop
order.

Swap to `for bj: for bi: for bk` to make B[*, bj] dead at end of each bj.
But then C(0..7, *) is written during bi=0 of each bj, BEFORE the B cells
go dead. Same timing problem. The fix would be to also reorder writes —
but we'd then lose the bi=0 contiguity that enables A-aliasing for C
outputs.

Even ignoring loop-order issues, the upper bound is 96 cells × ~8 cost-
delta-per-cell = ~768 savings, or 1.1% reduction → 68,929. Marginal at
best, structurally hard to realize.

DOMINANT COST: SCRATCH REGION
=============================

The scratch region (addresses 1-38) accounts for 60% of total cost.
This is from:
  - 4096 muls × 2 sA/sB reads = 8192 scratch reads
  - 4096 adds × 2 sC/tmp reads = 8192 scratch reads (some 3-arg)
  - 256 sB load copies (reads from B → scratch addr)
  - 32 sC initialization writes (no reads to scratch)
  - 256 final-add 3-arg adds replace half of sC writeback

To reduce this you'd need to *reduce read count itself* — that's what
Strassen attempts but loses to derived-op overhead. Or shrink the sC
footprint (fewer sC cells means lower addresses). That's the lower-
bound regime; the 69,697 IR is already structure-preserving optimal.

CONCLUSION
==========

Direction "inner-Strassen-2x2" definitively does not improve on 69,697.
Best in this direction: 204,570 (2.94x worse).

B-cell aliasing has hard upper bound of ~1% improvement (68,929) and is
blocked by loop-order timing constraints. Not worth implementing.

DIRECTION DONE.
"""
from __future__ import annotations

if __name__ == "__main__":
    print(__doc__)
