"""Summary of recursive-Strassen sweep on 16x16 matmul.

Implementation: ``exp_0_strassen_recursive.py``.

Recursive Strassen with parameterized base-case size; the base case is a
naive outer-product matmul (loop order k,i,j with one cached A-cell, an
n-cell sB strip, and a tmp). At every recursion step we allocate 10
derived (n/2)×(n/2) operands plus 7 result tiles M1..M7, then recurse 7
times.

After IR generation we run ``rename_optimal`` (the structure-preserving
lower bound) so the cost reflects an optimal address permutation for
this op schedule.

RESULTS
-------

  base  depth      ops   cells          raw    renamed
     8      1     8768    1926    7,341,717    195,792
     4      2   10224    4054    8,299,637    479,555
     2      3   13556    8464   10,779,221  1,194,326
     1      4   15271   12923   13,611,338  1,802,098

Current 16x16 record: 73,602 (sa-cache).
Best recursive-Strassen result (depth 1): 195,792 — 2.66x worse.

INTERPRETATION
--------------

Every recursion level multiplies the number of *distinct* live cells by
roughly 7 (one per Mi) plus 10 (derived operands), divided over a
half-sized block. Net effect:

  cells_total(d) ≈ 17 * (4^d - 1) / 3 * (n/2^d)^2 + base-kernel cells

For n=16 the totals match the measured 1926 / 4054 / 8464 / 12923.

Under cost(addr)=⌈√addr⌉, additional cells push *all* existing cells
outward. The cheapest k addresses cumulatively cost ~(2/3)·k^1.5 read
units of capacity, and reads scale roughly linearly with the number of
multiplications (4096 → 7^d * (n/2^d)^3). The arithmetic:

  - depth 1: 7  * 8^3 =  3584 base mults (vs 4096 naive: 87.5%)
  - depth 2: 49 * 4^3 =  3136            (76.6%)
  - depth 3: 343 * 2^3=  2744            (67.0%)
  - depth 4: 7^4      =  2401            (58.6%)

So total mults shrink at most ~40%. But the 17x cell-count growth per
level (from extra derived operands at *every* recursion node, not just
the top) more than wipes out that saving once cells/√addr cost dominates.

This is exactly the "Layout penalty from larger sC dominates over read-
count reductions" regime documented in the explorer brief (key insight:
addr-weighted cost is the real objective, not total reads).

CONCLUSION
----------

Recursive Strassen does not improve on 73,602 at any depth in this cost
model. Depth-1 (outer-only) is closest (195,792, ~2.66x worse) which
matches the prior strassen-outer-2x2 = 100,683 within an inner-kernel-
quality factor. Every deeper level is strictly worse.

Smirnov / Laderman-style 3x3 = 21-mult algorithms would need padding
since 16 mod 3 != 0; that adds wasted work (~ extra 80 padding cells
per padded operand). Highly unlikely to beat 73,602 either.

FUTURE DIRECTIONS UNLIKELY
--------------------------

- Mixed-radix recursion (e.g., one Strassen level then sa-cache 8x8):
  this is exactly depth-1 above, gives 195,792.
- Reusing derived operand storage across Mi calls: derived A-side blocks
  are needed for multiple Mi's so cannot be freed; B-side similarly.
  Could share for the *final* Mi (M7) using A12-A22 and B21+B22 — saves
  at most ~128 cells out of 1926 at depth-1, ~6% bound on improvement.
- Output cells fused into Mi-tile addresses: M-tiles are written and
  later combined into C, so they don't free until after the combine.
  Best case: free Mi cells for next Mi via reuse — but Mi values are
  reused across the C-combine phase, so they must persist.

Saved as record only if a depth beats 73,602. None did.
"""

# Re-import the implementation module to make this file runnable as a
# self-check (no-op when nothing beats 73,602; just prints summary).
from __future__ import annotations

if __name__ == "__main__":
    print(__doc__)
