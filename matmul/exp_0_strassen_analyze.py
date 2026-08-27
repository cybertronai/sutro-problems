"""Analyze where the 135554-cost Strassen reads go."""
from __future__ import annotations

import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from exp_0_strassen_v1 import build_strassen_ir  # noqa: E402
from addr_renamer import rename_optimal, count_reads  # noqa: E402
from matmul import _cost  # noqa: E402

ir = build_strassen_ir()
new_ir, info = rename_optimal(ir)

reads = count_reads(new_ir)
# reads keyed by NEW addr post-rename (since new_ir has them).
# Bucket by cost(addr) and sum.

by_cost = Counter()
total = 0
for addr, n in reads.items():
    c = _cost(addr)
    by_cost[c] += c * n
    total += c * n

print(f"Total reads cost (incl outputs in count_reads?): see below")
print(f"  cost(addr) | sum reads*cost | num cells | avg reads/cell")
for c in sorted(by_cost.keys()):
    cells_at_this_cost = sum(1 for a in reads if _cost(a) == c)
    reads_here = sum(reads[a] for a in reads if _cost(a) == c)
    print(f"  {c:>10} | {by_cost[c]:>14,} | {cells_at_this_cost:>9} | {reads_here / max(cells_at_this_cost, 1):.1f}")

# Top 30 hottest cells
print()
print("Top 30 hottest (new_addr, reads):")
for addr, n in sorted(reads.items(), key=lambda kv: -kv[1])[:30]:
    print(f"  addr={addr:>5}  reads={n:>5}  cost={_cost(addr)}")

# Print the top cells in original IR to see what they correspond to.
orig_reads = count_reads(ir)
print()
print("Top 30 hottest cells (original IR):")
for addr, n in sorted(orig_reads.items(), key=lambda kv: -kv[1])[:30]:
    print(f"  orig_addr={addr:>7}  reads={n:>5}")
