"""Quick breakdown of the Strassen v1 cost structure."""
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
total_cost = sum(_cost(a) * n for a, n in reads.items())
print(f"total reads cost = {total_cost:,}")

# Bucket: scratchpad (1-6), inner-sC (7-38), derived/M operands, bulk A/B/C
buckets = {
    "scratchpad cost1-2 (sA, tmp, sB)": 0,
    "inner sC region (cost3-6)": 0,
    "derived/Mi at cost 7-12": 0,
    "rest (cost 13+)": 0,
}
for a, n in reads.items():
    c = _cost(a)
    contrib = c * n
    if c <= 2:
        buckets["scratchpad cost1-2 (sA, tmp, sB)"] += contrib
    elif c <= 6:
        buckets["inner sC region (cost3-6)"] += contrib
    elif c <= 12:
        buckets["derived/Mi at cost 7-12"] += contrib
    else:
        buckets["rest (cost 13+)"] += contrib

for k, v in buckets.items():
    print(f"  {k:<40} {v:>10,}")

# Per-cell distribution: histogram of read-counts.
print()
hist = Counter()
for a, n in reads.items():
    hist[n] += 1
print("read-count histogram (# cells at each read-count):")
for n in sorted(hist):
    print(f"  reads={n:>5}: {hist[n]} cells")
