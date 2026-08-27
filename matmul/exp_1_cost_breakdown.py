"""Detailed cost breakdown of layout A (current 69,697)."""
from __future__ import annotations
import math


def cost(addr):
    return math.isqrt(addr - 1) + 1


def breakdown(groups, label):
    print(f"\n{label}:")
    addr = 1
    total = 0
    for count, reads in groups:
        block_cost = 0
        addrs_in_block = []
        for _ in range(count):
            addrs_in_block.append(addr)
            block_cost += reads * cost(addr)
            addr += 1
        avg_cost_per_addr = (sum(cost(a) for a in addrs_in_block) /
                             len(addrs_in_block))
        print(f"  {count:4d} cells × {reads:4d} reads each "
              f"@ addrs {addrs_in_block[0]}-{addrs_in_block[-1]} "
              f"(avg cost {avg_cost_per_addr:.2f}): "
              f"contributes {block_cost:>6,}")
        total += block_cost
    print(f"  TOTAL: {total:,}")
    return total


# Layout A: current 69,697.
A = [(1, 4096), (1, 3840), (4, 1024), (32, 120),
     (160, 5), (96, 4), (256, 2), (96, 1)]
breakdown(A, "Layout A (current 69,697)")
