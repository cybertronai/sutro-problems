"""Analyze the 69,697 IR's cost composition by address range."""
from __future__ import annotations

import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REC = os.path.join(HERE, "records", "record_69697_lane0.ir")


def parse_ir(text):
    lines = text.strip().splitlines()
    body = lines[1:-1]
    inputs = set(int(x) for x in lines[0].split(","))
    outputs = [int(x) for x in lines[-1].split(",")]
    return body, inputs, outputs


def analyze():
    with open(REC) as f:
        text = f.read()
    body, inputs, outputs = parse_ir(text)
    read_addrs = []
    for line in body:
        op, args = line.split(None, 1)
        a = [int(x) for x in args.split(",")]
        # First arg is dest; rest are reads (for copy: src is read)
        if op == "copy":
            # 2-arg: copy dst, src. src is read.
            read_addrs.append(a[1])
        elif op in ("mul",):
            # mul dst, a, b: 2 reads
            read_addrs.extend(a[1:])
        elif op in ("add", "sub"):
            # 3-arg: add dst, a, b: 2 reads
            # 2-arg: add dst, a: dst+a are reads
            if len(a) == 3:
                read_addrs.extend(a[1:])
            elif len(a) == 2:
                read_addrs.append(a[0])
                read_addrs.append(a[1])
            else:
                raise ValueError(f"unexpected: {line}")
        else:
            raise ValueError(f"unknown op: {op}")
    # Output reads
    read_addrs.extend(outputs)
    # Cost
    counter = Counter(read_addrs)
    total = 0
    for addr, n in counter.items():
        c = math.ceil(math.sqrt(addr))
        total += c * n

    # Bucket by address ranges
    buckets = {
        "scratch (1-38)": (1, 38),
        "A inputs (39-294)": (39, 294),
        "B inputs (295-550)": (295, 550),
        "C bulk (551+)": (551, 10**9),
    }
    for label, (lo, hi) in buckets.items():
        sub = sum(math.ceil(math.sqrt(addr)) * n for addr, n in counter.items()
                  if lo <= addr <= hi)
        nreads = sum(n for addr, n in counter.items() if lo <= addr <= hi)
        ncells = sum(1 for addr, n in counter.items() if lo <= addr <= hi)
        print(f"  {label}: cells={ncells}  reads={nreads}  cost={sub:,}")
    print(f"Total cost: {total:,}")
    print(f"Distinct addresses read: {len(counter)}")
    print(f"Total reads: {sum(counter.values())}")


if __name__ == "__main__":
    analyze()
