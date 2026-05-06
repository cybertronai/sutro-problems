#!/usr/bin/env python3
"""Per-submission access-distance plots.

For each `submissions/*.ir` file, walk the IR and collect the v0 read
distance ⌈√addr⌉ for every operand read (binary-op sources, copy src,
final output reads). Save a 2-panel PNG next to the IR:

    [left]  histogram: how many reads happen at each distance
    [right] CDF:       cumulative cost share vs distance
                       (so you can see how much of the total cost
                        comes from the long-distance tail)

Run:
    python3 matmul/plot_access_distances.py
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import matmul as mm


HERE = Path(__file__).parent
SUBMISSIONS = HERE / "submissions"


def collect_read_distances(ir: str) -> List[int]:
    """Replay the IR exactly the way matmul._simulate does, recording
    ⌈√addr⌉ for every operand read."""
    input_addrs, ops, output_addrs = mm._parse(ir)
    distances: List[int] = []
    for op, oprs in ops:
        if op == "copy":
            _, src = oprs
            distances.append(mm._cost(src))
            continue
        # add/sub/mul
        if len(oprs) == 3:
            _, s1, s2 = oprs
        else:
            dest, s2 = oprs
            s1 = dest
        distances.append(mm._cost(s1))
        distances.append(mm._cost(s2))
    for a in output_addrs:
        distances.append(mm._cost(a))
    return distances


def plot_one(ir_path: Path, out_path: Path) -> int:
    distances = np.array(collect_read_distances(ir_path.read_text()))
    total_cost = int(distances.sum())
    n_reads = len(distances)

    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(11, 4))

    # Histogram (left): reads per distance bucket
    edges = np.arange(distances.min(), distances.max() + 2) - 0.5
    ax_h.hist(distances, bins=edges, color="#3b78b4",
              edgecolor="white", linewidth=0.5)
    ax_h.set_xlabel("distance")
    ax_h.set_ylabel("count")
    ax_h.set_title(f"{ir_path.name}\n{n_reads:,} reads, total cost {total_cost:,}")
    ax_h.grid(axis="y", alpha=0.3)

    # CDF (right): cumulative share of total cost vs distance threshold
    sorted_d = np.sort(distances)
    cumulative_cost = np.cumsum(sorted_d) / total_cost
    ax_c.plot(sorted_d, cumulative_cost, color="#cc4c4c", linewidth=1.5)
    ax_c.set_xlabel("distance")
    ax_c.set_ylabel("cumulative cost share")
    ax_c.set_title("CDF (share of total cost ≤ distance)")
    ax_c.set_ylim(0, 1.02)
    ax_c.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return total_cost


def main() -> None:
    out_dir = HERE / "access_distance_plots"
    out_dir.mkdir(exist_ok=True)
    print(f"{'submission':<32}{'reads':>10}{'total_cost':>14}  out")
    print("-" * 78)
    for ir_path in sorted(SUBMISSIONS.glob("*.ir")):
        out_path = out_dir / (ir_path.stem + ".png")
        total_cost = plot_one(ir_path, out_path)
        n_reads = len(collect_read_distances(ir_path.read_text()))
        print(f"{ir_path.name:<32}{n_reads:>10,}{total_cost:>14,}  "
              f"{out_path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
