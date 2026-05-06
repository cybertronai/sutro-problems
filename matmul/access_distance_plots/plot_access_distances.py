#!/usr/bin/env python3
"""Per-submission access-distance plots.

For each `../submissions/*.ir` file, walk the IR and collect the v0 read
distance ⌈√addr⌉ for every operand read (binary-op sources, copy src,
final output reads). Save a 2-panel PNG into this directory:

    [left]  histogram: how many reads happen at each distance
    [right] CDF:       cumulative cost share vs distance
                       (so you can see how much of the total cost
                        comes from the long-distance tail)

Also emits a single combined CDF (`combined_cdf.png`) overlaying four
representative 16×16 submissions on one axis.

Run:
    python3 matmul/access_distance_plots/plot_access_distances.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
MATMUL = HERE.parent
SUBMISSIONS = MATMUL / "submissions"

# Make `import matmul` resolve to the parent package's matmul.py.
sys.path.insert(0, str(MATMUL))
import matmul as mm  # noqa: E402

COMBINED = [
    "baseline_16x16.ir",
    "tiled_16x16.ir",
    "hierarchical_16x16.ir",
    "sa_cache_16x16.ir",
]


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

    edges = np.arange(distances.min(), distances.max() + 2) - 0.5
    ax_h.hist(distances, bins=edges, color="#3b78b4",
              edgecolor="white", linewidth=0.5)
    ax_h.set_xlabel("distance")
    ax_h.set_ylabel("count")
    ax_h.set_title(f"{ir_path.name}\n{n_reads:,} reads, total cost {total_cost:,}")
    ax_h.grid(axis="y", alpha=0.3)

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


def plot_combined_cdf(ir_names: List[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3b78b4", "#e0863c", "#5fa055", "#cc4c4c"]
    for name, color in zip(ir_names, colors):
        ir_path = SUBMISSIONS / name
        distances = np.sort(collect_read_distances(ir_path.read_text()))
        total = distances.sum()
        cumulative = np.cumsum(distances) / total
        ax.plot(distances, cumulative, label=f"{name}  (cost {total:,})",
                color=color, linewidth=1.8)
    ax.set_xlabel("distance")
    ax.set_ylabel("cumulative cost share")
    ax.set_title("CDF — share of total cost at distance ≤ x")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    print(f"{'submission':<32}{'reads':>10}{'total_cost':>14}  out")
    print("-" * 78)
    for ir_path in sorted(SUBMISSIONS.glob("*.ir")):
        out_path = HERE / (ir_path.stem + ".png")
        total_cost = plot_one(ir_path, out_path)
        n_reads = len(collect_read_distances(ir_path.read_text()))
        print(f"{ir_path.name:<32}{n_reads:>10,}{total_cost:>14,}  "
              f"{out_path.name}")

    combined_path = HERE / "combined_cdf.png"
    plot_combined_cdf(COMBINED, combined_path)
    print(f"\nCombined CDF: {combined_path.name}  ({len(COMBINED)} submissions)")


if __name__ == "__main__":
    main()
