#!/usr/bin/env python3
"""Per-submission access-distance plots for the symmetry problem.

For each `../../submissions/*.ir` file, walk the v3 IR and collect the
read distance ⌈√addr⌉ for every operand read (instruction sources and
final output reads). Save a 2-panel PNG into this directory:

    [left]  histogram: how many reads happen at each distance
    [right] CDF:       cumulative read count vs distance

Also emits a single combined CDF (`combined_cdf.png`) overlaying all
submissions on one axis.

Run:
    python3 symmetry/doc/access_distance/plot_access_distance.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
SYMMETRY = HERE.parent.parent
SUBMISSIONS = SYMMETRY / "submissions"

sys.path.insert(0, str(SYMMETRY))
import symmetry as sym  # noqa: E402

# v3 instruction read-operand counts (by head keyword)
# "set" has zero reads; everything else listed below.
_BINARY_OPS = {"add", "sub", "mul", "div", "and", "or", "xor"}
_UNARY_OPS  = {"copy", "not", "abs"}


def collect_read_distances(ir: str) -> List[int]:
    """Return ⌈√addr⌉ for every operand read in the v3 IR trace."""
    text = ir.replace(";", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []

    output_addrs = [int(x) for x in lines[-1].split(",") if x.strip()]
    distances: List[int] = []

    for ln in lines[1:-1]:
        head, _, rest = ln.partition(" ")
        raw = [x.strip() for x in rest.split(",") if x.strip()] if rest else []

        if head == "set":
            reads = []

        elif head == "cmp":
            # cmp dest, a, b, pred  — reads a and b
            reads = [int(raw[1]), int(raw[2])]

        elif head == "select":
            # select dest, cond, true_val, false_val  — reads all three sources
            reads = [int(raw[1]), int(raw[2]), int(raw[3])]

        elif head in _UNARY_OPS:
            # copy/not/abs dest, src
            reads = [int(raw[1])]

        elif head in _BINARY_OPS:
            if len(raw) == 3:
                # 3-operand: op dest, s1, s2
                reads = [int(raw[1]), int(raw[2])]
            else:
                # 2-operand in-place: op dest, s2  (dest is also read)
                reads = [int(raw[0]), int(raw[1])]

        else:
            reads = []

        for addr in reads:
            distances.append(sym._cost(addr))

    for addr in output_addrs:
        distances.append(sym._cost(addr))

    return distances


def plot_one(ir_path: Path, out_path: Path) -> int:
    distances = np.array(collect_read_distances(ir_path.read_text()))
    total_cost = int(distances.sum())
    n_reads = len(distances)

    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(11, 4))

    edges = np.arange(distances.min(), distances.max() + 2) - 0.5
    ax_h.hist(distances, bins=edges, color="#3b78b4",
              edgecolor="white", linewidth=0.5)
    ax_h.set_xlabel("distance  ⌈√addr⌉")
    ax_h.set_ylabel("count")
    ax_h.set_title(f"{ir_path.name}\n{n_reads:,} reads, total cost {total_cost:,}")
    ax_h.grid(axis="y", alpha=0.3)

    sorted_d = np.sort(distances)
    cumulative = np.arange(1, len(sorted_d) + 1)
    ax_c.plot(sorted_d, cumulative, color="#cc4c4c", linewidth=1.5)
    ax_c.set_xlabel("distance  ⌈√addr⌉")
    ax_c.set_ylabel("cumulative reads")
    ax_c.set_title("CDF (reads at distance ≤ x)")
    ax_c.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return total_cost


def plot_combined_cdf(ir_paths: List[Path], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("turbo")
    rows = []
    for p in ir_paths:
        distances = np.sort(collect_read_distances(p.read_text()))
        rows.append((p.name, distances, int(distances.sum())))
    rows.sort(key=lambda r: r[2])
    n = len(rows)
    for i, (name, distances, total) in enumerate(rows):
        color = cmap(0.05 + 0.9 * i / max(n - 1, 1))
        ax.plot(distances, np.arange(1, len(distances) + 1),
                label=f"{name}  (cost {total:,})", color=color, linewidth=1.5)
    ax.set_xlabel("distance  ⌈√addr⌉")
    ax.set_ylabel("cumulative reads")
    ax.set_title(f"CDF — reads at distance ≤ x  ({n} submissions)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    ir_paths = sorted(SUBMISSIONS.glob("*.ir"))
    print(f"{'submission':<30}{'reads':>10}{'total_cost':>14}  out")
    print("-" * 70)
    for ir_path in ir_paths:
        out_path = HERE / (ir_path.stem + ".png")
        total_cost = plot_one(ir_path, out_path)
        n_reads = len(collect_read_distances(ir_path.read_text()))
        print(f"{ir_path.name:<30}{n_reads:>10,}{total_cost:>14,}  {out_path.name}")

    combined_path = HERE / "combined_cdf.png"
    plot_combined_cdf(ir_paths, combined_path)
    print(f"\nCombined CDF: {combined_path.name}  ({len(ir_paths)} submissions)")


if __name__ == "__main__":
    main()
