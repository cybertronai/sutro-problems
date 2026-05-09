#!/usr/bin/env python3
"""Per-submission access-distance plots for sparse-parity.

For each `../../submissions/*.ir` file, walk the IR and record the v3
read distance ⌈√addr⌉ for every operand read (binary/unary sources,
cmp/select operands, final output reads — ``set`` writes a literal
and reads nothing). Save a 2-panel PNG into this directory:

    [left]  histogram: how many reads happen at each distance
    [right] CDF:       count of reads at distance ≤ x

Also emits two combined CDFs — one per problem size — overlaying every
submission of that size on a single axis (legend sorted by total cost,
cheapest first). Size is detected from the IR's input count: small
declares ``M_TRAIN*N_BITS + M_TRAIN + M_TEST*N_BITS`` for the small
spec (112) and medium for the medium spec (584).

Run:
    python3 sparse-parity/doc/access_distance_plots/plot_access_distances.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
SPARSE = HERE.parent.parent
SUBMISSIONS = SPARSE / "submissions"

sys.path.insert(0, str(SPARSE))
import sparse_parity as sp  # noqa: E402

_BINARY = {"add", "sub", "mul", "div", "and", "or", "xor"}
_UNARY = {"copy", "not", "abs"}

# Map input-count → label for figuring out which problem size the IR
# was authored against.
_SIZE_BY_INPUT_COUNT: Dict[int, str] = {
    sp._n_inputs(sp.SMALL):  "small",
    sp._n_inputs(sp.MEDIUM): "medium",
}


def _ir_size(ir: str) -> str:
    input_addrs, _, _ = sp._parse(ir)
    return _SIZE_BY_INPUT_COUNT.get(len(input_addrs), "unknown")


def collect_read_distances(ir: str) -> List[int]:
    """Replay the IR exactly the way sparse_parity's simulator counts
    reads, recording ⌈√addr⌉ for every operand read."""
    _, ops, output_addrs = sp._parse(ir)
    distances: List[int] = []
    for op, oprs in ops:
        if op == "set":
            continue
        if op == "cmp":
            _, a, b, _pred = oprs
            distances.append(sp._cost(a))
            distances.append(sp._cost(b))
            continue
        if op == "select":
            _, c, t, f = oprs
            distances.append(sp._cost(c))
            distances.append(sp._cost(t))
            distances.append(sp._cost(f))
            continue
        if op in _UNARY:
            _, src = oprs
            distances.append(sp._cost(src))
            continue
        if op in _BINARY:
            if len(oprs) == 3:
                _, s1, s2 = oprs
            else:
                dest, s2 = oprs
                s1 = dest
            distances.append(sp._cost(s1))
            distances.append(sp._cost(s2))
            continue
        raise ValueError(f"unknown op {op!r} while collecting distances")
    for a in output_addrs:
        distances.append(sp._cost(a))
    return distances


def plot_one(ir_path: Path, out_path: Path) -> Tuple[int, int]:
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
    cumulative_count = np.arange(1, len(sorted_d) + 1)
    ax_c.plot(sorted_d, cumulative_count, color="#cc4c4c", linewidth=1.5)
    ax_c.set_xlabel("distance")
    ax_c.set_ylabel("count")
    ax_c.set_title("CDF (reads at distance ≤ x)")
    ax_c.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return total_cost, n_reads


def plot_combined_cdf(ir_paths: List[Path], out_path: Path, title: str) -> int:
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("turbo")
    rows = []
    for ir_path in ir_paths:
        distances = np.sort(collect_read_distances(ir_path.read_text()))
        rows.append((ir_path.name, distances, int(distances.sum())))
    rows.sort(key=lambda r: r[2])  # cheapest first
    n = len(rows)
    for i, (name, distances, total) in enumerate(rows):
        color = cmap(0.05 + 0.9 * i / max(n - 1, 1))
        cumulative = np.arange(1, len(distances) + 1)
        ax.plot(distances, cumulative, label=f"{name}  (cost {total:,})",
                color=color, linewidth=1.6)
    # Log y so submissions whose read counts differ by 10× (e.g. baseline
    # vs packed-decoder) all stay visible instead of bunching at the bottom.
    ax.set_yscale("log")
    ax.set_xlabel("distance")
    ax.set_ylabel("count (log)")
    ax.set_title(f"{title}  ({n} submissions)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return n


def main() -> None:
    by_size: Dict[str, List[Path]] = {"small": [], "medium": [], "unknown": []}
    print(f"{'submission':<32}{'size':>8}{'reads':>10}{'total_cost':>14}  out")
    print("-" * 86)
    for ir_path in sorted(SUBMISSIONS.glob("*.ir")):
        ir = ir_path.read_text()
        size = _ir_size(ir)
        out_path = HERE / (ir_path.stem + ".png")
        total_cost, n_reads = plot_one(ir_path, out_path)
        by_size[size].append(ir_path)
        print(f"{ir_path.name:<32}{size:>8}{n_reads:>10,}{total_cost:>14,}  "
              f"{out_path.name}")

    for size in ("small", "medium"):
        paths = by_size[size]
        if not paths:
            continue
        combined_path = HERE / f"combined_cdf_{size}.png"
        n = plot_combined_cdf(
            paths, combined_path,
            f"CDF — reads at distance ≤ x  ({size})",
        )
        print(f"\nCombined {size} CDF: {combined_path.name}  ({n} submissions)")


if __name__ == "__main__":
    main()
