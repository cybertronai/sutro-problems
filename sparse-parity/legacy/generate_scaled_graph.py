#!/usr/bin/env python3
"""Energy-vs-accuracy curve for the scaled (n=32) sparse-parity tier.

Sweeps the reference ISD circuit over its two knobs (restarts T, outputs
computed f) plus the capped enumeration family, evaluates everything on
the deterministic dev suite (with two alternate suite keys overlaid to
show sampling noise), and writes ``doc/scaled32_energy_vs_accuracy.png``
(and ``.svg``) plus a markdown frontier table on stdout.

Runs in under a minute:  python3 generate_scaled_graph.py
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

import approx_sparse_parity as ap
import scaled_sparse_parity as sp

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)  # sparse-parity/ — doc/ and the mask module live there
sys.path.insert(0, PARENT)
OUT_PNG = os.path.join(PARENT, "doc", "scaled32_energy_vs_accuracy.png")
OUT_SVG = os.path.join(PARENT, "doc", "scaled32_energy_vs_accuracy.svg")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}

T_SWEEP = [1, 2, 3, 4, 5, 6]
F_SWEEP = [32, 64, 128, 192, 256]
Q_SWEEP = [600, 1200, 1797]          # capped enumeration: <=0.9% of candidates
ALT_KEYS = ["scaled-dev-b", "scaled-dev-c"]


@dataclass
class Series:
    label: str
    color: str
    points: List[Tuple[int, float]]   # (cost, raw accuracy)


def _eval(ir: str, suite_key: str = sp.DEV_SUITE_KEY):
    res = sp.evaluate_scaled(ir, suite_key=suite_key)
    return res.cost, res.raw_accuracy


def collect():
    series = [
        Series(
            "ISD — more Gaussian-elimination restarts (T=1…6)",
            SERIES["blue"],
            [_eval(sp.generate_isd(T)) for T in T_SWEEP],
        ),
        Series(
            "ISD, T=3 — compute fewer outputs (f=32…256)",
            SERIES["orange"],
            [_eval(sp.generate_isd(3, f)) for f in F_SWEEP],
        ),
        Series(
            "capped enumeration (≤1,797 of 201,376 candidates)",
            SERIES["aqua"],
            [_eval(ap.generate_mask_baseline(q, spec=sp.SCALED32)) for q in Q_SWEEP],
        ),
    ]
    # sampling-noise overlay: the ISD restart sweep on two other suite keys
    alt_curves = [
        [_eval(sp.generate_isd(T), key) for T in T_SWEEP] for key in ALT_KEYS
    ]
    return series, alt_curves


def pareto(points):
    frontier, best = [], -1.0
    for cost, acc in sorted(points):
        if acc > best + 1e-12:
            frontier.append((cost, acc))
            best = acc
    return frontier


def _fmt_cost(v, _pos=None):
    if v >= 1e6:
        return f"{v/1e6:g}M"
    if v >= 1e3:
        return f"{v/1e3:g}k"
    return f"{v:g}"


def plot(series: List[Series], alt_curves) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    front = pareto([p for s in series for p in s.points])
    ax.step(
        [a for _, a in front], [c for c, _ in front], where="pre",
        color=BASELINE, linewidth=1.2, zorder=1.5,
        label="Pareto frontier  E*(target)",
    )

    # sampling-noise overlay: faint replicas of the restart sweep under
    # two alternate suite keys (same energy, jittered accuracy).
    for curve in alt_curves:
        ax.plot(
            [a for _, a in curve], [c for c, _ in curve],
            color=SERIES["blue"], alpha=0.25, linewidth=1.2, zorder=2,
            marker="o", markersize=3.5, markeredgewidth=0,
        )

    for s in series:
        ax.plot(
            [a for _, a in s.points], [c for c, _ in s.points],
            color=s.color, linewidth=2, zorder=3, label=s.label,
            marker="o", markersize=5.5, markeredgecolor=SURFACE,
            markeredgewidth=1.2, solid_capstyle="round",
        )

    def _round_cost(v):
        return f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"

    cost, acc = series[0].points[-1]
    ax.annotate(
        f"T=6 restarts\n{_round_cost(cost)} reads", (acc, cost),
        xytext=(-8, -4), textcoords="offset points", ha="right", va="top",
        fontsize=8.5, color=INK_2,
    )
    cost, acc = series[2].points[-1]
    ax.annotate(
        "enumeration ceiling under the cap:\n≤0.9% of candidates, ~50% accuracy", (acc, cost),
        xytext=(10, -2), textcoords="offset points", ha="left", va="top",
        fontsize=8.5, color=INK_2,
    )

    ax.set_yscale("log")
    ax.set_xlim(0.498, 0.60)
    ax.set_ylim(2e6, 4.8e7)
    ax.set_ylabel(
        "Energy — static read cost (simplified Dally model, v3 ISA)",
        color=INK_2, fontsize=10,
    )
    ax.set_xlabel(
        "Test accuracy (128 sampled secrets × 8 reps × 256 rows, dev suite)",
        color=INK_2, fontsize=10,
    )
    ax.set_title(
        "Scaled sparse parity — energy required vs accuracy\n"
        "n=32 bits, k=5 hidden, 18 train / 256 test · 250k-instruction cap",
        color=INK, fontsize=11.5, loc="left", pad=30,
    )

    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_cost))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([2e6, 5e6, 1e7, 2e7, 4e7]))
    xticks = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax2 = ax.secondary_xaxis(
        "top", functions=(lambda a: 2 * a - 1, lambda e: (e + 1) / 2)
    )
    ax2.set_xlabel("normalized advantage  η = 2·acc − 1",
                   color=INK_2, fontsize=9, labelpad=6)
    ax2.xaxis.set_major_locator(FixedLocator([2 * t - 1 for t in xticks]))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))

    for a in (ax, ax2):
        a.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.set_axisbelow(True)

    legend = ax.legend(
        loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2,
        handlelength=1.6, borderaxespad=0.2,
    )
    for line in legend.get_lines():
        line.set_linewidth(2.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, facecolor=SURFACE)
    fig.savefig(OUT_SVG, facecolor=SURFACE)
    plt.close(fig)


def frontier_table(series: List[Series]) -> str:
    labeled = []
    for s, knobs, fmt in (
        (series[0], T_SWEEP, "ISD, T={}"),
        (series[1], F_SWEEP, "ISD T=3, f={}"),
        (series[2], Q_SWEEP, "enumeration, q={}"),
    ):
        for (cost, acc), kn in zip(s.points, knobs):
            labeled.append((cost, acc, fmt.format(kn)))
    front = set(pareto([(c, a) for c, a, _ in labeled]))
    lines = [
        "| Energy (reads) | Accuracy | Advantage | Cheapest strategy |",
        "| -: | -: | -: | - |",
    ]
    seen = set()
    for cost, acc, label in sorted(labeled):
        if (cost, acc) in front and (cost, acc) not in seen:
            seen.add((cost, acc))
            lines.append(f"| {cost:,} | {acc:.2%} | {2*acc-1:.3f} | {label} |")
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    series, alt_curves = collect()
    plot(series, alt_curves)
    print(frontier_table(series))
    print()
    for s_ in series:
        pts = " ".join(f"({c/1e6:.2f}M,{a:.3f})" for c, a in s_.points)
        print(f"  {s_.label}: {pts}")
    main_curve = [a for _, a in series[0].points]
    spread = max(
        abs(a - b)
        for curve in alt_curves
        for (_, a), b in zip(curve, main_curve)
    )
    print()
    print(f"max ISD-curve accuracy spread across 3 suite keys: {spread:.4f}")
    print(f"total {time.time() - t0:.0f}s; wrote {os.path.relpath(OUT_PNG, PARENT)}")


if __name__ == "__main__":
    main()
