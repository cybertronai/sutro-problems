#!/usr/bin/env python3
"""Energy-vs-recovery curve for the mask (test-set-free, n=32) tier.

Sweeps the three reference families -- the GE + null-space Gray scan
(reaches 100%), ISD restarts, and capped candidate enumeration -- on the
deterministic dev suite, overlays two alternate suite keys on the scan
sweep to show sampling noise, and writes
``doc/mask32_energy_vs_recovery.png`` (and ``.svg``) plus a markdown
frontier table on stdout.

Runs in a couple of minutes:  python3 generate_mask_graph.py
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

import mask_sparse_parity as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "doc", "mask32_energy_vs_recovery.png")
OUT_SVG = os.path.join(HERE, "doc", "mask32_energy_vs_recovery.svg")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}

S_SWEEP = [0, 1023, 2047, 4095, 8191, 16383]
T_SWEEP = [1, 4, 8, 16, 32]
Q_SWEEP = [5000, 15000]
ALT_KEYS = ["mask-dev-b", "mask-dev-c"]
ALT_S = [0, 4095, 16383]


@dataclass
class Series:
    label: str
    color: str
    points: List[Tuple[int, float]]   # (cost, recovery)


def _eval(ir: str, suite_key: str = mp.DEV_SUITE_KEY):
    res = mp.evaluate_mask(ir, suite_key=suite_key)
    return res.cost, res.recovery


def collect():
    series = [
        Series(
            "GE + null-space Gray scan (s = 0…16,383 steps)",
            SERIES["blue"],
            [_eval(mp.generate_scan(s)) for s in S_SWEEP],
        ),
        Series(
            "ISD — Gaussian-elimination restarts (T = 1…32)",
            SERIES["orange"],
            [_eval(mp.generate_isd_mask(T)) for T in T_SWEEP],
        ),
        Series(
            "capped enumeration (≤15,000 of 201,376 candidates)",
            SERIES["aqua"],
            [_eval(mp.generate_enum_mask(q)) for q in Q_SWEEP],
        ),
    ]
    alt_curves = [
        [_eval(mp.generate_scan(s), key) for s in ALT_S] for key in ALT_KEYS
    ]
    return series, alt_curves


def pareto(points):
    frontier, best = [], -1.0
    for cost, rec in sorted(points):
        if rec > best + 1e-12:
            frontier.append((cost, rec))
            best = rec
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
        [r for _, r in front], [c for c, _ in front], where="pre",
        color=BASELINE, linewidth=1.2, zorder=1.5,
        label="Pareto frontier  E*(target)",
    )

    for curve in alt_curves:
        ax.plot(
            [r for _, r in curve], [c for c, _ in curve],
            color=SERIES["blue"], alpha=0.25, linewidth=1.2, zorder=2,
            marker="o", markersize=3.5, markeredgewidth=0,
        )

    for s in series:
        ax.plot(
            [r for _, r in s.points], [c for c, _ in s.points],
            color=s.color, linewidth=2, zorder=3, label=s.label,
            marker="o", markersize=5.5, markeredgecolor=SURFACE,
            markeredgewidth=1.2, solid_capstyle="round",
        )

    def _round_cost(v):
        return f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"

    cost, rec = series[0].points[-1]
    ax.annotate(
        f"full scan: 100% recovery\n@ {_round_cost(cost)} reads", (rec, cost),
        xytext=(-8, -6), textcoords="offset points", ha="right", va="top",
        fontsize=8.5, color=INK_2,
    )
    cost, rec = series[2].points[-1]
    ax.annotate(
        "enumeration: ≤8% recovery,\ndominated everywhere", (rec, cost),
        xytext=(10, 4), textcoords="offset points", ha="left",
        fontsize=8.5, color=INK_2,
    )

    ax.set_yscale("log")
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(1e6, 6.5e7)
    ax.set_ylabel(
        "Energy — static read cost (simplified Dally model, v3 ISA)",
        color=INK_2, fontsize=10,
    )
    ax.set_xlabel(
        "Secret recovery rate — exact 32-bit mask match "
        "(128 sampled secrets × 8 reps, dev suite)",
        color=INK_2, fontsize=10,
    )
    ax.set_title(
        "Mask sparse parity — energy required vs recovery\n"
        "n=32 bits, k=5 hidden, 18 train, no test set · 2M-instruction cap",
        color=INK, fontsize=11.5, loc="left", pad=12,
    )

    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_cost))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([1e6, 2e6, 5e6, 1e7, 2e7, 5e7]))
    xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

    for a in (ax,):
        a.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
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
        (series[0], S_SWEEP, "scan, s={}"),
        (series[1], T_SWEEP, "ISD, T={}"),
        (series[2], Q_SWEEP, "enumeration, q={}"),
    ):
        for (cost, rec), kn in zip(s.points, knobs):
            labeled.append((cost, rec, fmt.format(kn)))
    front = set(pareto([(c, r) for c, r, _ in labeled]))
    lines = [
        "| Energy (reads) | Recovery | Cheapest strategy |",
        "| -: | -: | - |",
    ]
    seen = set()
    for cost, rec, label in sorted(labeled):
        if (cost, rec) in front and (cost, rec) not in seen:
            seen.add((cost, rec))
            lines.append(f"| {cost:,} | {rec:.1%} | {label} |")
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    series, alt_curves = collect()
    plot(series, alt_curves)
    print(frontier_table(series))
    main_pts = {s: r for (_, r), s in zip(series[0].points, S_SWEEP)}
    spread = max(
        abs(r - main_pts[s])
        for curve in alt_curves
        for (_, r), s in zip(curve, ALT_S)
    )
    print()
    print(f"max scan-curve recovery spread across 3 suite keys: {spread:.4f}")
    print(f"total {time.time() - t0:.0f}s; wrote {os.path.relpath(OUT_PNG, HERE)}")


if __name__ == "__main__":
    main()
