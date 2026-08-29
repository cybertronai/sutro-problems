#!/usr/bin/env python3
"""Generate the accuracy-vs-energy curve for approximate sparse parity.

Sweeps the two approximate baseline families over their knobs (candidates
searched, outputs computed), evaluates every IR on the deterministic public
suite, and plots test accuracy (y) against static read cost (x, the energy
proxy of the simplified Dally model).  Writes
``doc/approx_accuracy_vs_energy.png`` (and ``.svg``) and prints a markdown
table of the Pareto frontier.

Runs in a few seconds:  python3 generate_graph.py
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

import approx_sparse_parity as ap
import mask_sparse_parity as mp
import scaled_sparse_parity as sp

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)  # sparse-parity/ — doc/ and the mask module live there
sys.path.insert(0, PARENT)
OUT_PNG = os.path.join(PARENT, "doc", "approx_accuracy_vs_energy.png")
OUT_SVG = os.path.join(PARENT, "doc", "approx_accuracy_vs_energy.svg")

REPETITIONS = ap.DEV_REPETITIONS  # 220 secrets x 8 reps = 1,760 instances

# Chart tokens (light mode; see the repo-independent palette notes in the
# PR/report -- slots 1-3 are the all-pairs colorblind-validated trio).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
          "yellow": "#eda100"}

Q_SWEEP = [11, 22, 44, 66, 88, 110, 132, 154, 176, 198, 220]
T_SWEEP = [1, 2, 4, 8, 12]      # ISD restarts (12 distinct subsets at n=12)
S_SWEEP = [0, 3, 7, 15]         # Gray-scan steps (2^(n-m) = 16 solutions)
OP_CAP = 100_000


@dataclass
class Series:
    label: str
    color: str
    points: List[Tuple[int, float]]  # (cost, raw accuracy), sweep order


def _sweep(gen: Callable[[int, int], str], knobs) -> List[Tuple[int, float]]:
    pts = []
    for q, f in knobs:
        res = ap.evaluate(gen(q, f), repetitions=REPETITIONS)
        pts.append((res.cost, res.raw_accuracy))
    return pts


def _eval_ir(ir: str) -> Tuple[int, float]:
    res = ap.evaluate(ir, repetitions=REPETITIONS)
    return res.cost, res.raw_accuracy


def collect() -> Tuple[List[Series], Tuple[int, float]]:
    naive, mask = ap.generate_approx_baseline, ap.generate_mask_baseline
    series = [
        Series(
            "try-each-candidate — search fewer candidates",
            SERIES["blue"],
            _sweep(naive, [(q, 32) for q in Q_SWEEP]),
        ),
        Series(
            "ISD — Gaussian-elimination restarts (T = 1…12)",
            SERIES["orange"],
            [_eval_ir(sp.generate_isd(T, spec=ap.APPROX, op_cap=OP_CAP))
             for T in T_SWEEP],
        ),
        Series(
            "mask decoder — search fewer candidates",
            SERIES["aqua"],
            _sweep(mask, [(q, 32) for q in Q_SWEEP]),
        ),
        Series(
            "GE + null-space Gray scan (s = 0…15)",
            SERIES["yellow"],
            [_eval_ir(mp.generate_scan(sv, spec=ap.APPROX, joint=True,
                                       op_cap=OP_CAP))
             for sv in S_SWEEP],
        ),
    ]
    res = ap.evaluate(naive(0, 0), repetitions=REPETITIONS)
    return series, (res.cost, res.raw_accuracy)


def pareto(points: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Lower envelope: cheapest energy achieving each accuracy level."""
    frontier: List[Tuple[int, float]] = []
    best = -1.0
    for cost, acc in sorted(points):
        if acc > best + 1e-12:
            frontier.append((cost, acc))
            best = acc
    return frontier


def check_suite_independence() -> float:
    """The baselines' accuracy is exact by construction: identical under
    every suite key.  Returns the max deviation observed (should be 0)."""
    ir = ap.generate_mask_baseline(110, 16)
    accs = [
        ap.evaluate(ir, repetitions=REPETITIONS, suite_key=key).raw_accuracy
        for key in ("public", "alt-a", "alt-b")
    ]
    return max(accs) - min(accs)


def _fmt_cost(v, _pos=None) -> str:
    if v >= 1e6:
        return f"{v/1e6:g}M"
    if v >= 1e3:
        return f"{v/1e3:g}k"
    return f"{v:g}"


def plot(series: List[Series]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    all_points = [p for s in series for p in s.points]
    front = pareto(all_points)

    # Pareto frontier E*(target): least energy achieving each accuracy
    # target -- a step underlay in recessive ink (energy jumps at each
    # accuracy level a strategy unlocks).
    ax.step(
        [a for _, a in front], [c for c, _ in front], where="pre",
        color=BASELINE, linewidth=1.2, zorder=1.5,
        label="Pareto frontier  E*(target)",
    )

    for s in series:
        x = [a for _, a in s.points]
        y = [c for c, _ in s.points]
        ax.plot(
            x, y, color=s.color, linewidth=2, zorder=3, label=s.label,
            marker="o", markersize=5.5, markeredgecolor=SURFACE,
            markeredgewidth=1.2, solid_capstyle="round",
        )

    def _round_cost(v: float) -> str:
        return f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"

    # Selective direct labels at the (near-)exact endpoints.
    for s, name, xy, ha, va in (
        (series[0], "try-each-candidate", (-8, 14), "right", "top"),
        (series[2], "mask decoder", (2, -12), "right", "top"),
    ):
        cost, acc = s.points[-1]
        ax.annotate(
            f"{name}\nexact @ {_round_cost(cost)} reads",
            (acc, cost), xytext=xy, textcoords="offset points",
            ha=ha, va=va, fontsize=8.5, color=INK_2,
        )
    # scan: label the flat stretch (fixed GE cost, near-free sweep)
    cost1, acc1 = series[3].points[1]
    ax.annotate(
        f"GE + Gray scan: flat @ {_round_cost(cost1)} reads, 99% at s=15",
        (acc1, cost1), xytext=(-4, 8), textcoords="offset points",
        ha="left", va="bottom", fontsize=8.5, color=INK_2,
    )

    ax.set_yscale("log")
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(3.5e4, 4.5e6)
    ax.set_ylabel(
        "Energy — static read cost (simplified Dally model, v3 ISA)",
        color=INK_2, fontsize=10,
    )
    ax.set_xlabel("Test accuracy target (aggregate: 220 secrets × 8 reps × 32 rows)",
                  color=INK_2, fontsize=10)
    ax.set_title(
        "Approximate sparse parity — energy required vs accuracy\n"
        "n=12 bits, k=3 hidden, 8 train / 32 test · deterministic public suite",
        color=INK, fontsize=11.5, loc="left", pad=30,
    )

    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_cost))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator(
        [5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 4e6]))
    xticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Top axis: the same scale relabeled as normalized advantage.
    ax2 = ax.secondary_xaxis(
        "top", functions=(lambda a: 2 * a - 1, lambda e: (e + 1) / 2)
    )
    ax2.set_xlabel("normalized advantage  η = 2·acc − 1",
                   color=INK_2, fontsize=9, labelpad=6)
    ax2.xaxis.set_major_locator(FixedLocator([2 * t - 1 for t in xticks]))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))

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
        loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2,
        handlelength=1.6, borderaxespad=0.2,
    )
    for line in legend.get_lines():
        line.set_linewidth(2.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, facecolor=SURFACE)
    fig.savefig(OUT_SVG, facecolor=SURFACE)
    plt.close(fig)


def frontier_table(series: List[Series], chance: Tuple[int, float]) -> str:
    labeled = [(chance[0], chance[1], "constant guess")]
    sweeps = [("q", Q_SWEEP), ("T", T_SWEEP), ("q", Q_SWEEP), ("s", S_SWEEP)]
    for s, (knob, knobs) in zip(series, sweeps):
        short = s.label.split(" — ")[0]
        for (cost, acc), k in zip(s.points, knobs):
            labeled.append((cost, acc, f"{short}, {knob}={k}"))
    front = set(pareto([(c, a) for c, a, _ in labeled]))
    lines = [
        "| Energy (reads) | Accuracy | Advantage | Cheapest strategy |",
        "| -: | -: | -: | - |",
    ]
    seen = set()
    for cost, acc, label in sorted(labeled):
        if (cost, acc) in front and (cost, acc) not in seen:
            seen.add((cost, acc))
            lines.append(
                f"| {cost:,} | {acc:.1%} | {2 * acc - 1:.3f} | {label} |"
            )
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    series, chance = collect()
    t1 = time.time()
    spread = check_suite_independence()
    plot(series)
    t2 = time.time()

    print(frontier_table(series, chance))
    print()
    print(f"suite: {220 * REPETITIONS:,} instances "
          f"({220 * REPETITIONS * 32:,} labels), key '{ap.PUBLIC_SUITE_KEY}'")
    print(f"baseline accuracy spread across 3 suite keys: {spread:.2e}")
    print(f"evaluated {sum(len(s.points) for s in series) + 1} IRs "
          f"in {t1 - t0:.1f}s; total {t2 - t0:.1f}s")
    print(f"wrote {os.path.relpath(OUT_PNG, PARENT)} and "
          f"{os.path.relpath(OUT_SVG, HERE)}")


if __name__ == "__main__":
    main()
