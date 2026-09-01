#!/usr/bin/env python3
"""Energy-vs-recovery curve for the mask (test-set-free, n=32) tier.

Sweeps the two reference families -- the GE + null-space Gray scan
(reaches 100%) and ISD restarts -- on the deterministic dev suite and
writes ``doc/mask32_energy_vs_recovery.png`` (and ``.svg``), plus
``doc/mask32_bands.json`` with every measured point and the cheapest
solution at each recovery band (20/40/60/80/100%), plus a markdown
band table on stdout.

Runs in a couple of minutes:  python3 doc/generate_mask_graph.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# Keep generated SVG IDs stable and omit wall-clock metadata so rerunning this
# script with the same inputs produces a reviewable artifact diff.
matplotlib.rcParams["svg.hashsalt"] = "sutro-sparse-parity-mask32-v1"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))       # the tier module lives one level up

import mask_sparse_parity as mp

_SUB = os.path.join(os.path.dirname(HERE), "submissions")
sys.path.insert(0, _SUB)
import packedsis, packedwalk, septwalk  # noqa: E402

OUT_PNG = os.path.join(HERE, "mask32_energy_vs_recovery.png")
OUT_SVG = os.path.join(HERE, "mask32_energy_vs_recovery.svg")
OUT_JSON = os.path.join(HERE, "mask32_bands.json")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = {"blue": "#2a78d6", "orange": "#eb6834",
           "green": "#2e9e4f", "purple": "#8259b8",
           "teal": "#1f8a8a", "magenta": "#c2548a",
           "crimson": "#c0392b", "olive": "#7f8c1f",
           "slate": "#4a6572", "amber": "#c9971e",
           "brick": "#8c4a3a"}

# Dense sweeps feed the band table; the marker subset keeps the plot quiet.
S_SWEEP = [0, 127, 255, 383, 511, 767, 1023, 1535, 2047, 3071, 4095,
           6143, 8191, 10239, 12287, 16383]
S_MARKERS = [0, 1023, 2047, 4095, 8191, 16383]
T_SWEEP = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
T_MARKERS = [1, 4, 8, 16, 32]
W_SWEEP = [0, 1, 2, 3, 4, 5]
W_MARKERS = [1, 2, 3, 5]
R_SWEEP = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
R_MARKERS = [1, 8, 16, 32]
SIS2_SWEEP = [1, 2]
SIS2_MARKERS = [1]
SIS3_SWEEP = [1, 2, 3, 4]
SIS3_MARKERS = [1, 4]
# Bit-packed families (7/8 bits per cell instead of 1): same algorithms as
# above, cheaper circuits. Knobs mirror each submission's own report.
PSIS_SWEEP = [0, 2, 4, 6, 8, 10, 12, 14]      # packedsis.py: g2 (partial cap-2 walk)
PSIS_MARKERS = [0, 8, 14]
PSIS3_SWEEP = [3]                              # packedsis.py: full cap-3 walk
PSIS3_MARKERS = [3]
PW_SWEEP = [0, 1, 2, 3]                       # packedwalk.py: cap, n_sets=1
PW_MARKERS = [1, 2, 3]
# The promoted cap-2 artifact uses seed 5; the other committed sweep points
# retain seed 0.  Keep the generated JSON/figure tied to the actual artifacts
# instead of silently plotting the retired seed-0 cap-2 point.
PW_SEEDS = {0: 0, 1: 0, 2: 5, 3: 0}
SEPT_SWEEP = [0, 1, 2, 3, 4, 5]                # septwalk.py: weight_cap
SEPT_MARKERS = [1, 3, 5]
BANDS = [0.2, 0.4, 0.6, 0.8, 1.0]


@dataclass
class Series:
    label: str
    color: str
    knob: str
    knobs: List[int]
    markers: List[int]
    points: List[Tuple[int, float, int]]   # (cost, recovery, ops)


def _eval(ir: str):
    res = mp.evaluate_mask(mp.optimize_layout(ir))
    return res.cost, res.recovery, len(ir.splitlines()) - 2


def _eval_raw(ir: str):
    """Like _eval, but for generators whose own pipeline already applies
    its layout pass (packedsis/packedwalk/septwalk) -- re-running the
    generic optimize_layout on top would relabel an already
    phase-optimal address space and typically raises cost slightly."""
    res = mp.evaluate_mask(ir)
    return res.cost, res.recovery, len(ir.splitlines()) - 2


def collect() -> List[Series]:
    return [
        Series(
            "Gray scan", SERIES["blue"], "s", S_SWEEP, S_MARKERS,
            [_eval(mp.generate_scan(s)) for s in S_SWEEP],
        ),
        Series(
            "ISD restarts", SERIES["orange"], "T", T_SWEEP, T_MARKERS,
            [_eval(mp.generate_isd_mask(T)) for T in T_SWEEP],
        ),
        Series(
            "Weight-ordered scan", SERIES["green"], "cap", W_SWEEP, W_MARKERS,
            [_eval(mp.generate_scan(0, walk="weight", weight_cap=w))
             for w in W_SWEEP],
        ),
        Series(
            "Random ISD", SERIES["purple"], "T", R_SWEEP, R_MARKERS,
            [_eval(mp.generate_isd_mask(T, subset_seed=0)) for T in R_SWEEP],
        ),        Series(
            "Static-IS walk (cap=2)", SERIES["teal"], "T", SIS2_SWEEP,
            SIS2_MARKERS,
            [_eval(mp.generate_sis_mask(T, 2)) for T in SIS2_SWEEP],
        ),
        Series(
            "Static-IS walk (cap=3)", SERIES["magenta"], "T", SIS3_SWEEP,
            SIS3_MARKERS,
            [_eval(mp.generate_sis_mask(T, 3)) for T in SIS3_SWEEP],
        ),
        Series(
            "Packed SIS (partial walk)", SERIES["crimson"], "g2",
            PSIS_SWEEP, PSIS_MARKERS,
            [_eval_raw(packedsis.generate_packed_sis(cap=2, seed=13, g2=g2))
             for g2 in PSIS_SWEEP],
        ),
        Series(
            "Packed SIS (cap=3 walk)", SERIES["brick"], "cap",
            PSIS3_SWEEP, PSIS3_MARKERS,
            [_eval_raw(packedsis.generate_packed_sis(cap=3, seed=13))],
        ),
        Series(
            "Bit-packed SIS walk", SERIES["olive"], "cap", PW_SWEEP,
            PW_MARKERS,
            [_eval_raw(packedwalk.generate(1, cap, seed=PW_SEEDS[cap]))
             for cap in PW_SWEEP],
        ),
        Series(
            "Septet-packed walk", SERIES["amber"], "cap", SEPT_SWEEP,
            SEPT_MARKERS,
            [_eval_raw(septwalk.generate_staged(weight_cap=w))
             for w in SEPT_SWEEP],
        ),
    ]


def _fmt_cost(v, _pos=None):
    if v >= 1e6:
        return f"{v/1e6:g}M"
    if v >= 1e3:
        return f"{v/1e3:g}k"
    return f"{v:g}"


def plot(series: List[Series]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for s in series:
        xs = [r for _, r, _ in s.points]
        ys = [c for c, _, _ in s.points]
        ax.plot(xs, ys, color=s.color, linewidth=2, zorder=3, label=s.label,
                solid_capstyle="round")
        mx = [r for (_, r, _), kn in zip(s.points, s.knobs) if kn in s.markers]
        my = [c for (c, _, _), kn in zip(s.points, s.knobs) if kn in s.markers]
        ax.plot(mx, my, color=s.color, linewidth=0, zorder=4, marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)

    # direct labels at the curve ends, in ink -- identity never color-alone
    by_label = {s.label: s for s in series}
    ends = [
        ("Static-IS walk (cap=3)", (6, 10), "left", "bottom"),
        ("Static-IS walk (cap=2)", (8, -8), "left", "top"),
        ("Gray scan", (-10, 6), "right", "bottom"),
        ("ISD restarts", (8, -2), "left", "top"),
        ("Weight-ordered scan", (-10, 6), "right", "bottom"),
        ("Random ISD", (8, -2), "left", "top"),
    ]
    for label, (dx, dy), ha, va in ends:
        if label not in by_label:
            continue
        end = by_label[label].points[-1]
        ax.annotate(label, (end[1], end[0]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=9.5, color=INK_2)

    # Linear y from zero. The families span 0.54M-29.2M (~54x), so a log
    # axis reads better, but a zero baseline makes the absolute size of each
    # gap legible instead of only its ratio.
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(0, 3.05e7)
    ax.set_ylabel("Energy — read cost (Dally model, v3 ISA)",
                  color=INK_2, fontsize=10)
    ax.set_xlabel("Secret recovery rate", color=INK_2, fontsize=10)
    ax.set_title(
        "Energy vs recovery — sparse parity, n=32 bits, k=5 secret",
        color=INK, fontsize=11.5, loc="left", pad=12,
    )

    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_cost))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(
        FixedLocator([0, 5e6, 1e7, 1.5e7, 2e7, 2.5e7, 3e7]))
    xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.set_axisbelow(True)

    # Upper left: on the linear-from-zero axis every curve sits low, so the
    # lower right (the log layout's home for the legend) is now occupied.
    legend = ax.legend(
        loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2,
        handlelength=1.6, borderaxespad=0.2,
    )
    for line in legend.get_lines():
        line.set_linewidth(2.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, facecolor=SURFACE)
    fig.savefig(OUT_SVG, facecolor=SURFACE, metadata={"Date": None})
    plt.close(fig)
    with open(OUT_SVG, encoding="utf-8") as f:
        svg = f.read()
    with open(OUT_SVG, "w", encoding="utf-8") as f:
        f.write("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")


def band_table(series: List[Series]):
    by_label = {s.label: s for s in series}
    methods = {
        "Gray scan": ("generate_scan", by_label["Gray scan"]),
        "ISD restarts": ("generate_isd_mask", by_label["ISD restarts"]),
        "Weight-ordered scan": ("generate_scan(0, walk='weight',"
                                " weight_cap=…)",
                                by_label["Weight-ordered scan"]),
        "Random ISD": ("generate_isd_mask(…, subset_seed=0)",
                       by_label["Random ISD"]),
        "Static-IS walk (cap=2)": ("generate_sis_mask(…, 2)",
                                  by_label["Static-IS walk (cap=2)"]),
        "Static-IS walk (cap=3)": ("generate_sis_mask(…, 3)",
                                  by_label["Static-IS walk (cap=3)"]),
        "Packed SIS (partial walk)": ("packedsis.generate_packed_sis(cap=2,"
                                       " seed=13, g2=…)",
                                       by_label["Packed SIS (partial walk)"]),
        "Packed SIS (cap=3 walk)": ("packedsis.generate_packed_sis(cap=3,"
                                     " seed=13)",
                                     by_label["Packed SIS (cap=3 walk)"]),
        "Bit-packed SIS walk": ("packedwalk.generate(1, …)",
                                 by_label["Bit-packed SIS walk"]),
        "Septet-packed walk": ("septwalk.generate_staged(weight_cap=…)",
                                by_label["Septet-packed walk"]),
    }
    fmt_call = {
        "Weight-ordered scan": lambda kn: f"generate_scan(0, walk='weight',"
                                          f" weight_cap={kn})",
        "Random ISD": lambda kn: f"generate_isd_mask({kn}, subset_seed=0)",
        "Static-IS walk (cap=2)": lambda kn: f"generate_sis_mask({kn}, 2)",
        "Static-IS walk (cap=3)": lambda kn: f"generate_sis_mask({kn}, 3)",
        "Packed SIS (partial walk)": lambda kn: (
            f"packedsis.generate_packed_sis(cap=2, seed=13, g2={kn})"),
        "Packed SIS (cap=3 walk)": lambda kn: (
            f"packedsis.generate_packed_sis(cap={kn}, seed=13)"),
        "Bit-packed SIS walk": lambda kn: (
            f"packedwalk.generate(1, {kn}, seed={PW_SEEDS[kn]})"),
        "Septet-packed walk": lambda kn: (
            f"septwalk.generate_staged(weight_cap={kn})"),
    }
    labeled = [
        {"method": name,
         "call": fmt_call.get(name, lambda kn, fn=fn: f"{fn}({kn})")(kn),
         "knob": kn, "cost": cost, "recovery": rec, "ops": ops}
        for name, (fn, s) in methods.items()
        for (cost, rec, ops), kn in zip(s.points, s.knobs)
    ]
    adjudicated_calls = {
        0.2: "packedsis.generate_packed_sis(cap=2, seed=13, g2=8)",
        0.4: "packedwalk.generate(1, 2, seed=5)",
        0.6: "packedsis.generate_packed_sis(cap=3, seed=13)",
        0.8: "septwalk.generate_staged(weight_cap=3)",
        1.0: "septwalk.generate_staged(weight_cap=5)",
    }
    bands = []
    for target in BANDS:
        eligible = [p for p in labeled if p["recovery"] >= target - 1e-9]
        if not eligible:
            continue
        best = min(eligible, key=lambda p: p["cost"])
        others = [p for p in eligible if p["method"] != best["method"]]
        runner = min(others, key=lambda p: p["cost"]) if others else None
        adjudicated = next(
            p for p in labeled if p["call"] == adjudicated_calls[target])
        bands.append({
            "target": target,
            "best": best,
            "adjudicated_best": adjudicated,
            "runner_up": runner,
        })
    return labeled, bands


def main() -> None:
    t0 = time.time()
    series = collect()
    plot(series)
    labeled, bands = band_table(series)
    with open(OUT_JSON, "w") as f:
        json.dump({"points": labeled, "bands": bands}, f, indent=1)

    print("| Target | Energy (reads) | Recovery | Ops | Cheapest solution |")
    print("| -: | -: | -: | -: | - |")
    for b in bands:
        p = b["adjudicated_best"]
        print(f"| {b['target']:.0%} | {p['cost']:,} | {p['recovery']:.1%} "
              f"| {p['ops']:,} | `{p['call']}` |")
    print()
    print(f"total {time.time() - t0:.0f}s; wrote "
          f"{os.path.relpath(OUT_PNG, HERE)}, {os.path.relpath(OUT_JSON, HERE)}")


if __name__ == "__main__":
    main()
