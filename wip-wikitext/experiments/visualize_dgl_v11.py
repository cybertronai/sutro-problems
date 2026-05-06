"""Static PNGs for the DGL v11 record (0.6839 char-acc, 93 kJ, 558 s).

Following the cybertronai/hinton-problems convention:
  * pure matplotlib, no global rcParams
  * one figure per concern, saved into viz_dgl_v11/ at dpi=120
  * per-layer curves cycled through an explicit tab10 hex palette,
    lw=1.5; light grid (alpha=0.3); small fonts; suptitle.

Run from the wip-wikitext directory:

    python3 experiments/visualize_dgl_v11.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
WIKI = HERE.parent
VIZ_DIR = HERE / "viz_dgl_v11"
RECORD = WIKI / "records" / "2026-05-06T10-33-29-dgl-greedy-layerwise-6f0bbd75"

# Hinton-problems palette: explicit tab10 hex, picked by hand.
PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
GREY = "0.7"
RED = "#d62728"
BLACK = "black"

V11_ACC, V11_KJ = 0.6839, 93.135
MODDED_ACC, MODDED_KJ = 0.7310, 55.345
E_BUDGET_KJ = 100.0


# --------------------------------------------------------------------- parse


def parse_run_log(log_path: Path) -> dict[int, dict[str, list[float]]]:
    """Extract per-layer per-step (step, loss, scale, elapsed) from run.log."""
    pat = re.compile(
        r"layer\s+(\d+)\s+step\s+(\d+)\s+loss\s+([\d.]+)\s+scale\s+([\d.]+)\s+elapsed\s+(\d+)s"
    )
    out: dict[int, dict[str, list[float]]] = {}
    for line in log_path.read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        layer, step, loss, scale, elapsed = m.groups()
        d = out.setdefault(int(layer), {"step": [], "loss": [], "scale": [], "elapsed": []})
        d["step"].append(int(step))
        d["loss"].append(float(loss))
        d["scale"].append(float(scale))
        d["elapsed"].append(int(elapsed))
    return out


# ---------------------------------------------- figure 1: training trajectory


def fig_training(per_layer: dict[int, dict[str, list[float]]]) -> None:
    """NLL vs global step + LR scale schedule, one line per layer."""
    n_layers = len(per_layer)
    steps_per_layer = max(max(d["step"]) for d in per_layer.values()) + 1

    fig = plt.figure(figsize=(11, 5.2), constrained_layout=False)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[3.0, 0.7], hspace=0.12,
        left=0.07, right=0.985, top=0.84, bottom=0.10,
    )
    ax_loss = fig.add_subplot(gs[0])
    ax_sch = fig.add_subplot(gs[1], sharex=ax_loss)

    for li in range(n_layers):
        d = per_layer[li]
        x_global = np.array(d["step"]) + li * steps_per_layer
        ax_loss.plot(
            x_global,
            d["loss"],
            color=PALETTE[li % len(PALETTE)],
            linewidth=1.5,
            label=f"L{li}",
            marker="o",
            markersize=2.6,
        )
        ax_sch.plot(
            x_global,
            d["scale"],
            color=PALETTE[li % len(PALETTE)],
            linewidth=1.2,
        )

    # phase boundaries between layers
    for li in range(1, n_layers):
        x = li * steps_per_layer
        ax_loss.axvline(x, color=BLACK, linestyle="--", linewidth=0.6, alpha=0.4)
        ax_sch.axvline(x, color=BLACK, linestyle="--", linewidth=0.6, alpha=0.4)

    # zoom y-axis past the cold-start spike (layer 0 step 0 ≈ 5.6 nats — log(256))
    ax_loss.set_ylim(1.05, 1.42)
    ax_loss.set_ylabel("local NLL  (nats / char)", fontsize=9)
    ax_loss.grid(alpha=0.3)
    ax_loss.tick_params(labelsize=7, labelbottom=False)
    ax_loss.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.005), fontsize=8,
        ncol=n_layers, frameon=False, columnspacing=1.4, handlelength=1.6,
    )
    ax_loss.set_title(
        "per-layer aux-head NLL vs global step  —  each layer trained in isolation, no backprop through the stack",
        fontsize=9, loc="right", pad=4,
    )

    # phase labels along the bottom of the loss panel
    ymin, ymax = ax_loss.get_ylim()
    for li in range(n_layers):
        ax_loss.text(
            li * steps_per_layer + steps_per_layer * 0.5,
            ymin + (ymax - ymin) * 0.04,
            f"layer {li}\n5000 Muon+AdamW steps",
            ha="center", va="bottom",
            fontsize=7, color="0.35",
        )

    # cold-start annotation (the 5.6→1.5 drop is clipped off-axis)
    ax_loss.annotate(
        "step 0 NLL ≈ 5.6\n(uniform-byte init)",
        xy=(0, 1.42), xytext=(800, 1.39),
        fontsize=7, color="0.4",
        arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6),
    )

    ax_sch.set_ylabel("LR\nscale", fontsize=8)
    ax_sch.set_xlabel("global step", fontsize=9)
    ax_sch.set_ylim(-0.05, 1.1)
    ax_sch.set_yticks([0.0, 0.5, 1.0])
    ax_sch.tick_params(labelsize=7)
    ax_sch.grid(alpha=0.3)

    fig.suptitle(
        "DGL v11 — greedy layerwise training of a 12.85M-param 4×d=512 char-LM on WikiText-103\n"
        f"final char-acc {V11_ACC:.4f}   |   training energy {V11_KJ:.1f} kJ ({V11_KJ/E_BUDGET_KJ:.0%} of budget)"
        f"   |   wall-clock 558 s   |   A100 SXM4 40 GB",
        fontsize=10, y=0.965,
    )
    out = VIZ_DIR / "dgl_v11_training.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------- figure 2: progression vs backprop


def fig_progression() -> None:
    """Stacked improvements v1 → v11; reference line at modded-backprop record."""
    series = [
        ("v1",  "plain DGL\n5×384 AdamW",                 0.6440),
        ("v2",  "+ warm-start\naux heads",                0.6553),
        ("v6",  "+ d=512,\n4 layers",                     0.6638),
        ("v7",  "+ Muon\non 2-D weights",                 0.6789),
        ("v9",  "+ CE readout\nfrom concat feats",        0.6803),
        ("v11", "+ 5000 vs 4000\nsteps / layer",          0.6839),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    xs = np.arange(len(series))
    accs = np.array([a for _, _, a in series])

    bar_colors = [GREY] * (len(series) - 1) + [PALETTE[0]]
    bars = ax.bar(xs, accs, color=bar_colors, edgecolor=BLACK, linewidth=0.6, width=0.6)

    # delta labels above each bar (skip last — gap arrow lives there)
    for i in range(1, len(series) - 1):
        d = accs[i] - accs[i - 1]
        ax.text(xs[i], accs[i] + 0.0025, f"+{d*100:.2f} pp",
                ha="center", fontsize=7, color="0.3")
    # last delta tucked inside the bar so the gap arrow has clear space above
    d_last = accs[-1] - accs[-2]
    ax.text(xs[-1], accs[-1] - 0.004, f"+{d_last*100:.2f} pp",
            ha="center", fontsize=7, color="white")

    for i, (_, _, a) in enumerate(series):
        ax.text(xs[i], a / 2 + 0.305, f"{a:.4f}",
                ha="center", va="center", fontsize=8, color="white")

    ax.axhline(MODDED_ACC, color=RED, linestyle="--", linewidth=1.2)
    ax.text(
        len(series) - 1 - 0.45, MODDED_ACC + 0.0015,
        f"modded backprop record: {MODDED_ACC:.4f} char-acc @ {MODDED_KJ:.0f} kJ",
        ha="right", fontsize=8, color=RED,
    )

    # gap annotation — arrow runs above the v11 bar; label sits inside the bar (left)
    gap_x = len(series) - 1 - 0.18
    ax.annotate(
        "",
        xy=(gap_x, MODDED_ACC),
        xytext=(gap_x, V11_ACC),
        arrowprops=dict(arrowstyle="<->", color=BLACK, lw=0.8),
    )
    ax.text(
        gap_x - 0.06, (MODDED_ACC + V11_ACC) / 2,
        "4.71 pp\ngap remaining",
        fontsize=7.5, color=BLACK, ha="right", va="center",
    )

    # tick labels — variant tag big, config (already \n-wrapped) subdued underneath
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for lbl, _, _ in series], fontsize=10, fontweight="bold")
    for i, (_, cfg, _) in enumerate(series):
        ax.text(xs[i], 0.616, cfg, ha="center", va="top", fontsize=7,
                color="0.3", linespacing=1.15)
    ax.set_ylabel("test char-accuracy  (60K-char slice)", fontsize=9)
    ax.set_ylim(0.602, 0.748)
    ax.tick_params(axis="y", labelsize=7, pad=2)
    ax.tick_params(axis="x", length=0, pad=24)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.suptitle(
        "DGL on WikiText-103: cumulative improvements toward the v11 high-water mark\n"
        "every config trained without end-to-end backprop; each bar is the best run of that variant",
        fontsize=10, y=0.965,
    )
    fig.subplots_adjust(top=0.86, bottom=0.22, left=0.07, right=0.985)
    out = VIZ_DIR / "dgl_v11_progression.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------- figure 3: per-layer NLL v11 vs v7


def fig_perlayer_nll() -> None:
    layers = [0, 1, 2, 3]
    v11 = [1.25, 1.15, 1.10, 1.14]
    v7 = [1.24, 1.17, 1.18, 1.12]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    w = 0.36
    xs = np.arange(len(layers))
    b7 = ax.bar(xs - w/2, v7, w, color=GREY, label="v7  (4000 steps/layer)", edgecolor=BLACK, linewidth=0.5)
    b11 = ax.bar(xs + w/2, v11, w, color=PALETTE[0], label="v11 (5000 steps/layer)", edgecolor=BLACK, linewidth=0.5)

    for bars, vals in ((b7, v7), (b11, v11)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.2f}",
                    ha="center", fontsize=7.5)

    # delta annotations (v11 - v7); green = improvement (lower NLL), red = regression
    for i, l in enumerate(layers):
        d = v11[i] - v7[i]
        col = "#2ca02c" if d < 0 else RED
        ax.text(
            i, 0.915, f"Δ {d:+.02f}",
            ha="center", fontsize=7.5, color=col,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2),
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([f"layer {l}" for l in layers], fontsize=8)
    ax.set_ylabel("final aux-head NLL  (nats / char)", fontsize=9)
    ax.set_ylim(0.88, 1.32)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    fig.suptitle(
        "where v11's +0.36 pp came from: mid-network layers had headroom under the per-layer ceiling\n"
        "boundary layers (0, 3) are unchanged; layer 2 falls below the 1.18 plateau for the first time",
        fontsize=9, y=0.965,
    )
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.10, right=0.98)
    out = VIZ_DIR / "dgl_v11_perlayer_nll.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------- figure 4: daily energy/accuracy Pareto


def fig_pareto() -> None:
    """All dgl-greedy-layerwise runs over 2026-05-06 in the (energy, acc) plane."""
    sub_dir = WIKI / "submissions"
    runs = []
    for d in sorted(sub_dir.glob("2026-05-06T*-dgl-greedy-layerwise-*")):
        rj = d / "result.json"
        if rj.exists():
            r = json.loads(rj.read_text())
            runs.append((r["training_energy_J"] / 1000.0, r["test_char_accuracy"], d.name))
    # the v11 record lives in records/, not submissions/
    rec_rj = RECORD / "result.json"
    rec = json.loads(rec_rj.read_text())
    v11 = (rec["training_energy_J"] / 1000.0, rec["test_char_accuracy"], "v11 record")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # exploration runs
    xs = [r[0] for r in runs]
    ys = [r[1] for r in runs]
    ax.scatter(xs, ys, s=42, color=GREY, edgecolor=BLACK, linewidth=0.5,
               label=f"earlier DGL-day runs (n={len(runs)})")

    # path through the day, in chronological order
    ax.plot(xs, ys, color=GREY, linewidth=0.8, alpha=0.6)

    # v11 highlight
    ax.scatter([v11[0]], [v11[1]], s=120, color=PALETTE[0], edgecolor=BLACK,
               linewidth=0.7, zorder=4, label="v11 record  0.6839 / 93 kJ")
    ax.annotate("v11", xy=(v11[0], v11[1]), xytext=(v11[0] - 4, v11[1] + 0.004),
                fontsize=9, color=PALETTE[0], fontweight="bold")

    # modded-backprop reference
    ax.scatter([MODDED_KJ], [MODDED_ACC], s=110, color=RED, marker="*",
               edgecolor=BLACK, linewidth=0.4, zorder=4,
               label=f"modded backprop  {MODDED_ACC:.4f} / {MODDED_KJ:.0f} kJ")

    # 100 kJ budget line
    ax.axvline(E_BUDGET_KJ, color=BLACK, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(E_BUDGET_KJ + 0.5, 0.595, "100 kJ budget", rotation=90, fontsize=7.5,
            color="0.3", va="bottom", ha="left")

    ax.set_xlabel("training energy  (kJ, NVML net of idle)", fontsize=9)
    ax.set_ylabel("test char-accuracy  (60K chars)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_xlim(45, 108)
    ax.set_ylim(0.59, 0.75)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper left", frameon=False)

    fig.suptitle(
        "DGL exploration on 2026-05-06: energy vs char-accuracy across all 9 jobs\n"
        "v11 is the high-water mark for no-backprop training within the 100 kJ budget",
        fontsize=10, y=0.965,
    )
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.09, right=0.98)
    out = VIZ_DIR / "dgl_v11_pareto.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------------------------------------- main


def main() -> None:
    VIZ_DIR.mkdir(exist_ok=True)
    per_layer = parse_run_log(RECORD / "run.log")
    fig_training(per_layer)
    fig_progression()
    fig_perlayer_nll()
    fig_pareto()


if __name__ == "__main__":
    main()
