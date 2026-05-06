"""GIF animation of DGL v11's training procedure.

Two panels driven by the same per-step run.log data:

  * left  — network schematic (tok_emb → 4 transformer blocks, each with
            its own per-layer aux head). The active layer is colour-
            highlighted; frozen blocks are grey; future blocks are
            outlined only. The detach() barrier between the frozen
            prefix and the active block is drawn as a red bar (no
            gradient crosses it). The aux-head arrow and the local
            backward path on the active layer are colour-cycled to
            match the right panel.
  * right — per-layer aux-head NLL curves, drawn progressively as the
            global step advances. The current point is marked with a
            larger dot in the active layer's colour.

A bottom strip shows global progress / 20,000 steps and the current
phase, layer-step, elapsed seconds, and WSD LR scale.

Saved at wip-wikitext/experiments/viz_dgl_v11/dgl_v11_animation.gif via
matplotlib's PillowWriter (no ffmpeg dep). A static final-frame PNG is
also written for easy embedding.

Run from anywhere — paths resolve from __file__:

    python3 wip-wikitext/experiments/animate_dgl_v11.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
WIKI = HERE.parent
VIZ_DIR = HERE / "viz_dgl_v11"
RECORD = WIKI / "records" / "2026-05-06T10-33-29-dgl-greedy-layerwise-6f0bbd75"

# Same explicit tab10 hex palette as the static figures.
PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
GREY_FILL = "#e6e6e6"
GREY_EDGE = "#b0b0b0"
GREY_TEXT = "#7a7a7a"
DETACH_RED = "#d62728"
BLACK = "black"

N_LAYERS = 4
STEPS_PER_LAYER = 5000
TOTAL_STEPS = N_LAYERS * STEPS_PER_LAYER


# --------------------------------------------------------------------- parse


def parse_run_log(log_path: Path) -> list[dict]:
    """Flatten the run.log into a chronological list of frame dicts."""
    pat = re.compile(
        r"layer\s+(\d+)\s+step\s+(\d+)\s+loss\s+([\d.]+)\s+scale\s+([\d.]+)\s+elapsed\s+(\d+)s"
    )
    frames = []
    for line in log_path.read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        layer, step, loss, scale, elapsed = m.groups()
        layer = int(layer)
        step = int(step)
        frames.append({
            "layer": layer,
            "step_in_layer": step,
            "loss": float(loss),
            "scale": float(scale),
            "elapsed_s": int(elapsed),
            "global_step": layer * STEPS_PER_LAYER + step,
        })
    frames.sort(key=lambda f: f["global_step"])
    return frames


# --------------------------------------------------------------- left panel


# Schematic axes are in [0, 5.6] x [-0.1, 5.0]. Boxes are wide enough to
# hold a two-line label without clipping at fontsize 8.
BLOCK_X = [0.55, 1.55, 2.55, 3.55, 4.55]   # [tok_emb, B0, B1, B2, B3]
BLOCK_Y_BOT, BLOCK_Y_TOP = 3.0, 4.2
HEAD_Y_BOT, HEAD_Y_TOP = 1.85, 2.45
NLL_Y_BOT, NLL_Y_TOP = 0.55, 1.15
BLOCK_W = 0.92


def _box(ax, cx, cy_bot, cy_top, *, text, fill, edge, text_color, fontsize=8.5,
         fontweight="normal", lw=1.0, zorder=2):
    # Less aggressive corner rounding so the usable interior is wider.
    p = FancyBboxPatch(
        (cx - BLOCK_W / 2, cy_bot),
        BLOCK_W, cy_top - cy_bot,
        boxstyle="round,pad=0.005,rounding_size=0.03",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(p)
    ax.text(cx, (cy_bot + cy_top) / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight=fontweight, zorder=zorder + 1)


def draw_schema(ax, active_layer: int, step_in_layer: int, current_loss: float):
    ax.clear()
    ax.set_xlim(0, 5.6)
    ax.set_ylim(-0.1, 5.0)
    ax.axis("off")

    active_color = PALETTE[active_layer % len(PALETTE)]

    # tok_emb: live during the layer-0 phase only (it's added to the layer-0
    # AdamW group), frozen for every later phase.
    if active_layer == 0:
        _box(ax, BLOCK_X[0], BLOCK_Y_BOT, BLOCK_Y_TOP,
             text="tok_emb\n(AdamW)",
             fill=active_color, edge=BLACK, text_color="white",
             fontweight="bold", lw=1.3)
    else:
        _box(ax, BLOCK_X[0], BLOCK_Y_BOT, BLOCK_Y_TOP,
             text="tok_emb\n(frozen)",
             fill=GREY_FILL, edge=GREY_EDGE, text_color=GREY_TEXT)

    # 4 transformer blocks
    for i in range(N_LAYERS):
        x = BLOCK_X[i + 1]
        if i < active_layer:
            _box(ax, x, BLOCK_Y_BOT, BLOCK_Y_TOP,
                 text=f"Block {i}\nfrozen",
                 fill=GREY_FILL, edge=GREY_EDGE, text_color=GREY_TEXT)
        elif i == active_layer:
            _box(ax, x, BLOCK_Y_BOT, BLOCK_Y_TOP,
                 text=f"Block {i}\nMuon+AdamW",
                 fill=active_color, edge=BLACK, text_color="white",
                 fontweight="bold", fontsize=7.5, lw=1.4, zorder=3)
            # WSD progress bar across the bottom of the active block
            frac = step_in_layer / max(1, STEPS_PER_LAYER - 1)
            bar_w = BLOCK_W - 0.10
            ax.add_patch(Rectangle(
                (x - bar_w / 2, BLOCK_Y_BOT + 0.04),
                bar_w * frac, 0.06,
                facecolor="white", edgecolor="none", alpha=0.7, zorder=4,
            ))
        else:
            _box(ax, x, BLOCK_Y_BOT, BLOCK_Y_TOP,
                 text=f"Block {i}\nuntrained",
                 fill="white", edge=GREY_EDGE, text_color=GREY_TEXT, lw=0.8)

    # Aux heads + NLL boxes (one per layer; warm-started from the previous
    # layer's head at the start of each phase).
    for i in range(N_LAYERS):
        x = BLOCK_X[i + 1]
        if i < active_layer:
            head_fill, head_edge, head_text = GREY_FILL, GREY_EDGE, GREY_TEXT
            head_label = "head\n(frozen)"
            nll_fill, nll_edge, nll_text = GREY_FILL, GREY_EDGE, GREY_TEXT
            nll_label = "NLL\n(not used)"
        elif i == active_layer:
            head_fill, head_edge, head_text = active_color, BLACK, "white"
            head_label = (
                f"aux head {i}\n(warm-start)" if i > 0 else f"aux head {i}"
            )
            nll_fill, nll_edge, nll_text = active_color, BLACK, "white"
            nll_label = f"NLL  {current_loss:.2f}"
        else:
            head_fill, head_edge, head_text = "white", GREY_EDGE, GREY_TEXT
            head_label = f"aux head {i}\n(unused)"
            nll_fill, nll_edge, nll_text = "white", GREY_EDGE, GREY_TEXT
            nll_label = "—"

        _box(ax, x, HEAD_Y_BOT, HEAD_Y_TOP,
             text=head_label,
             fill=head_fill, edge=head_edge, text_color=head_text,
             fontsize=7.5,
             fontweight="bold" if i == active_layer else "normal",
             lw=1.2 if i == active_layer else 0.8)
        _box(ax, x, NLL_Y_BOT, NLL_Y_TOP,
             text=nll_label,
             fill=nll_fill, edge=nll_edge, text_color=nll_text,
             fontsize=8 if i == active_layer else 7.5,
             fontweight="bold" if i == active_layer else "normal",
             lw=1.2 if i == active_layer else 0.8)

    # Forward path between blocks: tok_emb → B0 → B1 → ... → B3.
    # Frozen prefix is grey; the inter-block edge entering the active
    # block is colour-tinted so the live forward path is obvious.
    for i in range(N_LAYERS):
        x_from = BLOCK_X[i]
        x_to = BLOCK_X[i + 1]
        y = (BLOCK_Y_BOT + BLOCK_Y_TOP) / 2
        if i < active_layer:
            color, lw, ls = GREY_EDGE, 1.4, "-"
        elif i == active_layer:
            color, lw, ls = active_color, 1.8, "-"
        else:
            color, lw, ls = GREY_EDGE, 0.8, ":"
        ax.add_patch(FancyArrowPatch(
            (x_from + BLOCK_W / 2, y), (x_to - BLOCK_W / 2, y),
            arrowstyle="->", mutation_scale=12,
            color=color, lw=lw, linestyle=ls, zorder=1,
        ))

    # detach() barrier — between the frozen prefix and the active block
    # (only meaningful when active_layer > 0). Red bar in the inter-block
    # gap; label safely above the block row so it can't overlap any text.
    if active_layer > 0:
        bar_x = (BLOCK_X[active_layer] + BLOCK_X[active_layer + 1]) / 2
        y_mid = (BLOCK_Y_BOT + BLOCK_Y_TOP) / 2
        ax.plot([bar_x, bar_x], [y_mid - 0.32, y_mid + 0.32],
                color=DETACH_RED, lw=2.4, solid_capstyle="butt", zorder=5)
        ax.text(bar_x, BLOCK_Y_TOP + 0.42, ".detach()",
                ha="center", fontsize=7.5, color=DETACH_RED, fontweight="bold")
        ax.add_patch(FancyArrowPatch(
            (bar_x, BLOCK_Y_TOP + 0.30), (bar_x, BLOCK_Y_TOP + 0.05),
            arrowstyle="->", mutation_scale=8,
            color=DETACH_RED, lw=0.8, zorder=5,
        ))

    # Vertical aux-head connections: each block → its aux head → its NLL.
    # The active layer's arrows are DOUBLE-headed: forward (data flowing
    # down) AND backward (gradient flowing up) within the live subgraph.
    # Frozen and untrained layers stay single-headed (forward only).
    for i in range(N_LAYERS):
        x = BLOCK_X[i + 1]
        if i < active_layer:
            color, lw, ls, style = GREY_EDGE, 0.9, "-", "->"
        elif i == active_layer:
            color, lw, ls, style = active_color, 1.7, "-", "<->"
        else:
            color, lw, ls, style = GREY_EDGE, 0.7, ":", "->"
        ax.add_patch(FancyArrowPatch(
            (x, BLOCK_Y_BOT), (x, HEAD_Y_TOP),
            arrowstyle=style, mutation_scale=10,
            color=color, lw=lw, linestyle=ls, zorder=1,
        ))
        ax.add_patch(FancyArrowPatch(
            (x, HEAD_Y_BOT), (x, NLL_Y_TOP),
            arrowstyle=style, mutation_scale=10,
            color=color, lw=lw, linestyle=ls, zorder=1,
        ))

    # Small "∂loss" gradient label on the active column, between block
    # and head — close to the column, no looping arrow needed since the
    # double-headed arrows already encode the backward direction.
    x_a = BLOCK_X[active_layer + 1]
    ax.text(
        x_a + BLOCK_W / 2 + 0.04,
        (BLOCK_Y_BOT + HEAD_Y_TOP) / 2,
        "∂loss",
        ha="left", va="center", fontsize=7,
        color=active_color, fontweight="bold",
    )

    # Header labels above the block row. "forward →" lives ABOVE the
    # row of "layer N" headers so the .detach() callout below has room.
    # Skip the active layer's "layer N" header when the .detach() label
    # lives there, to avoid overlap.
    ax.text(2.85, 4.92, "forward  →",
            ha="center", fontsize=9, color="0.3", fontweight="bold")
    ax.text(BLOCK_X[0], BLOCK_Y_TOP + 0.18, "embed",
            ha="center", fontsize=7.5, color="0.4")
    for i in range(N_LAYERS):
        if active_layer > 0 and i == active_layer:
            continue
        ax.text(BLOCK_X[i + 1], BLOCK_Y_TOP + 0.18, f"layer {i}",
                ha="center", fontsize=7.5, color="0.4")

    # Phase title inside the panel
    ax.text(
        2.85, -0.01,
        f"phase {active_layer + 1} / 4   ·   training block {active_layer}   ·   "
        f"step {step_in_layer:5d} / {STEPS_PER_LAYER}",
        ha="center", va="bottom", fontsize=9,
        color=active_color, fontweight="bold",
    )


# -------------------------------------------------------------- right panel


def draw_loss(ax, frames: list[dict], up_to_idx: int):
    ax.clear()

    ax.set_xlim(-300, TOTAL_STEPS + 300)
    ax.set_ylim(1.05, 1.42)
    ax.set_xlabel("global step", fontsize=9)
    ax.set_ylabel("local NLL  (nats / char)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    for li in range(1, N_LAYERS):
        ax.axvline(li * STEPS_PER_LAYER,
                   color=BLACK, linestyle="--", linewidth=0.6, alpha=0.4)

    # Per-layer drawn-so-far traces
    drawn_first_xy = {}   # layer_idx -> (x, y) of the first drawn point
    for li in range(N_LAYERS):
        xs = [f["global_step"] for f in frames[:up_to_idx + 1] if f["layer"] == li]
        ys = [f["loss"]        for f in frames[:up_to_idx + 1] if f["layer"] == li]
        if not xs:
            continue
        ax.plot(xs, ys,
                color=PALETTE[li % len(PALETTE)],
                linewidth=1.5, marker="o", markersize=2.6)
        drawn_first_xy[li] = (xs[0], ys[0])

    cur = frames[up_to_idx]
    # Highlight current frame's data point
    ax.scatter([cur["global_step"]], [cur["loss"]],
               s=80, color=PALETTE[cur["layer"] % len(PALETTE)],
               edgecolor=BLACK, linewidth=0.9, zorder=5)
    # Playhead vertical line at the current global step
    ax.axvline(cur["global_step"], color=PALETTE[cur["layer"] % len(PALETTE)],
               linewidth=0.8, alpha=0.5, zorder=1)

    # Inline curve labels — one per layer, placed just above the curve's
    # first drawn point. Replaces the legend AND the "layer N" headers.
    for li, (x0, y0) in drawn_first_xy.items():
        ax.text(
            x0 + 350, min(y0 + 0.018, 1.41),
            f"L{li}",
            fontsize=8.5, color=PALETTE[li % len(PALETTE)],
            fontweight="bold", va="bottom",
        )

    # Cold-start callout (5.6-nat point sits off-axis); only shown at idx=0
    if up_to_idx == 0:
        ax.annotate("step 0 NLL ≈ 5.6\n(uniform-byte init)",
                    xy=(0, 1.42), xytext=(2200, 1.40),
                    fontsize=7, color="0.4",
                    arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6))


# -------------------------------------------------------- progress strip


def draw_progress(ax, frames: list[dict], up_to_idx: int):
    ax.clear()
    ax.set_xlim(0, TOTAL_STEPS)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.tick_params(axis="x", labelsize=7)
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("0.6")

    cur = frames[up_to_idx]

    for li in range(N_LAYERS):
        ax.add_patch(Rectangle(
            (li * STEPS_PER_LAYER, 0.15), STEPS_PER_LAYER, 0.7,
            facecolor=PALETTE[li % len(PALETTE)] if li <= cur["layer"] else "white",
            edgecolor=GREY_EDGE, lw=0.6,
            alpha=0.85 if li == cur["layer"] else (0.45 if li < cur["layer"] else 1.0),
        ))
        ax.text(li * STEPS_PER_LAYER + STEPS_PER_LAYER * 0.5, 0.5,
                f"layer {li}",
                ha="center", va="center",
                fontsize=8, color="white" if li <= cur["layer"] else GREY_TEXT,
                fontweight="bold" if li == cur["layer"] else "normal")

    # playhead
    ax.plot([cur["global_step"]] * 2, [0.05, 0.95],
            color=BLACK, lw=1.3, zorder=5)
    ax.scatter([cur["global_step"]], [0.5],
               s=70, color="white",
               edgecolor=BLACK, linewidth=1.0, zorder=6)

    ax.set_xlabel(
        f"global step {cur['global_step']:,} / {TOTAL_STEPS:,}   ·   "
        f"elapsed {cur['elapsed_s']} s   ·   LR scale {cur['scale']:.2f}",
        fontsize=8.5,
    )


# ------------------------------------------------------------- main / build


def build_animation():
    VIZ_DIR.mkdir(exist_ok=True)
    frames = parse_run_log(RECORD / "run.log")
    print(f"loaded {len(frames)} log frames "
          f"(layer 0..{N_LAYERS-1}, "
          f"global_step {frames[0]['global_step']}..{frames[-1]['global_step']})")

    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[5.0, 0.7],
        width_ratios=[1.1, 1.4],
        hspace=0.32, wspace=0.18,
        left=0.05, right=0.985, top=0.86, bottom=0.10,
    )
    ax_schema = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_prog = fig.add_subplot(gs[1, :])

    fig.suptitle(
        "DGL v11 — greedy layerwise training of a 4×d=512 char-LM, no end-to-end backprop\n"
        "left: which block is live this phase  ·  right: per-layer aux-head NLL drawn as training proceeds  ·  bottom: global progress",
        fontsize=10, y=0.965,
    )

    def update(idx: int):
        cur = frames[idx]
        draw_schema(ax_schema, cur["layer"], cur["step_in_layer"], cur["loss"])
        draw_loss(ax_loss, frames, idx)
        draw_progress(ax_prog, frames, idx)
        return ()

    anim = animation.FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=180,   # ms / frame
        blit=False,
    )

    out_gif = VIZ_DIR / "dgl_v11_animation.gif"
    print(f"writing {out_gif} ({len(frames)} frames @ 6 fps) ...")
    anim.save(out_gif, writer=animation.PillowWriter(fps=6), dpi=90)
    print(f"wrote {out_gif}")

    # Final-frame PNG (for embedding in markdown)
    update(len(frames) - 1)
    out_png = VIZ_DIR / "dgl_v11_animation_final.png"
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    build_animation()
