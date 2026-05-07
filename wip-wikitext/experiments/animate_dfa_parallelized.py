"""GIF animation of DFA Parallelized's training procedure.

Same visual language as animate_dgl_v11.py, but for the DFA-Parallelized
recipe (submission_dfa.py): all 6 transformer blocks train concurrently
every step, with `.detach()` between every pair so each block has an
independent autograd graph rooted at its detached input. Per-block
synthetic gradients are produced by projecting the output error
`e = (softmax(logits) - one_hot(y)) / (B*T)` through a frozen
sign-projection feedback matrix `B_L` per block. tok_emb gets the same
DFA signal via `B_emb` (= `B_0`); head + ln_f get *real* gradients via
`loss.backward()` (they sit below the last detach cut).

Two panels driven by submission_dfa_4ksteps_2026-05-06.log:

  * left  — network schematic (tok_emb → 6 blocks → ln_f+head). All
            blocks are colour-highlighted simultaneously (every block
            is live every step). Red `.detach()` bars sit between
            every pair, including tok_emb→B0 and B5→ln_f. Below each
            trainable block + tok_emb is a small `B_L (sign)` feedback
            matrix box; arrows from a shared "error e" bus on the
            right project up through each B_L into its block as a
            synthetic gradient. The head box receives a real
            backprop arrow from the loss.
  * right — single global training NLL curve over 4000 steps, drawn
            progressively. Curve segments are coloured by WSD phase
            (warmup / stable / decay).

A bottom strip shows global progress / 4000 steps and the current
WSD phase, elapsed seconds, and LR scale.

Saved at wip-wikitext/experiments/viz_dfa_parallelized/
dfa_parallelized_animation.gif (PillowWriter, no ffmpeg dep). A
static final-frame PNG is also written.

    python3 wip-wikitext/experiments/animate_dfa_parallelized.py
    python3 wip-wikitext/experiments/animate_dfa_parallelized.py --hires
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
WIKI = HERE.parent
VIZ_DIR = HERE / "viz_dfa_parallelized"
LOG = WIKI / "submissions" / "submission_dfa_4ksteps_2026-05-06.log"

# tab10 palette extended to 6 + 1 (one per block + tok_emb).
PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
           "#8c564b", "#e377c2", "#17becf"]
EMB_COLOR = "#7f7f7f"
HEAD_COLOR = "#404040"

GREY_FILL = "#e6e6e6"
GREY_EDGE = "#b0b0b0"
GREY_TEXT = "#7a7a7a"
DETACH_RED = "#d62728"
REAL_GRAD = "#0a8f4a"
BLACK = "black"

N_LAYERS = 6
TOTAL_STEPS = 4000
WARMUP_STEPS = 200
DECAY_START = 3200       # 4000 - int(0.2*4000)


# --------------------------------------------------------------------- parse


def parse_run_log(log_path: Path) -> list[dict]:
    """Flatten the run.log into a chronological list of frame dicts."""
    pat = re.compile(
        r"step\s+(\d+)\s+loss\s+([\d.]+)\s+scale\s+([\d.]+)\s+elapsed\s+(\d+)s"
    )
    frames = []
    for line in log_path.read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        step, loss, scale, elapsed = m.groups()
        frames.append({
            "step": int(step),
            "loss": float(loss),
            "scale": float(scale),
            "elapsed_s": int(elapsed),
        })
    frames.sort(key=lambda f: f["step"])
    return frames


def wsd_phase(step: int) -> str:
    if step < WARMUP_STEPS:
        return "warmup"
    if step < DECAY_START:
        return "stable"
    return "decay"


PHASE_COLOR = {
    "warmup": "#ff7f0e",
    "stable": "#1f77b4",
    "decay":  "#9467bd",
}


# --------------------------------------------------------------- left panel
#
# Schematic axes are in [0, 9.6] x [-0.4, 5.0]. 8-column forward row
# (tok_emb + 6 blocks + head_ln_f). Below the trainable boxes is a row
# of small B_L feedback-matrix boxes connected to a shared error bus
# that runs back from the head's loss.

BOX_W = 0.92
BOX_GAP_X = 1.10                     # centre-to-centre spacing
N_FWD = 1 + N_LAYERS + 1             # tok_emb + 6 blocks + head
LEFT_PAD = 0.55
BLOCK_X = [LEFT_PAD + i * BOX_GAP_X for i in range(N_FWD)]   # 8 centres

BLOCK_Y_BOT, BLOCK_Y_TOP = 3.6, 4.55
BL_Y_BOT, BL_Y_TOP = 2.20, 2.85
BUS_Y = 1.45


def _box(ax, cx, cy_bot, cy_top, *, text, fill, edge, text_color, fontsize=8.0,
         fontweight="normal", lw=1.0, zorder=2, w=BOX_W):
    p = FancyBboxPatch(
        (cx - w / 2, cy_bot),
        w, cy_top - cy_bot,
        boxstyle="round,pad=0.005,rounding_size=0.03",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(p)
    ax.text(cx, (cy_bot + cy_top) / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight=fontweight, zorder=zorder + 1)


def draw_schema(ax, step: int, scale: float, current_loss: float):
    ax.clear()
    ax.set_xlim(0, BLOCK_X[-1] + LEFT_PAD + 0.05)
    ax.set_ylim(-0.4, 5.05)
    ax.axis("off")

    # ---- forward row: tok_emb, B0..B5, head ---------------------------------
    # tok_emb (column 0)
    _box(ax, BLOCK_X[0], BLOCK_Y_BOT, BLOCK_Y_TOP,
         text="tok_emb\n(AdamW)",
         fill=EMB_COLOR, edge=BLACK, text_color="white",
         fontweight="bold", fontsize=7.5, lw=1.2)

    # 6 transformer blocks (columns 1..6) — every block is LIVE every step
    for i in range(N_LAYERS):
        x = BLOCK_X[i + 1]
        color = PALETTE[i]
        _box(ax, x, BLOCK_Y_BOT, BLOCK_Y_TOP,
             text=f"Block {i}\nMuon+AdamW",
             fill=color, edge=BLACK, text_color="white",
             fontweight="bold", fontsize=7.0, lw=1.3, zorder=3)

    # ln_f + head + softcap (column 7) — receives REAL gradient via loss.backward()
    _box(ax, BLOCK_X[-1], BLOCK_Y_BOT, BLOCK_Y_TOP,
         text="ln_f + head\n(real grad)",
         fill=HEAD_COLOR, edge=BLACK, text_color="white",
         fontweight="bold", fontsize=7.2, lw=1.3)

    # WSD progress bar across every active box (all blocks + tok_emb + head
    # all train every step, so the WSD bar is global, not per-block).
    frac = step / max(1, TOTAL_STEPS - 1)
    for i in range(N_FWD):
        x = BLOCK_X[i]
        bar_w = BOX_W - 0.10
        ax.add_patch(Rectangle(
            (x - bar_w / 2, BLOCK_Y_BOT + 0.04),
            bar_w * frac, 0.05,
            facecolor="white", edgecolor="none", alpha=0.6, zorder=4,
        ))

    # Forward arrows between successive boxes (live every step).
    fwd_y = (BLOCK_Y_BOT + BLOCK_Y_TOP) / 2
    for i in range(N_FWD - 1):
        x_from = BLOCK_X[i]
        x_to = BLOCK_X[i + 1]
        # Edge tint: tok_emb→B0 and final B5→head are coloured neutrally;
        # inter-block edges take the downstream block's colour.
        if i == 0:
            color = EMB_COLOR
        elif i + 1 == N_FWD - 1:
            color = HEAD_COLOR
        else:
            color = PALETTE[i]
        ax.add_patch(FancyArrowPatch(
            (x_from + BOX_W / 2, fwd_y), (x_to - BOX_W / 2, fwd_y),
            arrowstyle="->", mutation_scale=12,
            color=color, lw=1.6, linestyle="-", zorder=1,
        ))

    # `.detach()` bars between EVERY successive pair (every block input is
    # detached, plus head input is detached). This is the structural
    # difference vs. DGL v11 (which has only one detach cut per phase).
    for i in range(N_FWD - 1):
        bar_x = (BLOCK_X[i] + BLOCK_X[i + 1]) / 2
        ax.plot([bar_x, bar_x], [fwd_y - 0.30, fwd_y + 0.30],
                color=DETACH_RED, lw=2.2, solid_capstyle="butt", zorder=5)
    # Single label, centred above the row, instead of one label per bar
    # (with 7 detach bars individual labels would crowd the figure).
    ax.text(
        (BLOCK_X[0] + BLOCK_X[-1]) / 2,
        BLOCK_Y_TOP + 0.42,
        ".detach()  between every pair  (each block has its own autograd graph)",
        ha="center", fontsize=7.8, color=DETACH_RED, fontweight="bold",
    )

    # Header labels above the block row
    ax.text(BLOCK_X[0], BLOCK_Y_TOP + 0.18, "embed",
            ha="center", fontsize=7.5, color="0.4")
    for i in range(N_LAYERS):
        ax.text(BLOCK_X[i + 1], BLOCK_Y_TOP + 0.18, f"layer {i}",
                ha="center", fontsize=7.5, color="0.4")
    ax.text(BLOCK_X[-1], BLOCK_Y_TOP + 0.18, "head",
            ha="center", fontsize=7.5, color="0.4")
    ax.text(0.05, 4.92, "forward  →",
            ha="left", fontsize=9, color="0.3", fontweight="bold")

    # ---- B_L feedback row ---------------------------------------------------
    # One small feedback-matrix box under tok_emb and each block (NOT under
    # head, since head receives a real gradient instead).
    BL_W = 0.78
    for i in range(N_LAYERS + 1):           # tok_emb + 6 blocks
        x = BLOCK_X[i]
        if i == 0:
            label = "B_emb\n= B_0"
            color = EMB_COLOR
        else:
            label = f"B_{i-1}  (sign)"
            color = PALETTE[i - 1]
        _box(ax, x, BL_Y_BOT, BL_Y_TOP,
             text=label,
             fill=color, edge=BLACK, text_color="white",
             fontsize=6.8, fontweight="bold", lw=1.0, w=BL_W)

        # Vertical synthetic-gradient arrow: B_L → block (upward).
        ax.add_patch(FancyArrowPatch(
            (x, BL_Y_TOP), (x, BLOCK_Y_BOT - 0.01),
            arrowstyle="->", mutation_scale=10,
            color=color, lw=1.5, linestyle="-", zorder=1,
        ))

    # "synthetic ∂loss" label, placed under the bus near the leftmost
    # column where it has horizontal room (panel left edge is at x=0).
    ax.text(
        BLOCK_X[0], BUS_Y - 0.30,
        "synthetic ∂loss  =  e · B_Lᵀ",
        ha="center", va="top", fontsize=7.5,
        color="0.25", fontweight="bold",
    )

    # ---- error bus + loss element ------------------------------------------
    # Horizontal "error bus" connecting all B_L bottoms to a shared `e` node.
    bus_x_lo = BLOCK_X[0]
    bus_x_hi = BLOCK_X[N_LAYERS] + 0.55
    ax.plot([bus_x_lo, bus_x_hi], [BUS_Y, BUS_Y],
            color="0.35", lw=1.0, linestyle="-", zorder=1)
    # Drops from each B_L matrix down to the bus.
    for i in range(N_LAYERS + 1):
        x = BLOCK_X[i]
        ax.plot([x, x], [BL_Y_BOT, BUS_Y],
                color="0.35", lw=0.9, linestyle="-", zorder=1)

    # Error node `e` at the right end of the bus
    e_x = bus_x_hi + 0.20
    e_y = BUS_Y
    ax.add_patch(FancyBboxPatch(
        (e_x - 0.40, e_y - 0.22), 0.80, 0.44,
        boxstyle="round,pad=0.005,rounding_size=0.05",
        facecolor="white", edgecolor=BLACK, linewidth=1.2, zorder=2,
    ))
    ax.text(e_x, e_y, "e =  softmax(z)−y\n         B·T",
            ha="center", va="center",
            fontsize=6.8, color=BLACK, fontweight="bold")
    # bus → e
    ax.add_patch(FancyArrowPatch(
        (bus_x_hi, BUS_Y), (e_x - 0.40, e_y),
        arrowstyle="<-", mutation_scale=10,
        color="0.35", lw=1.0, zorder=1,
    ))

    # head ↓ to error node (real grad path: head feeds into loss, which
    # produces e). Drawn as a thicker green arrow downward from the head
    # box to the e box.
    head_x = BLOCK_X[-1]
    ax.add_patch(FancyArrowPatch(
        (head_x, BLOCK_Y_BOT - 0.01), (e_x, e_y + 0.22),
        arrowstyle="->", mutation_scale=11,
        color=REAL_GRAD, lw=1.5, zorder=2,
    ))
    ax.text(
        (head_x + e_x) / 2 + 0.05, (BLOCK_Y_BOT + e_y + 0.22) / 2,
        "loss.backward()\n→ real ∂loss",
        ha="left", va="center", fontsize=6.8,
        color=REAL_GRAD, fontweight="bold",
    )

    # Phase title inside the panel
    phase = wsd_phase(step)
    ax.text(
        (BLOCK_X[0] + BLOCK_X[-1]) / 2, -0.25,
        f"WSD phase: {phase}   ·   step {step:5d} / {TOTAL_STEPS}   ·   "
        f"loss {current_loss:.3f}   ·   LR scale {scale:.2f}",
        ha="center", va="bottom", fontsize=9,
        color=PHASE_COLOR[phase], fontweight="bold",
    )


# -------------------------------------------------------------- right panel


def draw_loss(ax, frames: list[dict], up_to_idx: int):
    ax.clear()

    ax.set_xlim(-100, TOTAL_STEPS + 100)
    ax.set_ylim(1.55, 5.85)
    ax.set_xlabel("global step", fontsize=9)
    ax.set_ylabel("training NLL  (nats / char)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    # WSD phase shading
    ax.axvspan(0, WARMUP_STEPS,
               color=PHASE_COLOR["warmup"], alpha=0.08, zorder=0)
    ax.axvspan(WARMUP_STEPS, DECAY_START,
               color=PHASE_COLOR["stable"], alpha=0.05, zorder=0)
    ax.axvspan(DECAY_START, TOTAL_STEPS,
               color=PHASE_COLOR["decay"], alpha=0.10, zorder=0)
    ax.axvline(WARMUP_STEPS, color=BLACK, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.axvline(DECAY_START,  color=BLACK, linestyle="--", linewidth=0.5, alpha=0.4)

    # Phase region labels (top of panel)
    for label, x in [("warmup", WARMUP_STEPS / 2),
                     ("stable", (WARMUP_STEPS + DECAY_START) / 2),
                     ("decay",  (DECAY_START + TOTAL_STEPS) / 2)]:
        ax.text(x, 5.72, label,
                ha="center", fontsize=7.5,
                color=PHASE_COLOR[label], fontweight="bold")

    # Per-phase curve segments drawn-so-far
    drawn = frames[:up_to_idx + 1]
    for phase, color in PHASE_COLOR.items():
        xs = [f["step"] for f in drawn if wsd_phase(f["step"]) == phase]
        ys = [f["loss"] for f in drawn if wsd_phase(f["step"]) == phase]
        if not xs:
            continue
        ax.plot(xs, ys,
                color=color,
                linewidth=1.6, marker="o", markersize=3.0)

    # Bridge segments across phase boundaries so the curve looks continuous.
    # Draw thin grey connectors between the last point of phase k and the
    # first point of phase k+1 if both are present in `drawn`.
    for k in range(len(drawn) - 1):
        a, b = drawn[k], drawn[k + 1]
        if wsd_phase(a["step"]) != wsd_phase(b["step"]):
            ax.plot([a["step"], b["step"]], [a["loss"], b["loss"]],
                    color="0.55", linewidth=1.0, linestyle="-", zorder=2)

    # Highlight current frame's data point
    cur = frames[up_to_idx]
    cur_color = PHASE_COLOR[wsd_phase(cur["step"])]
    ax.scatter([cur["step"]], [cur["loss"]],
               s=85, color=cur_color,
               edgecolor=BLACK, linewidth=0.9, zorder=5)
    ax.axvline(cur["step"], color=cur_color,
               linewidth=0.8, alpha=0.5, zorder=1)

    # Cold-start callout, only at the first frame
    if up_to_idx == 0:
        ax.annotate("step 0 NLL ≈ 5.6\n(uniform-byte init)",
                    xy=(0, 5.6), xytext=(800, 5.0),
                    fontsize=7, color="0.4",
                    arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6))

    # Plateau callout (only after we cross step 1500 or so)
    if cur["step"] >= 2000:
        ax.annotate("plateau ≈ 1.78\n(feedback-alignment\nlimit on this transformer)",
                    xy=(2200, 1.77), xytext=(1100, 2.6),
                    fontsize=7, color="0.4",
                    arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6))


# -------------------------------------------------------- progress strip


def draw_progress(ax, frames: list[dict], up_to_idx: int):
    ax.clear()
    ax.set_xlim(0, TOTAL_STEPS)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.tick_params(axis="x", labelsize=7)
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("0.6")

    cur = frames[up_to_idx]

    # WSD phase regions
    regions = [
        ("warmup", 0, WARMUP_STEPS),
        ("stable", WARMUP_STEPS, DECAY_START),
        ("decay",  DECAY_START, TOTAL_STEPS),
    ]
    cur_phase = wsd_phase(cur["step"])
    for name, lo, hi in regions:
        active = (lo <= cur["step"] < hi) or (name == "decay" and cur["step"] >= DECAY_START)
        completed = cur["step"] >= hi
        ax.add_patch(Rectangle(
            (lo, 0.15), hi - lo, 0.7,
            facecolor=PHASE_COLOR[name] if (active or completed) else "white",
            edgecolor=GREY_EDGE, lw=0.6,
            alpha=0.85 if active else (0.45 if completed else 1.0),
        ))
        ax.text((lo + hi) / 2, 0.5,
                name,
                ha="center", va="center",
                fontsize=8,
                color="white" if (active or completed) else GREY_TEXT,
                fontweight="bold" if active else "normal")

    # playhead
    ax.plot([cur["step"]] * 2, [0.05, 0.95],
            color=BLACK, lw=1.3, zorder=5)
    ax.scatter([cur["step"]], [0.5],
               s=70, color="white",
               edgecolor=BLACK, linewidth=1.0, zorder=6)

    ax.set_xlabel(
        f"global step {cur['step']:,} / {TOTAL_STEPS:,}   ·   "
        f"elapsed {cur['elapsed_s']} s   ·   LR scale {cur['scale']:.2f}   ·   "
        f"phase: {cur_phase}",
        fontsize=8.5,
    )


# ------------------------------------------------------------- main / build


def build_animation(suffix: str = "", dpi_gif: int = 90, dpi_png: int = 120):
    VIZ_DIR.mkdir(exist_ok=True)
    frames = parse_run_log(LOG)
    print(f"loaded {len(frames)} log frames "
          f"(step {frames[0]['step']}..{frames[-1]['step']})")

    fig = plt.figure(figsize=(15.5, 6.0))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[5.0, 0.7],
        width_ratios=[1.55, 1.20],
        hspace=0.30, wspace=0.16,
        left=0.04, right=0.985, top=0.86, bottom=0.10,
    )
    ax_schema = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_prog = fig.add_subplot(gs[1, :])

    fig.suptitle(
        "DFA Parallelized — all 6 blocks train concurrently with .detach() between every pair, no end-to-end backprop\n"
        "left: synthetic-grad architecture (B_L sign-projection per block)  ·  right: global training NLL  ·  bottom: WSD progress",
        fontsize=10, y=0.965,
    )

    def update(idx: int):
        cur = frames[idx]
        draw_schema(ax_schema, cur["step"], cur["scale"], cur["loss"])
        draw_loss(ax_loss, frames, idx)
        draw_progress(ax_prog, frames, idx)
        return ()

    anim = animation.FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=240,   # ms / frame — slightly slower than v11 since fewer frames
        blit=False,
    )

    out_gif = VIZ_DIR / f"dfa_parallelized_animation{suffix}.gif"
    print(f"writing {out_gif} ({len(frames)} frames @ 4 fps, dpi={dpi_gif}) ...")
    anim.save(out_gif, writer=animation.PillowWriter(fps=4), dpi=dpi_gif)
    print(f"wrote {out_gif}")

    update(len(frames) - 1)
    out_png = VIZ_DIR / f"dfa_parallelized_animation_final{suffix}.png"
    fig.savefig(out_png, dpi=dpi_png)
    print(f"wrote {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if "--hires" in sys.argv:
        build_animation(suffix="_hires", dpi_gif=200, dpi_png=240)
    else:
        build_animation()
