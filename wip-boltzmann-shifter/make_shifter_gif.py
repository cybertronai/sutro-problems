"""
Generate an animated GIF illustrating the shifter task.

For each frame, draws two rings (V1 above, V2 below) of N binary cells.
V2 is V1 shifted (with wraparound) by -1, 0, or +1. The animation cycles
through the three shift classes for several example patterns, with motion
arrows and a label showing which class the frame represents.

Output: shifter.gif in the same directory.
Dependencies: numpy, matplotlib, pillow.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


# ----------------------------------------------------------------------
# Bit-pattern helpers
# ----------------------------------------------------------------------

def bits(n: int, N: int) -> np.ndarray:
    """Convert an integer to an N-bit array (LSB first)."""
    return np.array([(n >> i) & 1 for i in range(N)], dtype=int)


SHIFTS = [-1, 0, 1]
LABELS = {-1: "shift left  ( −1 )", 0: "no shift  ( 0 )", 1: "shift right  ( +1 )"}
COLORS = {-1: "#a83232", 0: "#404040", 1: "#3a78a8"}


# ----------------------------------------------------------------------
# Frame plan
# ----------------------------------------------------------------------

def build_frame_plan(patterns: list[int], N: int,
                      hold_frames: int = 8,
                      slide_frames: int = 3,
                      fadeout_frames: int = 2) -> list[dict]:
    """Build a flat list of frames. Each frame is a dict describing what to draw.

    For every (pattern, shift) pair we emit:
      - `hold_frames` static frames showing V1 only (V2 absent)
      - `slide_frames` interpolating frames showing V2 fading in
      - `hold_frames` static frames showing the resulting V1 + shifted V2
      - `fadeout_frames` interpolating frames fading V2 back out to 0
    """
    frames = []
    for p in patterns:
        v1 = bits(p, N)
        for s in SHIFTS:
            v2 = np.roll(v1, s)
            # Phase A: V1 alone
            for _ in range(hold_frames):
                frames.append({"v1": v1, "v2": None, "shift": s, "phase": "v1_only", "t": 0.0})
            # Phase B: V2 fading in (shows arrows + animated alpha)
            for k in range(1, slide_frames + 1):
                t = k / slide_frames
                frames.append({"v1": v1, "v2": v2, "shift": s, "phase": "slide", "t": t})
            # Phase C: hold final
            for _ in range(hold_frames):
                frames.append({"v1": v1, "v2": v2, "shift": s, "phase": "hold", "t": 1.0})
            # Phase D: fast fade-out before transitioning to next scene
            for k in range(1, fadeout_frames + 1):
                t = 1.0 - k / fadeout_frames
                frames.append({"v1": v1, "v2": v2, "shift": s, "phase": "slide", "t": t})
    return frames


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------

def cell_color(bit: int, alpha: float = 1.0) -> tuple[float, float, float, float]:
    if bit == 1:
        return (0.13, 0.13, 0.13, alpha)
    return (0.97, 0.97, 0.97, alpha)


def draw_row(ax, y: float, vec: np.ndarray | None, *, alpha: float = 1.0,
             label: str = "") -> None:
    """Draw a row of N bit cells centered horizontally on `ax`."""
    if vec is None:
        return
    N = len(vec)
    cell_w = 1.0
    for i, b in enumerate(vec):
        rect = mpatches.FancyBboxPatch(
            (i + 0.05, y + 0.05), cell_w - 0.1, cell_w - 0.1,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=(0.2, 0.2, 0.2, alpha),
            facecolor=cell_color(int(b), alpha=alpha),
        )
        ax.add_patch(rect)
        ax.text(i + 0.5, y + 0.5, str(int(b)),
                ha="center", va="center",
                color=(0.95 if b == 1 else 0.2,) * 3 + (alpha,),
                fontsize=14, fontfamily="monospace", fontweight="bold")
    if label:
        ax.text(-0.5, y + 0.5, label, ha="right", va="center",
                fontsize=13, fontfamily="monospace",
                color=(0.2, 0.2, 0.2, alpha))


def draw_arrows(ax, y_top: float, y_bot: float, N: int, shift: int,
                alpha: float) -> None:
    """Draw arrows from V1 cells to their corresponding V2 cells under `shift`."""
    if shift == 0:
        return
    color = COLORS[shift]
    for i in range(N):
        j = (i + shift) % N
        # straight arrow if no wraparound, curved if wrapping
        wraps = abs(j - i) > 1
        if wraps:
            # draw a small "wraps around" mark instead of a long arrow
            ax.annotate("",
                        xy=(j + 0.5, y_bot + 0.85),
                        xytext=(i + 0.5, y_top + 0.15),
                        arrowprops=dict(
                            arrowstyle="->", lw=1.2,
                            color=(*[c / 255 for c in [int(color[1:3], 16),
                                                         int(color[3:5], 16),
                                                         int(color[5:7], 16)]], alpha * 0.5),
                            connectionstyle="arc3,rad=0.4",
                        ))
        else:
            ax.annotate("",
                        xy=(j + 0.5, y_bot + 0.85),
                        xytext=(i + 0.5, y_top + 0.15),
                        arrowprops=dict(
                            arrowstyle="->", lw=1.4,
                            color=(*[c / 255 for c in [int(color[1:3], 16),
                                                         int(color[3:5], 16),
                                                         int(color[5:7], 16)]], alpha),
                        ))


def render_frame(ax, frame: dict, N: int) -> None:
    ax.clear()
    ax.set_xlim(-3.5, N + 0.5)
    ax.set_ylim(-1.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    y_top, y_bot = 2.5, 0.5
    s = frame["shift"]
    t = frame["t"]
    phase = frame["phase"]
    v1 = frame["v1"]
    v2 = frame["v2"]

    # Title strip (above the cells)
    ax.text(N / 2, 4.6, "Boltzmann shifter task",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=(0.15, 0.15, 0.15))

    # Class label, color-coded (below the cells)
    ax.text(N / 2, -0.9, LABELS[s], ha="center", va="center",
            fontsize=15, fontfamily="monospace", fontweight="bold",
            color=COLORS[s])

    # V1 always full opacity
    draw_row(ax, y_top, v1, alpha=1.0, label="V1")

    # V2 fades in during slide phase
    if phase == "v1_only":
        return
    alpha = t if phase == "slide" else 1.0
    draw_row(ax, y_bot, v2, alpha=alpha, label="V2")
    draw_arrows(ax, y_top, y_bot, N, s, alpha=alpha)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def make_gif(out_path: Path, N: int = 8, fps: int = 12) -> None:
    rng = np.random.default_rng(7)
    # pick a few visually distinct patterns:
    patterns = [
        0b00010110,   # 22
        0b01101001,   # 105
        0b11000110,   # 198
        0b00111100,   # 60
    ]
    if N != 8:
        patterns = [int(rng.integers(1, 2**N - 1)) for _ in range(4)]

    frame_plan = build_frame_plan(patterns, N,
                                   hold_frames=6,
                                   slide_frames=3,
                                   fadeout_frames=0)

    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=110)
    fig.patch.set_facecolor("white")

    def update(i):
        render_frame(ax, frame_plan[i], N)
        return []

    anim = FuncAnimation(fig, update, frames=len(frame_plan),
                         interval=1000 / fps, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    print(f"wrote {out_path}  ({len(frame_plan)} frames @ {fps} fps)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--out", default="shifter.gif")
    args = p.parse_args()
    make_gif(Path(args.out), N=args.N, fps=args.fps)
