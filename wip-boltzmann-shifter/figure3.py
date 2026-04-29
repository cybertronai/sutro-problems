"""
Reproduce the layout/style of Figure 3 from Hinton & Sejnowski (PDP v1 ch 7)
using my own trained shifter RBM.

Each hidden unit is rendered in a "Hinton diagram" style:
    Top row:   [threshold | ...gap... | y_L  y_N  y_R]
    Bottom:    [V1 row of 8 weights]
               [V2 row of 8 weights]

White square = positive weight
Black square = negative weight
Square size ∝ |weight|
Background = mid-gray

24 hidden units, 6x4 grid layout.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import shifter_rbm as sh


def train_24(N=8, epochs=300, seed=0):
    return sh.train(N=N, n_hidden=24, n_epochs=epochs,
                    lr=0.03, momentum=0.7, batch_size=32,
                    seed=seed, verbose=True)


def hinton_unit_panel(ax, w_v1, w_v2, w_y, w_thresh,
                       vmax, cell=1.0, pad=0.2):
    """
    Draw one unit's weights in the PDP Fig-3 style on axis `ax`.
    Layout:
        row 0 (top):      [thresh]           [y_L y_N y_R]
        row 1:            — blank —
        row 2 (V1 row):   [v1[0]...v1[7]]
        row 3 (V2 row):   [v2[0]...v2[7]]
    Weights are drawn as centered squares with side proportional to |w|/vmax,
    white if positive, black if negative.
    """
    N = len(w_v1)
    ncols = max(N, 4 + 3)   # at least enough for thresh + gap + 3 Y cells
    nrows = 4                # top row, gap, V1, V2

    # background
    ax.add_patch(Rectangle((0, 0), ncols, nrows, facecolor="#c8c8c8",
                           edgecolor="black", lw=0.7, zorder=0))

    def draw_cell(col, row, w):
        if np.isnan(w) or w == 0:
            return
        s = min(0.95, abs(w) / vmax) * cell
        cx = col + 0.5
        cy = (nrows - 1 - row) + 0.5
        color = "white" if w > 0 else "black"
        ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s,
                               facecolor=color, edgecolor="none", zorder=2))

    # top row: threshold at col 0
    draw_cell(0, 0, w_thresh)
    # three output weights in the middle of top row
    # place them centered-right: columns ncols-3, ncols-2, ncols-1
    y_start = ncols - 3
    for k, w in enumerate(w_y):
        draw_cell(y_start + k, 0, w)

    # V1 row at row 2 (bottom-1)
    for i, w in enumerate(w_v1):
        draw_cell(i, 2, w)
    # V2 row at row 3 (bottom)
    for i, w in enumerate(w_v2):
        draw_cell(i, 3, w)

    # thin dividers between top row and V1 row
    ax.plot([0, ncols], [nrows - 1, nrows - 1], color="black", lw=0.3, zorder=1)
    ax.plot([0, ncols], [2, 2], color="black", lw=0.3, zorder=1)

    ax.set_xlim(0, ncols); ax.set_ylim(0, nrows)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_figure3(rbm, N, grid=(4, 6), out_path="figure3.png"):
    """Render all hidden units in a grid matching Fig 3 layout (6 cols × 4 rows)."""
    assert rbm.nh == grid[0] * grid[1], f"rbm.nh={rbm.nh} ≠ {grid[0]*grid[1]}"

    W = rbm.W    # (nv, nh)
    bh = rbm.bh  # (nh,)
    nv_y = 3

    # per-unit weight vectors
    W_v1 = W[:N, :]
    W_v2 = W[N:2 * N, :]
    W_y  = W[2 * N:2 * N + nv_y, :]

    # sort by preferred shift, then by L/N/R subgroup magnitude for readability
    pref = np.argmax(W_y, axis=0)
    strength = W_y.max(axis=0) - W_y.min(axis=0)
    order = sorted(range(rbm.nh), key=lambda j: (pref[j], -strength[j]))

    # global scale — use 95th percentile of |W| so small weights are visible
    vmax = float(np.percentile(np.abs(np.concatenate([W.ravel(), bh.ravel()])),
                               97))
    rows, cols = grid
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 1.7, rows * 1.3))
    fig.suptitle("Shifter hidden-unit weights (24 units, Hinton-diagram style)\n"
                 "top-left: threshold   ·   top-right trio: [L, N, R] output weights\n"
                 "bottom two rows: V1 and V2 receptive field   ·   "
                 "white = +, black = −, size ∝ |w|",
                 fontsize=9)

    for k, j in enumerate(order):
        ax = axes.flat[k]
        hinton_unit_panel(ax,
                           w_v1=W_v1[:, j],
                           w_v2=W_v2[:, j],
                           w_y=W_y[:, j],
                           w_thresh=bh[j],
                           vmax=vmax)
        pref_labels = {0: "L", 1: "N", 2: "R"}
        ax.set_title(f"unit {j} → {pref_labels[pref[j]]}",
                     fontsize=7, pad=2)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02,
                        wspace=0.15, hspace=0.3)
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")

    # print a legible summary of the "easiest to interpret" unit
    # (analogous to the chapter's "top-left" unit analysis)
    # Find the unit with the clearest single-pair V1↔V2 response
    best = None
    for j in range(rbm.nh):
        v1w = W_v1[:, j]; v2w = W_v2[:, j]
        # peak pair: max_{i1,i2} |v1w[i1]| * |v2w[i2]| with same sign
        score_pos = np.outer(np.maximum(v1w, 0), np.maximum(v2w, 0))
        score_neg = np.outer(-np.minimum(v1w, 0), -np.minimum(v2w, 0))
        score = np.maximum(score_pos, score_neg)
        idx = np.unravel_index(np.argmax(score), score.shape)
        if best is None or score[idx] > best[2]:
            best = (j, idx, score[idx])
    j, (i1, i2), _ = best
    shift_offset = (i2 - i1) % N
    if shift_offset == N - 1: shift_label = "left (-1)"
    elif shift_offset == 1:   shift_label = "right (+1)"
    elif shift_offset == 0:   shift_label = "none (0)"
    else:                     shift_label = f"offset +{shift_offset}"
    print(f"\nMost interpretable unit: #{j}")
    print(f"  Strongest V1↔V2 pair: V1[{i1}] ↔ V2[{i2}]  (offset = {shift_offset}, consistent with {shift_label})")
    print(f"  Output preference:     L={W_y[0,j]:+.2f}  N={W_y[1,j]:+.2f}  R={W_y[2,j]:+.2f}")
    print(f"  Threshold:             {bh[j]:+.2f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="viz")
    args = ap.parse_args()

    print(f"# Training 24-hidden-unit shifter (N={args.N})...")
    rbm = train_24(N=args.N, epochs=args.epochs, seed=args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    # report final accuracy so we know the network actually learned
    V1, V2, Y = sh.make_shifter_data(args.N)
    mask = np.concatenate([np.ones(2 * args.N), np.zeros(3)]).astype(np.float32)
    acc = sh.evaluate(rbm, V1, V2, Y, args.N, mask, n_gibbs=150)
    print(f"Final accuracy: {acc*100:.1f}%")

    render_figure3(rbm, args.N, grid=(4, 6),
                   out_path=os.path.join(args.outdir, "figure3_reproduction.png"))
