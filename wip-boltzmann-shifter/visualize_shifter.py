"""
Visualize what the shifter RBM's hidden units have learned.

For each hidden unit, plot its incoming weights organized as:
    V1 row  (8 weights)        ← top input ring
    V2 row  (8 weights)        ← bottom (shifted) ring
    Y bar   (3 weights)        ← shift class preference (left / none / right)

Hidden units are sorted by which shift class they prefer (Y argmax).

A hidden unit that detects "shift right by 1" should show a diagonal pattern:
positive weights on V1[i] and V2[i+1] (or both negative), and a strong Y[right]
preference.

Outputs:
    hidden_units.png      — per-unit receptive fields
    weights_heatmap.png   — full visible×hidden weight matrix
    confusion.png         — confusion matrix on the test set
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import shifter_rbm as sh


def train_quick(N=8, n_hidden=64, epochs=200, seed=0):
    return sh.train(N=N, n_hidden=n_hidden, n_epochs=epochs,
                    lr=0.03, momentum=0.7, batch_size=32, seed=seed,
                    verbose=False)


def plot_hidden_units(rbm, N, out_path="hidden_units.png"):
    """One small panel per hidden unit, sorted by preferred shift."""
    nh = rbm.nh
    # W: (nv, nh)  →  per-unit weight vectors of shape (nv,)
    W = rbm.W.copy()
    nv_y = 3

    # split each unit's weight vector into V1, V2, Y
    W_v1 = W[:N, :]            # (N, nh)
    W_v2 = W[N:2*N, :]          # (N, nh)
    W_y  = W[2*N:2*N+nv_y, :]   # (3, nh)

    # sort hidden units by preferred shift class (argmax of Y weights)
    pref_class = np.argmax(W_y, axis=0)        # 0=left, 1=none, 2=right
    pref_strength = W_y.max(axis=0) - W_y.min(axis=0)
    order = sorted(range(nh), key=lambda j: (pref_class[j], -pref_strength[j]))

    # symmetric color scale around 0
    vmax = float(np.abs(W).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    cols = 8
    rows = (nh + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.4))
    fig.suptitle(f"Hidden unit receptive fields (sorted by preferred shift)\n"
                 f"top: V1 (8 bits) · middle: V2 (8 bits) · bottom: Y [L N R]",
                 fontsize=11)
    classes = ["L", "N", "R"]
    class_colors = {0: "#a83232", 1: "#404040", 2: "#3a78a8"}

    for k, j in enumerate(order):
        ax = axes.flat[k]
        # build a (3 + 1, max(N, 3)) tile: pad Y row
        tile = np.full((3, N), np.nan, dtype=float)
        tile[0, :] = W_v1[:, j]
        tile[1, :] = W_v2[:, j]
        # Y row centered
        y_pad = (N - nv_y) // 2
        tile[2, y_pad:y_pad + nv_y] = W_y[:, j]

        im = ax.imshow(tile, cmap="RdBu_r", norm=norm, aspect="auto",
                       interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        c = pref_class[j]
        ax.set_title(f"#{j} → {classes[c]}", fontsize=8,
                     color=class_colors[c])
        # draw a frame around the Y row
        ax.axhline(1.5, color="black", lw=0.5)
        ax.axvline(y_pad - 0.5, color="black", lw=0.4, ymin=0, ymax=0.33)
        ax.axvline(y_pad + nv_y - 0.5, color="black", lw=0.4, ymin=0, ymax=0.33)

    # hide unused subplots
    for k in range(nh, rows * cols):
        axes.flat[k].axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="weight")
    plt.subplots_adjust(left=0.02, right=0.9, top=0.9, bottom=0.05,
                        wspace=0.3, hspace=0.5)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return order, pref_class


def plot_weights_heatmap(rbm, N, order, out_path="weights_heatmap.png"):
    """Full visible × hidden weight matrix as a heatmap with annotations."""
    W = rbm.W[:, order]
    nv = W.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, rbm.nh * 0.18), 6))
    vmax = float(np.abs(W).max())
    im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    # horizontal lines separating V1/V2/Y blocks
    ax.axhline(N - 0.5, color="black", lw=1)
    ax.axhline(2 * N - 0.5, color="black", lw=1)
    yticks = list(range(N)) + list(range(N)) + ["L", "N", "R"]
    ax.set_yticks(range(nv))
    ax.set_yticklabels([f"v1[{i}]" for i in range(N)] +
                        [f"v2[{i}]" for i in range(N)] +
                        ["y_left", "y_none", "y_right"], fontsize=8)
    ax.set_xlabel("hidden unit (sorted by preferred shift)")
    ax.set_ylabel("visible unit")
    ax.set_title("RBM weight matrix W (visible × hidden)")
    plt.colorbar(im, ax=ax, label="weight")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_confusion(rbm, N, out_path="confusion.png"):
    """Confusion matrix on the full test set."""
    V1, V2, Y = sh.make_shifter_data(N)
    clamp_mask = np.concatenate([np.ones(2 * N), np.zeros(3)]).astype(np.float32)
    M = V1.shape[0]
    cm = np.zeros((3, 3), dtype=int)
    for i in range(M):
        v_init = np.concatenate([V1[i], V2[i], np.zeros(3, dtype=np.float32)])
        v_mean = rbm.conditional_fill(v_init, clamp_mask, n_gibbs=200)
        pred = int(np.argmax(v_mean[2 * N:]))
        true = int(np.argmax(Y[i]))
        cm[true, pred] += 1
    classes = ["left (-1)", "none (0)", "right (+1)"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    ax.set_xticks(range(3)); ax.set_xticklabels(classes)
    ax.set_yticks(range(3)); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted shift"); ax.set_ylabel("true shift")
    overall = np.trace(cm) / cm.sum()
    ax.set_title(f"Shifter confusion matrix\noverall acc = {overall*100:.1f}%")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({cm.sum()} cases, acc {overall*100:.1f}%)")
    return cm


def plot_hidden_activations_per_shift(rbm, N, out_path="hidden_activations.png"):
    """For each shift class, show mean hidden activation across all input patterns.
    Reveals which units 'fire for left' vs 'fire for right' vs 'fire for none'."""
    V1, V2, Y = sh.make_shifter_data(N)
    X = np.concatenate([V1, V2, Y], axis=1)
    ph = rbm.ph_given_v(X)            # (M, nh)
    classes = ["left (-1)", "none (0)", "right (+1)"]
    means = np.stack([ph[Y[:, k] == 1].mean(axis=0) for k in range(3)])  # (3, nh)
    diff = means - means.mean(axis=0, keepdims=True)  # selectivity
    fig, axes = plt.subplots(2, 1, figsize=(max(8, rbm.nh * 0.16), 5))

    im0 = axes[0].imshow(means, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    axes[0].set_yticks(range(3)); axes[0].set_yticklabels(classes)
    axes[0].set_xlabel("hidden unit")
    axes[0].set_title("Mean hidden activation per shift class")
    plt.colorbar(im0, ax=axes[0], label="P(h=1)")

    vmax = float(np.abs(diff).max())
    im1 = axes[1].imshow(diff, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    axes[1].set_yticks(range(3)); axes[1].set_yticklabels(classes)
    axes[1].set_xlabel("hidden unit")
    axes[1].set_title("Selectivity (mean − overall mean): red = fires more for this class")
    plt.colorbar(im1, ax=axes[1], label="Δ activation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=".")
    args = p.parse_args()

    print(f"# Training RBM (N={args.N}, hidden={args.hidden}, "
          f"epochs={args.epochs}, seed={args.seed})...")
    rbm = train_quick(N=args.N, n_hidden=args.hidden,
                      epochs=args.epochs, seed=args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    order, pref = plot_hidden_units(rbm, args.N,
                                    os.path.join(args.outdir, "hidden_units.png"))
    plot_weights_heatmap(rbm, args.N, order,
                         os.path.join(args.outdir, "weights_heatmap.png"))
    plot_hidden_activations_per_shift(rbm, args.N,
                                      os.path.join(args.outdir, "hidden_activations.png"))
    plot_confusion(rbm, args.N, os.path.join(args.outdir, "confusion.png"))

    # also save weights for later inspection
    np.savez(os.path.join(args.outdir, "rbm_weights.npz"),
             W=rbm.W, bv=rbm.bv, bh=rbm.bh)
    print(f"saved {args.outdir}/rbm_weights.npz")

    # Print summary of what each unit prefers
    classes = ["left", "none", "right"]
    counts = np.bincount(pref, minlength=3)
    print("\nHidden-unit shift preference distribution:")
    for k, c in enumerate(classes):
        print(f"  {c:5s}: {counts[k]} units")
