"""
Shifter network — reproduction via Restricted Boltzmann Machine (CD-1).

Same problem setup as the Hinton & Sejnowski 1986 shifter experiment:
  Two rings of N binary units V1, V2 where V2 is V1 shifted by one of
  {-1, 0, +1} positions. The network must infer the shift direction.

Architecture: RBM with visible = [V1 | V2 | Y] and a layer of hidden units.
Training: Contrastive Divergence (CD-1) — same fundamental learning rule
(positive phase minus negative phase) as the original Boltzmann machine,
but with the efficient RBM sampling structure.

This is a faithful reproduction of the problem and learning principle,
implemented from scratch.
"""

from __future__ import annotations
import argparse
import time
import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_shifter_data(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All N-bit shifter cases: 2^N patterns x 3 shifts."""
    patterns = np.array([[(p >> i) & 1 for i in range(N)] for p in range(2**N)],
                        dtype=np.float32)
    V1, V2, Y = [], [], []
    for p in patterns:
        for k, s in enumerate([-1, 0, 1]):
            V1.append(p)
            V2.append(np.roll(p, s))
            y = np.zeros(3, dtype=np.float32); y[k] = 1.0
            Y.append(y)
    return np.array(V1), np.array(V2), np.array(Y)


# ----------------------------------------------------------------------
# RBM
# ----------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class RBM:
    def __init__(self, n_visible: int, n_hidden: int, rng=None):
        self.nv = n_visible
        self.nh = n_hidden
        self.rng = rng or np.random.default_rng(0)
        self.W = 0.01 * self.rng.standard_normal((n_visible, n_hidden)).astype(np.float32)
        self.bv = np.zeros(n_visible, dtype=np.float32)
        self.bh = np.zeros(n_hidden, dtype=np.float32)

    def ph_given_v(self, v):   # v: (B, nv)
        return sigmoid(v @ self.W + self.bh)

    def pv_given_h(self, h):   # h: (B, nh)
        return sigmoid(h @ self.W.T + self.bv)

    def sample(self, p):
        return (self.rng.random(p.shape) < p).astype(np.float32)

    def cd1(self, v0, lr, momentum, vW, vbv, vbh, clamp_mask=None, clamp_values=None):
        """One CD-1 step on a minibatch v0: (B, nv). Returns updated momentum buffers."""
        # positive phase
        ph0 = self.ph_given_v(v0)
        h0 = self.sample(ph0)

        # negative phase: one step reconstruction
        pv1 = self.pv_given_h(h0)
        # if some visible units are clamped (e.g. during conditional sampling), keep them fixed
        if clamp_mask is not None:
            pv1 = pv1 * (1 - clamp_mask) + clamp_values * clamp_mask
        v1 = self.sample(pv1)
        if clamp_mask is not None:
            v1 = v1 * (1 - clamp_mask) + clamp_values * clamp_mask
        ph1 = self.ph_given_v(v1)

        # gradients (averaged over batch, using probabilities on hidden side for lower variance)
        B = v0.shape[0]
        dW  = (v0.T @ ph0 - v1.T @ ph1) / B
        dbv = (v0 - v1).mean(axis=0)
        dbh = (ph0 - ph1).mean(axis=0)

        vW  = momentum * vW  + lr * dW
        vbv = momentum * vbv + lr * dbv
        vbh = momentum * vbh + lr * dbh
        self.W  += vW
        self.bv += vbv
        self.bh += vbh
        return vW, vbv, vbh, (dW, dbv, dbh)

    # ---- Gibbs-based inference for clamped-subset queries ---------------
    def conditional_fill(self, v_init, clamp_mask, n_gibbs=50):
        """Clamp a subset of visible units and let the rest settle by Gibbs.
        Returns mean visible state over the last half of Gibbs steps."""
        v = v_init.copy()
        accum = np.zeros_like(v)
        n_accum = 0
        for t in range(n_gibbs):
            ph = self.ph_given_v(v)
            h = self.sample(ph)
            pv = self.pv_given_h(h)
            v = self.sample(pv)
            # re-clamp
            v = v * (1 - clamp_mask) + v_init * clamp_mask
            if t >= n_gibbs // 2:
                # use probabilities on unclamped units for lower-variance mean
                v_mean = pv * (1 - clamp_mask) + v_init * clamp_mask
                accum += v_mean
                n_accum += 1
        return accum / n_accum


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(N=8, n_hidden=24, n_epochs=60, lr=0.05, momentum=0.5,
          batch_size=16, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    V1, V2, Y = make_shifter_data(N)
    X = np.concatenate([V1, V2, Y], axis=1)  # (M, 2N+3)
    M, nv = X.shape
    nv_y = 3
    nv_inputs = 2 * N  # V1 + V2 portion

    rbm = RBM(nv, n_hidden, rng=rng)
    vW = np.zeros_like(rbm.W); vbv = np.zeros_like(rbm.bv); vbh = np.zeros_like(rbm.bh)

    # mask: 1 for clamped units during positive phase (input + label, i.e. all visible)
    if verbose:
        print(f"# N={N}: {M} training cases, visible={nv}, hidden={n_hidden}")

    clamp_eval_mask = np.concatenate([np.ones(nv_inputs), np.zeros(nv_y)]).astype(np.float32)

    for epoch in range(n_epochs):
        t0 = time.time()
        idx = rng.permutation(M)
        total_recon = 0.0
        for i in range(0, M, batch_size):
            batch = X[idx[i:i + batch_size]]
            vW, vbv, vbh, grads = rbm.cd1(batch, lr, momentum, vW, vbv, vbh)
            # track reconstruction error as a sanity signal
            ph = rbm.ph_given_v(batch)
            pv = rbm.pv_given_h(ph)
            total_recon += np.mean((batch - pv) ** 2) * batch.shape[0]
        recon = total_recon / M

        if verbose and (epoch + 1) % 5 == 0:
            acc = evaluate(rbm, V1, V2, Y, N, clamp_eval_mask, n_gibbs=80)
            print(f"epoch {epoch+1:3d}  recon_mse={recon:.4f}  "
                  f"acc={acc*100:5.1f}%  ({time.time()-t0:.2f}s)")

    return rbm


def evaluate(rbm, V1, V2, Y, N, clamp_mask, n_gibbs=80):
    """Clamp V1+V2 on the visible layer, let Y settle, argmax."""
    M = V1.shape[0]
    correct = 0
    for i in range(M):
        v_init = np.concatenate([V1[i], V2[i], np.zeros(3, dtype=np.float32)])
        v_mean = rbm.conditional_fill(v_init, clamp_mask, n_gibbs=n_gibbs)
        y_pred = v_mean[2 * N:]
        if int(np.argmax(y_pred)) == int(np.argmax(Y[i])):
            correct += 1
    return correct / M


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rbm = train(N=args.N, n_hidden=args.hidden, n_epochs=args.epochs,
                lr=args.lr, momentum=args.momentum, batch_size=args.batch,
                seed=args.seed)

    # final full eval with more Gibbs sweeps
    V1, V2, Y = make_shifter_data(args.N)
    clamp_mask = np.concatenate([np.ones(2 * args.N), np.zeros(3)]).astype(np.float32)
    acc = evaluate(rbm, V1, V2, Y, args.N, clamp_mask, n_gibbs=200)
    print(f"\nFinal accuracy (N={args.N}, {V1.shape[0]} cases, 200 Gibbs sweeps): "
          f"{acc*100:.2f}%")

    # confusion matrix per shift class
    print("\nPer-class accuracy:")
    classes = ['left (-1)', 'none (0)', 'right (+1)']
    for k in range(3):
        mask = Y[:, k] == 1
        sub_V1, sub_V2, sub_Y = V1[mask], V2[mask], Y[mask]
        sub_acc = evaluate(rbm, sub_V1, sub_V2, sub_Y, args.N, clamp_mask, n_gibbs=200)
        print(f"  {classes[k]:12s}  {sub_acc*100:5.1f}% ({int(sub_acc*len(sub_Y))}/{len(sub_Y)})")
