"""
Shifter network — reproduction of the classic Boltzmann machine experiment from
Hinton & Sejnowski (PDP Vol 1, Ch 7, 1986).

Problem:
  Two rings of N binary units (V1 and V2). V2 is a copy of V1 shifted by one
  of {-1, 0, +1} positions (with wraparound). The network has K hidden units
  and 3 output units (one-hot {left, none, right}). Training data is the full
  enumeration of {V1 patterns} x {3 shifts}.

Learning rule (Hinton & Sejnowski 1983):
  Δw_ij  ∝  <s_i s_j>_clamped  -  <s_i s_j>_free
  Sampling: simulated annealing then equilibrium gibbs sweeps.

Author: implemented from scratch, no text copied from the chapter.
"""

from __future__ import annotations
import argparse
import math
import time
import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_shifter_data(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build all training cases for an N-bit shifter.

    Returns:
        V1: (M, N) binary, the input ring
        V2: (M, N) binary, the shifted ring
        Y:  (M, 3) one-hot, [left, none, right] (shift by -1, 0, +1)
    """
    patterns = np.array([[(p >> i) & 1 for i in range(N)] for p in range(2**N)],
                        dtype=np.float32)
    shifts = [-1, 0, 1]
    V1, V2, Y = [], [], []
    for p in patterns:
        for k, s in enumerate(shifts):
            V1.append(p)
            V2.append(np.roll(p, s))
            y = np.zeros(3, dtype=np.float32); y[k] = 1.0
            Y.append(y)
    return np.array(V1), np.array(V2), np.array(Y)


# ----------------------------------------------------------------------
# Boltzmann machine
# ----------------------------------------------------------------------

class BoltzmannMachine:
    """
    Single fully-connected layer of binary units {0,1} with a symmetric weight
    matrix W (no self-connections) and biases b. Some units are "visible"
    (clamped during the positive phase), the rest are "hidden".
    """

    def __init__(self, n_units: int, visible_idx: np.ndarray, rng=None):
        self.n = n_units
        self.visible_idx = np.asarray(visible_idx, dtype=int)
        mask = np.zeros(n_units, dtype=bool); mask[self.visible_idx] = True
        self.hidden_idx = np.where(~mask)[0]
        self.W = np.zeros((n_units, n_units), dtype=np.float32)
        self.b = np.zeros(n_units, dtype=np.float32)
        self.rng = rng or np.random.default_rng(0)

    # ---- core sampling ----------------------------------------------------

    def _gibbs_sweep(self, s: np.ndarray, T: float, frozen: np.ndarray | None = None):
        """One asynchronous Gibbs sweep at temperature T. Updates `s` in place.
        `frozen` is a boolean mask of indices NOT to update."""
        order = self.rng.permutation(self.n)
        for i in order:
            if frozen is not None and frozen[i]:
                continue
            # net input from current state of all other units
            net = self.W[i] @ s + self.b[i]
            p = 1.0 / (1.0 + math.exp(-net / T)) if abs(net) < 50 * T else (1.0 if net > 0 else 0.0)
            s[i] = 1.0 if self.rng.random() < p else 0.0

    def _anneal_and_sample(self,
                           init: np.ndarray,
                           frozen: np.ndarray | None,
                           schedule: list[tuple[float, int]],
                           equil_sweeps: int) -> np.ndarray:
        """Run an annealing schedule, then collect a co-occurrence matrix
        averaged over equilibrium sweeps at the final temperature."""
        s = init.copy()
        for T, n_sweeps in schedule:
            for _ in range(n_sweeps):
                self._gibbs_sweep(s, T, frozen)
        T_final = schedule[-1][0]
        co = np.zeros((self.n, self.n), dtype=np.float32)
        means = np.zeros(self.n, dtype=np.float32)
        for _ in range(equil_sweeps):
            self._gibbs_sweep(s, T_final, frozen)
            co += np.outer(s, s)
            means += s
        co /= equil_sweeps
        means /= equil_sweeps
        return co, means

    # ---- training step ---------------------------------------------------

    def positive_phase(self, clamp_values: np.ndarray, schedule, equil_sweeps):
        """Clamp the visible units to `clamp_values` (length = len(visible_idx))
        and sample the hidden units."""
        s = self.rng.integers(0, 2, size=self.n).astype(np.float32)
        s[self.visible_idx] = clamp_values
        frozen = np.zeros(self.n, dtype=bool); frozen[self.visible_idx] = True
        return self._anneal_and_sample(s, frozen, schedule, equil_sweeps)

    def negative_phase(self, schedule, equil_sweeps):
        """Free phase — no units clamped."""
        s = self.rng.integers(0, 2, size=self.n).astype(np.float32)
        return self._anneal_and_sample(s, None, schedule, equil_sweeps)

    # ---- inference -------------------------------------------------------

    def clamp_subset_and_sample(self,
                                clamp_idx: np.ndarray,
                                clamp_values: np.ndarray,
                                schedule,
                                equil_sweeps) -> np.ndarray:
        """Clamp only a subset of units; return mean state."""
        s = self.rng.integers(0, 2, size=self.n).astype(np.float32)
        s[clamp_idx] = clamp_values
        frozen = np.zeros(self.n, dtype=bool); frozen[clamp_idx] = True
        _, means = self._anneal_and_sample(s, frozen, schedule, equil_sweeps)
        return means


# ----------------------------------------------------------------------
# Training driver
# ----------------------------------------------------------------------

def train(N: int = 4,
          n_hidden: int = 12,
          n_epochs: int = 30,
          lr: float = 0.02,
          momentum: float = 0.9,
          weight_decay: float = 0.0002,
          schedule=((40, 2), (30, 2), (20, 2), (15, 2), (10, 2), (6, 3), (3, 5)),
          equil_sweeps: int = 30,
          seed: int = 0,
          verbose: bool = True):
    """Train a Boltzmann machine on the N-bit shifter problem."""
    rng = np.random.default_rng(seed)

    # Layout: [V1 (N) | V2 (N) | Y (3) | hidden (n_hidden)]
    n_v1, n_v2, n_y = N, N, 3
    n_visible = n_v1 + n_v2 + n_y
    n_units = n_visible + n_hidden
    visible_idx = np.arange(n_visible)

    bm = BoltzmannMachine(n_units, visible_idx, rng=rng)
    # zero init (Hinton's original choice) — symmetry broken by stochastic sampling
    bm.W = np.zeros((n_units, n_units), dtype=np.float32)
    vW = np.zeros_like(bm.W)  # momentum buffer for weights
    vb = np.zeros_like(bm.b)  # momentum buffer for biases

    V1, V2, Y = make_shifter_data(N)
    M = V1.shape[0]
    if verbose:
        print(f"# Shifter N={N}: {M} training cases ({2**N} patterns x 3 shifts)")
        print(f"# Network: {n_v1}+{n_v2}+{n_y} visible + {n_hidden} hidden = {n_units} units")

    schedule_list = list(schedule)

    for epoch in range(n_epochs):
        t0 = time.time()
        pos_co = np.zeros_like(bm.W)
        pos_b = np.zeros_like(bm.b)
        neg_co = np.zeros_like(bm.W)
        neg_b = np.zeros_like(bm.b)

        # positive phase: average over training cases
        order = rng.permutation(M)
        for idx in order:
            clamp = np.concatenate([V1[idx], V2[idx], Y[idx]])
            co, m = bm.positive_phase(clamp, schedule_list, equil_sweeps)
            pos_co += co
            pos_b += m
        pos_co /= M; pos_b /= M

        # negative phase: same number of free samples
        for _ in range(M):
            co, m = bm.negative_phase(schedule_list, equil_sweeps)
            neg_co += co
            neg_b += m
        neg_co /= M; neg_b /= M

        # update (with momentum + weight decay)
        dW = pos_co - neg_co
        np.fill_diagonal(dW, 0)
        dW = (dW + dW.T) / 2  # symmetrize the gradient
        vW = momentum * vW + lr * (dW - weight_decay * bm.W)
        vb = momentum * vb + lr * (pos_b - neg_b)
        bm.W += vW
        bm.b += vb

        if verbose:
            acc = evaluate(bm, V1, V2, Y, n_v1, n_v2, n_y, schedule_list, equil_sweeps)
            print(f"epoch {epoch+1:3d}  ΔW_norm={np.linalg.norm(dW):.4f}  "
                  f"acc={acc*100:5.1f}%  ({time.time()-t0:.1f}s)")

    return bm


def evaluate(bm, V1, V2, Y, n_v1, n_v2, n_y, schedule, equil_sweeps,
             max_cases: int | None = 64) -> float:
    """Clamp V1 and V2, let Y settle, count correct argmax."""
    M = V1.shape[0]
    idxs = np.arange(M)
    if max_cases and M > max_cases:
        idxs = bm.rng.choice(M, size=max_cases, replace=False)
    y_start = n_v1 + n_v2
    clamp_idx = np.arange(0, y_start)  # V1 + V2 clamped, Y free
    correct = 0
    for i in idxs:
        clamp_vals = np.concatenate([V1[i], V2[i]])
        means = bm.clamp_subset_and_sample(clamp_idx, clamp_vals, schedule, equil_sweeps)
        pred = int(np.argmax(means[y_start:y_start + n_y]))
        true = int(np.argmax(Y[i]))
        correct += (pred == true)
    return correct / len(idxs)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=4, help="ring width (default 4)")
    p.add_argument("--hidden", type=int, default=12, help="number of hidden units")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--decay", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--equil", type=int, default=30, help="equilibrium sweeps for stats")
    args = p.parse_args()

    bm = train(N=args.N,
               n_hidden=args.hidden,
               n_epochs=args.epochs,
               lr=args.lr,
               momentum=args.momentum,
               weight_decay=args.decay,
               equil_sweeps=args.equil,
               seed=args.seed)

    V1, V2, Y = make_shifter_data(args.N)
    final_acc = evaluate(bm, V1, V2, Y,
                         args.N, args.N, 3,
                         schedule=[(20,1),(15,1),(12,1),(10,1),(8,3)],
                         equil_sweeps=10,
                         max_cases=None)
    print(f"\nFinal accuracy (full {V1.shape[0]} cases): {final_acc*100:.2f}%")
