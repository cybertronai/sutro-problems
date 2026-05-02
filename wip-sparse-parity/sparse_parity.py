"""Sparse-parity prototype: data generation, two solvers, Dally-style cost.

Problem: given X in {-1,+1}^(m x n) and y = prod_{j in S} X[:,j] for some
unknown subset S of |S| = k, recover S. Equivalently a GF(2) linear-parity
problem (map +1 -> 0, -1 -> 1, multiply -> XOR).

We expose:
  - generate(n_bits, k_sparse, n_samples, seed): X, y, secret_indices
  - solve_bruteforce(X, y, k, cost): O(C(n,k) * m * k) reads
  - solve_gf2(X, y, k, cost):       O(m * n + (k+1)^2 * n) GF(2) ops on
                                    a (k+1) x n matrix

Cost model: a `Cost` accumulator tracks reads of the (X, y) arrays, weighted
by the simplified-Dally rule cost(addr) = ceil(sqrt(addr)) where the
arrays are laid out linearly in a single address space:

    X bulk at addresses 1..m*n      (row-major)
    y bulk at addresses m*n+1..m*n+m

This is the same cost discipline as ../matmul, so totals here are directly
comparable to matmul costs (e.g. baseline_4x4 = 1,316; baseline_16x16 = 340,704).
"""
from __future__ import annotations

import math
import random
from itertools import combinations
from typing import Callable, List, Sequence, Tuple


def _cost_of(addr: int) -> int:
    if addr <= 0:
        raise ValueError(f"address must be positive, got {addr}")
    return math.ceil(math.sqrt(addr))


class Cost:
    """Address-aware read counter. Mirrors matmul.score_*'s cost model."""

    def __init__(self, m: int, n: int) -> None:
        self.m = m
        self.n = n
        self._x_base = 1
        self._y_base = m * n + 1
        self.reads = 0
        self.read_cost = 0

    def read_x(self, i: int, j: int) -> None:
        self.reads += 1
        self.read_cost += _cost_of(self._x_base + i * self.n + j)

    def read_y(self, i: int) -> None:
        self.reads += 1
        self.read_cost += _cost_of(self._y_base + i)


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------

def generate(
    n_bits: int, k_sparse: int, n_samples: int, seed: int = 0,
) -> Tuple[List[List[int]], List[int], List[int]]:
    """Random {-1,+1} inputs, label = product of X[:, secret].

    Returns (X, y, secret) with X shape (n_samples, n_bits) as nested lists,
    y of length n_samples, secret as a sorted list of k_sparse indices.

    With m_samples >= k_sparse + 1 random samples, the secret is unique
    with high probability (over GF(2) the rank-deficiency event has
    probability ~2^-(m-k)).
    """
    if k_sparse > n_bits:
        raise ValueError("k_sparse must be <= n_bits")
    rng = random.Random(seed)
    secret = sorted(rng.sample(range(n_bits), k_sparse))
    X: List[List[int]] = []
    y: List[int] = []
    for _ in range(n_samples):
        row = [rng.choice([-1, 1]) for _ in range(n_bits)]
        prod = 1
        for j in secret:
            prod *= row[j]
        X.append(row)
        y.append(prod)
    return X, y, secret


# --------------------------------------------------------------------------
# Solver A: brute-force subset search
# --------------------------------------------------------------------------

def solve_bruteforce(
    X: Sequence[Sequence[int]],
    y: Sequence[int],
    k: int,
    cost: Cost,
) -> List[int]:
    """Try every k-subset of columns; return the first that produces the
    observed labels on every sample. Reads X[i,j] for j in subset and y[i]
    once per (subset, row) pair, with early termination on mismatch."""
    m = len(X)
    n = len(X[0])
    for subset in combinations(range(n), k):
        ok = True
        for i in range(m):
            prod = 1
            for j in subset:
                cost.read_x(i, j)
                prod *= X[i][j]
            cost.read_y(i)
            if prod != y[i]:
                ok = False
                break
        if ok:
            return list(subset)
    raise RuntimeError("no k-subset matched the labels")


# --------------------------------------------------------------------------
# Solver B: GF(2) Gaussian elimination
# --------------------------------------------------------------------------

def solve_gf2(
    X: Sequence[Sequence[int]],
    y: Sequence[int],
    k: int,
    cost: Cost,
) -> List[int]:
    """Solve A s = b (mod 2) where A[i,j] = (1 - X[i,j]) // 2 and
    b[i] = (1 - y[i]) // 2. Returns the indices where s == 1.

    Reads each X[i,j] and y[i] once during the load phase. Subsequent
    Gaussian elimination operates on the loaded GF(2) matrix and is not
    charged against (X, y) bulk reads -- treat it as scratchpad work.
    """
    m = len(X)
    n = len(X[0])
    A = [[0] * n for _ in range(m)]
    b = [0] * m
    for i in range(m):
        for j in range(n):
            cost.read_x(i, j)
            A[i][j] = 0 if X[i][j] == 1 else 1
        cost.read_y(i)
        b[i] = 0 if y[i] == 1 else 1

    # Standard row-reduction over GF(2). Augmented matrix [A | b].
    M = [row + [b_i] for row, b_i in zip(A, b)]
    pivot_col_for_row: List[int] = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        pivot = next((i for i in range(r, m) if M[i][c] == 1), None)
        if pivot is None:
            continue
        M[r], M[pivot] = M[pivot], M[r]
        for i in range(m):
            if i != r and M[i][c] == 1:
                M[i] = [a ^ b_ for a, b_ in zip(M[i], M[r])]
        pivot_col_for_row.append(c)
        r += 1

    # Pick any solution s with the right Hamming weight by scanning the
    # free columns: prefer the all-pivot solution (free vars = 0). For
    # m = k+1 random samples this almost always recovers s exactly.
    s = [0] * n
    free_cols = [c for c in range(n) if c not in pivot_col_for_row]
    for row_i, pcol in enumerate(pivot_col_for_row):
        s[pcol] = M[row_i][n]  # b column after reduction
    secret = [j for j, sj in enumerate(s) if sj == 1]
    if len(secret) != k:
        # Free-vars present and our default assignment got the wrong
        # weight. Try the 2^free_vars settings (cheap when m ~ k+1, free
        # cols ~ n - r is small in practice for m close to n).
        for mask in range(1, 1 << len(free_cols)):
            s_try = list(s)
            for bit, fc in enumerate(free_cols):
                if (mask >> bit) & 1:
                    # toggling free var fc flips pivot rows that had a 1
                    # in column fc after reduction
                    s_try[fc] ^= 1
                    for row_i, pcol in enumerate(pivot_col_for_row):
                        if M[row_i][fc] == 1:
                            s_try[pcol] ^= 1
            if sum(s_try) == k:
                secret = [j for j, sj in enumerate(s_try) if sj == 1]
                break
        else:
            raise RuntimeError(
                f"no GF(2) solution of weight {k}; need more samples"
            )
    return secret
