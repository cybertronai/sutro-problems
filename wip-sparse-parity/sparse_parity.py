"""Sparse parity: recover k=3 secret bits among n=16 from m=16 training rows.

Each row of X is in {-1,+1}^16. The label is the product of the k secret
bits: ``y[i] = prod(X[i, j] for j in secret)``. The training set is chosen
(by rejection sampling) so that the secret subset is the *unique* 3-subset
of columns consistent with the training labels — therefore the brute-force
solver always recovers it, and the 64-row test set is classified at 100%.
"""
from __future__ import annotations

from itertools import combinations
from random import Random
from typing import List, Sequence, Tuple

N_BITS = 16
K_SECRET = 3
M_TRAIN = 16
M_TEST = 64


def _label(row: Sequence[int], subset: Sequence[int]) -> int:
    p = 1
    for j in subset:
        p *= row[j]
    return p


def _identifiable(X: Sequence[Sequence[int]], y: Sequence[int], k: int) -> bool:
    """True iff exactly one k-subset of columns explains every label in y."""
    n = len(X[0])
    matches = 0
    for subset in combinations(range(n), k):
        if all(_label(row, subset) == y_i for row, y_i in zip(X, y)):
            matches += 1
            if matches > 1:
                return False
    return matches == 1


def generate(seed: int = 0) -> Tuple[
    List[List[int]], List[int], List[List[int]], List[int], List[int]
]:
    """Return ``(X_train, y_train, X_test, y_test, secret)``.

    The training rows are resampled until the secret is the unique
    weight-k subset matching y_train. The expected number of resamples
    is ~1 (E[false subsets] ≈ 0.0085 for n=16, k=3, m=16).
    """
    rng = Random(seed)
    secret = sorted(rng.sample(range(N_BITS), K_SECRET))
    while True:
        X_train = [
            [rng.choice((-1, 1)) for _ in range(N_BITS)]
            for _ in range(M_TRAIN)
        ]
        y_train = [_label(row, secret) for row in X_train]
        if _identifiable(X_train, y_train, K_SECRET):
            break
    X_test = [
        [rng.choice((-1, 1)) for _ in range(N_BITS)]
        for _ in range(M_TEST)
    ]
    y_test = [_label(row, secret) for row in X_test]
    return X_train, y_train, X_test, y_test, secret


def solve_bruteforce(
    X: Sequence[Sequence[int]],
    y: Sequence[int],
    k: int = K_SECRET,
) -> List[int]:
    """Return the unique k-subset of columns matching every label in y."""
    n = len(X[0])
    for subset in combinations(range(n), k):
        if all(_label(row, subset) == y_i for row, y_i in zip(X, y)):
            return list(subset)
    raise RuntimeError("no k-subset matches the training labels")


def predict(X: Sequence[Sequence[int]], subset: Sequence[int]) -> List[int]:
    return [_label(row, subset) for row in X]


def accuracy(y_pred: Sequence[int], y_true: Sequence[int]) -> float:
    return sum(a == b for a, b in zip(y_pred, y_true)) / len(y_true)


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te, secret = generate(seed=0)
    found = solve_bruteforce(X_tr, y_tr)
    acc = accuracy(predict(X_te, found), y_te)
    print(f"secret    = {secret}")
    print(f"recovered = {found}")
    print(f"test acc  = {acc:.0%}")
