# Sparse parity

- The [Sparse Parity Challenge](https://github.com/cybertronai/sparse-parity-challenge) (Sutro Group) asks: given random ±1 inputs and a label that's the product of *k* secret bits, find those bits.
- Here we fix the smallest version that's still meaningful: **n = 16 bits, k = 3 secret bits, m = 16 train, 64 test**.
- The training set is constructed so that **exactly one** 3-subset of bit positions is consistent with the training labels — so 100% test accuracy is reachable by any solver that finds that subset.

## Problem

Given a training matrix `X ∈ {-1, +1}^(16 × 16)` and labels `y ∈ {-1, +1}^16` with

    y[i] = ∏_{j ∈ S} X[i, j]

for some unknown `S ⊆ {0,…,15}` of size `3`, recover `S`. Then predict the labels of a held-out test set `X_test ∈ {-1, +1}^(64 × 16)`.

The training set is selected (via rejection sampling on the random rows) so that the secret subset `S` is the **unique** 3-subset of columns whose products reproduce every training label. With this guarantee the brute-force solver — enumerate all `C(16, 3) = 560` subsets, return the one that matches — always recovers `S` exactly, and test accuracy is 100%.

## API

```python
from sparse_parity import generate, solve_bruteforce, predict, accuracy

X_train, y_train, X_test, y_test, secret = generate(seed=0)

found = solve_bruteforce(X_train, y_train)   # → [6, 12, 15]
acc   = accuracy(predict(X_test, found), y_test)  # → 1.0
```

## Reference solver

`solve_bruteforce` enumerates the `C(16, 3) = 560` candidate subsets and returns the unique one that reproduces every training label. Worst-case work: `560 × 16 × 3 ≈ 27k` multiplications. Because `generate` guarantees identifiability, this solver scores 100% on the 64-example test set for every seed.

## Record History

| Date       | Solver                    | Test acc | Contributors |
| -          | -                         | -:       | -            |
| 2026-05-06 | `solve_bruteforce`        | 100%     | [@yaroslavvb](https://github.com/yaroslavvb) |
