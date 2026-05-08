# Sparse parity

- The [Sparse Parity Challenge](https://github.com/cybertronai/sparse-parity-challenge) (Sutro Group) asks: given random ±1 inputs and a label that's the product of *k* secret bits, find those bits.
- Here we fix a tiny version: **n = 3 bits (2 secret + 1 noise), k = 2, m = 5 train, 5 test**.
- The training set is constructed so that **exactly one** 2-subset of bit positions is consistent with the training labels — so 100% test accuracy is reachable by any solver that finds that subset.

## Problem

Given a training matrix `X ∈ {-1, +1}^(5 × 3)` and labels `y ∈ {-1, +1}^5` with

    y[i] = ∏_{j ∈ S} X[i, j]

for some unknown `S ⊆ {0, 1, 2}` of size `2`, recover `S`. Then predict the labels of a held-out test set `X_test ∈ {-1, +1}^(5 × 3)`.

The training set is selected (via rejection sampling on the random rows) so that the secret subset `S` is the **unique** 2-subset of columns whose products reproduce every training label. With this guarantee the brute-force solver — enumerate all `C(3, 2) = 3` subsets, return the one that matches — always recovers `S` exactly, and test accuracy is 100%.

## API

```python
from sparse_parity import generate, solve_bruteforce, predict, accuracy

X_train, y_train, X_test, y_test, secret = generate(seed=0)

found = solve_bruteforce(X_train, y_train)   # → [1, 2]
acc   = accuracy(predict(X_test, found), y_test)  # → 1.0
```

## Reference solver

`solve_bruteforce` enumerates the `C(3, 2) = 3` candidate subsets and returns the unique one that reproduces every training label. Worst-case work: `3 × 5 × 2 = 30` multiplications. Because `generate` guarantees identifiability, this solver scores 100% on the 5-example test set for every seed.

## Record History

| Date       | Solver                    | Test acc | Contributors |
| -          | -                         | -:       | -            |
| 2026-05-07 | `solve_bruteforce`        | 100%     | [@yaroslavvb](https://github.com/yaroslavvb) |
