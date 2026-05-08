# Sparse parity

- The [Sparse Parity Challenge](https://github.com/cybertronai/sparse-parity-challenge) (Sutro Group) asks: given random ±1 inputs and a label that's the product of *k* secret bits, find those bits. Equivalently over `{0, 1}` with parity = XOR.
- Here we fix a tiny version: **n = 3 bits (2 secret + 1 noise), k = 2, m = 5 train, 5 test**.
- The training set is constructed so that **exactly one** 2-subset of bit positions is consistent with the training labels — so 100% test accuracy is reachable by any solver that finds that subset.
- We measure not just correctness but **energy**, using the [simplified Dally model](https://github.com/cybertronai/simplified-dally-model) and the [v1 instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets/v1) (`add`, `sub`, `mul`, `copy`, `and`, `or`, `xor`, `not`). Parity is a single `xor` per test row, so v1 is the natural fit.

## Problem

Given a training matrix `X ∈ {0, 1}^(5 × 3)` and labels `y ∈ {0, 1}^5` with

    y[i] = XOR_{j ∈ S} X[i, j]

for some unknown `S ⊆ {0, 1, 2}` of size `2`, recover `S`. Then predict the labels of a held-out test set `X_test ∈ {0, 1}^(5 × 3)`.

The training set is selected (via rejection sampling on the random rows) so that the secret subset `S` is the **unique** 2-subset of columns consistent with `y_train`. With this guarantee the brute-force solver always recovers `S` exactly, and test accuracy is 100%.

## Cost model

Same simplified Dally model as `../matmul`. Processor at the origin; memory laid out as a 2D upper half-plane indexed by positive integers; the cell at linear index `addr` sits at Manhattan distance `⌈√addr⌉` from the core. Each operand read pays that distance; writes and arithmetic are free; inputs are placed for free at caller-specified addresses; every output address pays one standard read at exit.

Three-address-code IR. v1 ops:

| Op     | Form                  | Effect                       |
|--------|-----------------------|------------------------------|
| `add`  | `add d, a, b`         | `mem[d] = mem[a] + mem[b]`   |
| `sub`  | `sub d, a, b`         | `mem[d] = mem[a] - mem[b]`   |
| `mul`  | `mul d, a, b`         | `mem[d] = mem[a] * mem[b]`   |
| `and`  | `and d, a, b`         | `mem[d] = mem[a] & mem[b]`   |
| `or`   | `or  d, a, b`         | `mem[d] = mem[a] \| mem[b]`  |
| `xor`  | `xor d, a, b`         | `mem[d] = mem[a] ^ mem[b]`   |
| `copy` | `copy d, a`           | `mem[d] = mem[a]`            |
| `not`  | `not  d, a`           | `mem[d] = ~mem[a]`           |

Two-operand short form for the binary ops: `xor d, a` ≡ `xor d, d, a`. `;` works as a line separator so single-line IR strings are fine.

The 35 input values are fed to the IR in this fixed order:

    X_train (row-major, 15 values)
    y_train               (5 values)
    X_test  (row-major, 15 values)

The IR declares 35 input addresses (one per value, in the same order) and is responsible for laying them out. Outputs: 5 prediction values that must equal `y_test` exactly.

## API

```python
from sparse_parity import (
    generate, solve_bruteforce, predict, accuracy,
    score_sparse_parity, generate_baseline,
)

# Python reference
X_train, y_train, X_test, y_test, secret = generate(seed=0)
found = solve_bruteforce(X_train, y_train)         # → [1, 2]
acc   = accuracy(predict(X_test, found), y_test)   # → 1.0

# IR scoring
ir   = generate_baseline()                          # naive predictor IR
cost = score_sparse_parity(ir)                      # → 91
```

## Reference solver

`solve_bruteforce` enumerates the `C(3, 2) = 3` candidate subsets and returns the unique one that reproduces every training label. Worst-case work: `3 × 5 × 2 = 30` XORs. Because `generate` guarantees identifiability, this solver scores 100% on the 5-example test set for every seed.

## Baseline IR

`generate_baseline` discovers `S = solve_bruteforce(X_train, y_train)` in Python (free, just like matmul's algorithm choice is free) and emits an IR that predicts each test row with a single `xor` over the two secret columns of `X_test`. Layout:

    X_train at addrs 1..15   (row-major)
    y_train at addrs 16..20
    X_test  at addrs 21..35  (row-major)
    pred    at addrs 36..40  (output)

For seed=0 (`secret = [1, 2]`):

    1,2,3,...,35           # input addresses
    xor 36,22,23           # pred[0] = X_test[0,1] ^ X_test[0,2]
    xor 37,25,26           # pred[1] = X_test[1,1] ^ X_test[1,2]
    xor 38,28,29
    xor 39,31,32
    xor 40,34,35
    36,37,38,39,40         # output addresses

Cost: 5 xors × ~11 read-cost + 5 output reads (×7 avg) = **91**.

## Record History

| Date       | Cost | Submission                       | Contributors                                 | Description                        |
| -          | -:   | -                                | -                                            | -                                  |
| 2026-05-07 |   91 | [ir](submissions/baseline.ir)    | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_baseline` (naive layout, v1 `xor`) |
