# Sparse parity (WIP)

> *Work in progress.* This folder is a small probe at framing the
> [Sparse Parity Challenge](https://github.com/cybertronai/sparse-parity-challenge)
> as a Sutro-problems-style energy-cost benchmark — the same shape as
> [`../matmul`](../matmul), with a Dally-flavored read-cost ruler.

## Motivation

[Sparse Parity Challenge](https://github.com/cybertronai/sparse-parity-challenge) (Sutro Group):

> Given random {-1, +1} inputs and a label that's the product of *k*
> secret bits, find those bits. The catch: we measure not just accuracy
> and speed, but **data movement** — how much energy your solution
> costs at the hardware level.

A neural-network SGD baseline costs ~1.3M DMD; the best known algebraic
solver (Sequential Elimination) is at ~19K. The challenge's leaderboard
fixes the configuration; here we want the opposite — find the
configuration whose optimal-solver cost lines up with our matmul
references, so the two problems can be compared head-to-head on the same
ruler.

## Problem

Given a matrix `X ∈ {-1, +1}^(m × n)` and labels `y ∈ {-1, +1}^m` with

    y[i] = ∏_{j ∈ S} X[i, j]

for some unknown `S ⊆ {0,…,n-1}` of size `k`, recover `S`.

The user-flagged starting point is **`k = 3` secret (known-class) bits
inside `n - k = 8` noise bits, so `n = 11`.**

## Cost model

Same Dally-style read accounting as `../matmul`:

    cost(addr) = ⌈√addr⌉

with the (X, y) buffers laid out linearly:

    X bulk at addresses 1..m·n      (row-major)
    y bulk at addresses m·n+1..m·n+m

So totals here are directly comparable to matmul costs
(baseline_4x4 = 1,316; baseline_16x16 = 340,704; tiled_16x16 = 133,783).

Only reads of the input buffers are charged. Scratchpad/GF(2) work is
treated as free for now; this is a WIP and that part of the model can
be tightened up once we actually want to write IRs.

## API

```python
from sparse_parity import Cost, generate, solve_bruteforce, solve_gf2

X, y, secret = generate(n_bits=11, k_sparse=3, n_samples=12, seed=0)
cost = Cost(m=12, n=11)
found = solve_gf2(X, y, k=3, cost=cost)
assert sorted(found) == secret
print(cost.read_cost)   # → ~1,222
```

Two reference solvers (both reach 100% across 5 seeds at every config below):

- `solve_bruteforce` — enumerate all `C(n, k)` subsets, validate against
  every sample with early termination on mismatch. Cost dominated by
  `X` reads inside the inner loop.
- `solve_gf2` — map +1/−1 to 0/1, solve `A·s = b (mod 2)` by Gaussian
  elimination on the loaded GF(2) matrix. Reads `X` and `y` once each
  during the load phase.

## Exploration

`python3 explore.py` (5 seeds per config, all hit 100% accuracy):

| label        |  n |  k |  m |     bruteforce |        gf2 |
|--------------|---:|---:|---:|---------------:|-----------:|
| user-target  | 11 |  3 | 12 |          2,959 |      1,222 |
| tiny         |  8 |  2 |  8 |            750 |        444 |
| medium       | 16 |  3 | 16 |         22,006 |      3,128 |
| medium-k4    | 20 |  4 | 24 |        184,771 |      7,797 |
| large        | 32 |  3 | 32 |        208,362 |     23,408 |

### Where this lands vs. matmul

| matmul reference  |     cost | sparse-parity match (≈ same cost)               |
|-------------------|---------:|-------------------------------------------------|
| `baseline_4x4`    |    1,316 | **`user-target` GF(2)**: 1,222 (n=11, k=3, m=12) |
| `tiled_16x16`     |  133,783 | **`medium-k4` brute force**: 184,771 (n=20, k=4, m=24) |
| `baseline_16x16`  |  340,704 | between `medium-k4` and `large` brute force     |

So the user-suggested 3-known/8-unknown shape (n=11, k=3) is genuinely
a "4×4 matmul"-difficulty problem under the GF(2) solver — and a
brute-force variant at (n=20, k=4) lands in the 16×16 matmul range.

## Open questions / next steps

1. **Tighten the cost model.** Charge GF(2) elimination work too, by
   either writing an IR over the same v0 instruction set as `../matmul`
   or by accounting for scratchpad reads with a cheap fixed cost.
2. **Pick a target config.** Do we want one record table (one `(n,k,m)`)
   or a small ladder of difficulties matching matmul's 4×4 / 16×16
   split? The exploration above suggests `(11, 3, 12)` and `(20, 4, 24)`
   are natural anchors.
3. **Better solvers.** The challenge leaderboard has solvers (KM
   influence estimation, sequential elimination) far below GF(2)
   Gaussian elimination on raw DMD — once the cost model is tightened
   we should port them in to set the record-history floor.
