# Sutro problems

A collection of small, self-contained problems used as benchmarks for the Sutro Group's energy-efficient learning research.

## Structure

Each problem lives in **its own folder**, with **its own README** describing: the problem, reference implementation, history of records.

To add a new problem, create a new directory and add a `README.md` at its root.

## Problems

- [`matmul/`](matmul/) — cheapest matrix multiplication under a
  simplified Dally explicit-communication cost model
  
- [`wip-boltzmann-shifter/`](wip-boltzmann-shifter/) — the shift-direction
  inference task from Hinton & Sejnowski's Boltzmann-machine chapter
  (PDP Vol 1, 1986). _(work in progress)_

- [`wip-sparse-parity/`](wip-sparse-parity/) — recover the *k* secret bits
  whose product equals the label, from random {-1,+1} samples. Frames the
  [Sparse Parity Challenge](https://github.com/cybertronai/sparse-parity-challenge)
  on the same Dally-style read-cost ruler as `matmul/`. _(work in progress)_
