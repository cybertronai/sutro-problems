# sutro-problems

A collection of small, self-contained problems used as benchmarks for the Sutro Group's energy-efficient learning research.

## Structure

Each problem lives in **its own folder**, with **its own README** describing: the problem, reference implementation, history of records.

To add a new problem, create a new directory and add a `README.md` at its root.

## Problems

- [`matmul/`](matmul/) — cheapest matrix multiplication under a
  simplified Dally explicit-communication cost model (data movement
  priced; arithmetic free). Records: 4×4 naive 1,316; 16×16 naive
  340,704; 16×16 tiled 133,783.
  
- [`wip-boltzmann-shifter/`](wip-boltzmann-shifter/) — the shift-direction
  inference task from Hinton & Sejnowski's Boltzmann-machine chapter
  (PDP Vol 1, 1986). _(work in progress)_
