[View this project on GitHub ↗](https://github.com/cybertronai/sutro-problems)

# Sutro problems

A collection of small, self-contained problems used as benchmarks for the [Sutro Group](https://docs.google.com/document/d/1B9867EN6Bg4ZVQK9vI_ZqykZ5HEtMAHJ7zBGGas4szQ/edit?tab=t.0#heading=h.j6rssh3enbtd)'s energy-efficient learning research.

## Structure

Each problem lives in **its own folder**, with **its own README** describing: the problem, reference implementation, history of records.

To add a new problem, create a new directory and add a `README.md` at its root.

## Problems

- [`matmul/`](matmul/) — 4x4 and 16x16 matmul
- [`sparse-parity/`](sparse-parity/) — approximate sparse parity: recover the k secret bit positions at the lowest energy

## Design notes

- [Fixed or moving I/O order?](https://cybertronai.github.io/sutro-problems/docs/dataflow-io-order.html) — how data should enter a submission, which dataflow class the model already belongs to, and what fixing the I/O order costs and buys in relevance to buildable hardware
- [Sparse parity under the spatial cost model](https://cybertronai.github.io/sutro-problems/docs/spatial-model-analysis.html) — literature anchors, and the effect of distance-priced memory on the implemented decoders
