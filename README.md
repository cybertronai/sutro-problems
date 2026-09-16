# Sutro problems

A collection of small, self-contained problems used as benchmarks for the [Sutro Group](https://docs.google.com/document/d/1B9867EN6Bg4ZVQK9vI_ZqykZ5HEtMAHJ7zBGGas4szQ/edit?tab=t.0#heading=h.j6rssh3enbtd)'s energy-efficient learning research.

## Problems

- [`matmul/`](matmul/) — 4x4 and 16x16 matmul
- [`sparse-parity/`](sparse-parity/) — approximate sparse parity: recover the k secret bit positions at the lowest energy
- [`mnist/`](mnist/) — learn from labeled images and predict test digits

## Contributing

`main` holds only the problems and their record submissions. Open a pull request
against `main` only for a result that qualifies for a leaderboard table.
Experiments, prototypes, harness proposals, and unfinished or non-record results
belong on other branches; push them there and link them from an issue or a
pull request against that branch instead of merging them into `main`.
