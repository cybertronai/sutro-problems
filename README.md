[View this project on GitHub ↗](https://github.com/cybertronai/sutro-problems)

# Sutro problems

A collection of small, self-contained problems used as benchmarks for the [Sutro Group](https://docs.google.com/document/d/1B9867EN6Bg4ZVQK9vI_ZqykZ5HEtMAHJ7zBGGas4szQ/edit?tab=t.0#heading=h.j6rssh3enbtd)'s energy-efficient learning research.

## Problems

- [`matmul/`](matmul/) — 4x4 and 16x16 matmul
- [`sparse-parity/`](sparse-parity/) — approximate sparse parity: recover the k secret bit positions at the lowest energy
- [`mnist/`](mnist/) — learn from labeled images and predict test digits: 3×3 (600/600), 9×9 (6,000/6,000), and classic 28×28 (60,000/10,000), with reference results and W&B curves
