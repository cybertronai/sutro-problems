# Symmetry

$$
\underbrace{x_0\; x_1\; x_2\; x_3\; x_4\; x_5}_{\text{6-bit pattern}}
\;\xrightarrow{\text{palindrome?}}\;
\begin{cases}1 & x_0{=}x_5,\; x_1{=}x_4,\; x_2{=}x_3 \\ 0 & \text{otherwise}\end{cases}
$$

- Given a 6-bit pattern, classify it as a palindrome (1) or not (0).
- Exactly 8 of the 64 possible patterns are palindromes.
- What is the most energy-efficient way to implement this classifier?
- To measure energy, use the simplified version of Bill Dally's [model](https://github.com/cybertronai/simplified-dally-model), v3 [instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets), 8-bits.

This is the target classification task from Rumelhart, Hinton & Williams (1986),
where a 6-2-1 MLP was trained to detect 6-bit palindromes.  The function has
a degree-3 polynomial form over ±1 inputs

$$
f(x) = \frac{(1 + x_0 x_5)(1 + x_1 x_4)(1 + x_2 x_3)}{8}
$$

requiring only multiplication and addition — no sigmoid needed.

## API

```python
import symmetry

# Verify your IR classifies all 64 patterns correctly and return its cost.
ir   = symmetry.generate_baseline_six()   # 6-bit (canonical)
cost = symmetry.score_six(ir)             # → 20

ir   = symmetry.generate_baseline_eight() # 8-bit (harder)
cost = symmetry.score_eight(ir)           # → 29
```

The IR receives `n_bits` inputs (0 or 1) at addresses `1..n_bits` and must
produce exactly one output (0 or 1).  It is tested against every possible
input pattern — all 2^n_bits of them.

## Six-bit, 100% target (canonical RHW1986 task)

3 mirror pairs, 8/64 palindromes.

| Date       | Cost | Time  | Submission             | Contributors | Description            |
| -          | -:   | -:    | -                      | -            | -                      |
| 2026-05-11 |   20 |       | [ir](submissions/baseline_six.ir) | baseline | `generate_baseline_six` (3× cmp eq + 2× and) |

## Eight-bit, 100% target

4 mirror pairs, 16/256 palindromes.

| Date       | Cost | Time  | Submission             | Contributors | Description             |
| -          | -:   | -:    | -                      | -            | -                       |
| 2026-05-11 |   29 |       | [ir](submissions/baseline_eight.ir) | baseline | `generate_baseline_eight` (4× cmp eq + 3× and) |

## Background

The symmetry problem is the canonical example from §3 of Rumelhart, Hinton &
Williams (1986) *Learning representations by back-propagating errors* (Nature
323, 533–536).  A 6-2-1 MLP with sigmoid units was trained to detect 6-bit
palindromes; the 2 hidden units reliably learn to encode 3-bit parity over the
three mirror pairs.

The polynomial identity above shows the function is learnable by a degree-3
circuit using only mul and add — the `mul`/`add` subset of the Dally v3 IR.
This is what makes it a natural benchmark: the optimal IR solution may look
nothing like a trained MLP.

The companion ByteDMD analysis (hinton-problems `v2-bytedmd/`) measures the
data-movement cost of *learning* symmetry with gradient descent vs alternatives,
while this problem measures the cost of *computing* the target function once
it is known.
