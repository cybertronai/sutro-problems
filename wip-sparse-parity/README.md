# Sparse parity

- Given some labeled examples of sparse parity, and 8x more unlabeled ones.
- What is the most energy-efficient way to label them?
- To measure energy, use [simplified version](https://github.com/cybertronai/simplified-dally-model) of Bill Dally's *Parallel Explicit Communication Model*

$$
\begin{array}{cl|c}
 & \text{bits} & \text{sparse parity} \\
\text{train} &
\left\lbrace \begin{array}{ccccc}
1 & 0 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 & 1
\end{array} \right. &
\begin{array}{c}
1 \\
0
\end{array} \\
\\
\begin{array}{c}
\text{test} \\
\text{8x larger}
\end{array} &
\left\lbrace \begin{array}{ccccc}
1 & 0 & 0 & 0 & 1 \\
1 & 1 & 1 & 0 & 0
\end{array} \right. &
\begin{array}{c}
? \\
?
\end{array}
\end{array}
$$

## API

```python
import sparse_parity

# Verify your IR predicts y_test correctly and return its read-cost.
ir   = sparse_parity.generate_baseline_small()   # n=3, k=2, 4 train / 32 test
cost = sparse_parity.score_small(ir)             # → 6,918

ir   = sparse_parity.generate_baseline_medium()  # n=8, k=3, 8 train / 64 test
cost = sparse_parity.score_medium(ir)            # → 816,251
```

## Small Record History

| Date       | Cost  | Submission                                                                   | Contributors                                 | Description                                              |
| -          | -:    | -                                                                            | -                                            | -                                                        |
| 2026-05-07 | 6,918 | [ir](submissions/baseline_small.ir), [report](submissions/baseline_small.md) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_baseline_small` (try-each-candidate, v2 ops)   |

## Medium Record History

| Date       | Cost    | Submission                                                                     | Contributors                                 | Description                                                |
| -          | -:      | -                                                                              | -                                            | -                                                          |
| 2026-05-07 | 816,251 | [ir](submissions/baseline_medium.ir), [report](submissions/baseline_medium.md) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_baseline_medium` (try-each-candidate, v2 ops)    |
