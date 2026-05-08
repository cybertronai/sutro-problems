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
ir   = sparse_parity.generate_baseline()       # naive try-each-candidate
cost = sparse_parity.score_sparse_parity(ir)   # → 6,918
```

## Small Record History

| Date       | Cost  | Submission                                                       | Contributors                                 | Description                                       |
| -          | -:    | -                                                                | -                                            | -                                                 |
| 2026-05-07 | 6,918 | [ir](submissions/baseline.ir), [report](submissions/baseline.md) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_baseline` (try-each-candidate, v2 ops)  |

## Medium Record History

| Date | Cost | Submission | Contributors | Description |
| -    | -:   | -          | -            | -           |
