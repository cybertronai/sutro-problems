# Sparse parity

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

- Given some labeled examples of sparse parity, and 8x more unlabeled ones.
- What is the most energy-efficient way to label them?
- To measure energy, use simplified version of Bill Dally's [model](https://github.com/cybertronai/simplified-dally-model), v3 [instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets), 8-bit scalars

## API

```python
import sparse_parity

# Verify your IR predicts y_test correctly and return its read-cost.
ir   = sparse_parity.generate_baseline_small()   # small
cost = sparse_parity.score_small(ir)             # → 6,918

ir   = sparse_parity.generate_baseline_medium()  # medium
cost = sparse_parity.score_medium(ir)            # → 816,251
```

## Small Record History

2 hidden bits, 3 total bits, 4 train examples, 32 test.

| Date       | Cost   | Time   | Submission                                                                   | Contributors                                 | Description                                      |
| -          | -:     | -:     | -                                                                            | -                                            | -                                                |
| 2026-05-07 |  6,918 | 4.2 ms | [ir](submissions/baseline_small.ir), [report](submissions/baseline_small.md) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_baseline_small` (try-each-candidate)   |
| 2026-05-08 |  1,932 | 3.8 ms | [ir](submissions/small_pack_best.ir), [report](submissions/small_pack_report.md), [py](submissions/small_pack_generator.py) | [@sjbaebae](https://github.com/sjbaebae) | low-address row decoder + scheduled output/test aliasing ★ best |
| 2026-05-08 | 22,238 | 6.7 ms | [ir](submissions/ge_small.ir), [report](submissions/ge_small.md), [py](submissions/ge_small.py) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_ge_small` (GF(2) Gaussian elimination) |

## Medium Record History

3 hidden bits, 8 total bits, 8 train examples, 64 test.

| Date       | Cost    | Time  | Submission                                                                     | Contributors                                 | Description                                       |
| -          | -:      | -:    | -                                                                              | -                                            | -                                                 |
| 2026-05-07 | 816,251 | 27 ms | [ir](submissions/baseline_medium.ir), [report](submissions/baseline_medium.md) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_baseline_medium` (try-each-candidate)   |
| 2026-05-08 | 473,046 | 13 ms | [ir](submissions/ge_medium.ir), [report](submissions/ge_medium.md), [py](submissions/ge_medium.py) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_ge_medium` (GF(2) Gaussian elimination) |
| 2026-05-08 |  16,084 | 5.6 ms | [ir](submissions/ge_medium_packed.ir), [report](submissions/ge_medium_packed.md), [py](submissions/ge_medium_packed.py) | [@sjbaebae](https://github.com/sjbaebae) | packed-column candidate check |
| 2026-05-08 |  15,960 | 5.5 ms | [ir](submissions/predpack_medium.ir), [report](submissions/predpack_medium.md), [py](submissions/predpack_medium.py) | [@sjbaebae](https://github.com/sjbaebae) | packed-column decoder + pair-XOR reuse |
| 2026-05-08 |  15,691 | 5.4 ms | [ir](submissions/predpack_tuned_medium.ir), [report](submissions/predpack_tuned.md), [py](submissions/predpack_tuned.py) | [@sjbaebae](https://github.com/sjbaebae) | pair-XOR reuse + address/liveness tuning ★ best |

[access_distance](doc/access_distance/) — per-submission read-distance histogram + CDF for every IR above.
