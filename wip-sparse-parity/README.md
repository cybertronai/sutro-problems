# Sparse parity

- The [Sparse Parity Challenge](https://github.com/cybertronai/sparse-parity-challenge) (Sutro Group) asks: given random `{0, 1}` inputs and a label that's the XOR of *k* secret bits, find those bits.
- Here we fix a tiny version: **n = 3 bits (2 secret + 1 noise), k = 2, m = 4 train, 32 test**. The training set is constructed (via rejection sampling) so that **exactly one** 2-subset of bit positions is consistent with the training labels — so 100% test accuracy is reachable by any solver that finds that subset.
- To measure energy, use [simplified version](https://github.com/cybertronai/simplified-dally-model) of Bill Dally's *Parallel Explicit Communication Model*, with the [v2 instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets/v2) (`add`, `sub`, `mul`, `copy`, `and`, `or`, `xor`, `not`, `set`).
- The scorer is **robust to different inputs**: it runs the IR against canonical instances covering every possible secret subset, so a predictor that hard-codes the secret for one seed cannot pass.

## Illustration

A toy example with `m_train = 2` and `m_test = 2` (the actual problem has 4 train rows and 32 test rows; predictions are the XOR of two secret columns):

$$
\begin{array}{cl|c}
& \text{bits} & \text{sparse parity} \\
\text{train} & \left\lbrace \begin{array}{ccccc} 1 & 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 & 1 \end{array} \right. & \begin{array}{c} 1 \\ 0 \end{array} \\
\\
\begin{array}{c} \text{test} \\ \text{8x larger} \end{array} & \left\lbrace \begin{array}{ccccc} 1 & 0 & 0 & 0 & 1 \\ 1 & 1 & 1 & 0 & 0 \end{array} \right. & \begin{array}{c} ? \\ ? \end{array}
\end{array}
$$

## API

```python
import sparse_parity

# Verify your IR predicts y_test correctly for every canonical secret
# and return its (data-independent) read-cost.
ir = sparse_parity.generate_baseline()      # try-each-candidate, v2 xor/and/or/set
cost = sparse_parity.score_sparse_parity(ir)
```

## Record History

| Date       | Cost  | Submission                                      | Contributors                                 | Description                                       |
| -          | -:    | -                                               | -                                            | -                                                 |
| 2026-05-07 | 6,918 | [ir](submissions/baseline.ir), [report](submissions/baseline.md) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_baseline` (try-each-candidate, v2 ops)  |
