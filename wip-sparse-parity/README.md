# Sparse parity

- The [Sparse Parity Challenge](https://github.com/cybertronai/sparse-parity-challenge) (Sutro Group) asks: given random `{0, 1}` inputs and a label that's the XOR of *k* secret bits, find those bits.
- Here we fix a tiny version: **n = 3 bits (2 secret + 1 noise), k = 2, m = 5 train, 5 test**. The training set is constructed so that **exactly one** 2-subset of bit positions is consistent with the training labels — so 100% test accuracy is reachable by any solver that finds that subset.
- To measure energy, use [simplified version](https://github.com/cybertronai/simplified-dally-model) of Bill Dally's *Parallel Explicit Communication Model*, with the [v2 instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets/v2) (`add`, `sub`, `mul`, `copy`, `and`, `or`, `xor`, `not`, `set`).

## API

```python
import sparse_parity

# Verify your IR predicts y_test correctly and return its read-cost.
cost = sparse_parity.score_sparse_parity("1,2;xor 3,1,2;3")  # raises (35-input mismatch)

ir = sparse_parity.generate_baseline()      # naive, one xor per test row
cost = sparse_parity.score_sparse_parity(ir)
```

## Record History

| Date       | Cost | Submission                                      | Contributors                                 | Description                  |
| -          | -:   | -                                               | -                                            | -                            |
| 2026-05-07 |   91 | [ir](submissions/baseline.ir), [report](submissions/baseline.md) | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_baseline` (naive, v2 `xor`) |
