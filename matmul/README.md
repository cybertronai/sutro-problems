# Matmul

- DeepMind's [AlphaTensor](https://github.com/google-deepmind/alphatensor) discover a better 4x4 matrix multiplication algorithm in terms of FLOPs. 
- What is the best algorithm when we care about *energy* instead?
- To measure energy, use [simplified version](https://github.com/cybertronai/simplified-dally-model) of Bill Dally's *Parallel Explicit Communication Model* (v0 instruction set)

## API

```python
import matmul

# Verify your IR computes A @ B correctly and return its read-cost.
cost = matmul.score_1x1("1,2;mul 3,1,2;3")    # 5

ir = matmul.generate_baseline_4x4()      # naive triple loop, 4×4
cost = matmul.score_4x4(ir)

ir = matmul.generate_baseline_16x16()    # naive triple loop, 16×16
cost = matmul.score_16x16(ir)

ir = matmul.generate_tiled_16x16()       # 4×4 scratchpad-cached tiles
cost = matmul.score_16x16(ir)
```

Submissions go into subdirectories, files at top-level shouldn't be modified except for adding entries to tables below --

## 4×4 Record History

| #  | Cost  | Date       | Submission                                          | Contributors                                 | Description                              |
| -  | -:    | -          | -                                                   | -                                            | -                                        |
| 1  | 1,316 | 2026-04-29 | [ir](submissions/baseline_4x4.ir), report       | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_baseline_4x4` (naive)          |
| 2  |   800 | 2026-04-30 | [ir](submissions/outer_product_4x4.ir), report  | [@sjbaebae](https://github.com/sjbaebae)     | `generate_outer_product_4x4` (size-1 sA) |

## 16×16 Record History

| #  | Cost    | Date       | Submission                                          | Contributors                                 | Description                                   |
| -  | -:      | -          | -                                                   | -                                            | -                                             |
| 1  | 340,704 | 2026-04-29 | [ir](submissions/baseline_16x16.ir), report     | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_baseline_16x16` (naive)             |
| 2  | 133,783 | 2026-04-29 | [ir](submissions/tiled_16x16.ir), report        | [@yaroslavvb](https://github.com/yaroslavvb) | `generate_tiled_16x16` (4×4 tiles)            |
| 3  | 110,743 | 2026-04-30 | [ir](submissions/tiled_16x16_opt1.ir), report   | [@SethTS](https://github.com/SethTS)         | `generate_tiled_16x16_opt1` (tmp@1)           |
| 4  |  80,217 | 2026-04-30 | [ir](submissions/hierarchical_16x16.ir), report | [@sjbaebae](https://github.com/sjbaebae)     | `generate_hierarchical_16x16` (asym. reload)  |
| 5  |  73,602 | 2026-04-30 | [ir](submissions/sa_cache_16x16.ir), report     | [@adotzh](https://github.com/adotzh)         | sA-cache + sB scratchpad (rank2) ★ best      |
