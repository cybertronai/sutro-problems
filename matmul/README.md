# Matmul

- DeepMind's [AlphaTensor](https://github.com/google-deepmind/alphatensor) discover a better 4x4 matrix multiplication algorithm in terms of FLOPs. 
- What is the best algorithm when we care about *energy* instead?
- To measure energy, use [simplified version](https://github.com/cybertronai/simplified-dally-model) of Bill Dally's *Parallel Explicit Communication Model*

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

Submissions live under `ir/` as standalone scripts (one per IR file).
Run e.g. `python matmul/ir/outer_product_4x4.py` to regenerate
`ir/outer_product_4x4.ir` and verify its cost.


## 4×4 Record History

| date       | method                                        | IR                                                    | cost  |
|------------|-----------------------------------------------|-------------------------------------------------------|------:|
| 2026-04-29 | `generate_baseline_4x4` (naive)               | [`ir/baseline_4x4.ir`](ir/baseline_4x4.ir)            | 1,316 |
| 2026-04-29 | `generate_outer_product_4x4` (size-1 sA)      | [`ir/outer_product_4x4.ir`](ir/outer_product_4x4.ir)  |   800 |

## 16×16 Record History

| date       | method                                        | IR                                                        | cost    |
|------------|-----------------------------------------------|-----------------------------------------------------------|--------:|
| 2026-04-29 | `generate_baseline_16x16` (naive)             | [`ir/baseline_16x16.ir`](ir/baseline_16x16.ir)            | 340,704 |
| 2026-04-29 | `generate_tiled_16x16` (4×4 tiles)            | [`ir/tiled_16x16.ir`](ir/tiled_16x16.ir)                  | 133,783 |
| 2026-04-29 | `generate_hierarchical_16x16` (asym. reload)  | [`ir/hierarchical_16x16.ir`](ir/hierarchical_16x16.ir)    |  80,217 |
