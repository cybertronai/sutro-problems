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


## 4×4 Record History

| #  | Cost  | Description                         | Date       | IR                                          | Contributors |
| -  | -:    | -                                   | -          | -                                           | -            |
| 1  | 1,316 | `generate_baseline_4x4` (naive)     | 2026-04-29 | [`ir/baseline_4x4.ir`](ir/baseline_4x4.ir)  | [@yaroslavvb](https://github.com/yaroslavvb) |

## 16×16 Record History

| #  | Cost    | Description                         | Date       | IR                                                 | Contributors |
| -  | -:      | -                                   | -          | -                                                  | -            |
| 1  | 340,704 | `generate_baseline_16x16` (naive)   | 2026-04-29 | [`ir/baseline_16x16.ir`](ir/baseline_16x16.ir)     | [@yaroslavvb](https://github.com/yaroslavvb) |
| 2  | 133,783 | `generate_tiled_16x16` (4×4 tiles)  | 2026-04-29 | [`ir/tiled_16x16.ir`](ir/tiled_16x16.ir)           | [@yaroslavvb](https://github.com/yaroslavvb) |
| 3  | 110,743 | `generate_tiled_16x16_opt1` (tmp@1) | 2026-04-30 | [`ir/tiled_16x16_opt1.ir`](ir/tiled_16x16_opt1.ir) | [@SethTS](https://github.com/SethTS) |
