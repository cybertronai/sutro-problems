# energy-efficient-matmul

- DeepMind's AlphaTensor https://github.com/google-deepmind/alphatensor discover a better 4x4 matrix multiplication algorithm in terms of arithmetic operations. 

- What is best algorithm when we care about *energy* rather than *FLOP count*?

- For energy, use simplified version of Bill Dally's proposed *Parallel Explicit Communication Model* [cybertronai/simplified-dally-model](https://github.com/cybertronai/simplified-dally-model).


## API

```python
from matmul import (
    score_1x1, score_4x4, score_16x16,
    generate_baseline_4x4, generate_baseline_16x16,
    generate_tiled_16x16,
)

# Verify your IR computes A @ B correctly and return its read-cost.
cost = matmul.score_1x1("1,2;mul 3,1,2;3")    # 5  (1+2 + 2)
cost = matmul.score_4x4(my_ir_text)
cost = matmul.score_16x16(my_ir_text)
```

### Baselines

```python
ir = generate_baseline_4x4()      # naive triple loop, 4×4
ir = generate_baseline_16x16()    # naive triple loop, 16×16
ir = generate_tiled_16x16()       # 4×4 scratchpad-cached tiles
```

## Current best results

| matrices | algorithm                           | cost     |
|----------|-------------------------------------|---------:|
| 4×4      | `generate_baseline_4x4` (naive)     |   1,316  |
| 16×16    | `generate_baseline_16x16` (naive)   | 340,704  |
| 16×16    | `generate_tiled_16x16` (4×4 tiles)  | 133,783  |
