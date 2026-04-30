# Matmul

- DeepMind's AlphaTensor https://github.com/google-deepmind/alphatensor discover a better 4x4 matrix multiplication algorithm in terms of FLOPs. 
- What is the best algorithm when we care about *energy* rather than *FLOPs*?
- Use [simplified version](https://github.com/cybertronai/simplified-dally-model) of Bill Dally's *Parallel Explicit Communication Model* to measure energy.


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

| date       | method                              | IR                                          | cost  |
|------------|-------------------------------------|---------------------------------------------|------:|
| 2026-04-29 | `generate_baseline_4x4` (naive)     | [`ir/baseline_4x4.ir`](ir/baseline_4x4.ir)  | 1,316 |

## 16×16 Record History

| date       | method                              | IR                                              | cost    |
|------------|-------------------------------------|-------------------------------------------------|--------:|
| 2026-04-29 | `generate_baseline_16x16` (naive)   | [`ir/baseline_16x16.ir`](ir/baseline_16x16.ir)  | 340,704 |
| 2026-04-29 | `generate_tiled_16x16` (4×4 tiles)  | [`ir/tiled_16x16.ir`](ir/tiled_16x16.ir)        | 133,783 |

The tiled record is **2.55× cheaper** than the naive 16×16 baseline
despite issuing 32 % more instructions — the extra `mov` traffic
loading 4×4 A/B tiles into addrs 1..32 is more than paid back by the
inner `mul`/`add` reads now hitting distance-1..6 cells instead of
distance-15..23 cells. That's the energy-vs-FLOPs gap the problem is
designed to expose.

To beat 133,783 on 16×16, drop a `generate_<your_method>()` into
`matmul.py`, rerun `python matmul.py` to regenerate the IR file under
`ir/`, and add a row above.

## Files

| File | Purpose |
|------|---------|
| `matmul.py` | Scorer, parser, simulator, and the three baseline generators (`generate_baseline_4x4`, `generate_baseline_16x16`, `generate_tiled_16x16`). Run `python matmul.py` to regenerate the record-history IR files under `ir/`. |
| `ir/*.ir`   | The IR text emitted by each entry in the record-history tables; viewable inline on GitHub. |
| `README.md` | This file. |
