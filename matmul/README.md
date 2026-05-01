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

| #  | Cost  | Description                              | Date       | IR                                                    | Contributors |
| -  | -:    | -                                        | -          | -                                                     | -            |
| 1  | 1,316 | `generate_baseline_4x4` (naive)          | 2026-04-29 | [`ir/baseline_4x4.ir`](ir/baseline_4x4.ir)            | [@yaroslavvb](https://github.com/yaroslavvb) |
| 2  |   800 | `generate_outer_product_4x4` (size-1 sA) | 2026-04-30 | [`ir/outer_product_4x4.ir`](ir/outer_product_4x4.ir)  | [@sjbaebae](https://github.com/sjbaebae) |

## 16×16 Record History

| #  | Cost    | Description                                   | Date       | IR                                                                                             | Contributors |
| -  | -:      | -                                             | -          | -                                                                                              | -            |
| 1  | 340,704 | `generate_baseline_16x16` (naive)             | 2026-04-29 | [`ir/baseline_16x16.ir`](ir/baseline_16x16.ir)                                                | [@yaroslavvb](https://github.com/yaroslavvb) |
| 2  | 133,783 | `generate_tiled_16x16` (4×4 tiles)            | 2026-04-29 | [`ir/tiled_16x16.ir`](ir/tiled_16x16.ir)                                                      | [@yaroslavvb](https://github.com/yaroslavvb) |
| 3  | 110,743 | `generate_tiled_16x16_opt1` (tmp@1)           | 2026-04-30 | [`ir/tiled_16x16_opt1.ir`](ir/tiled_16x16_opt1.ir)                                            | [@SethTS](https://github.com/SethTS) |
| 4  |  80,217 | `generate_hierarchical_16x16` (asym. reload)  | 2026-04-30 | [`ir/hierarchical_16x16.ir`](ir/hierarchical_16x16.ir)                                        | [@sjbaebae](https://github.com/sjbaebae) |
| 5  |  73,858 | sA-cache + sB scratchpad (rank2)              | 2026-04-30 | [`ir/new_record_sa_cache_73858.ir`](ir/new_record_sa_cache_73858.ir)                           | [@adotzh](https://github.com/adotzh) |
| 6  |  73,602 | sA-cache + sB scratchpad (rank2) ★ best      | 2026-04-30 | [`ir/new_record_sa_cache_73602.ir`](ir/new_record_sa_cache_73602.ir)                           | [@adotzh](https://github.com/adotzh) |

## Exploration Summary

The agent ran a multi-direction search session on 2026-04-30. Below are all directions tried and their outcomes.

### Completed Directions

| direction | best score | outcome |
|-----------|----------:|---------|
| Non-square tiles (Ti=8, Tj=4, Tk=1) | 82,477 | Superseded by sA-cache approach |
| **sA-cache single-cell + sB scratchpad** | **73,602** | **Current record. Cache A[ii,bk] at addr 1 (cost 1) before inner jj loop. sB at 3–6, sC at 7–38. Analytically proven optimal for this tiling family.** |
| A-tile reuse across bj (loop reorder bi>bk>bj) | 147,365 | Expensive bulk-C reload kills savings from fewer A copies |
| Strassen 2×2 within tiles (M-values at addr 50+) | 196,026 | M-values at addr 50+ (cost 8) too expensive to read repeatedly |
| Strassen with M-values at addr 1–7 | 169,174 | Better than naive Strassen but still worse than tiling record |

### Pending Directions (not yet attempted)

| direction | description |
|-----------|-------------|
| Column streaming (Ti=16, Tj=1) | Cache A[:,k] at addr 2–17; load B[k,j] to addr 1; each A value reused 16× at cost 2–5 |
| Three-level cascade (16×16 → 4×4 → 2×2) | Innermost 2×2 at addr 1–4 (cost 1–2), loaded from 4×4 mid-scratchpad |
| Algebraic pre-summation | Pre-compute pairwise sums/differences (free adds) to reduce distinct source reads before muls |

### Key Insights

- **Address cost model**: `cost(addr) = floor(sqrt(addr-1)) + 1`. Addr 1 costs 1, addr 2–4 cost 2, addr 5–9 cost 3, etc. Keeping hot values at the lowest addresses is critical.
- **Additions are free**; only reads from memory (copy/mul operands) incur cost.
- **sA-cache trick**: caching `A[ii,bk]` at addr 1 (cost 1) before the inner `jj` loop reduces per-multiplication A-read cost from 5–17 to 1. This is the dominant optimization.
- **Strassen disappoints**: The 7-multiplication benefit is outweighed by the cost of reading shared M-values from higher addresses multiple times.
- **73,602 appears to be optimal** for any parametric tiling framework — exhaustive analysis of address assignments and loop structures confirms no further improvement is possible within these algorithmic families.
