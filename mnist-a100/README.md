[View this project on GitHub ↗](https://github.com/cybertronai/sutro-problems/tree/main/mnist-a100)

# MNIST end-to-end on an A100

Architecture + optimizer + kernel codesign

- Learn to read digits from 10,000 labelled 9x9 MNIST images, then label 10,000 more, from scratch, on one A100.
- Every call runs in a fresh, sandboxed process on a fresh draw, whitened and secretly rotated.
- Ranked by time per call. Energy per call above idle is reported beside it ([how](energy/README.md)).

<img width="1084" height="392" alt="Screenshot 2026-09-29 at 5 14 36 PM" src="https://github.com/user-attachments/assets/51940abe-372c-4434-8003-d61cc1b4733c" />


## API

```python
import mnist

def my_method(train_x, train_y, test_x):
    # train_x (10000, 60) float32, train_y (10000,) int64 in 0..9, test_x (10000, 60) float32, on the GPU
    ...
    return labels  # (10000,) integer tensor on the GPU

if __name__ == "__main__":
    ms = mnist.score(my_method, difficulty=1)  # 1 (loosest) to 5; ms per call, or raises mnist.Disqualified
```

```bash
python example.py                                            # on your own GPU
python run_modal.py example.py:mlp --difficulty 1 --runs 3   # official: three Modal A100-80GB runs, median
```

The rules are in [`mnist.py`](mnist.py)'s docstring.

Each row is a separate submission by the listed entrant; source attribution is
in its report. The new variants below have one verification run each and are
provisional. ★ identifies an established record.

## Difficulty = 1 (error ≤ 5.40%)

| Date | mJ | ms | Submission | Entrant | Description |
| - | -: | -: | - | - | - |
| 2026-09-25 | 29,662 | 700.1 | [py](example.py), [report](energy/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | `example.py` baseline: MLP 60-1024-1024-10, 400 steps |
| 2026-09-26 | 2,648 | 191.3 | [py](energy/entries/mlp_k1_w1024_s100_b512.py), [report](energy/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | MLP 60-1024-1024-10, 100 steps |
| 2026-09-26 | 2,131 | 61.7 | [py](submissions/graph-mlp-20260926/fast_mlp.py), [report](submissions/graph-mlp-20260926/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | MLP 60-256-256-10, 200 steps in a CUDA graph ★ best |
| 2026-09-30 | 3,634 | 351.0 | [py](submissions/improved-20260930/mlp_d1.py), [report](submissions/improved-20260930/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | MLP, 200 steps, TF32 and fused AdamW; one sandboxed run |

## Difficulty = 2 (error ≤ 3.40%)

| Date | mJ | ms | Submission | Entrant | Description |
| - | -: | -: | - | - | - |
| 2026-09-26 | 200,870 | 927.3 | [py](energy/entries/mlp_k16_w1024_s400_b512.py), [report](energy/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | 16 MLPs 60-1024-1024-10, 400 steps |
| 2026-09-26 | 14,436 | 252.1 | [py](energy/entries/mlpg_k4_w256_s800_b512.py), [report](energy/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | 4 MLPs 60-256-256-10, 800 steps in a CUDA graph |
| 2026-09-29 | 8,071 | 38.3 | [py](submissions/kernel-pcg-20260929/kernel_pcg.py), [report](submissions/kernel-pcg-20260929/README.md) | [@islamborghini](https://github.com/islamborghini) | Non-neural RBF kernel ridge, one learned metric update, 16-step Nyström-preconditioned CG ★ best |
| 2026-09-30 | 132,635 | 761.9 | [py](submissions/improved-20260930/mlp_d2.py), [report](submissions/improved-20260930/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | 8 MLPs, 500 steps, EMA 0.992; one sandboxed run |

## Difficulty = 3 (error ≤ 2.70%)

| Date | mJ | ms | Submission | Entrant | Description |
| - | -: | -: | - | - | - |
| 2026-09-28 | 1,159,909 | 15,818.8 | [py](submissions/ladder-graph-20260928/ladder.py), [report](submissions/ladder-graph-20260928/README.md) | [@SethTS](https://github.com/SethTS) | Ladder 60-1000-500-250-250-250-10, 5,000 steps in a CUDA graph |
| 2026-09-28 | 676,850 | 6,556.9 | [py](submissions/ladder-triton-20260928/ladder_d3.py), [report](submissions/ladder-triton-20260928/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, 6,500 steps, fused Triton kernels in a CUDA graph |
| 2026-09-29 | 259,847 | 1,895.4 | [py](submissions/ladder-fast-20260929/ladder_d3.py), [report](submissions/ladder-fast-20260929/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, batch 1,000, 1,200 steps, TF32, fused Triton kernels in a CUDA graph |
| 2026-09-29 | 201,792 | 1,443.3 | [py](submissions/ladder-sampled-20260929/ladder_d3.py), [report](submissions/ladder-sampled-20260929/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, batch 1,000, 900 steps, labelled rows sampled by loss, TF32, fused Triton kernels in a CUDA graph ★ best |
| 2026-09-30 | 253,761 | 1,732.2 | [py](submissions/improved-20260930/ladder_d3.py), [report](submissions/improved-20260930/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | Ladder, 1,100 steps; 8.3% fewer updates; one sandboxed run |

## Difficulty = 4 (error ≤ 2.30%)

| Date | mJ | ms | Submission | Entrant | Description |
| - | -: | -: | - | - | - |
| 2026-09-28 | 1,143,255 | 11,094.6 | [py](submissions/ladder-triton-20260928/ladder_d4.py), [report](submissions/ladder-triton-20260928/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, 11,000 steps, fused Triton kernels in a CUDA graph |
| 2026-09-29 | 547,382 | 3,784.2 | [py](submissions/ladder-fast-20260929/ladder_d4.py), [report](submissions/ladder-fast-20260929/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, batch 1,000, 2,400 steps, TF32, fused Triton kernels in a CUDA graph |
| 2026-09-29 | 400,682 | 2,882.5 | [py](submissions/ladder-sampled-20260929/ladder_d4.py), [report](submissions/ladder-sampled-20260929/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, batch 1,000, 1,800 steps, labelled rows sampled by loss, TF32, fused Triton kernels in a CUDA graph ★ best |
| 2026-09-30 | 460,763 | 3,481.3 | [py](submissions/improved-20260930/ladder_d4.py), [report](submissions/improved-20260930/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | Ladder, 2,200 steps; 8.3% fewer updates; one sandboxed run |

## Difficulty = 5 (error ≤ 1.90%)

| Date | mJ | ms | Submission | Entrant | Description |
| - | -: | -: | - | - | - |
| 2026-09-28 | 4,410,730 | 36,201.2 | [py](submissions/ladder-triton-20260928/ladder_d5.py), [report](submissions/ladder-triton-20260928/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, 36,000 steps, fused Triton kernels in a CUDA graph |
| 2026-09-29 | 2,059,576 | 14,130.6 | [py](submissions/ladder-fast-20260929/ladder_d5.py), [report](submissions/ladder-fast-20260929/README.md) | [@SethTS](https://github.com/SethTS) | Ladder, batch 1,000, 9,000 steps, TF32, fused Triton kernels in a CUDA graph ★ best |
| 2026-09-30 | 1,824,760 | 13,264.3 | [py](submissions/improved-20260930/ladder_d5.py), [report](submissions/improved-20260930/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | Ladder, 8,400 steps; 6.7% fewer updates; one sandboxed run |
