[View this project on GitHub ↗](https://github.com/cybertronai/sutro-problems/tree/main/mnist-a100)

# MNIST on an A100

- Learn to read digits from 10,000 labelled 9x9 MNIST images, then label 10,000 more, from scratch, on one A100.
- Every call runs in a fresh, sandboxed process on a fresh draw, whitened and secretly rotated.
- Ranked by time per call. Energy per call above idle is reported beside it ([how](energy/README.md)).

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

## Difficulty = 1 (error ≤ 5.40%)

| Date | mJ | ms | Submission | Contributors | Description |
| - | -: | -: | - | - | - |
| 2026-09-25 | 29,662 | 700.1 | [py](example.py), [report](energy/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | `example.py` baseline: MLP 60-1024-1024-10, 400 steps |
| 2026-09-26 | 2,648 | 191.3 | [py](energy/entries/mlp_k1_w1024_s100_b512.py), [report](energy/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | MLP 60-1024-1024-10, 100 steps |
| 2026-09-26 | 2,131 | 61.7 | [py](submissions/graph-mlp-20260926/fast_mlp.py), [report](submissions/graph-mlp-20260926/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | MLP 60-256-256-10, 200 steps in a CUDA graph ★ best |

## Difficulty = 2 (error ≤ 3.40%)

| Date | mJ | ms | Submission | Contributors | Description |
| - | -: | -: | - | - | - |
| 2026-09-26 | 200,870 | 927.3 | [py](energy/entries/mlp_k16_w1024_s400_b512.py), [report](energy/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | 16 MLPs 60-1024-1024-10, 400 steps |
| 2026-09-26 | 14,436 | 252.1 | [py](energy/entries/mlpg_k4_w256_s800_b512.py), [report](energy/README.md) | [@yaroslavvb](https://github.com/yaroslavvb) | 4 MLPs 60-256-256-10, 800 steps in a CUDA graph ★ best |

## Difficulty = 3 (error ≤ 2.70%)

| Date | mJ | ms | Submission | Contributors | Description |
| - | -: | -: | - | - | - |
| 2026-09-28 | 1,159,909 | 15,818.8 | [py](submissions/ladder-graph-20260928/ladder.py), [report](submissions/ladder-graph-20260928/README.md) | [@SethTS](https://github.com/SethTS) | Ladder 60-1000-500-250-250-250-10, 5,000 steps in a CUDA graph ★ best |

## Difficulty = 4 (error ≤ 2.30%)

No entry yet.

## Difficulty = 5 (error ≤ 1.90%)

No entry yet.
