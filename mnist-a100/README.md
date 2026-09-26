[View this project on GitHub ↗](https://github.com/cybertronai/sutro-problems/tree/main/mnist-a100)

# MNIST on an A100

- Learn to read digits from 10,000 labelled 9x9 MNIST images, then label 10,000 more, from scratch, on one A100.
- Every call runs in a fresh, sandboxed process on a fresh draw, whitened and secretly rotated: the only way to be fast is to learn fast.
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

Difficulty 1 to 5 caps the mean MNIST error at 5.40, 3.40, 2.70, 2.30 and 1.90%, what a
[Ladder network](../mnist/experiments/release-cutoffs-20260925/README.md#five-cutoffs-from-200-to-10000-labels-at-24000-steps)
reaches from 200, 532, 1,414, 3,761 and 10,000 labels. Nothing under the
60 s limit reaches 3 to 5 yet. The rules are in [`mnist.py`](mnist.py)'s docstring.

## 5.40%

| Date | mJ | ms | Submission |
| - | -: | -: | - |
| 2026-09-25 | 29,662 | 700.1 | [example.py](example.py), [report](energy/README.md) |
| 2026-09-26 | 2,648 | 191.3 | [mlp_k1_w1024_s100_b512.py](energy/entries/mlp_k1_w1024_s100_b512.py), [report](energy/README.md) |
| 2026-09-26 | 2,429 | 61.6 | [fast_mlp.py](energy/entries/fast_mlp.py), [report](energy/README.md) |

## 3.40%

| Date | mJ | ms | Submission |
| - | -: | -: | - |
| 2026-09-26 | 200,870 | 927.3 | [mlp_k16_w1024_s400_b512.py](energy/entries/mlp_k16_w1024_s400_b512.py), [report](energy/README.md) |
| 2026-09-26 | 14,436 | 252.1 | [mlpg_k4_w256_s800_b512.py](energy/entries/mlpg_k4_w256_s800_b512.py), [report](energy/README.md) |
