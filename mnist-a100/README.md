# MNIST on an A100

Learn to read digits from 10,000 labelled examples, then label 10,000 more, as
fast as you can on an A100. Every call starts from scratch, in a fresh process,
on a fresh and secretly scrambled draw. The only way to be fast is to learn fast.

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

[`example.py`](example.py) is a complete method, a two-layer MLP in 40 lines.
Run everything from this directory; the repository root has another `mnist/`.

```bash
pip install torch numpy                                   # all the scorer itself needs
python example.py                                         # score it on your own CUDA GPU
python mnist.py example.py:mlp --difficulty 2             # the same scorer from the shell, at any difficulty
pip install modal && modal setup                          # once, for official A100 times
python run_modal.py example.py:mlp --difficulty 1         # on a Modal A100-80GB, sandboxed: an official time
python run_modal.py example.py:mlp --difficulty 1 --runs 3    # three containers; a record takes the median
```

Put your method in its own `.py` file: the scorer imports it again in a fresh,
sandboxed process for every call. The first run downloads MNIST,
Fashion-MNIST and KMNIST (about 55 MB) into `~/.cache/sutro-mnist`.
`mnist.draw()` returns one draw as numpy arrays `(train_x, train_y, test_x, test_y)`
for quick experiments; feed your method with `torch.as_tensor(a, device="cuda")`.
A run is 15 timed calls, each in its own process after an untimed warm-up:
about 3 minutes for the example.

## Difficulties

A difficulty is an error band. To pass, the mean test error over a run's MNIST
calls must be at most the band. Each band is what a Ladder network, the best
known method, reaches when trained for 24,000 steps on only N labelled images.
Your method always gets all 10,000.

| Difficulty | Mean error at most | The Ladder needs | Fastest known method, A100-80GB |
| :-: | -: | -: | - |
| 1 (default) | 5.40% | 200 labels | MLP, 61 ms per call (CUDA graph) |
| 2 | 3.40% | 532 labels | MLP ensemble, 250 ms (CUDA graph) |
| 3 | 2.70% | 1,414 labels | Ladder, about 99 s: over the 60 s limit |
| 4 | 2.30% | 3,761 labels | Ladder, about 171 s: over the limit |
| 5 | 1.90% | 10,000 labels | Ladder, about 8 minutes: over the limit |

Difficulties 3 to 5 have no qualifying method yet: the best MLP under 60 s gets
2.75%. Difficulty 5 sits on the Ladder's own error, so even a rerun of the
Ladder would pass only about two runs in three. The bands and times come from
[`mnist/experiments/release-cutoffs-20260925`](../mnist/experiments/release-cutoffs-20260925/README.md#five-cutoffs-from-200-to-10000-labels-at-24000-steps);
the MLP times were measured in the popcorn3 harness on the same draws, bands and
CUDA-event timing; this port adds a fresh process per call, the clock floor and stricter source rules.

## What the method sees

Each call is a fresh draw: 10,000 training and 10,000 test images from the
60,000-image MNIST training set, box-averaged to 9x9. The draw arrives as
`z = Q W (x - mu)`: 60 numbers per image, whitened onto the top 60 principal
directions of that draw's training images and turned by a secret random
rotation. The class labels are secretly permuted too. Your file is imported
fresh for every call, so nothing learned in one call is there for the next.

## Rules

* **Score:** the mean time per call on the slower of MNIST and the hold-out.
  Each call is timed with CUDA events after an L2 flush, and the scorer times
  the whole round trip itself; a dataset's time is never less than the
  scorer's clock minus 0.1 ms + 0.5%, so hiding work from the events cannot
  lower a score.
* **A run:** 11 MNIST calls and 4 hold-out calls (Fashion-MNIST or KMNIST,
  released the same way) in a secret order. Each call runs in a fresh process
  after one untimed warm-up on the other foreign dataset, which is the time to
  compile, autotune and capture CUDA graphs. The whole run has a 45-minute limit.
* **Pass:** mean MNIST error at most the band, no MNIST draw more than 1.5
  points worse, and the hold-out at least 15% correct.
* **Same work every call:** within a dataset, the slowest call at most 2x the
  median + 2 ms and the fastest at least half the median - 2 ms. 60 s per call.
* **Small, inert file:** at most 20,480 bytes. At import it may only import,
  define, assign constants and set torch flags, plus a final
  `if __name__ == "__main__":` block. The scorer prints review flags (large
  high-entropy literals, decoders, network imports) for a human to read.
* **Sandbox:** as root on Linux x86-64 (as on Modal), the method runs as an
  unprivileged user with no network, no access to the scorer or the dataset
  files, and its files deleted after every call. Elsewhere (macOS, or Linux as a
  normal user) it runs unsandboxed with a warning, so the time is for your
  information only; Windows is not supported.

These are the protections of the popcorn3 KernelBot harness
(`sutro-mnist-medium/3.0.0`), tightened after a red-team review of this port:
a fresh process per call, a floor from the scorer's clock, and stricter
source rules. `mnist.py`'s docstring gives the detail. The release hides
coordinates, not identity, so a method holding the 60,000-image pool could
re-identify the images; the sandbox and the size limit keep the pool away. A
method small enough to fit can still carry a compact prior trained offline,
so records are read before they stand.

## Records

| Difficulty | Date | ms per call | Method | Contributors |
| :-: | - | -: | - | - |
| 1 | 2026-09-25 | 700.1 (median of 3: 678.8, 700.1, 723.0) | [`example.py`](example.py), two-hidden-layer MLP, 96.7% | baseline |

## Tests

```bash
python -m pytest -q test_mnist.py   # CPU only, no network, about a minute
```
