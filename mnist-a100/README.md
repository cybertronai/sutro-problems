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

[`example.py`](example.py) is a complete method, a two-hidden-layer MLP in 40 lines.
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
about 3 minutes for the example. A passing run then measures the energy column,
about 80 s more wherever the GPU's energy counter can be read (`--no-energy` or
`mnist.score(..., energy=False)` skips it).

## Difficulties

A difficulty is an error band. To pass, the mean test error over a run's MNIST
calls must be at most the band. Each band is what a Ladder network, the best
known method, reaches when trained for 24,000 steps on only N labelled images.
Your method always gets all 10,000.

| Difficulty | Mean error at most | The Ladder needs | Fastest known method, A100-80GB | Its energy per call |
| :-: | -: | -: | - | -: |
| 1 (default) | 5.40% | 200 labels | MLP, 61 ms per call (CUDA graph) | 2,429 mJ |
| 2 | 3.40% | 532 labels | MLP ensemble, 250 ms (CUDA graph) | 14,436 mJ |
| 3 | 2.70% | 1,414 labels | Ladder, about 99 s: over the 60 s limit | not measured |
| 4 | 2.30% | 3,761 labels | Ladder, about 171 s: over the limit | not measured |
| 5 | 1.90% | 10,000 labels | Ladder, about 8 minutes: over the limit | not measured |

Difficulties 3 to 5 have no qualifying method yet: the best MLP under 60 s gets
2.75%. Difficulty 5 sits on the Ladder's own error, so even a rerun of the
Ladder would pass only about two runs in three. The bands and times come from
[`mnist/experiments/release-cutoffs-20260925`](../mnist/experiments/release-cutoffs-20260925/README.md#five-cutoffs-from-200-to-10000-labels-at-24000-steps);
the MLP times were measured in the popcorn3 harness on the same draws, bands and
CUDA-event timing; this port adds a fresh process per call, the clock floor and stricter source rules.
The energies are medians of three runs of this scorer
([`energy/`](energy/README.md)); both methods are also the lowest-energy known at their difficulty.

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
* **Energy, a column beside the score:** the GPU board's energy per call above
  idle, measured after a passing run (next section). It ranks nothing.

These are the protections of the popcorn3 KernelBot harness
(`sutro-mnist-medium/3.0.0`), tightened after a red-team review of this port:
a fresh process per call, a floor from the scorer's clock, and stricter
source rules. `mnist.py`'s docstring gives the detail. The release hides
coordinates, not identity, so a method holding the 60,000-image pool could
re-identify the images; the sandbox and the size limit keep the pool away. A
method small enough to fit can still carry a compact prior trained offline,
so records are read before they stand.

## Energy

After a passing run, the scorer reads the board's NVML energy counter and prints
the method's energy per call, in mJ above idle, as the last line:
`energy 29661.574 mJ per call above idle`. How it is measured:

* **A window, not a call.** The A100's counter moves every 100 ms, so a call of
  tens of ms cannot be read on its own. One more fresh process runs the method,
  after its own untimed warm-up, back to back on fresh MNIST draws for 20 s (at
  least 3 calls), and the window's energy is split over its calls.
* **Above idle.** Idle power is measured for 5 s before and after the window,
  after 3 s to settle, with every process of the method frozen (SIGSTOP) and its
  CUDA context still open: an idle A100-SXM4 drew 60 W with no context and 67 W
  with one. What the same round trip costs with an empty method (staging the
  draw, the L2 flush, the reply; 13-21 mJ) is subtracted too.
* **Telemetry checked first.** Before the method is imported, the same process
  runs 5 s of FP32 matmul on constant operands. A healthy A100 reads 6-11 J per
  10^12 FLOPs above idle (8.1-8.8 J on six boards in earlier audits, 7.2-9.2 J on
  the fifteen measured for `energy/`); outside that band the column is left empty.
* **Same work as the timed calls.** The column is also left empty if the GPU is
  busy while the method is frozen, another process holds a CUDA context, a draw in
  the window falls below the per-draw floor, or the window's calls take more than
  2x (or under half) the timed calls' median.
* **Board energy only.** It leaves out the host CPU, memory and power supply.
  Modal's A100-80GB is sometimes an SXM4 board and sometimes a PCIe card, and the
  same CUDA-graph MLP read 1.4 J per call on a PCIe card and 2.4-2.6 J on SXM4
  boards, so a record takes the median of its runs, as for time.

`python run_modal.py FILE:FUNCTION --runs 3 --json DIR` also saves each run's
calls and energy windows, from which [`energy/summarize.py`](energy/summarize.py)
recomputes every energy. [`energy/README.md`](energy/README.md) has the energy of
every known method under 60 s per call and the measurements behind this protocol.

## Records

| Difficulty | Date | ms per call | mJ per call | Method | Contributors |
| :-: | - | -: | -: | - | - |
| 1 | 2026-09-25 | 700.1 (median of 3: 678.8, 700.1, 723.0) | 29,662 | [`example.py`](example.py), two-hidden-layer MLP, 96.7% | baseline |

The energy is the median over a record's runs; for a record set before the column
existed, over three later runs of the same file ([`energy/`](energy/README.md)).

## Tests

```bash
python -m pytest -q test_mnist.py   # CPU only, no network, about a minute
```
