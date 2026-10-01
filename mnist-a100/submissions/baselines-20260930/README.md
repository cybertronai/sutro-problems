# Baseline reproductions for all five difficulties — 2026-09-30

The leaderboard entries for these exact reproductions are superseded by
[Yaroslav's improved variants](../improved-20260930/README.md). The source and
measurements below are retained as historical evidence. Attribution to the
upstream implementation is separate from the entrant's own measured score.

Five standalone entries package existing learning procedures for the five
MNIST A100 categories. These are reproductions: no hyperparameters were tuned,
no new learning method is claimed, and this submission makes no speed-record
claim. Each file exposes `custom_kernel(train_x, train_y, test_x)` and includes
the corresponding `#!POPCORN leaderboard mnist-a100-N` header.

## Sources and attribution

The source snapshot is repository commit `6aba49a`.

| Difficulty | Error band | Procedure and original contributor | Source file |
| ---: | ---: | --- | --- |
| 1 | 5.40% | MLP 60-1024-1024-10, 400 steps, batch 512; @yaroslavvb | `../../example.py` |
| 2 | 3.40% | 16-member eager MLP ensemble, 400 steps, batch 512; @yaroslavvb | `../../energy/entries/mlp_k16_w1024_s400_b512.py` |
| 3 | 2.70% | Ladder, 1,200 steps, batch 1,000, learning rate 0.008; @SethTS | `../ladder-fast-20260929/ladder_d3.py` |
| 4 | 2.30% | Ladder, 2,400 steps, batch 1,000, learning rate 0.008; @SethTS | `../ladder-fast-20260929/ladder_d4.py` |
| 5 | 1.90% | Ladder, 9,000 steps, batch 1,000, learning rate 0.006; @SethTS | `../ladder-fast-20260929/ladder_d5.py` |

The Ladder entries retain the upstream TF32 matmuls, fused Triton kernels,
CUDA graphs, architecture, objective, and schedule. The MLP entries retain
their upstream initialization, training, and prediction procedures.

The only changes are submission headers, attribution comments, and a
`custom_kernel` wrapper calling the original `mlp` or `ladder` function.
Difficulty 1 also removes the original `__main__` scoring launcher and its
`import mnist`. Original
docstrings describe prior runs; the measurements below belong to these exact
submission files. Source and upstream SHA-256 hashes are recorded in
`evidence/provenance.json`.

## Validation

Each entry passed one full sandboxed Modal A100-80GB run, including energy,
using the unchanged `mnist-a100/1.2.0` scorer. These are single-run baseline
measurements, rather than medians of three independent runs.

| Difficulty | Result | ms/call | mJ/call above idle | MNIST error | Hold-out accuracy | Evidence |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | Pass | 799.8 | 22,890 | 3.36% | kmnist: 94.85% | [run JSON](evidence/d1/run-1.json) |
| 2 | Pass | 918.1 | 205,886 | 3.26% | fashion: 87.62% | [run JSON](evidence/d2/run-1.json) |
| 3 | Pass | 1,865.4 | 241,778 | 2.39% | kmnist: 96.23% | [run JSON](evidence/d3/run-1.json) |
| 4 | Pass | 3,796.8 | 499,472 | 2.07% | fashion: 87.74% | [run JSON](evidence/d4/run-1.json) |
| 5 | Pass | 14,151.9 | 1,980,209 | 1.75% | kmnist: 97.45% | [run JSON](evidence/d5/run-1.json) |

A run evaluates 11 fresh MNIST calls and four Fashion-MNIST or KMNIST hold-out
calls in secret order. Each call starts a fresh sandboxed worker, performs an
untimed foreign-dataset warm-up, and trains from scratch on the supplied
10,000 labelled examples. The scorer applies the accuracy, per-draw error,
timing, and dispersion checks; the score is the slower dataset's mean time,
subject to the scorer's independent clock floor. Energy is measured after a
passing run with NVML windows, subtracting idle and empty-call overhead.

All files fit the 20,480-byte source cap: difficulty 1 is 1,632 bytes,
difficulty 2 is 4,547 bytes, and difficulties 3–5 are 19,856 bytes each.
The expected review flag for each Ladder file is
`19,856 bytes, close to the 20,480-byte limit`; it is advisory. Difficulties
1 and 2 have no source review flags. The scorer SHA-256 is
`1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064`.

## Reproduce

From `mnist-a100/`, with Modal configured:

```bash
python run_modal.py submissions/baselines-20260930/baseline_d1.py:custom_kernel --difficulty 1 --runs 1 --json submissions/baselines-20260930/evidence/d1
python run_modal.py submissions/baselines-20260930/baseline_d2.py:custom_kernel --difficulty 2 --runs 1 --json submissions/baselines-20260930/evidence/d2
python run_modal.py submissions/baselines-20260930/baseline_d3.py:custom_kernel --difficulty 3 --runs 1 --json submissions/baselines-20260930/evidence/d3
python run_modal.py submissions/baselines-20260930/baseline_d4.py:custom_kernel --difficulty 4 --runs 1 --json submissions/baselines-20260930/evidence/d4
python run_modal.py submissions/baselines-20260930/baseline_d5.py:custom_kernel --difficulty 5 --runs 1 --json submissions/baselines-20260930/evidence/d5
```

Energy is enabled by default. Fresh draws and different A100 boards can change
accuracy, time, and energy on reruns; these measurements do not guarantee
that every future run passes a category's band.

To check the saved results locally with Python 3.11 or newer, run
`python submissions/baselines-20260930/verify_results.py` from `mnist-a100/`.
This uses only the standard library: it checks source hashes, compares the
learning code's AST with upstream, applies the scorer's source checks, and
recomputes each verdict and score from the 15 saved calls, and independently
recomputes energy from the saved NVML windows.

Files: [difficulty 1](baseline_d1.py), [difficulty 2](baseline_d2.py),
[difficulty 3](baseline_d3.py), [difficulty 4](baseline_d4.py),
[difficulty 5](baseline_d5.py), [provenance](evidence/provenance.json).
Original contributors: [@yaroslavvb](https://github.com/yaroslavvb),
[@SethTS](https://github.com/SethTS).
