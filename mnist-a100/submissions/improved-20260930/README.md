# Yaroslav's faster variants for all five difficulties — 2026-09-30

These five entries are submitted by **@yaroslavvb**. Each makes a small,
measured change to an existing procedure; no new algorithm or overall speed
record is claimed. They are derivative hyperparameter and execution variants,
with separate source files and measurements from the earlier reproductions.

@SethTS wrote the original Ladder implementations used for difficulties 3–5.
That source credit is retained here and in the code. Seth's original
submissions and scores remain separate entries; attribution of these variants
to Yaroslav does not turn the originals into joint submissions.

## What changed

| Difficulty | Error band | Source contributor and procedure | Changes in this entry |
| ---: | ---: | --- | --- |
| 1 | 5.40% | @yaroslavvb, `example.py` MLP | 400 → 200 steps; TF32 matmuls; fused AdamW |
| 2 | 3.40% | @yaroslavvb, 16-member MLP ensemble | 16 → 8 members; 400 → 500 steps; EMA 0.99 → 0.992 |
| 3 | 2.70% | @SethTS, `ladder-fast-20260929` | 1,200 → 1,100 steps |
| 4 | 2.30% | @SethTS, `ladder-fast-20260929` | 2,400 → 2,200 steps |
| 5 | 1.90% | @SethTS, `ladder-fast-20260929` | 9,000 → 8,400 steps |

Other learning settings stay unchanged. The Ladder variants retain Seth's
architecture, objective, batch size, learning rates, proportional decay
schedule, fused Triton kernels, and CUDA graphs. Every call initializes its
parameters and optimizer state anew and fits only the supplied data. No
development examples, test labels, or fitted weights enter the submissions.

## Paired development comparisons

These are unsandboxed development measurements on public seeded draws, not
official scores. Each original/candidate pair ran on the same GPU and draws:
six MNIST draws for each MLP comparison and three for each Ladder comparison.
Times are synchronized wall-clock means; reductions compare each pair only.

| Difficulty | Original ms | Variant ms | Less time | Original → variant MNIST error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 520.610 | 235.413 | 54.8% | 3.400% → 3.830% |
| 2 | 922.961 | 782.466 | 15.2% | 3.323% → 3.257% |
| 3 | 1,862.956 | 1,708.970 | 8.3% | 2.347% → 2.493% |
| 4 | 3,774.703 | 3,456.154 | 8.4% | 2.017% → 2.007% |
| 5 | 14,116.684 | 13,209.568 | 6.4% | 1.693% → 1.673% |

One earlier D2 variant used eight members and 400 steps: 519.548 ms and
3.338% error, versus its paired original's 915.952 ms and 3.328% error.
It was rejected because its margin below 3.40% was narrow. The final variant
uses 500 steps and the matching EMA decay. This was the only additional
measured candidate; its source and all measurements are retained. Fresh
draws can still change qualification, especially near a category boundary.

The first MLP probe completed all GPU calls but failed while deserializing
its final version metadata locally. Its JSON was recovered from the complete
printed measurements; the log, recovery note, and corrected controller are
retained. The second MLP probe completed normally.

## Official validation

Each category passed one full sandboxed Modal A100-80GB run with the
unchanged `mnist-a100/1.2.0` scorer and energy enabled. These are single-run
measurements, not medians across three independent runs.

| Difficulty | Result | Score ms/call | mJ/call above idle | MNIST error | Hold-out accuracy |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Pass | 350.984 | 3,633.545 | 3.784% | fashion: 87.33% |
| 2 | Pass | 761.895 | 132,634.979 | 3.020% | kmnist: 95.09% |
| 3 | Pass | 1,732.218 | 253,760.696 | 2.578% | kmnist: 96.14% |
| 4 | Pass | 3,481.336 | 460,763.342 | 2.078% | fashion: 87.98% |
| 5 | Pass | 13,264.305 | 1,824,759.863 | 1.719% | kmnist: 97.56% |

A full run checks 11 fresh MNIST calls and four hold-out calls, each in a
fresh sandboxed process after a foreign-dataset warm-up. The score uses the
slower dataset's mean and the scorer's independent clock floor. Energy is
measured separately after a pass, subtracting idle and empty-call overhead.
All sources fit the 20,480-byte cap; the Ladder size flags are advisory.

## Reproduce and verify

From `mnist-a100/`, with Modal configured, run:

```bash
python run_modal.py submissions/improved-20260930/mlp_d1.py:custom_kernel --difficulty 1 --runs 1 --json submissions/improved-20260930/evidence/d1
python run_modal.py submissions/improved-20260930/mlp_d2.py:custom_kernel --difficulty 2 --runs 1 --json submissions/improved-20260930/evidence/d2
python run_modal.py submissions/improved-20260930/ladder_d3.py:custom_kernel --difficulty 3 --runs 1 --json submissions/improved-20260930/evidence/d3
python run_modal.py submissions/improved-20260930/ladder_d4.py:custom_kernel --difficulty 4 --runs 1 --json submissions/improved-20260930/evidence/d4
python run_modal.py submissions/improved-20260930/ladder_d5.py:custom_kernel --difficulty 5 --runs 1 --json submissions/improved-20260930/evidence/d5
python submissions/improved-20260930/verify_results.py
```

The verifier needs only Python 3.11+ and its standard library. It checks source
and scorer hashes, enforces the documented changes through AST comparisons,
applies source checks, recomputes verdicts and scores with the current scorer,
and independently recalculates energy from saved windows. It requires all
five completed run files. Reruns can vary with fresh draws and A100 hardware.

Sources: [D1](mlp_d1.py), [D2](mlp_d2.py), [D3](ladder_d3.py),
[D4](ladder_d4.py), [D5](ladder_d5.py), [provenance](evidence/provenance.json).
Development: [MLP report](evidence/dev-mlp-report.txt),
[first MLP probe](evidence/dev-mlp-probe.json),
[final D2 probe](evidence/dev-mlp-probe-500.json),
[rejected D2 source](evidence/dev-mlp-d2-k8-s400.py),
[Ladder D3](evidence/dev-ladder-d3.json), [D4](evidence/dev-ladder-d4.json),
[D5](evidence/dev-ladder-d5.json).
Official records: [D1](evidence/d1/run-1.json), [D2](evidence/d2/run-1.json),
[D3](evidence/d3/run-1.json), [D4](evidence/d4/run-1.json), [D5](evidence/d5/run-1.json).
