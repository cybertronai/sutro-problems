# Retuned Ladder submission — difficulties 3, 4 and 5, 2026-09-29

The Triton Ladder of [`../ladder-triton-20260928`](../ladder-triton-20260928/README.md), with its 2016 hyperparameters retuned for the A100: **batch 1,000 instead of 250, peak learning rate 0.008 instead of 0.002 (0.006 at difficulty 5), TF32 matmuls, and far fewer steps.** The model, the objective and the schedule's shape are unchanged. Each row is the median of three sandboxed Modal A100-80GB runs of the unchanged scorer 1.2.0; all nine runs passed.

| Difficulty | Band | Steps | ms/call | mJ/call above idle | MNIST error, 33 calls | Previous best | File |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 2.70% | 1,200 | **1,895.4** | 259,847 | 2.44% | 6,556.9 ms, 676,850 mJ | [ladder_d3.py](ladder_d3.py) |
| 4 | 2.30% | 2,400 | **3,784.2** | 547,382 | 2.06% | 11,094.6 ms, 1,143,255 mJ | [ladder_d4.py](ladder_d4.py) |
| 5 | 1.90% | 9,000 | **14,130.6** | 2,059,576 | 1.76% | 36,201.2 ms, 4,410,730 mJ | [ladder_d5.py](ladder_d5.py) |

Against the Triton entries this is 2.6-3.5× faster and uses 2.1-2.6× less energy per call. Against the recipe as the cutoffs study timed it (about 482 s for 24,000 steps), difficulty 5 is about 34× faster: roughly 4× from the kernels and 8× from the retuning.

## The idea: fewer, bigger steps

The recipe takes thousands of small steps (250 labelled and 250 unlabelled rows), and on an A100 each one leaves the GPU mostly idle, so the time is set by the number of steps, not by the arithmetic. Three changes follow from that:

1. **Batch 1,000.** A step does 4× the work for about 2.2× the time in FP32 (2.22 ms against 1.00 ms) and 1.6× with TF32. Batch 2,000 was no faster than 1,000: by then the matmuls fill the GPU.
2. **A higher learning rate.** Larger batches give less noisy gradients and take larger steps; the recipe's rate was already too low at batch 250.
3. **TF32 matmuls.** Once the matmuls dominate, tensor cores speed them up 1.35-1.4×. The recipe left TF32 off because it normalises with eps 1e-10; accuracy is unchanged within noise.

The spare accuracy then paid for fewer epochs. Energy fell with time this round: the Triton entries had cut time without cutting energy, because the fused step ran the GPU harder.

## How the settings were chosen

Every comparison below uses the same dev draws (`mnist.draw` seeds 301-306, or 301-311), one learner seed, and graph-captured Triton steps on one A100 per probe. Each probe's script and output are in [`evidence/sweeps`](evidence/sweeps/); they ran against [`ladder_tb.py`](evidence/sweeps/ladder_tb.py), which is these files before the constants were set, or against `../ladder-triton-20260928/ladder_d5.py`.

**Schedule and learning rate** (batch 250, FP32, [probe_sched](evidence/sweeps/probe_sched.log), [probe_sched2](evidence/sweeps/probe_sched2.log)):

| Schedule | Peak lr | 5,000 steps | 6,500 steps |
| --- | ---: | ---: | ---: |
| Recipe: constant for 2/3, then linear decay | 0.002 | 2.585% | 2.457% |
| Late decay: constant for 85% | 0.002 | 2.577% | — |
| Early decay: constant for 1/3 | 0.002 | 2.692% | 2.515% |
| Shifted lognormal, peak at 10% (σ 0.65) | 0.002 | 4.182% | 3.818% |
| Shifted lognormal, peak at 10% (σ 0.65) | 0.004 | 3.560% | 3.188% |
| Recipe | 0.003 | 2.370% | — |
| Recipe | 0.004 | 2.302% | — |

A higher rate beat the recipe on all six draws; where the decay starts barely matters once it is late. Schedules that spend less time at a high rate lose.

**Batch size** (162 epochs, FP32, best rate per batch, [probe_batch](evidence/sweeps/probe_batch.log)), and **TF32** ([probe_tf32](evidence/sweeps/probe_tf32.log)):

| Batch | Steps | lr | FP32 error | FP32 s/call | TF32 error | TF32 s/call |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 6,520 | 0.004 | 2.153% | 6.51 | — | — |
| 500 | 3,260 | 0.006 | 2.147% | 4.42 | 2.130% | 3.27 |
| 1,000 | 1,630 | 0.008 | 2.228% | 3.62 | **2.200%** | **2.56** |
| 2,000 | 815 | 0.008 | 2.268% | 3.77 | 2.293% | 2.67 |

**Steps per difficulty** (batch 1,000, TF32, [probe_epochs](evidence/sweeps/probe_epochs.log)); the rule, as in the cutoffs study, was the fewest steps whose dev mean sits at least 0.15 points under the band:

| Steps | lr | Draws | Mean error | s/call | Picked for |
| ---: | ---: | ---: | ---: | ---: | --- |
| 600 | 0.008 | 6 | 3.060% | 0.93 | |
| 900 | 0.008 | 6 | 2.615% | 1.40 | |
| 1,200 | 0.008 | 6 | 2.363% | 1.86 | difficulty 3 |
| 2,400 | 0.008 | 6 | 1.998% | 3.72 | difficulty 4 |
| 3,600 | 0.008 | 6 | 1.822% | 5.58 | |
| 6,000 | 0.008 | 11 | 1.878% | 9.31 | |
| 9,000 | 0.008 | 11 | 1.780% | 13.93 | |
| 9,000 | 0.006 | 11 | 1.733% | 13.93 | difficulty 5 |

## Runs

| Difficulty | Run | GPU | Score, ms/call | Energy, mJ/call | MNIST accuracy | Hold-out |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| 3 | 1 | A100-SXM4-80GB | 1,887.417 | 267,234 | 97.48% (107,227/110,000) | KMNIST 95.82% |
| 3 | 2 | A100 80GB PCIe | 1,906.757 | 259,847 | 97.67% (107,438/110,000) | Fashion-MNIST 87.23% |
| 3 | 3 | A100 80GB PCIe | 1,895.432 | 241,941 | 97.54% (107,292/110,000) | KMNIST 96.30% |
| 4 | 1 | A100-SXM4-80GB | 3,781.352 | 553,117 | 97.90% (107,693/110,000) | KMNIST 96.81% |
| 4 | 2 | A100-SXM4-80GB | 3,785.116 | 529,626 | 97.91% (107,704/110,000) | KMNIST 96.71% |
| 4 | 3 | A100-SXM4-80GB | 3,784.243 | 547,382 | 98.02% (107,819/110,000) | Fashion-MNIST 87.55% |
| 5 | 1 | A100 80GB PCIe | 14,201.382 | 2,059,576 | 98.18% (108,000/110,000) | Fashion-MNIST 87.98% |
| 5 | 2 | A100 80GB PCIe | 13,995.891 | 1,892,444 | 98.33% (108,168/110,000) | Fashion-MNIST 88.05% |
| 5 | 3 | A100-SXM4-80GB | 14,130.640 | 2,297,345 | 98.20% (108,017/110,000) | Fashion-MNIST 87.98% |

The nine runs landed on nine distinct boards (GPU UUIDs in the records) and passed every energy check, with telemetry references of 7.49-9.16 J/TFLOP. Records and output: [difficulty 3](evidence/d3/), [difficulty 4](evidence/d4/), [difficulty 5](evidence/d5/).

## Changes to the code

Relative to `../ladder-triton-20260928`: the constants (`STEPS`, `BATCH = 1000`, `LR`), `torch.backends.cuda.matmul.allow_tf32 = True`, and the Triton programs' column block, now sized from the batch (`TILE // rows`, 8 columns at 250 rows and 2 at 1,000) so each program still holds about 2,048 values in registers. The kernels are otherwise identical. The files are 19,517 bytes.

## Caveats

1. **Difficulty 5's margin is 0.14 points**, just under the 0.15 target: the three runs read 1.82%, 1.67% and 1.80%. A rerun on unlucky draws could miss the band.
2. **The tuning used one learner seed** and 6-11 dev draws per setting. The best learning rate may sit between the values tried, and batch 1,000 was the only large batch tuned for step count.
3. **Not a new learning method.** This is the same Ladder with its hyperparameters retuned for the hardware.

## Reproduce

From `mnist-a100/`, with Modal configured:

```bash
python run_modal.py submissions/ladder-fast-20260929/ladder_d3.py:ladder --difficulty 3 --runs 3
python run_modal.py submissions/ladder-fast-20260929/ladder_d4.py:ladder --difficulty 4 --runs 3
python run_modal.py submissions/ladder-fast-20260929/ladder_d5.py:ladder --difficulty 5 --runs 3
```

The runs used a copy of `run_modal.py` with the function timeout at 2,700 s and files named `ladder_fast_d3.py` etc.; the source hash in each record matches the file here.

## Cost

Modal bills by UTC day ([billing](evidence/modal-billing.json)). The batch, TF32 and step sweeps and the nine record runs fell on 2026-09-29 UTC: **$3.81** when this was committed, possibly still settling. The two schedule probes ran on 2026-09-28 UTC, inside that day's $9.92, which mostly covers the Triton entries.

SHA-256:

- `ladder_d3.py`: `ac5b661e664668699398d1988c90921625f7417de67779b5c9b1878e9c6365de`
- `ladder_d4.py`: `2e2d0eda0aeff1b9af965f4fc360e4fb461bef3d9ccb3200c02020bbe76027b3`
- `ladder_d5.py`: `b4810f555fa2eaed3f56baf17af198e89a955d1cd009a951fca51ec7304996a7`
- Scorer 1.2.0: `1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064`
