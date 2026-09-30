# Sampled Ladder submission — difficulties 3 and 4, 2026-09-29

The retuned Ladder of [`../ladder-fast-20260929`](../ladder-fast-20260929/README.md) (batch 1,000, TF32, fused Triton kernels), with one change: **each step draws its labelled rows in proportion to their recent loss, annealed back to uniform before training ends.** It spends early steps on the examples the network still gets wrong, and needs fewer steps. Each row is the median of three sandboxed Modal A100-80GB runs of the unchanged scorer 1.2.0; all six runs passed.

| Difficulty | Band | Steps | Mix | ms/call | mJ/call above idle | MNIST error, 33 calls | Previous best | File |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 2.70% | 900 | 0.9 | **1,443.3** | 201,792 | 2.41% | 1,895.4 ms, 259,847 mJ | [ladder_d3.py](ladder_d3.py) |
| 4 | 2.30% | 1,800 | 0.5 | **2,882.5** | 400,682 | 2.11% | 3,784.2 ms, 547,382 mJ | [ladder_d4.py](ladder_d4.py) |

Both are 24% faster than the previous entries and use 22-27% less energy per call. Difficulty 5 is unchanged: at its 9,000-step budget the sampler gains nothing (below).

## The sampler

Every step already computes each labelled row's cross-entropy; the step records it into a per-row score table (`scores`, reset each call). The next step draws its 1,000 labelled rows with

    p_i = (1 - mix_t) / n + mix_t · score_i / Σ score

where `mix_t` is 0 for the first 10% of steps (scores warm up), `MIX` until 60%, then falls linearly to 0 at 90%, so the last tenth of training samples uniformly. The draw is a cumulative sum and a binary search (`torch.searchsorted`), so it runs inside the CUDA graph; it adds about 0.04 s per call. Unlabelled rows keep the per-epoch shuffle. The idea follows the "skip low-value flops" framing of the 2026-09-28 NanoGPT speedrun record, whose sampled softmax also grows back to the full vocabulary by the end.

## How the settings were chosen

Paired probes on the same dev draws (`mnist.draw` seeds 301-306), batch 1,000, TF32, one A100 each; scripts and output in [`evidence/sweeps`](evidence/sweeps/) ([probe_samp](evidence/sweeps/probe_samp.log)):

| Labelled rows | 1,200 steps | 2,400 steps | 9,000 steps |
| --- | ---: | ---: | ---: |
| Per-epoch shuffle (previous entries) | 2.353% | 2.015% | 1.695% |
| Uniform with replacement | 2.365% | 2.015% | 1.738% |
| Loss-proportional, mix 0.5, annealed | 2.267% | **1.883%** | **1.683%** |
| Loss-proportional, mix 0.9, annealed | **2.228%** | 1.957% | 1.758% |
| Loss-proportional, mix 0.5, held to the end | 2.375% | 2.080% | 1.838% |

* **It helps at short and medium budgets:** about 0.13 points at 1,200 and 2,400 steps, better on five of six draws each time. Sampling mechanics alone change nothing (uniform with replacement matches the shuffle).
* **The annealing is essential.** Held to the end, the same mix is no better than uniform at 1,200 steps and worse at longer budgets (+0.07 and +0.10 points): the bias toward hard rows has to be removed before training ends.
* **The best mix falls as runs get longer**, and at 9,000 steps (difficulty 5) nothing beats the shuffle.

Step counts were then checked on eleven draws (seeds 301-311, [probe_samp_steps](evidence/sweeps/probe_samp_steps.log)), keeping the rule of a dev mean at least 0.15 points under the band:

| Difficulty | Steps | Mix | Mean error, 11 draws | s/call |
| ---: | ---: | ---: | ---: | ---: |
| 3 | **900** | 0.9 | **2.444%** | 1.45 |
| 3 | 1,000 | 0.9 | 2.328% | 1.62 |
| 3 | 1,100 | 0.9 | 2.349% | 1.78 |
| 4 | **1,800** | 0.5 | **2.068%** | 2.91 |
| 4 | 2,000 | 0.5 | 2.014% | 3.23 |
| 4 | 2,200 | 0.5 | 1.990% | 3.55 |

At mix 0.9 an occasional draw lands high (2.96% at 1,100 steps), far from the per-draw limit but a reason not to push the mix further.

**Also tried: weight averaging** ([probe_avg](evidence/sweeps/probe_avg.log)). Uniform tail averages over the last 25% or 50%, an EMA from halfway, and a constant rate with a tail average all matched or trailed the recipe's linear decay at 1,200, 2,400 and 9,000 steps: the decay to zero already does what averaging would.

## Runs

| Difficulty | Run | GPU | Score, ms/call | Energy, mJ/call | MNIST accuracy | Hold-out |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| 3 | 1 | A100-SXM4-80GB | 1,442.142 | 201,792 | 97.59% (107,352/110,000) | KMNIST 96.21% |
| 3 | 2 | A100-SXM4-80GB | 1,443.302 | 202,587 | 97.58% (107,339/110,000) | KMNIST 96.30% |
| 3 | 3 | A100 80GB PCIe | 1,449.412 | 187,447 | 97.60% (107,364/110,000) | KMNIST 96.45% |
| 4 | 1 | A100-SXM4-80GB | 2,884.058 | 400,682 | 97.85% (107,636/110,000) | Fashion-MNIST 87.51% |
| 4 | 2 | A100-SXM4-80GB | 2,880.149 | 433,464 | 97.83% (107,610/110,000) | Fashion-MNIST 87.68% |
| 4 | 3 | A100-SXM4-80GB | 2,882.534 | 380,017 | 98.00% (107,803/110,000) | KMNIST 96.87% |

Six distinct boards (GPU UUIDs in the records), every energy check passed, telemetry references 7.64-8.66 J/TFLOP. Records and output: [difficulty 3](evidence/d3/), [difficulty 4](evidence/d4/).

## Changes to the code

Relative to `../ladder-fast-20260929`: the sampler (score table, `MIX` and its per-step table, the draw in `step`), per-row cross-entropy returned from `loss`, the step counts, and the labelled per-epoch shuffle removed since sampling replaces it. Removing it changes which random numbers feed the unlabelled shuffle, so these files are not bit-identical to the probe code; the record runs measure the files as submitted. Files are 19,794 and 19,795 bytes.

## Caveats

1. **One learner seed; six draws for the sampler comparison, eleven for step counts.** The best mix and anneal window were not tuned beyond the values above.
2. **TF32 runs are not bit-reproducible across boards**: the same configuration on different A100 hosts differs in the last digits (for example 2.353% here against 2.363% in #99's sweep), likely from cuBLAS choosing different kernels.

## Reproduce

From `mnist-a100/`, with Modal configured:

```bash
python run_modal.py submissions/ladder-sampled-20260929/ladder_d3.py:ladder --difficulty 3 --runs 3
python run_modal.py submissions/ladder-sampled-20260929/ladder_d4.py:ladder --difficulty 4 --runs 3
```

The runs used a copy of `run_modal.py` with the function timeout at 2,700 s and files named `ladder_sfast_d3.py` and `ladder_sfast_d4.py`; the source hash in each record matches the file here.

## Cost

Modal bills by UTC day ([billing](evidence/modal-billing.json)). The weight-averaging and sampling probes fell on 2026-09-29 UTC ($4.30 that day, shared with #99's records); the eleven-draw step check and these six record runs fell on 2026-09-30 UTC ($2.15 when this was committed, including a few cents for an unrelated start-up timing probe).

SHA-256:

- `ladder_d3.py`: `0c11521f8ef34e4356ac335160fc1537f16e6805dbbeba9870646614a869452c`
- `ladder_d4.py`: `1bfb7939b04712957c2f97aab558b0774b6400484a73551258eb7364c554bfca`
- Scorer 1.2.0: `1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064`
