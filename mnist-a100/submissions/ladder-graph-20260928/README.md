# CUDA-graph Ladder submission — 2026-09-28

> **Superseded the same day** by [`../ladder-triton-20260928`](../ladder-triton-20260928/README.md): the same recipe with its step fused into Triton kernels, at difficulty 3 in 6,556.9 ms and first entries at difficulties 4 and 5. This entry stays as the plain-PyTorch port and the faithfulness check the Triton entries rest on.

The Ladder network that sets the difficulty bands passes difficulty 3 in **15,818.770 ms per call**, the median of three sandboxed Modal A100-80GB runs of the unchanged scorer 1.2.0, at **1,159,909 mJ per call above idle**. It is the first entry at difficulty 3. The recipe is unchanged from `ladder-xlong-s11` of [`release-cutoffs-20260925`](../../../mnist/experiments/release-cutoffs-20260925/README.md), trained for 5,000 steps instead of 24,000; what changed is how a step runs: it is captured once in a CUDA graph and replayed.

| Run | GPU | Score, ms/call | Energy, mJ/call | MNIST accuracy | Hold-out |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | A100-SXM4-80GB | 15,830.109 | 1,159,909 | 97.48% (107,233/110,000) | Fashion-MNIST 86.78% |
| 2 | A100-SXM4-80GB | 15,774.046 | 1,142,580 | 97.34% (107,070/110,000) | KMNIST 96.24% |
| 3 | A100-SXM4-80GB | 15,818.770 | 1,162,986 | 97.50% (107,250/110,000) | KMNIST 96.00% |

Mean MNIST error over the 330,000 predictions is 2.56% (8,447 wrong), against the 2.70% band. The margin is thin: across seven scored runs at 5,000 steps (these three, an earlier one, and three with the Triton kernels) the 11-call mean ranged over 2.50-2.72%, and one Triton run was disqualified by 24 predictions, which is why the Triton entry uses 6,500 steps. Each run landed on a different board (GPU UUIDs in the records) and passed every energy check: telemetry references of 8.02-8.43 J/TFLOP. Within a run the 15 calls are within 0.3% of each other. Records: [run 1](evidence/run-1.json), [run 2](evidence/run-2.json), [run 3](evidence/run-3.json), [output](evidence/runs.log).

## The Ladder fits the 60 s limit once its step is graph-captured

The cutoffs study timed this recipe at 10-27 ms per step on Modal A100s and concluded that only the Ladder reaches difficulties 3-5, in about 99 s, 171 s and 482 s per call, all over the harness's 60 s limit. That timing ran the step eagerly. The step is small (about 4.4 GFLOP) but launches several hundred kernels, so it is launch-bound. Measured on one A100-SXM4-80GB with CUDA events, one draw per row:

| Step | ms/step | 5,000 steps | 9,000 steps | 24,000 steps |
| --- | ---: | --- | --- | --- |
| Captured in a CUDA graph (this file) | 3.14 | 2.58%, 15.7 s | 1.98%, 28.2 s | 2.02%, 75.2 s |
| Graph + `torch.compile` on the loss | 1.54 | — | — | 1.91%, 37.0 s |

Both were later scored with the Triton kernels instead of `torch.compile`; see [`../ladder-triton-20260928`](../ladder-triton-20260928/README.md).

## Method and attribution

`ladder.py` ports `below-2pct/ladder_model.py` and `neural.py`'s `_fit_ladder_once` (@yaroslavvb) to the three-argument API: the fully supervised AMLP[2,2] Ladder of Pezeshki et al. (ICML 2016), 60-1000-500-250-250-250-10, noise 0.3 at the input and every layer, inputs scaled by 0.6, input reconstruction weight 2000, Adam at 0.002 with the published 100/150 linear decay, 250 labelled and 250 unlabelled rows per step, every minibatch full, BatchNorm calibrated on the training rows at the end. The unlabelled rows come from `train_x` and `test_x` together, as in the recipe that sets the bands (transductive; `test_y` is never seen).

Differences from that code, none in the model or the objective:

* The whole step (noisy encoder, decoder, loss, backward, fused Adam) is captured in a CUDA graph on the untimed warm-up call and replayed. Every call re-initialises weights, Adam moments, the step counter and the random streams in place; only the graph is reused.
* Each epoch's shuffles are drawn up front on the GPU and read by step number.
* The recipe runs a clean forward pass every step to update BatchNorm running statistics. `calibrate_bn` overwrites those statistics before prediction, so the pass never reaches an output; it is dropped. (Dropping it would speed up the reference code too.)
* Adam is PyTorch's fused implementation.

**Faithfulness check.** On three dev draws (`mnist.draw` seeds 0-2), at 2,000 steps, the port averages 3.43% query error and the reference code 3.49% (3.45/3.78, 3.43/3.48, 3.40/3.21); the study's curve gives 3.68% at 2,000 steps. On the A100, graph replay and eager execution of the same code give identical error (7.25% at 400 steps).

The source is 10,890 bytes, passes the source checker, and contains no training examples or pretrained weights.

## Snags found

1. **`torch.compile` does not fit the warm-up limits.** Compiling the loss halves the step time, but takes about 80 s in every fresh worker, and the sandbox deletes `/tmp` between calls, so nothing is cached. Later warm-ups get 90 s (`LATER_WARMUP_MAX_CALL_MS`), so a 24,000-step warm-up would fail. A harness-owned inductor cache per run, or a longer later warm-up, would make the allow-listed decorator usable. Not yet tested inside the sandbox.
2. **Difficulty 5 has little slack**, as the cutoffs study notes: the band is 0.03 points above the Ladder's own 1.870%. One compiled 24,000-step draw read 1.91%.
3. **Energy.** The board averages about 140 W during the method's window, about 73 W above idle, for 16 s a call: about 1.16 kJ per call, against 14 J for the difficulty 2 MLP.

## Reproduce

From `mnist-a100/`, with Modal configured:

```bash
python run_modal.py submissions/ladder-graph-20260928/ladder.py:ladder --difficulty 3 --runs 3
```

The runs used a copy of `run_modal.py` with the function timeout lowered from 7,200 s to 1,200 s, and the file named `ladder_d3.py`; the source hash in each record matches `ladder.py`.

## Cost

Modal's billing API reports **$1.54** for the three scored runs ([billing](evidence/modal-billing.json)). The work behind it cost $0.82 more: two probe containers ($0.33) and one earlier scored run that also passed ($0.50: 15,856 ms, MNIST 97.47%).

SHA-256:

- Submission: `925623e253d14fafd38ef658ba20f11d7988dc0efe000521b92290b532f681cf`
- Scorer 1.2.0: `1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064`
