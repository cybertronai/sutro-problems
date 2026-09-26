# CUDA-graph MLP submission — 2026-09-26

A 60→256→256→10 MLP trained from scratch passes difficulty 1 in **61.388 ms per call**, the median of three sandboxed Modal A100-80GB runs. The three runs used the unchanged official scorer from repository commit `5714d1a`. Each run contains 11 MNIST calls and four foreign hold-out calls, with a fresh process and foreign warmup for every timed call.

| Evidence attempt | Score, ms/call | MNIST accuracy | Hold-out | Hold-out accuracy | Result |
| --- | ---: | ---: | --- | ---: | --- |
| 2 | 61.388 | 95.07% (104,582/110,000) | Fashion-MNIST | 85.98% | Pass |
| 3 | 61.269 | 95.13% (104,638/110,000) | Fashion-MNIST | 85.67% | Pass |
| 4 | 61.590 | 95.30% (104,828/110,000) | KMNIST | 91.56% | Pass |

Scores are the scorer's final ranked times: the slower of the MNIST and hold-out means, including its wall-clock floor where applicable. Passing also checks the worst MNIST draw, hold-out accuracy, timing dispersion, and per-call limits. Only difficulty 1 was tested.

## Method and attribution

This submission ports the repository's `mlpg-k1-w256-s200-b512` cutoff experiment (credited below) to the three-argument `mnist-a100` API. Its algorithm is unchanged: 200 minibatches of 512, AdamW, a cosine learning-rate schedule with warmup, input noise, dropout, label smoothing, and EMA inference. One training step is captured in a CUDA graph during the untimed warmup.

Every invocation copies the current training data and labels and resets model weights, Adam moments and counters, EMA weights, the schedule counter, and CUDA random state. The graph is reused within the process; learned state is not reused. The submission includes no training examples or pretrained weights. Its source is 5,729 bytes and passes the source checker with no review flags.

## Reproduce

From `mnist-a100/`, with Modal configured:

```bash
python run_modal.py submissions/graph-mlp-20260926/fast_mlp.py:fast_mlp --difficulty 1 --runs 3
```

The included `verify_modal.py` runs that same scorer with sandboxing required, records source/scorer hashes and complete output, and reserves budget before launching each attempt. It uses separate single-use containers, function retries disabled (`retries=0`), a 600-second function timeout and a 120-second startup timeout. The checked-in ledger preserves this task's reservations. A rerun writes a separate ledger and evidence directory:

```bash
python submissions/graph-mlp-20260926/verify_modal.py --runs 3 --output /tmp/mnist-a100-verification
```

The verification controller sets `cpu=(1, 2)` and `memory=(4096, 8192)`, with the same CUDA/PyTorch image as `run_modal.py`. It runs the unchanged `mnist.py` in a clean subprocess with `MNIST_SANDBOX=required`.

## Budget and verification

Four launch attempts reserve **$3.70**: $0.80 each plus $0.50 for CPU image building, below the requested $5 limit. Attempt 1 failed during controller startup and was stopped before any scoring calls; its full reservation remains in the ledger. The controller import was repaired without changing the submission or scorer. Attempts 2–4 are the three scored runs. These are conservative reservations, not a provider billing receipt. Modal's billing API reports **$0.22687950** in compute across these four app IDs (about **$0.23**), including the failed startup. All four apps are stopped with zero tasks; the itemized billing snapshot is included below.

All three passing containers reported NVIDIA A100-SXM4-80GB, Python 3.13.0, PyTorch 2.12.0+cu130, and sandbox on. Their GPU UUIDs are distinct and recorded in the evidence. The median is 11.40× faster than the listed 700.1 ms example baseline; aggregate MNIST accuracy across 330,000 predictions is 95.166%.

The local CPU scorer suite initially passed 49 tests; its stdout-isolation test hit the timing-dispersion gate on macOS. That test passed on rerun. The scorer was not modified. Local validation records both outcomes.

SHA-256:

- Submission: `44cb0c03d4ac869e99186c584d234c8743c12565e0a3b49cf3172bd5c8a82325`
- Official scorer: `2833ba470776314eabd00f8315561f31da16f839740f5776b58fb8b132248022`

## Files

- [Standalone submission](fast_mlp.py)
- [Budgeted verification controller](verify_modal.py)
- [Original algorithm](../../../mnist/experiments/release-cutoffs-20260925/mlp_timing/submissions/mlpg-k1-w256-s200-b512.py)
- [Attempt 2 evidence](evidence/run-2.json), [attempt 3 evidence](evidence/run-3.json), [attempt 4 evidence](evidence/run-4.json)
- [Aggregate results](evidence/summary.json), [Modal billing](evidence/modal-billing.json)
- [Budget ledger](evidence/budget.json), [startup failure](evidence/startup-failure.log), [local validation](evidence/local-validation.json)
