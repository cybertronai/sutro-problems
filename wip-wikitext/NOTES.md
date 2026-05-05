# Developer notes — wip-wikitext

Onboarding notes for someone picking this up cold. Pairs with
[`README.md`](README.md) (problem definition) and
[`RUNBOOK.md`](RUNBOOK.md) (Lambda provisioning steps); this file is
the *why* and the gotchas, not the *what*.

Status as of **2026-05-05**: v0 implementation landed; first baseline
training run in flight on a Lambda A100 SXM4 40GB; record-history
table not yet populated. See [Status](#status) below.

---

## Design philosophy

Five decisions that shape every other choice in the folder.

### 1. Streaming `CharModel` API, not batched `score(text)`

The model exposes ``reset()`` / ``predict()`` / ``observe(char)``. The
runner drives the loop; the model never sees a character before
emitting that character's distribution.

**Why:** future-peeking has to be *structurally* impossible, not
*policy* impossible. We trust submitters but worry their coding agents
will quietly introduce bidirectional attention or test-set leakage and
the human won't notice. A batched `score(full_text) -> dist[N, 256]`
API can't defend against that without extra audit machinery; a
streaming API defends by construction.

KV-caching keeps the streaming path O(1) marginal per char inside the
model — no throughput penalty for the safety property.

### 2. Empirical energy measurement, not proxies

We measure NVML's monotonic energy counter
(`nvmlDeviceGetTotalEnergyConsumption`, Volta+) on a pinned hardware
platform. **No theoretical approximator.** A wall-clock-time × TDP
proxy was discussed and explicitly rejected — its biases (under-rewarding
memory-efficient methods, over-rewarding overclocking) erase exactly
what the benchmark exists to measure.

This drives the provider choice (NVML must be exposed, ruling out
Modal and other serverless GPU hosts).

### 3. Char-accuracy, not perplexity or x-entropy

Greedy argmax accuracy at every position, computed against the held-out
test split. Reason: tokenization-agnostic. Any model that produces
`P(next_char | prefix)` is comparable, regardless of whether it
internally uses BPE / SentencePiece / raw bytes.

### 4. Training-from-scratch only

No pre-trained weights. WikiText overlaps WebText (GPT-2's training
corpus); allowing pre-trained init poisons the comparison.

### 5. Training energy only, not inference energy

The submitter wraps **training** with `EnergyMeter`. Eval is run for
correctness only and not energy-charged in v0. Reason: matches the
"energy-efficient *learning*" framing; inference accounting can be
layered on later as a tier-2 leaderboard.

---

## Gotchas

### WikiText-103 canonical S3 URL is dead — use HuggingFace

The "canonical" URL,

    https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip

returns an S3 PermanentRedirect. The redirect target
(`research.metamind.io.s3.amazonaws.com/...`) breaks SSL SNI because the
S3 wildcard cert doesn't cover the dotted-bucket hostname — **both
direct paths fail**.

Workaround (now baked into `RUNBOOK.md` step 2 and the
`load_wikitext103` error message): pull from HuggingFace via the
`datasets` library and write out `wiki.{train,valid,test}.raw`:

```python
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
text = "\n".join(ds["train"]["text"])
```

`load_wikitext103` itself still expects the local raw files (so the
submission Docker doesn't need a `datasets` dep at run time); the HF
fetch is a one-shot host-side step.

### NVML energy counter is virtualized away on most clouds

`nvmlDeviceGetTotalEnergyConsumption` is exposed only on bare-metal or
near-bare-metal hosts. Confirmed exposed (as of 2026-05-05): Lambda
On-Demand, RunPod Secure, AWS p4d/p4de, GCP a2, CoreWeave. Confirmed
*not* exposed: Modal. Treat any "serverless GPU" provider as
unverified until you've run `verify_nvml.py` there.

This is the single biggest constraint on provider choice and the
reason RUNBOOK.md pins Lambda (with RunPod Secure as documented
fallback).

### A100 80GB SXM4 has frequently-zero capacity on Lambda

The README pins A100 80GB. The 40GB SXM4 SKU is the same chip, same
400W TDP, half the memory — energy numbers should be directly
comparable. The current in-flight baseline is on 40GB. If/when 80GB
returns and we want to re-anchor, the cost difference per run should
be small.

**Important:** never substitute a *different-chip* GPU (e.g. H100) for
A100 without explicit user approval. TDP, tensor-core efficiency, and
memory hierarchy all differ — the energy numbers stop being
comparable. There's a memory entry on this.

### Streaming eval is slow — standard eval is a 60K-char slice

The streaming `evaluate()` runs at ~460–660 chars/s on an A100 SXM4
40GB for the small-config transformer (4.94M params). The full 1.3M
test split takes 33–47 min at that rate — longer than the 17-min
training phase that produced the model.

Standard eval is therefore the **first 60,000 chars** of the test
split (`--max-test-chars 60000`, the default in `run_eval.py`). That
finishes in ~2 min and gives a 95% CI of ±0.4–1.3pp on accuracy
(naive Bernoulli SE inflated 5–10× for char-level autocorrelation),
which is tight enough to rank submissions. The same slice is used
for dev iteration and for the record table — no two-tier confusion.

Pass `--max-test-chars 0` to score the full 1.3M chars if you want
the (much) tighter CI; not required for the leaderboard.

Why streaming is slow at all: each char is a single forward pass with
one new query, plus softmax + dict construction in Python. GPU sits
at ~36% utilization because kernels are tiny. Quick wins not yet
attempted (left as follow-ups): pre-compute the byte→char map in
`predict()`, run inference under bf16 autocast, capture the per-token
forward as a CUDA graph.

The progress indicator (`progress_every` parameter on `evaluate()`)
makes the wait visible but does not speed it up.

### Lambda doesn't auto-stop instances

Lambda bills per minute and does *not* auto-terminate idle instances.
Always tear down via the API or web console as soon as a run
completes, or you'll pay overnight rates for nothing.

### Original `run_eval.py` had no checkpointing

If eval crashed after training, the trained model was lost. Fixed
(2026-05-05): `--save-model PATH` flag added; pass it to write the
state dict + config to disk after training so eval can be retried
without retraining.

---

## Status

### Done

- `wikitext.py` — `CharModel` ABC, `evaluate()` with progress
  indicator, `EnergyMeter` (NVML, no fudged proxy on no-GPU hosts),
  `load_wikitext103()`.
- `baseline_ngram.py` — stupid-backoff char n-gram, no torch dep.
- `baseline_transformer.py` — small GPT-2-style char transformer,
  three configs (`tiny` / `small` / `gpt2` ≈ 0.6M / 5M / 22M params),
  weight tying, bf16 autocast in training, KV-cached streaming.
- `run_eval.py` — CLI runner; trains under `EnergyMeter`, optionally
  saves a checkpoint, then evaluates.
- `test_wikitext.py` — 7 tests, all passing on CPU. Includes a
  structural test that `predict()` is always called before the
  matching `observe()`.
- `Dockerfile` — submitter harness template, pins
  `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` + `nvidia-ml-py`.
- `verify_nvml.py` — verification script. Run on Lambda A100 SXM4
  40GB on 2026-05-05: idle 45 W, 30 s stress drew 11.6 kJ at 329 W
  avg, counter monotonic ✓.
- `RUNBOOK.md` — Lambda provisioning + verification + train +
  capture procedure with cost estimates.

### In flight (as of 2026-05-05)

- Re-run of `small`-config baseline (4.94M params, 30K steps) on
  Lambda A100 SXM4 40GB with progress indicator and `--save-model`.
  First run got the training number (186,868 J / 1032 s) but its eval
  was killed without progress visibility. The re-run is at step
  ~11,600 / 30,000 at last check.

### Outstanding

- Populate first row of the record-history table in `README.md`
  once the in-flight run finishes.
- Speedups for streaming eval (pre-compute byte→char map, bf16
  inference, CUDA graphs).
- Add `wip-wikitext/` to root `README.md` "Problems" list.
- Pick **leaderboard framing**: fixed-budget vs fixed-floor (or
  both).
- Pick **target numbers**: budget level (e.g. 1 MJ? 100 kJ?) or
  accuracy floor.
- Designate **official evaluator** (who runs the re-evaluation pass).
- Pick **reproduction tolerance** (energy ±X%, accuracy ±Y points).
- Optional: run `gpt2` config (~22M params, ~$10 on A100) once
  `small` is in the table.
- Optional: re-run on A100 80GB once Lambda capacity returns, to
  match the README's pinned platform exactly.

---

## File map

| file                       | purpose                                                          |
|----------------------------|------------------------------------------------------------------|
| `README.md`                | Problem definition, design rationale, open items                 |
| `RUNBOOK.md`               | Lambda provisioning + train + capture procedure                  |
| `NOTES.md`                 | (this file) Why decisions, gotchas, status                       |
| `wikitext.py`              | `CharModel` ABC, `evaluate()`, `EnergyMeter`, data loader        |
| `baseline_ngram.py`        | Char n-gram baseline (no torch)                                  |
| `baseline_transformer.py`  | GPT-2-style char transformer baseline (PyTorch)                  |
| `run_eval.py`              | CLI: train under `EnergyMeter`, optionally checkpoint, eval      |
| `test_wikitext.py`         | Pytest-and-stdlib-runnable tests                                 |
| `verify_nvml.py`           | NVML energy-counter verification script                          |
| `Dockerfile`               | Submitter harness template                                       |

---

## How to extend

To add a new baseline:

1. Subclass `CharModel`. Implement `reset` / `predict` / `observe`.
2. Add a training entry point that takes `text: str` and returns the
   trained model.
3. Add a branch in `run_eval.py` to wire the CLI through.
4. Run `test_wikitext.py` to make sure the streaming contract still
   holds.

To submit a record:

1. Build a Docker image extending `Dockerfile` with your training
   code.
2. Run on a Lambda A100 (see `RUNBOOK.md`) with `--network=none`
   during the eval phase. Test set must not be present in the
   container during training.
3. Capture (training_energy_J, test_char_acc) from the runner's final
   output.
4. PR with: log file under `submissions/`, your run's
   `verify_nvml.py` JSON line, and a row appended to the
   record-history table.
