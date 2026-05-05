# wikitext (WIP)

> *Work in progress.* The v0 reference scorer, both baselines, and the
> Docker harness exist (this folder). What's still pending is the first
> Lambda A100 run that produces a real (energy, char-acc) record — see
> [`RUNBOOK.md`](RUNBOOK.md) for the manual procedure, and the empty
> Record History table at the bottom for the slot it lands in.

## Motivation

Maximize next-character prediction accuracy on WikiText-103 under a
limited training-energy budget.

Strong character-level next-token prediction tends to require
language understanding regardless of the model class chosen — unlike
e.g. sparse-parity, where a solver can score perfectly using
operator-specific tricks. That makes char-accuracy a defensible proxy
for "is your training procedure energy-efficient at *language*?", not
just at the specific benchmark task.

Char-accuracy (greedy argmax over `P(next_char | prefix)`) is chosen
over token-accuracy or cross-entropy specifically so the metric is
**tokenization-agnostic**: any model that can produce a next-character
distribution is comparable, regardless of internal vocabulary.

## Problem

Train a character-level language model from scratch on **WikiText-103**.
Use the standard train/valid/test split. The model exposes a streaming
next-character distribution; the runner scores it on the **first
60,000 chars** of the held-out test split by greedy-argmax
char-accuracy. (60K is fixed across submissions for comparability;
~2 min eval, ±0.4–1.3pp 95% CI. The full 1.3M-char split is available
via `--max-test-chars 0` but is not required for the leaderboard.)

Two leaderboard framings ship side-by-side; both report the same two
numbers, only the constraint differs:

| framing          | knob          | metric              |
|------------------|---------------|---------------------|
| **fixed-budget** | `E_max` joules  | maximize char-acc   |
| **fixed-floor**  | `acc_min`       | minimize joules     |

(v0 leaves the actual `E_max` / `acc_min` thresholds open until we
have a real reference number from the Lambda A100 run — see Record
History below.)

## API

```python
import wikitext

# Streaming next-character interface. Future-peeking is structurally
# impossible: the runner calls predict() for position i strictly
# before observe() commits the ground-truth char at position i.
class CharModel:
    def reset(self) -> None: ...
    def predict(self) -> dict[str, float]: ...   # P(next_char | so_far)
    def observe(self, char: str) -> None: ...    # commit ground-truth

# Greedy-argmax char-accuracy; one stream, one reset() at start.
result = wikitext.evaluate(model, test_text)
print(result.accuracy)

# nvmlDeviceGetTotalEnergyConsumption-based meter; reports
# E_run - idle_watts * duration in joules. None on hosts without NVML.
meter = wikitext.EnergyMeter()
with meter.measure() as m:
    train_my_model()
print(m.energy_joules)
```

The two reference baselines plug into `CharModel`:

```python
from baseline_ngram import NGramModel
from baseline_transformer import train_transformer, TransformerModel

ngram = NGramModel(n=5);              ngram.train(train_text)
xfmr  = train_transformer(train_text, config="small", n_steps=30_000)
xfmr_streamer = TransformerModel(xfmr)   # KV-cached streaming wrapper
```

## Energy measurement

- **Hardware**: pinned **Lambda On-Demand A100 80GB**. Documented
  fallback if capacity unavailable: RunPod Secure A100 80GB.
- **Counter**: `nvmlDeviceGetTotalEnergyConsumption` — monotonic
  millijoule counter exposed on Volta+. Read at run start, read at
  run end, subtract. No sampling-rate error, no power-draw integration.
- **Idle subtraction**: calibrate `idle_power × duration` once per
  host (≈50 W on A100); subtract from `E_run`.
- **Reported energy**: `E_run − E_idle`, in joules.
- **Scope**: training only — inference-time energy is not charged in
  v0.

[`verify_nvml.py`](verify_nvml.py) confirms the counter is exposed,
monotonic, and produces plausible Watts on the chosen SKU before any
record-class run.

## Anti-cheat

The streaming `CharModel` API makes within-document future-peeking
structurally impossible: the model emits position-`i`'s distribution
before being told the ground-truth at position `i`. This defends
against a coding agent unintentionally introducing bidirectional
attention, batched-prefix scoring, etc.

Throughput is preserved via internal KV-cache: each
`observe`/`predict` pair is `O(1)` marginal — same wall-time as a single
batched forward pass. Submitters parallelize across documents by
instantiating N model instances and feeding N independent streams.

Beyond that:

- Eval container runs with `--network=none`.
- Test set is **not present** in the container during training;
  mounted read-only only during the eval phase.
- Pre-trained weights disallowed (train-from-scratch only).

That's it for v0. We trust the submitter; the design above defends
against unintentional cheating from coding agents.

## Files

| file                       | purpose                                                          |
|----------------------------|------------------------------------------------------------------|
| `wikitext.py`              | `CharModel` ABC, streaming `evaluate`, `EnergyMeter`, data loader |
| `baseline_ngram.py`        | n-gram baseline with stupid-backoff smoothing (no torch dep)     |
| `baseline_transformer.py`  | small GPT-2-style transformer with KV-cached streaming (PyTorch) |
| `run_eval.py`              | CLI: trains a baseline (energy-measured), then evals             |
| `verify_nvml.py`           | NVML energy-counter verification for a target host               |
| `test_wikitext.py`         | tests for the evaluator + n-gram                                 |
| `Dockerfile`               | submitter harness template (PyTorch 2.5.1 + CUDA 12.4)           |
| `RUNBOOK.md`               | manual Lambda A100 procedure for the first record run            |
| `submissions/`             | training logs + NVML JSON evidence per submission                |

## Running

```bash
# Tests, locally (no GPU needed).
python3 test_wikitext.py

# Quick smoke run on the full pipeline (works on CPU; energy=NOT MEASURED).
python3 run_eval.py --data-dir /path/to/wikitext-103-raw --baseline ngram --n 5

# Reference transformer run on a Lambda A100 (see RUNBOOK.md).
python3 run_eval.py --data-dir /path/to/wikitext-103-raw \
    --baseline transformer --config small --n-steps 30000
```

## Submission format

A Docker image + entrypoint that:

1. Trains from scratch on WikiText-103 train split.
2. Exposes a `CharModel` matching the API above.
3. Prints the final `training energy (J)` / `test char-accuracy`
   block in the same format as `run_eval.py`.

Submitter pays for their training run on the pinned Lambda SKU. The
runner re-runs the image, reproduces the reported (energy, accuracy)
within tolerance, and updates the record table.

## Open items

These block "promote out of WIP", not "ship the v0 scorer":

- **Primary leaderboard**: pick fixed-budget or fixed-floor (or run
  both indefinitely). Currently both are reported; the
  primary-vs-secondary call waits on having actual records to point at.
- **Target numbers**: pick `E_max` or `acc_min`. Needs the first real
  Lambda A100 run for an anchor.
- **Official re-evaluator**: who runs the reproduction pass on
  submitted images.
- **Reproduction tolerance**: ±X% on energy, ±Y points on accuracy.
  Needs run-to-run variance numbers from a real host.

## Record History

*(no records yet — first row lands after the Lambda A100 run in
[`RUNBOOK.md`](RUNBOOK.md))*

| Date | Energy (J) | Char-acc | Config | Submission | Contributor |
|------|-----------:|---------:|--------|------------|-------------|
| —    |          — |        — | —      | —          | —           |
