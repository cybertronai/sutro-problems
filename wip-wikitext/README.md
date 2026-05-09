# wikitext (WIP)

> *Work in progress.* The v0 reference scorer, both baselines, and the
> Modal A100 harness exist (this folder). What's still pending is the
> first record — submit one with [`submit.py`](submit.py); the empty
> Record History table at the bottom is where it lands.

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

`E_max` is pinned at **100 kJ** in [`task.py`](task.py) (≈5 min × 329 W
avg net on the pinned A100 SXM4). `acc_min` is unset until a baseline
record exists to anchor it; see Open items.

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

- **Hardware**: pinned **Modal A100 40GB SXM4** (`gpu="A100-40GB"`).
  Same silicon as Lambda's `gpu_1x_a100_sxm4`, so prior energy
  calibrations carry over. Documented fallback if Modal capacity is
  unavailable: RunPod Secure A100 40GB SXM4.
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
record-class run. `submit.py` invokes it inside the Modal container
on every run; first failure aborts before training spend.

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
| `task.py`                  | task-pinned constants (`TEST_CHARS`, `INSTANCE_TYPE`, `E_MAX_JOULES`) — single source of truth |
| `run_eval.py`              | CLI: trains a baseline or user submission (energy-measured), then evals |
| `submit.py`                | end-to-end submission orchestrator: defines a Modal A100 function and runs it |
| `fetch_data.py`            | HuggingFace WikiText-103 fetch (the canonical S3 URL is dead)    |
| `example_submission.py`    | reference submission file (5-gram wrapper) — copy and edit       |
| `submission_modded_nanogpt.py` | byte-vocab port of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (Muon + RoPE + ReLU²) for 1xA100-40GB |
| `verify_nvml.py`           | NVML energy-counter verification for a target host               |
| `test_wikitext.py`         | tests for the evaluator + n-gram                                 |
| `RUNBOOK.md`               | NVML verification + manual baseline experimentation on Modal A100 |
| `.env.example`             | optional CI env vars `submit.py` reads (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`) |
| `submissions/`             | result JSON + NVML JSON evidence per submission                  |

## Setup

Pre-requisites: Python 3.11 or higher and a Modal account.

```bash
pip install modal
modal token new      # opens a browser, writes ~/.modal.toml
```

For CI / non-interactive setups where `~/.modal.toml` isn't available,
set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the environment (or in
a `.env` file alongside `submit.py` — see [`.env.example`](.env.example)).

PyTorch + nvidia-ml-py + datasets and WikiText-103 itself are baked into
the Modal image at build time. Modal builds the image once on first
`submit.py` run (~1 min HuggingFace fetch); every subsequent run reuses
the cached layers, so editing your submission code does **not** trigger
a re-fetch. No Modal Volume to manage.

## Running

```bash
# Tests, locally (no GPU needed).
python3 test_wikitext.py

# Quick smoke run on the full pipeline (works on CPU; energy=NOT MEASURED).
# (Note: torch-based submissions also need `pip install torch` locally —
# submit.py runs an import-only precheck before any Modal spend.)
python3 run_eval.py --data-dir /path/to/wikitext-103-raw --baseline ngram --n 5

# Reference transformer run on a Modal A100 (see RUNBOOK.md).
python3 run_eval.py --data-dir /path/to/wikitext-103-raw \
    --baseline transformer --config small --n-steps 30000
```

## Submission

The standard path is [`submit.py`](submit.py). Write a Python file
that exposes:

```python
def train(train_text: str, valid_text: str | None = None) -> CharModel: ...
```

Then ship it:

```bash
cp example_submission.py my_submission.py    # edit to use your model
python3 submit.py my_submission.py
```

`submit.py` takes no model-sizing flags — training cost/timeout is
fixed (see `EST_INSTANCE_MIN` in `submit.py`). Per-config knobs (e.g.
`baseline_transformer`'s `config="tiny"|"small"|"gpt2"`, `n_steps`,
`peak_lr`, …) live inside your submission file's `train()` function.
The `--config` flag *only* applies to manual `run_eval.py` runs (see
[`RUNBOOK.md`](RUNBOOK.md) §3).

`submit.py` defines a Modal app with the harness *and* WikiText-103
baked into the image, calls a single A100-40GB function with your
submission file's bytes as the only argument, and the function runs the
pipeline end-to-end (NVML probe → train under `EnergyMeter` → 60K-char
eval) and returns the result dict. After the result lands locally,
`submit.py` saves the JSON to `submissions/` and appends a row to the
Record History below.

Modal builds the image, hosts it, runs the function, and auto-shuts-down
when it returns. No GHCR, no Docker login, no SSH key, no leaked-instance
cleanup. See [Setup](#setup) for one-time install instructions.

**Task constants** ([`task.py`](task.py)) define the leaderboard
contract: `TEST_CHARS=60_000`, `INSTANCE_TYPE=modal:A100-40GB`,
`E_MAX_JOULES=100_000`. Submitters cannot vary these — `submit.py`
reads them from `task.py` (baked into the image) and forwards them to
`run_eval.py` inside the container.

Submitter pays for their training run on the pinned Modal SKU. An
official re-evaluator (TBD — see Open items) re-runs the same Modal
function on the pushed submission and reproduces the reported (energy,
accuracy) within tolerance.

## Reproducing the modded-nanogpt submission

[`submission_modded_nanogpt.py`](submission_modded_nanogpt.py) is a port
of [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)'s
"simple" recipe to the byte-vocab / single-A100 / WikiText-103 regime.
Tricks ported verbatim from upstream:

- **Muon optimizer** (Newton-Schulz orthogonalized momentum) for the
  2-D block weights; AdamW for embeddings, lm_head, and 1-D scalars.
- Half-truncate **RoPE** with base-freq tuning (`base=1024`, second
  half of angular_freq zeroed).
- **QK RMSNorm before RoPE**, attention scale `0.12`.
- **ReLU² MLP**, RMSNorm pre-norm, soft-capped logits (cap 15).
- Zero-init projections + lm_head; "stable then decay" LR schedule
  (`cooldown_frac=0.7`).

Adaptations from upstream: `vocab_size` 50304 → 256 (raw bytes),
distributed all-gather stripped from Muon (single-GPU), and a streaming
KV-cache wrapper added so the same module serves training and the
`CharModel` per-byte interface.

Reproduce on the pinned Modal A100-40GB:

```bash
python3 submit.py submission_modded_nanogpt.py
```

Defaults (in `submission_modded_nanogpt.py`'s `TrainConfig`): ~22M
params, 2400 steps, batch 32, seq 1024 — should land within the
100 kJ training-energy budget pinned in [`task.py`](task.py).
`submit.py` prints the energy / accuracy block at the end and appends
a row to the [Record History](#record-history) table.

To experiment without forking the file, edit the `TrainConfig` defaults
(`model_dim`, `num_layers`, `n_steps`, per-group LRs, `cooldown_frac`,
`muon_wd`, …) and re-run.

**Estimated cost**: ~$0.35 per run at Modal's $2.10/hr A100-40GB rate
(~10 min wall clock for a fresh-image build, ~7 min for warm runs).

This is a *port*, not a 1:1 reproduction of the upstream record:
upstream targets FineWeb tokens on 8xH100 (3.28 cross-entropy in
under 90 s); this benchmark is bytes on 1xA100-40GB (greedy
char-accuracy under 100 kJ). LRs are still upstream defaults and almost
certainly want re-tuning for the byte / A100 / WikiText regime — that
is the obvious next record-improvement.

## Open items

These block "promote out of WIP", not "ship the v0 scorer":

- **Primary leaderboard**: pick fixed-budget or fixed-floor (or run
  both indefinitely). Currently both are reported; the
  primary-vs-secondary call waits on having actual records to point at.
- **`acc_min`**: needs an anchor from a real Lambda A100 baseline run.
- **Official re-evaluator**: who runs the reproduction pass on
  submitted images.
- **Reproduction tolerance**: ±X% on energy, ±Y points on accuracy.
  Needs run-to-run variance numbers from a real host.

## Record History

*(no records yet — first row lands after `submit.py` runs)*

| Date | Energy (J) | Char-acc | Config | Submission | Contributor |
|------|-----------:|---------:|--------|------------|-------------|
| —    |          — |        — | —      | —          | —           |
