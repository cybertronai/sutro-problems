# wikitext (WIP)

> *Work in progress.* The v0 reference scorer, both baselines, and the
> Docker harness exist (this folder). What's still pending is the first
> Lambda A100 record — submit one with [`submit.py`](submit.py); the
> empty Record History table at the bottom is where it lands.

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

- **Hardware**: pinned **Lambda On-Demand A100 40GB SXM4**
  (`gpu_1x_a100_sxm4`). Documented fallback if capacity unavailable:
  RunPod Secure A100 40GB SXM4.
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
| `task.py`                  | task-pinned constants (`TEST_CHARS`, `INSTANCE_TYPE`, `E_MAX_JOULES`) — single source of truth |
| `run_eval.py`              | CLI: trains a baseline or user submission (energy-measured), then evals |
| `submit.py`                | end-to-end submission orchestrator: build image → Lambda A100 → result |
| `entrypoint.sh`            | container entrypoint: data fetch → NVML probe → `run_eval.py`    |
| `fetch_data.py`            | HuggingFace WikiText-103 fetch (the canonical S3 URL is dead)    |
| `example_submission.py`    | reference submission file (5-gram wrapper) — copy and edit       |
| `fixtures/tiny/`           | tiny committed raw splits for local CPU smoke tests              |
| `verify_nvml.py`           | NVML energy-counter verification for a target host               |
| `test_wikitext.py`         | tests for the evaluator + n-gram                                 |
| `Dockerfile`               | submitter harness template (PyTorch 2.5.1 + CUDA 12.4)           |
| `RUNBOOK.md`               | NVML verification + manual baseline experimentation on Lambda A100 |
| `.env.example`             | env vars `submit.py` reads (`LAMBDA_API_KEY`, `GHCR_USER`)       |
| `submissions/`             | result JSON + training log + NVML JSON evidence per submission   |

## Running

Fast local path, CPU only:

```bash
# From the repo root. No Docker, cloud, GPU, HF download, or credentials.
nix develop -c python3 wip-wikitext/test_wikitext.py
nix develop -c python3 wip-wikitext/run_eval.py \
    --data-dir wip-wikitext/fixtures/tiny \
    --baseline ngram --n 3 --max-test-chars 300 --progress-every 0
```

Equivalent without Nix, if `python3` is already available:

```bash
cd wip-wikitext

# Tests, locally (no GPU needed).
python3 test_wikitext.py

# Quick smoke run on the tiny committed fixture (works on CPU; energy=NOT MEASURED).
python3 run_eval.py --data-dir fixtures/tiny \
    --baseline ngram --n 3 --max-test-chars 300 --progress-every 0

# Full WikiText-103 smoke run (works on CPU; energy=NOT MEASURED).
# Requires preparing wiki.{train,valid,test}.raw first; see RUNBOOK.md.
# (Note: torch-based submissions also need `pip install torch` locally —
# submit.py runs an import-only precheck before any Lambda spend.)
python3 run_eval.py --data-dir /path/to/wikitext-103-raw --baseline ngram --n 5

# Reference transformer run on a Lambda A100 (see RUNBOOK.md).
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

`submit.py` builds a Docker image around your file, pushes it to GHCR,
provisions a Lambda A100 (pinned `INSTANCE_TYPE` from `task.py`),
runs the harness end-to-end (NVML probe → data fetch → train under
`EnergyMeter` → 60K-char eval), pulls the result back, terminates the
instance in a `finally` block, and appends a row to the Record History
below.

**One-time setup** (see [`.env.example`](.env.example) for the env-var
template):

1. `LAMBDA_API_KEY` — cloud.lambda.ai → API keys.
2. `GHCR_USER` — your GitHub username; the image lands at
   `ghcr.io/<user>/wikitext-bench`.
3. `gh auth token | docker login ghcr.io -u $GHCR_USER --password-stdin`.
4. SSH key registered on the Lambda dashboard, private key in your
   local `ssh-agent`.
5. (New SKU only) run [`verify_nvml.py`](verify_nvml.py) once on the
   target host — see [`RUNBOOK.md`](RUNBOOK.md) §1 and the NVML gotcha
   in `NOTES.md`.

**Task constants** ([`task.py`](task.py)) define the leaderboard
contract: `TEST_CHARS=60_000`, `INSTANCE_TYPE=gpu_1x_a100_sxm4`,
`E_MAX_JOULES=100_000`. Submitters cannot vary these —
`entrypoint.sh` forwards them from `task.py` to `run_eval.py` inside
the container.

Submitter pays for their training run on the pinned Lambda SKU. An
official re-evaluator (TBD — see Open items) re-runs the pushed image
and reproduces the reported (energy, accuracy) within tolerance.

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
