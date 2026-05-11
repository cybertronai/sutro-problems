# wikitext (WIP)

Char-level WikiText-103 under a 100 kJ training-energy budget on a
pinned Modal A100-40GB. Maximize greedy-argmax char-accuracy on the
first 60K chars of test under the budget.

> *Work in progress.* No records yet — first row lands after
> `submit.py` runs.

## Setup

Pre-requisites: Python 3.11+ and a Modal account.

```bash
pip install modal
modal token new      # opens a browser, writes ~/.modal.toml
```

For CI / non-interactive setups, set `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` in the environment (or in a `.env` file alongside
`submit.py` — see [`.env.example`](.env.example)).

PyTorch-based submissions also need `pip install torch` locally;
`submit.py` runs an import-only precheck before any Modal spend.

## Local CPU smoke

Run this first if you only want to check the evaluator and n-gram
baseline locally. It does not require Docker, Modal, a GPU, NVML, cloud
credentials, or a WikiText-103 download.

```bash
# From the repo root.
nix develop -c python3 wip-wikitext/test_wikitext.py
nix develop -c python3 wip-wikitext/run_eval.py \
    --data-dir wip-wikitext/fixtures/tiny \
    --baseline ngram --n 3 --max-test-chars 300 --progress-every 0
```

Equivalent without Nix, if `python3` is already available:

```bash
cd wip-wikitext
python3 test_wikitext.py
python3 run_eval.py --data-dir fixtures/tiny \
    --baseline ngram --n 3 --max-test-chars 300 --progress-every 0
```

## Reproducing the modded-nanogpt submission

```bash
python3 submit.py submission_modded_nanogpt.py
```

[`submission_modded_nanogpt.py`](submission_modded_nanogpt.py) is a
byte-vocab port of the modded-nanogpt "simple" recipe (Muon + RoPE +
QK RMSNorm + ReLU² + zero-init projections + stable-then-decay LR) for
1xA100-40GB. Defaults (in `TrainConfig`): ~22M params, 2400 steps,
batch 32, seq 1024 — fits within the 100 kJ budget pinned in
[`task.py`](task.py). ~$0.35 / ~10 min per fresh-image run on Modal's
$2.10/hr A100-40GB rate.

A *port*, not a 1:1 reproduction: upstream targets FineWeb tokens on
8xH100 in <90 s; this is bytes on 1xA100-40GB under 100 kJ. LRs are
still upstream defaults and almost certainly want re-tuning — the
obvious next record-improvement. See the file's docstring for the full
list of tricks ported / adapted / dropped.

## Submitting your own model

Write a Python file exposing:

```python
def train(train_text: str, valid_text: str | None = None) -> CharModel: ...
```

then ship it:

```bash
cp example_submission.py my_submission.py    # edit to use your model
python3 submit.py my_submission.py
```

`submit.py` defines a Modal app that pulls a prebuilt public image
(`ghcr.io/ab-10/wikitext-bench`, source: [`Dockerfile`](Dockerfile))
containing torch + nvidia-ml-py + pyarrow with the WikiText-103 raw
splits already baked into `/data`. It calls a single A100-40GB
function with your file's bytes as the only argument and runs the
pipeline end-to-end (NVML probe → train under `EnergyMeter` → 60K-char
eval), returning the result dict. Modal caches the registry digest, so
cold start is just the one-time ~85s pull; harness / submission edits
do not re-pull or re-fetch the dataset.

After the result lands locally, `submit.py` saves the JSON to
`submissions/` and appends a row to the [Record History](#record-history).

Per-config knobs (model size, n_steps, peak_lr, …) live inside your
file's `train()` function. `submit.py` itself takes no model-sizing
flags — `task.py`'s `TEST_CHARS=60_000`,
`INSTANCE_TYPE=modal:A100-40GB`, `E_MAX_JOULES=100_000` are leaderboard
rules and submitters cannot vary them.

Submitter pays for their training run; an official re-evaluator (TBD —
see [Open items](#open-items)) re-runs the same Modal function on the
pushed submission to reproduce the reported (energy, accuracy) within
tolerance.

## Problem

Train a character-level language model from scratch on **WikiText-103**
using the standard train/valid/test split. The model exposes a
streaming next-character distribution; the runner scores it on the
**first 60,000 chars** of the held-out test split by greedy-argmax
char-accuracy. (60K is fixed for comparability; ~2 min eval, ±0.4–1.3pp
95% CI. Pass `--max-test-chars 0` to score the full 1.3M-char split;
not required for the leaderboard.)

Two leaderboard framings ship side-by-side; same two numbers, only the
constraint differs:

| framing          | knob          | metric              |
|------------------|---------------|---------------------|
| **fixed-budget** | `E_max` joules  | maximize char-acc   |
| **fixed-floor**  | `acc_min`       | minimize joules     |

`E_max` is pinned at **100 kJ** in [`task.py`](task.py) (≈5 min × 329 W
avg net on the pinned A100 SXM4). `acc_min` is unset until a baseline
record exists.

Char-accuracy (greedy argmax over `P(next_char | prefix)`) is chosen
over token-accuracy or cross-entropy specifically so the metric is
**tokenization-agnostic**: any model that produces a next-character
distribution is comparable, regardless of internal vocabulary.
Pre-trained weights are disallowed (train-from-scratch only) —
WikiText overlaps WebText, so allowing pretrained init poisons the
comparison.

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
  Energy numbers are comparable only on this pinned SKU and runner
  configuration unless a fallback provider is separately verified and
  recorded as a distinct `INSTANCE_TYPE`.
- **Counter**: `nvmlDeviceGetTotalEnergyConsumption` — monotonic
  millijoule counter exposed on Volta+. Read at run start, read at run
  end, subtract.
- **Idle subtraction**: calibrate `idle_power × duration` once per host
  (≈50 W on A100); subtract from `E_run`.
- **Reported energy**: `E_run − E_idle`, in joules.
- **Scope**: training only — inference-time energy is not charged in v0.

[`verify_nvml.py`](verify_nvml.py) confirms the counter is exposed,
monotonic, and produces plausible Watts on the chosen SKU before any
record-class run. `submit.py` invokes it inside the Modal container on
every run; first failure aborts before training spend.

## Anti-cheat

The streaming `CharModel` API makes within-document future-peeking
structurally impossible: the model emits position-`i`'s distribution
before being told the ground-truth at position `i`. This defends
against a coding agent unintentionally introducing bidirectional
attention, batched-prefix scoring, etc.

Throughput is preserved via internal KV-cache: each `observe`/`predict`
pair is `O(1)` marginal — same wall-time as a single batched forward
pass.

Beyond that:

- The current Modal runner bakes the fixed train/valid/test raw splits
  into `/data` for reproducibility and faster warm runs. This is not a
  test-hiding mechanism.
- No container-level network isolation or train/eval split remounting
  is implemented in v0. Add those before treating this as an adversarial
  leaderboard.
- Pre-trained weights disallowed (train-from-scratch only).

That's it for v0. We trust the submitter; the design above defends
against unintentional cheating from coding agents.

## Files

| file                       | purpose                                                          |
|----------------------------|------------------------------------------------------------------|
| `wikitext.py`              | `CharModel` ABC, streaming `evaluate`, `EnergyMeter`, data loader |
| `baseline_ngram.py`        | n-gram baseline with stupid-backoff smoothing (no torch dep)     |
| `baseline_transformer.py`  | small GPT-2-style transformer with KV-cached streaming (PyTorch) |
| `task.py`                  | task-pinned constants (`TEST_CHARS`, `INSTANCE_TYPE`, `E_MAX_JOULES`) |
| `run_eval.py`              | CLI: trains a baseline or user submission (energy-measured), then evals |
| `submit.py`                | end-to-end submission orchestrator: defines a Modal A100 function and runs it |
| `Dockerfile`               | builds `ghcr.io/ab-10/wikitext-bench` (torch + pyarrow + WikiText-103 baked into `/data`) — pulled by `submit.py` via `Image.from_registry` |
| `fetch_data.py`            | downloads WikiText-103 from `gs://wikitext-103-raw-v1` to a local dir — used to stage `wikitext-103-raw-v1/` for the Dockerfile build context |
| `bake_wikitext.py`         | parquet → `wiki.{split}.raw` helper used by `fetch_data.py`      |
| `example_submission.py`    | reference submission file (5-gram wrapper) — copy and edit       |
| `submission_modded_nanogpt.py` | byte-vocab port of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (Muon + RoPE + ReLU²) for 1xA100-40GB |
| `fixtures/tiny/`           | tiny committed raw splits for local CPU smoke tests              |
| `verify_nvml.py`           | NVML energy-counter verification for a target host               |
| `test_wikitext.py`         | tests for the evaluator + n-gram                                 |
| `RUNBOOK.md`               | NVML verification + manual baseline experimentation on Modal A100 |
| `.env.example`             | optional CI env vars `submit.py` reads (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`) |
| `submissions/`             | result JSON + NVML JSON evidence per submission                  |

## Open items

These block "promote out of WIP", not "ship the v0 scorer":

- **Primary leaderboard**: pick fixed-budget or fixed-floor (or run
  both indefinitely). Currently both are reported; the
  primary-vs-secondary call waits on having actual records to point at.
- **`acc_min`**: needs an anchor from a real Modal A100 baseline run.
- **Official re-evaluator**: who runs the reproduction pass on
  submitted records.
- **Reproduction tolerance**: ±X% on energy, ±Y points on accuracy.
  Needs run-to-run variance numbers from a real host.

## Record History

*(no records yet — first row lands after `submit.py` runs)*

| Date | Energy (J) | Char-acc | Config | Submission | Contributor |
|------|-----------:|---------:|--------|------------|-------------|
| 2026-05-11 |      7,249 | 0.6228 | example_submission | [json](submissions/example_submission_2026-05-11.json) | @you |
