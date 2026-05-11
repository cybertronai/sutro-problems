# Developer notes — wip-wikitext

Onboarding notes for someone picking this up cold. Pairs with
[`README.md`](README.md) (problem definition) and
[`RUNBOOK.md`](RUNBOOK.md) (Modal provisioning steps); this file is
the *why* and the gotchas, not the *what*.

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

This drives the provider choice (NVML must be exposed). `submit.py`
runs on Modal A100-40GB and gates each submission on a `verify_nvml`
probe at function start.

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

### WikiText-103 canonical S3 URL is dead — use the GCS mirror

The "canonical" URL,

    https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip

returns an S3 PermanentRedirect. The redirect target
(`research.metamind.io.s3.amazonaws.com/...`) breaks SSL SNI because the
S3 wildcard cert doesn't cover the dotted-bucket hostname — **both
direct paths fail**. We previously fell back to the
`datasets.load_dataset("Salesforce/wikitext", ...)` HuggingFace path,
but that hung inside Modal's image builder.

Workaround: pull the parquet shards (4358 / 3760 / 1.8M-row splits,
mirrored from the HF export) from the public GCS bucket
`gs://wikitext-103-raw-v1` and join the `text` column with `\n`:

```python
import io, urllib.request, pyarrow.parquet as pq
buf = urllib.request.urlopen(
    "https://storage.googleapis.com/wikitext-103-raw-v1/test-00000-of-00001.parquet"
).read()
text = "\n".join(pq.read_table(io.BytesIO(buf))["text"].to_pylist())
```

`load_wikitext103` itself still expects the local raw files. In the
Modal runner, the splits are pre-baked into `/data` of the public
`ghcr.io/ab-10/wikitext-bench` image (see [`Dockerfile`](Dockerfile)),
which `submit.py` pulls via `Image.from_registry(...)`.
[`fetch_data.py`](fetch_data.py) materialises the same splits locally
— used both for ad-hoc experiments and to stage the
`wikitext-103-raw-v1/` directory the Dockerfile `COPY`s from.

### NVML energy counter is virtualized away on some clouds

`nvmlDeviceGetTotalEnergyConsumption` is exposed only on some GPU
hosts. Treat any provider as unverified until you've run
`verify_nvml.py` there and recorded the SKU, driver stack, idle power,
and counter behavior.

`submit.py` runs `verify_nvml.py` inside the Modal container on every
submission and aborts on counter unavailability — so this constraint
is checked on each run, not just at provider-bring-up.

### Pinned SKU is A100 40GB SXM4 (`modal:A100-40GB`)

We pin the 40GB SXM4 because it has reliable capacity on Modal; the
80GB SXM4 SKU is more expensive without changing the chip. Same chip,
same 400W TDP — only memory differs — so submissions that fit in 40GB
produce energy numbers directly comparable to anything you'd measure
on 80GB.

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

### Modal auto-shuts-down; per-second billing

Modal bills per function-second of GPU time and tears the container
down automatically when the function returns. There's nothing to
terminate by hand.

### `--e-max-joules` is a soft cap, not a hard one

`EnergyMeter(e_max_joules=N)` arms a daemon thread that polls NVML
every 250 ms and, when running net energy crosses `N`, sends SIGUSR1
to the main thread; the handler raises `BudgetExceededError`. If
signal install fails (caller is on a worker thread, etc.) the
watchdog falls back to `os._exit(124)`.

Two structural limits to know:

- **Soft cap inside the submitter's process.** A submission that
  doesn't wrap training in `EnergyMeter.measure()`, or that catches
  `RuntimeError` broadly, bypasses it. Defending against that needs
  an *external* hard-floor SIGKILL at the container/wrapper level —
  not yet implemented (see Outstanding). For now we trust the
  submission contract; the in-process killswitch defends against
  honest-mistake over-budget runs, not adversarial ones.
- **No-op on CPU hosts.** Without NVML the watchdog never spawns.
  The flag is silently inert; `run_eval.py` prints a warning to make
  this visible.

The poll interval is a tradeoff: 250 ms means the energy reading at
kill can overshoot `N` by up to ~100 J on an A100 at 400 W. Tighten
via the `poll_interval_s` constructor arg if that overshoot matters
for your leaderboard tolerance.

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
  saves a checkpoint, then evaluates. `--e-max-joules N` arms an
  NVML-polling watchdog that kills training over budget and reports
  the run as DISQUALIFIED (exit 2).
- `test_wikitext.py` — stdlib-runnable tests, all passing on CPU.
  Includes a
  structural test that `predict()` is always called before the
  matching `observe()`.
- `task.py` — task-pinned constants (`TEST_CHARS`, `INSTANCE_TYPE`,
  `E_MAX_JOULES`, `ACC_MIN`). Single source of truth; submitters
  cannot vary these.
- `fetch_data.py` — local/manual fetch from `gs://wikitext-103-raw-v1`
  + write `wiki.{split}.raw` (the canonical S3 URL is dead — see Gotchas).
  Wraps `bake_wikitext.bake(out_dir)`. Used to stage the
  `wikitext-103-raw-v1/` directory the Dockerfile `COPY`s from.
- `Dockerfile` — builds `ghcr.io/ab-10/wikitext-bench` (python:3.11-slim
  + torch + nvidia-ml-py + pyarrow + WikiText-103 raw splits at
  `/data`). Pushed to ghcr.io and pulled by `submit.py` via
  `Image.from_registry`.
- `submit.py` — end-to-end orchestrator. Defines a Modal app +
  A100-40GB function using the prebuilt `ghcr.io/ab-10/wikitext-bench`
  image; the user's submission file is passed as bytes per call (no
  per-run image rebuild). The remote function verifies NVML, runs
  `run_eval.py`, and returns the result dict. `submit.py` saves the
  JSON to `submissions/` and appends one row to the Record History
  table.
- `submission_baseline.py` — baseline submission: byte-vocab
  modded-nanogpt port (~22M params, 2400 steps) sized to fit the
  100 kJ A100-40GB budget. The "first run" referenced in `README.md`.
- `verify_nvml.py` — verification script. Invoked inside the Modal
  container on every submission before training starts.
- `RUNBOOK.md` — manual Modal procedure for NVML verification on
  new SKUs and standalone baseline experimentation (not a submission
  path; submissions go through `submit.py`).

### Outstanding

- Speedups for streaming eval (pre-compute byte→char map, bf16
  inference, CUDA graphs).
- Pick **leaderboard framing**: fixed-budget vs fixed-floor (or
  both).
- Pick **`ACC_MIN`** for the fixed-floor framing — needs an anchor
  from a real Modal A100 baseline record. (`E_MAX_JOULES` is currently
  pinned at 100 kJ.)
- Designate **official evaluator** (who runs the re-evaluation pass).
- Pick **reproduction tolerance** (energy ±X%, accuracy ±Y points).
- Optional: run `gpt2` config (~22M params, ~$10 on A100) once
  `small` is in the table.
- **External hard-floor SIGKILL** at ~1.5× E_max wall-clock (or a
  fixed ceiling like 7.5 min) as a second-layer defense against
  submissions that bypass the in-meter watchdog. Lives wherever the
  official re-evaluator lives — likely a wall-clock cap on the
  Modal function (currently set to 30 min as a coarse 2× safety
  factor, not a tight enforcer), not in `wikitext.py`.

---

## File map

| file                       | purpose                                                          |
|----------------------------|------------------------------------------------------------------|
| `README.md`                | Problem definition, design rationale, open items                 |
| `RUNBOOK.md`               | Manual Modal procedure: NVML verification + baseline experiments |
| `NOTES.md`                 | (this file) Why decisions, gotchas, status                       |
| `wikitext.py`              | `CharModel` ABC, `evaluate()`, `EnergyMeter`, data loader        |
| `baseline_ngram.py`        | Char n-gram baseline (no torch)                                  |
| `baseline_transformer.py`  | GPT-2-style char transformer baseline (PyTorch)                  |
| `task.py`                  | Task-pinned constants (`TEST_CHARS`, `INSTANCE_TYPE`, `E_MAX_JOULES`) |
| `run_eval.py`              | CLI: train under `EnergyMeter`, optionally checkpoint, eval      |
| `submit.py`                | End-to-end Modal orchestrator (define app → invoke A100 fn → save result) |
| `Dockerfile`               | Builds `ghcr.io/ab-10/wikitext-bench` (torch + WikiText-103 baked into `/data`) |
| `bake_wikitext.py`         | parquet → `wiki.{split}.raw` helper used by `fetch_data.py`      |
| `fetch_data.py`            | Local/manual GCS WikiText-103 fetch helper                       |
| `submission_baseline.py`   | Baseline submission (modded-nanogpt port, 1xA100-40GB, ~22M params) |
| `test_wikitext.py`         | Pytest-and-stdlib-runnable tests                                 |
| `verify_nvml.py`           | NVML energy-counter verification script                          |
| `.env.example`             | Optional CI env vars `submit.py` reads                           |

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

1. Write your submission as a Python file exposing
   `train(train_text, valid_text=None) -> CharModel`. See
   `submission_baseline.py` for a working PyTorch reference.
2. Run `python3 submit.py path/to/your_submission.py`.
   `submit.py` defines a Modal app, ships your file as bytes to a
   pinned A100-40GB function, captures the returned result dict, and
   appends the row to the Record History table for you. Modal
   auto-shuts-down the container when the function returns.
3. PR with: the result JSON / nvml.json files dropped into
   `submissions/` by `submit.py`, plus the record-history row it
   appended to `README.md`.

If you need to drive the run by hand (debugging, NVML verification on
a new SKU, keeping a GPU warm), see `RUNBOOK.md` for the manual path.
