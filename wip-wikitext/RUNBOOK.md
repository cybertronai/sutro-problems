# Lambda A100 runbook

Manual Lambda A100 procedure for two **non-submission** uses:

1. Verify `nvmlDeviceGetTotalEnergyConsumption` is exposed on a new
   GPU SKU (currently the Lambda On-Demand A100 40GB SXM4,
   `gpu_1x_a100_sxm4`).
2. Train a from-scratch transformer baseline by hand — for debugging,
   timing experiments, or seeding a baseline number.

**This is not the submission path.** Records go through
[`submit.py`](submit.py), which orchestrates everything below
automatically. Use this runbook only when you need to drive the host
yourself.

It assumes you have a Lambda account, an SSH key registered, and the
Lambda web console.

---

## 0. Provision

Web console → **Launch instance** → **A100 (40 GB) SXM** → 1× GPU →
Ubuntu 22.04 image → your SSH key. Wait for it to boot, then:

```bash
ssh ubuntu@<instance-ip>
```

Cost is billed per minute. Tear down with **Terminate instance** in
the console as soon as the run finishes — Lambda does *not* auto-stop.

## 1. NVML verification (5 minutes, ≈$0.17)

```bash
# On the Lambda host
git clone <repo-url> sutro-problems
cd sutro-problems/wip-wikitext

# Native install (no Docker needed for verification).
pip install --user nvidia-ml-py torch
python3 verify_nvml.py
```

Expected output:

```
GPU: NVIDIA A100-SXM4-40GB
sampling idle power for 3s ...
  idle: 50-80 W
running 30s stress workload ...
  duration:       30.0 s
  energy delta:   ~9000-12000 J
  avg power:      ~300-400 W
  monotonic:      True
---
{"nvml_available": true, "energy_counter_supported": true, "monotonic": true, ...}
```

**Pass criteria** (script returns exit 0):

- `nvml_available == true`
- `energy_counter_supported == true`
- `monotonic == true`
- `100 W < stress_watts_avg < 700 W`

If any fail, capture the JSON line and the failure note from
`verify_nvml.py` — that's the data point we need to either confirm
Lambda works for this benchmark or fall back to RunPod Secure.

## 2. Download WikiText-103 (≈1 min)

The original `s3.amazonaws.com/research.metamind.io/wikitext/...` URL is
dead (PermanentRedirect → SNI mismatch on the dotted-bucket cert). Pull
from the HuggingFace mirror instead and write out `wiki.{split}.raw`
files in the layout `load_wikitext103` expects:

```bash
mkdir -p ~/data/wikitext-103-raw
pip install --user datasets

python3 - <<'PY'
from pathlib import Path
from datasets import load_dataset

out = Path.home() / "data" / "wikitext-103-raw"
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
for hf_split, fname in [
    ("train", "wiki.train.raw"),
    ("validation", "wiki.valid.raw"),
    ("test", "wiki.test.raw"),
]:
    (out / fname).write_text("\n".join(ds[hf_split]["text"]), encoding="utf-8")
PY

ls ~/data/wikitext-103-raw/   # wiki.{train,valid,test}.raw
```

## 3. Train a transformer baseline

Three configs ship in `baseline_transformer.py`. Pick one based on
how much A100 time you want to burn for the experiment:

| config | params | recommended steps | training wall-clock | training cost @ $1.99/hr |
|--------|--------|-------------------|---------------------|--------------------------|
| tiny   | ~0.6 M | 5,000             | ~5 min              | ~$0.17                   |
| small  | ~5 M   | 30,000            | ~45 min             | ~$1.49                   |
| gpt2   | ~22 M  | 100,000           | ~5 hr               | ~$10                     |

(Training wall-clock is rough — depends on CPU bottleneck on data
shuffling etc. Re-time after the first run. These numbers cover
training only; eval on the 60K-char slice adds ~2 min.)

Quick **tiny** smoke run (validates the full pipeline before any
expensive run):

```bash
cd ~/sutro-problems/wip-wikitext
python3 run_eval.py \
  --data-dir ~/data/wikitext-103-raw \
  --baseline transformer \
  --config tiny \
  --n-steps 5000
```

Full **small** baseline (representative training run):

```bash
python3 run_eval.py \
  --data-dir ~/data/wikitext-103-raw \
  --baseline transformer \
  --config small \
  --n-steps 30000 \
  | tee training.log
```

**Fixed-budget runs** — pass `--e-max-joules N` to arm the in-meter
watchdog. It polls NVML every 250 ms; once running net energy crosses
`N` joules the training is killed, the runner prints a
`DISQUALIFIED:` block, and exits with code 2. The leaderboard value
is `task.E_MAX_JOULES = 100_000`:

```bash
python3 run_eval.py \
  --data-dir ~/data/wikitext-103-raw \
  --baseline transformer --config small --n-steps 30000 \
  --e-max-joules 100000 \
  | tee training.log
```

The runner prints a final block like:

```
baseline           : transformer
training energy (J): 4,832,109.4
test char-accuracy : 0.6234
test chars         : 60,000
```

If `--e-max-joules` was set and the run went over, the block is
replaced by:

```
DISQUALIFIED: training energy budget exceeded (e_max=100,000 J, used≈100,142 J)
training duration  : 312.4s
training energy (J): 100,142.0  (at kill)
```

(exit code 2; eval is skipped — the partially-trained model is not
scored.)

## 4. Save artifacts and tear down

If the run is something you want to keep around — a verification log
on a new SKU, a baseline timing reference — copy the relevant files
back off the host (`scp`) before terminating. The leaderboard's
Record History is populated by `submit.py` only; manual runs do not
land there.

## 5. Tear down

```bash
# In the Lambda web console: Terminate instance.
```

Confirm in **Billing** that the instance is no longer accruing.
