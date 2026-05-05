# Lambda A100 runbook

Step-by-step for the two open items in `README.md`:

1. Verify `nvmlDeviceGetTotalEnergyConsumption` is exposed on the Lambda
   On-Demand A100 80GB SKU.
2. Train a from-scratch transformer baseline and produce the first
   record-history entry (energy, char-acc).

This is a manual procedure for now. It assumes you have a Lambda
account, an SSH key registered, and the Lambda CLI or the web console.

---

## 0. Provision

Web console → **Launch instance** → **A100 (80 GB) SXM** → 1× GPU →
Ubuntu 22.04 image → your SSH key. Wait for it to boot, then:

```bash
ssh ubuntu@<instance-ip>
```

Cost is billed per minute. Tear down with **Terminate instance** in
the console as soon as the run finishes — Lambda does *not* auto-stop.

## 1. NVML verification (5 minutes, ≈$0.15)

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
GPU: NVIDIA A100-SXM4-80GB
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
how much A100 time you want to burn for the first record:

| config | params | recommended steps | est. wall-clock | est. cost @ $1.79/hr |
|--------|--------|-------------------|-----------------|----------------------|
| tiny   | ~0.6 M | 5,000             | ~5 min          | ~$0.15               |
| small  | ~5 M   | 30,000            | ~45 min         | ~$1.35               |
| gpt2   | ~22 M  | 100,000           | ~5 hr           | ~$9                  |

(Wall-clock is rough — depends on CPU bottleneck on data shuffling
etc. Re-time after the first run.)

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

Full **small** baseline (recommended first record):

```bash
python3 run_eval.py \
  --data-dir ~/data/wikitext-103-raw \
  --baseline transformer \
  --config small \
  --n-steps 30000 \
  | tee training.log
```

The runner prints a final block like:

```
baseline           : transformer
training energy (J): 4,832,109.4
test char-accuracy : 0.6234
test chars         : 60,000
```

## 4. Capture the result

Save four artefacts back to the repo:

1. `submissions/<config>_<date>.log` — full `training.log` from
   the run.
2. The `verify_nvml.py` JSON output (one line) — proof the host
   passed verification.
3. A row appended to a "Record history" table in the README:

   | Date       | Energy (J) | Char-acc | Config    | Submission        | Contributor |
   |------------|-----------:|---------:|-----------|-------------------|-------------|
   | YYYY-MM-DD |  4,832,109 |  0.6234  | small     | [log](submissions/small_YYYY-MM-DD.log) | @you |

4. Tear down the Lambda instance.

## 5. Tear down

```bash
# In the Lambda web console: Terminate instance.
```

Confirm in **Billing** that the instance is no longer accruing.
