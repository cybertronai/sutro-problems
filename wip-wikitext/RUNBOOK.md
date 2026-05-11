# Modal A100 runbook

Manual Modal A100 procedure for two **non-submission** uses:

1. Verify `nvmlDeviceGetTotalEnergyConsumption` is exposed on a new
   GPU SKU (currently `gpu="A100-40GB"`).
2. Train a from-scratch transformer baseline by hand — for debugging,
   timing experiments, or seeding a baseline number.

**This is not the submission path.** Records go through
[`submit.py`](submit.py), which orchestrates everything below
automatically. Use this runbook only when you need to drive the
container yourself (e.g. to bisect a regression in the harness or
collect timing data outside the leaderboard contract).

It assumes you have a Modal account and have run `modal token new`.

---

## 0. Setup

```bash
pip install modal
modal token new      # opens browser, writes ~/.modal.toml
```

Modal bills per function-second of GPU time and auto-shuts-down when
the function returns. There's nothing to terminate by hand.

## 1. NVML verification (≈2 min, ≈$0.07)

A one-shot Modal function that runs [`verify_nvml.py`](verify_nvml.py)
on the pinned GPU and returns the JSON summary:

```bash
modal run -m submit::run_submission --help    # confirm wiring
```

The cleanest one-liner is to run `verify_nvml.py` in a Modal shell
on the same image:

```bash
modal shell --gpu A100-40GB submit.py
# inside the shell:
python verify_nvml.py
```

Expected output (last line is JSON):

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

**Pass criteria** (`verify_nvml.py` returns exit 0):

- `nvml_available == true`
- `energy_counter_supported == true`
- `monotonic == true`
- `100 W < stress_watts_avg < 700 W`

If any fail, capture the JSON line — that's the data point we need to
either confirm Modal works for this benchmark or fall back to RunPod
Secure.

## 2. WikiText-103 staging

`submit.py` does not download WikiText-103 at run time. The raw splits
live at `/data/wiki.{train,valid,test}.raw` inside the prebuilt
`ghcr.io/ab-10/wikitext-bench` image (see [`Dockerfile`](Dockerfile)),
which `submit.py` pulls via `Image.from_registry(...)`. Modal caches
the registry digest, so cold start is just the one-time pull
(~85s end-to-end for the full 6 GB image); subsequent runs reuse the
cached layers.

To rebuild the image (only needed when bumping torch / pyarrow /
WikiText-103 contents):

```bash
# Stage the .raw splits from the public GCS mirror into the build
# context — Dockerfile COPYs from this directory:
python fetch_data.py wikitext-103-raw-v1/

docker build -t ghcr.io/ab-10/wikitext-bench:latest -f Dockerfile .
docker push ghcr.io/ab-10/wikitext-bench:latest
```

To inspect the image-baked files, open a shell on the same image and
list `/data`.

## 3. Train a transformer baseline by hand

Three configs ship in `baseline_transformer.py`:

| config | params | recommended steps | training wall-clock | training cost @ $2.10/hr |
|--------|--------|-------------------|---------------------|--------------------------|
| tiny   | ~0.6 M | 5,000             | ~5 min              | ~$0.18                   |
| small  | ~5 M   | 30,000            | ~45 min             | ~$1.58                   |
| gpt2   | ~22 M  | 100,000           | ~5 hr               | ~$10.50                  |

(Wall-clock is rough — depends on CPU bottleneck on data shuffling
etc. Re-time after the first run. These cover training only; eval on
the 60K-char slice adds ~2 min.)

The harness function in `submit.py` only takes a submission file as
input. For ad-hoc transformer experiments without authoring a
submission, `modal shell --gpu A100-40GB submit.py` drops you into an
interactive container with the harness on `PYTHONPATH`; from there
`run_eval.py` works the same way the submission function invokes it:

```bash
# inside the modal shell:
python run_eval.py \
  --data-dir /data \
  --baseline transformer --config small --n-steps 30000 \
  --e-max-joules 100000 \
  | tee training.log
```

The runner prints a final block like:

```
submission         : baseline_transformer_small
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

## 4. Save artifacts

`modal shell` containers are ephemeral. Anything written inside the
container, including under `/workspace`, `/tmp`, or `/data`, should be
treated as temporary. Capture logs from stdout or copy them out before
exiting if you need to keep them.

For the leaderboard path, all artifacts (`run.log`, `nvml.json`,
`result.json`) are persisted automatically by `submit.py` to
`submissions/`. Manual `modal shell` runs do not land in Record
History.
