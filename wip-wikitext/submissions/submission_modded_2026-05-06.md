# submission_modded — method writeup

| | |
|--|--|
| **char-accuracy** | **0.7310** (greedy argmax, first 60,000 test chars) |
| **training energy** | 55,345 J (55 % of 100 kJ budget) |
| **training duration** | 308.7 s |
| **GPU** | NVIDIA A100-SXM4-40GB (Lambda On-Demand, us-east-1) |
| **date** | 2026-05-06T04:41:40Z |
| **commit** | 65f3812 (+ uncommitted submit.py fixes) |
| **artifacts** | `submission_modded_2026-05-06.{json,log,nvml.json}` |

## Approach

Char-level GPT-2-style decoder (vocab=256 raw bytes, d=384, 6 layers,
6 heads, head_dim=64, max_seq_len=512 → ≈10 M params), trained from
scratch on WikiText-103 train under a 100 kJ NVML energy budget.

The architecture is `baseline_transformer.py` re-imagined with the
subset of modded-nanogpt techniques that transfer to single-A100
char-level training. modded-nanogpt itself is fused tightly to its
8×H100 + BPE + FineWeb regime — FP8 matmuls, FlashAttention-3 varlen
sliding-window kernels, sparse-comms parameter-bank sharding,
multi-token prediction, value embeddings, YaRN, two-track schedule on
batch and window size — none of which apply here. The transferable
subset is small but high-value:

| modded-nanogpt technique | ported here | why / why not |
|--|--|--|
| **Muon optimizer** for hidden 2-D weights | ✅ | Newton-Schulz5 orthogonalization (single-GPU, non-sharded) on qkv / proj / fc / proj. Aspect-ratio LR scaling `√(max(1, fan_out/fan_in))`. AdamW for embeddings, lm_head, biases. |
| **QK-norm** (RMSNorm on q,k after RoPE) | ✅ | Stabilizes attention; cheap. |
| **ReLU² MLP** | ✅ | Drop-in for GELU; small but free gain. |
| **Logit softcapping** (Gemma-2 style) | ✅ | `30·tanh(z/30)`, applied at both train and eval so the streaming distribution matches the training distribution. |
| **Per-group LRs** (high for embeddings, low for hidden) | ✅ | AdamW lr=3e-3 on embeddings/lm_head, Muon lr=0.02 on hidden matmuls. |
| **WSD schedule** (warmup–stable–decay) | ✅ | 200-step linear warmup → flat → linear cooldown to 0 over the last 20 %. Better than cosine for short trapezoidal runs. |
| **Tied input/output embeddings** | ✅ | GPT-2 convention; halves embedding param footprint at this scale. |
| **bf16 autocast + tf32 matmul** | ✅ | A100 native, free 1.5-2× over fp32. |
| FP8 matmul / Polar Express NS | ❌ | Polar Express is the in-place upgrade to NS5; gap is small at this scale. FP8 is H100-only. |
| FlashAttention-3 varlen + sliding window | ❌ | Kernel not in the pinned PyTorch 2.5.1 base image; SDPA + dense causal is fast enough at L=512. |
| Sparse-comms parameter-bank sharding | ❌ | Single GPU. |
| Multi-token prediction | ❌ | Adds compute and complexity for small per-step gain at our budget. |
| Value embeddings (5×vocab × d) | ❌ | Trained-from-scratch on a 1-region budget — can't afford an extra 5×vocab embedding table. |
| YaRN, paired heads, attn skip layer 6 | ❌ | Tied to the specific 12-layer/16K-context speedrun config. |

## Streaming inference

`ModdedCharModel` mirrors `baseline_transformer.TransformerModel`'s
KV-cache + sliding-window-trim pattern: a true absolute-position
counter `_pos` is fed into every forward so RoPE rotates the new
query at its real position; once the cache reaches `max_len`, it's
trimmed to `max_len - 1` so the relative offset stays inside the
trained range for arbitrarily long test streams. `predict()` returns
`P(next_byte | observed_so_far)` over the 256 byte vocabulary
(restricted to bytes that decode as a single UTF-8 character so the
runner's char-accuracy score is well-defined).

Eval throughput: 143 char/s on the A100 — 420.8 s for the 60K-char
slice. That's ~3× slower than the score time on a tighter inference
loop (the obvious win is bf16 inference + a precomputed byte→char
map; deferred since the leaderboard scores training energy only).

## Hyperparameters

```
vocab_size = 256        max_seq_len  = 512
d_model    = 384        batch_size   = 64
n_layers   = 6          n_steps      = 3000
n_heads    = 6          warmup       = 200
softcap    = 30.0       cooldown_frac = 0.20

Muon  : lr=0.02,  momentum=0.95 (Nesterov), ns_steps=5, wd=0.0
AdamW : lr=3e-3,  betas=(0.9, 0.95), eps=1e-10, wd=0.0
Grad clip 1.0; bf16 autocast on CUDA; tf32 matmul precision.
```

3000 steps × 64-batch × 512-tokens = 98 M tokens of training data
seen. Training fit comfortably within the 100 kJ budget (308.7 s × ≈
180 W net = 55 kJ; 45 kJ headroom).

## Where the time went

| phase | wall-clock | energy |
|--|--|--|
| Lambda boot + image pull + NVML probe + data fetch | ~2 min (untimed) | not measured |
| training (Muon + AdamW, 3000 steps) | 308.7 s | 55,345 J |
| streaming eval (60K chars) | 420.8 s | not metered (per v0 design) |

## Follow-up directions

Roughly in order of expected return for the next iteration:

1. **More steps, same model** — 45 kJ unspent. Retuning to ~5500
   steps should fit and likely move acc to ~0.74-0.75.
2. **Bigger model** — d=512, L=6 (≈18 M params) within the same
   budget if step time permits. Muon scales well.
3. **`torch.compile`** — eaten by ~30-60 s of compile, but should
   give 1.3-1.5× step time, paying for itself at >2000 steps.
4. **Polar Express** orthogonalization replacing NS5 — small but
   free; modded-nanogpt's current default.
5. **bf16 inference** — should ~2× the streaming eval throughput;
   not on the leaderboard but useful for dev iteration.
6. **Trained-token-count scaling experiment** — fix the model, sweep
   n_steps to map the energy/accuracy Pareto curve.

## Reproduction

```bash
cd wip-wikitext
python3 submit.py submission_modded.py --yes \
    --wait-for-capacity --wait-timeout 3600
```

Caveats discovered during this submission, fixes still un-PR'd in
local working tree:

1. The GHCR package `ab-10/wikitext-bench` is private; the original
   cloud-init `docker pull` had no `docker login`, so submissions hung
   forever in the cloud-init silently. Patched submit.py to inject
   `gh auth token` → `docker login ghcr.io` into the cloud-init.
2. `wait_for_active` defaulted to 600 s; capacity-tight days saw the
   first launch timeout in `asia-south-1` before transitioning to
   `active`. Bumped to 1200 s.

Both worth their own PR; both made the difference between a hung run
and a clean one.
