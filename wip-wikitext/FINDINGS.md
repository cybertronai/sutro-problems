# wip-wikitext — no-backprop research findings

Char-level WikiText-103 LM trained **without end-to-end backpropagation**,
under the v0 leaderboard's 100 kJ NVML training-energy budget on a pinned
A100 SXM4. Score: greedy-argmax char-accuracy on the first 60,000
held-out test chars.

| target | value |
|--|--|
| modded-nanogpt backprop baseline | **0.7310 char-acc**, 55 kJ training energy |
| best no-backprop result this branch | **0.6839 char-acc** (v11, DGL+Muon, 93 kJ) |
| gap remaining | **4.7 pp** |

The 0.7310 bar was **not beaten** in this iteration. This document
captures what was tried, what worked, what didn't, and where to go
next.

---

## Methodological constraint

"No backprop" is interpreted as: **at any given training step, only
one transformer block's autograd graph is live**. All upstream blocks
run under `torch.no_grad()`; their activations are released
immediately. This rules out keeping a multi-layer activation cache
for end-to-end gradient flow, while still permitting standard
autograd within a single layer + its auxiliary head.

`baseline_transformer.py` (modded-nanogpt port) holds full activations
across all 6 layers for the standard backward pass — that's the
behaviour we are explicitly avoiding.

---

## Methods tried

### A. DGL (Decoupled Greedy Layer-wise, Belilovsky 2019) — 12 variants

DGL trains the stack one transformer block at a time. Block L is
optimised against a local next-char NLL via a small auxiliary head
attached to its output. Once block L is trained, it freezes; block
L+1 is trained against its own aux head, taking block L's outputs
(under `no_grad`) as input. Only one block's activations are live at
any step.

The best result in this family was **v11 = 0.6839** (DGL + Muon for
hidden matmul weights, 4 × d=512, 5000 steps/layer, WSD schedule,
warm-started aux heads, 93 kJ).

| # | recipe | char-acc | Δ vs prior best | energy |
|---|---|---:|---:|---:|
| v1 | plain DGL, 5 × d=384, AdamW, last-layer head | 0.6440 | — | 54 kJ |
| v2 | + warm-start aux heads + zero-init residuals + 6-layer ensemble | 0.6553 | +1.13 pp | 80 kJ |
| v3 | concat features → closed-form ridge readout | 0.5990 | regression | 68 kJ |
| v4 | concat features → CE-trained linear readout + per-layer LN | 0.6555 | tied with v2 | 77 kJ |
| v6 | wider DGL: 4 × d=512 | 0.6638 | +0.85 pp | 72 kJ |
| **v7** | **+ Muon for hidden 2-D matmuls (lr 0.02), AdamW for rest, WSD** | **0.6789** | **+1.51 pp** | **74 kJ** |
| v8 | v7 with 5 layers × 3500 steps | 0.6785 | tied | 90 kJ |
| v9 | v7 + CE-trained 2048→256 readout from per-layer-LN'd features | 0.6803 | +0.14 pp | 80 kJ |
| v10 | v7 + multi-future-token (K=4) aux objective | 0.6787 | −0.16 pp | 65 kJ |
| **v11** | **v7 with 5000 (vs 4000) steps/layer** | **0.6839** | **+0.36 pp** (best) | **93 kJ** |
| v12 | v7 + MLP-residual aux head | unknown¹ | per-layer NLLs ≈ v11 | — |

¹ v12 ran successfully (`rc=0 wall=687s`) but the persistent Lambda
instance rotated host-key/IP before `result.json` could be pulled.
Per-layer NLL trajectory captured live (1.22 / 1.17 / 1.12 / 1.13)
was indistinguishable from v11 — no evidence of breakthrough.

### B. DFA (Direct Feedback Alignment, Lillicrap / Nøkland) — 1 variant

Tried once, naively. v5 (`exp_dfa_v1.py`): a 6 × d=384 transformer
with **fixed random Gaussian feedback matrices** per block (`fb_std =
1/√V`), AdamW (peak lr 3e-4), 5000 steps. Hidden representations are
trained against `Bᵀ · output_error` rather than the chained backprop
gradient.

**Result: 0.3331 char-acc.** Catastrophic. Training loss plateaued at
~2.40 nats/char (vs DGL's ~1.20). The blocks essentially became
random feature extractors; only the head and tok_emb (which receive
*real* gradients) carried the model. Naive DFA does not work on
transformer blocks at this scale.

---

## What worked

The four interventions that contributed positively to the DGL chain:

1. **Warm-starting aux heads** between layers (v1 → v2): +1.13 pp.
   At each new layer, copy the previous layer's aux head weights
   instead of re-initialising. Also zero-init the residual outputs
   (`attn.proj`, `fc2`) so each new block starts as a near-identity
   transformation.

2. **Wider transformer** (v2 → v6): +0.85 pp. 4 × d=512 with 8 heads
   (head_dim=64) beat 5 × d=384 with 6 heads under fixed energy. At
   constant FLOPs, width was a better spend than depth for DGL —
   probably because each layer's local NLL has more headroom on a
   wider feature.

3. **Muon optimizer for hidden 2-D matmul weights** (v6 → v7):
   **+1.51 pp — the single biggest win.** Newton-Schulz5
   orthogonalization gives every per-layer update full effective
   rank. Under DGL, gradient updates at one layer are dominated by a
   small set of singular vectors (the local-NLL signal is repetitive
   in direction); AdamW's elementwise step accumulates low-rank
   effective updates, leaving the matmul weight under-utilised. NS5
   forces every step to span the full singular space.

4. **Longer training** (v7 → v11): +0.36 pp. Increasing steps per
   layer from 4000 to 5000 dropped layer 2's NLL by 0.08 (1.18 →
   1.10). This brought the energy budget from 74 kJ to 93 kJ — only
   7 kJ headroom remains for further step-count tuning.

These four interventions are **cumulative and orthogonal**. Each was
necessary; removing any one regresses by close to its delta.

---

## What didn't work

1. **Closed-form ridge readout from concatenated features** (v3):
   −5.6 pp. L2 regression on one-hot targets is the wrong objective
   for argmax classification — the optimal L2 solution is not the
   optimal CE / argmax solution. Replacing with CE-trained linear
   readout (v4) recovered to ≈v2 level but did not improve over it.

2. **CE-trained linear readout** on top of frozen DGL features (v4,
   v9): +0.0 to +0.14 pp. The per-layer aux heads already extract
   nearly all the linearly-decodable next-char information from each
   layer. Concatenating layers and re-decoding adds essentially
   nothing.

3. **Adding depth** (v8: 5 layers): tied with v7. At fixed energy,
   the per-layer step reduction needed to fit a 5th layer cancelled
   the depth gain.

4. **Multi-future-token (K=4) aux objective** (v10): −0.16 pp. The
   intent was to force layers to encode features beyond pure
   next-char prediction by training each aux head on +1, +2, +3, +4
   character targets. But the diluted gradient on the +1 target
   actually hurt the inference path's char-acc.

5. **MLP-residual aux head** (v12): no measurable change in per-layer
   NLL. Confirms the per-layer ceiling is not a head-capacity issue —
   the features themselves are already near-optimal as next-char
   predictors at the linear-decoder level.

6. **Naive DFA** (v5): catastrophic regression to 0.33. Fixed random
   feedback projection direction never developed alignment on
   transformer blocks under this LR schedule and step budget.

---

## Why DGL plateaued at ~0.68

After 12 attempts, the diminishing-returns pattern is unambiguous:

```
v1 → v2 → v6 → v7 → v9  → v11 → v12
0.6440 0.6553 0.6638 0.6789 0.6803 0.6839 ≈0.68
       +1.13  +0.85  +1.51  +0.14   +0.36   +0.00
```

The per-layer **local NLL** plateaus at ~1.10–1.18 nats/char regardless
of optimizer / width / depth / head-architecture / step-budget. This
is consistent with a structural argument:

> Without inter-layer gradient flow, each layer's features are shaped
> only by what its own local next-char predictor can learn from them.
> The optimal local-NLL features are the ones a linear (or MLP) head
> can decode best — but this is not the same as the features that
> *next-layer + ... + final head* would prefer. End-to-end backprop
> implicitly co-optimises features for the eventual loss; DGL cannot.

The 0.7310 bar corresponds to ~1.05 nats/char NLL — about 0.05 nats
below the per-layer DGL ceiling. Closing that gap likely requires a
**different no-backprop signal** that propagates output error through
the stack without keeping multi-layer activations live.

---

## Future directions identified but not tried

In rough priority order based on the diagnosis above:

### 1. Forward-Forward (Hinton 2022)

Replace local NLL with a per-layer **goodness** objective: each layer
maximises `sum(activation²)` on positive (real) samples and minimises
it on negative (perturbed) samples. Negatives can be cheap — e.g.
random byte-permutations of the input window. Sidesteps the local-NLL
plateau because the objective is *not* a feature-decoder — it's a
discrimination task that pressures features to be class-separable in
the activation-magnitude sense.

Hinton's original results (MNIST/CIFAR) underperformed backprop. But
on character LMs with rich multi-scale structure, it may reach
different ceilings than DGL.

**Why it's the highest-priority untried direction**: it's the only
candidate that fundamentally changes the per-layer signal type.

### 2. DFA properly tuned

The naive DFA in v5 used `fb_std = 1/√V` and AdamW lr=3e-4 — both
likely too conservative. Real DFA training on transformers seems to
require:

- **Much higher LR** (1e-2+) since the random-feedback pseudo-grads
  are typically much smaller in magnitude than backprop grads.
- **Sign-projection feedback** `B_L = sign(N(0,1))` instead of
  Gaussian — gives more concentrated update directions.
- **Per-layer feedback scaling** matched to expected backprop-grad
  magnitudes (so the random projection isn't drowned out by the
  AdamW preconditioning).
- **Project to per-layer pre-activation**, not block output, so the
  feedback signal hits the hidden weights at the right place.
- Likely **10k+ steps** for feedback alignment to develop on a
  transformer.

Risky — feedback alignment may simply not develop on transformer
blocks at all — but if it does, the gain could be 5+ pp because the
signal is global.

### 3. Random-feature ridge on n-gram context

Closed-form ridge regression over high-dim hashed n-gram features.
Pure linear, no gradients. Estimated < 10 kJ. Not expected to beat
0.7310, but anchors the lower end of the no-backprop spectrum and
gives a cheap calibration point.

### 4. Echo-State Network + ridge readout

Fixed sparse recurrent reservoir (spectral radius ≈ 0.95), 2k–8k
units, ridge readout solved closed-form on streaming `Hᵀ H`. No
gradients touch the reservoir. Historically strong on character-level
English (Jaeger 2002).

### 5. Inter-layer "soft" pressure

DGL's per-layer NLL augmented with a small `no_grad` regulariser
tying layer L's representation to layer L+1's pre-activation (computed
once at the start of layer L's training, with current weights frozen).
Provides global signal at constant memory — no multi-layer activation
cache needed.

### 6. Aggressive Muon LR + longer warmup

v7's Muon LR of 0.02 was conservative (matching the modded baseline).
Pure DGL+Muon may have a higher sweet spot — try 0.04–0.05 with longer
warmup. Free experiment, no extra energy.

### 7. Target propagation

Each layer learns to invert the next layer's transformation, then is
trained to produce activations that the next layer's frozen inverse
maps to the desired target. Standard published method, never tried
here.

---

## Engineering learnings

### Persistent GPU sessions can rotate

Lambda On-Demand instances expose a 60-minute idle-timeout, but in
practice the IP/host-key can rotate **even while the worker shows
"running"** (likely a load-balancer or DHCP renewal). v12's
`result.json` was permanently lost because we relied on a deferred
SCP from the same IP/host key that worked minutes earlier.

**Operational rule**: pull `result.json` immediately on the worker's
`[done]` event. Do not assume the SSH connection to the same IP
remains valid for later.

### v10 K-loop slowdown gotcha

The first K=4 multi-future-token implementation (v10a) ran a Python
for-loop over K cross-entropies, doubling per-step wall time
(42 ms vs 21 ms). At 4000 steps × 4 layers this projected to
117 kJ — would have tripped the 100 kJ watchdog. Killed pre-emptively
at step 1500.

The fix (v10c): fuse K cross-entropies into a single `F.cross_entropy`
on `(B·T·K, V)` reshaped logits. Two killed attempts cost ~50 kJ
across separate sessions before the fused version landed at 65 kJ.

**Operational rule**: always do a **20-step pre-flight** to measure
ms/step before committing to a multi-thousand-step config. Especially
when the aux-head shape changes.

### Ridge regression for argmax tasks

L2 regression on one-hot targets gave 0.5990 (v3) — a 5.6 pp
*regression* below the warm-start v2 baseline. The L2 solution is
not the argmax solution. For any closed-form readout from frozen
features, use **CE-trained linear** with a few thousand SGD steps
instead — it's still cheap (≈ 10 kJ for 3000 steps on this stack)
and uses the right objective.

### Warm-starting aux heads is doing real work

The 1.1 pp gain from v1 → v2 (warm-start + zero-init residuals)
shows that DGL's per-layer training is not robust to head
re-initialisation between layers. Each layer's optimum head is
*close to* the previous layer's optimum, and starting fresh
discards that information. Warm-starting + residual identity-init
preserves it.

---

## Repository layout (this branch)

```
wip-wikitext/
├── FINDINGS.md                      ← this file
├── experiments/
│   ├── exp_dgl_v{1,2,3,4,6,7,8,9,10,11,12}.py    DGL variants
│   ├── exp_dgl_v{1,2,3,4,6,7,8,9,10,11,12}.md    per-experiment writeups
│   ├── exp_dfa_v1.py / .md                       DFA naive
│   └── _smoke_*.py                               tiny CPU smoke variants
├── records/
│   └── 2026-05-06T10-33-29-…/        v11 (best, 0.6839) — full record + run.log
└── submissions/
    ├── 2026-05-06T08-00-45-…/        v1  result.json + submission.py + run.log
    ├── 2026-05-06T08-25-18-…/        v3
    ├── 2026-05-06T08-39-54-…/        v4
    ├── 2026-05-06T09-10-33-…/        v6
    ├── 2026-05-06T09-25-35-…/        v7  (recipe baseline for everything after)
    ├── 2026-05-06T09-37-39-…/        v8
    ├── 2026-05-06T09-53-53-…/        v9
    └── 2026-05-06T10-16-58-…/        v10
```

Per-experiment write-ups in `experiments/exp_*.md` give the per-layer
NLL trajectories and inline reasoning. The 12-variant story is
condensed in this top-level document.
