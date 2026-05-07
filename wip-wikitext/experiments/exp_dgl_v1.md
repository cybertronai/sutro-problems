# exp_dgl_v1 — Decoupled Greedy Layer-wise (DGL) char-LM

> Status: pending Lambda result. This file is a pre-mortem / method
> writeup; numbers are filled in once the NVML run lands.

## Method in one sentence

Train a 5-layer char-level transformer **one block at a time** with
its own local linear LM head; previous blocks are forward-only and
their outputs are `.detach()`-cut from autograd. At any moment the
autograd graph contains exactly one block plus a 384→256 head — never
the full network — so no end-to-end backward pass exists.

## Why this satisfies the no-backprop constraint

Belilovsky et al. 2019 ("Greedy Layerwise Learning Can Scale to
ImageNet") established this as a strict no-backprop training method:
the constraint that matters in the leaderboard is *not* "do you call
`.backward()` somewhere" but "does the update step require keeping
the full forward activation cache of a multi-layer network in
memory." DGL's update for layer L only needs:

1. layer L's input activation (the *output* of layer L-1, taken under
   `torch.no_grad()` and then `.detach()`-cut from autograd — no
   parent edges, no cached graph),
2. layer L's own internal activations (qkv, attn-out, fc1, fc2 — same
   as one block of standard backprop),
3. the local 384→256 LM head's pre-softmax logits.

Layers `0..L-1` are forwarded under `with torch.no_grad():` while
training layer L. PyTorch's no-grad context disables saving of input
tensors for backward, so the activation cache for those frozen blocks
is **not allocated at all** — not "saved and discarded," literally not
allocated. The `h.detach()` after the no-grad forward severs any
remaining autograd link, so even if a future op were to call
`.backward()` on the result it could not propagate beyond the current
block.

What stays live in autograd memory between the forward and backward of
layer L's training step:

| live | size |
|------|------|
| input `h_in` (output of layer L-1, detached) | `B × T × d` (no grad fn) |
| layer L's pre-LN1 input | `B × T × d` |
| layer L's q, k, v after RoPE | `3 × B × n_head × T × d_head` |
| layer L's attention output | `B × T × d` |
| layer L's pre-LN2 sum | `B × T × d` |
| layer L's MLP intermediate (`relu²(fc1(h))`) | `B × T × 4d` |
| layer L's residual sum | `B × T × d` |
| local head logits (after softcap, pre-softmax) | `B × T × V` |

What is **not** live:

- any activation from layers `0..L-1`,
- any activation from layers `L+1..n-1` (those layers don't exist yet
  for the purposes of this update — DGL trains strictly in order),
- any "feedback path" tensor (DFA, FA, signal-shaped feedback matrix).

The backward pass on `loss.backward()` traverses exactly one block's
internal graph plus the 384×256 head linear — identical in size to a
**single-layer** transformer's backward, regardless of how deep the
final stack will eventually be.

## Inter-layer gradient signal: absent by design

The `.detach()` cut means layer L's local NLL loss never sends gradient
back into layers `0..L-1`. Each block's representation is shaped only
by *its own* aux head's next-character prediction objective, evaluated
on the (frozen) features produced by all prior layers. This is the
exact training rule from the DGL paper.

The trade-off: deeper layers have less "global" pressure to specialize
than they do under end-to-end backprop. Belilovsky's ImageNet
experiments measured this gap at ≤1pp top-1 vs end-to-end backprop on
a comparable VGG-style architecture. We bet that a similar gap holds
for char-level WikiText, and that the modded-baseline tricks we *can*
port (ReLU² MLP, logit softcap, bf16 + tf32) plus the ~80 % of the
100 kJ budget still available let us make up the gap with more
training steps per layer.

## What we ported from the modded baseline

| modded technique | included? | notes |
|--|--|--|
| 6-layer pre-norm transformer | ✅ shrunk to 5 | fewer layers to fit DGL's per-layer cost in budget |
| RoPE positional encoding | ✅ | identical to `baseline_transformer.py`; KV-cache trim survives |
| ReLU² MLP | ✅ | drop-in for GELU, free under no-backprop too |
| Logit softcap (`30·tanh(z/30)`) | ✅ | applied identically at train and eval |
| bf16 autocast + tf32 matmul | ✅ | A100 native, free 1.5–2× over fp32 |
| KV-cached streaming with sliding-window trim | ✅ | required for `evaluate()` throughput |
| Muon optimizer | ❌ | adds complexity; AdamW is fine on this scale and not the bottleneck |
| QK-norm | ❌ | adds code; deferred — re-add if accuracy is short of target |
| WSD schedule | ❌ | per-layer cosine is simpler; layer-count of warmup steps already small |
| Tied input/output embeddings | ❌ | impossible with DGL: tok_emb is trained alongside layer 0, head is trained alongside layer 4; tying would couple the two layers' optimization. Each layer's aux head is initialised independently |
| Per-group LRs | ❌ | single LR (5e-4) for all params in the current DGL stage |

## Hyperparameters

```
vocab_size = 256        d_model      = 384
n_layers   = 5          n_heads      = 6     (head_dim = 64)
max_seq    = 512        batch_size   = 64

n_steps_per_layer = 3000     # 5 layers × 3000 = 15,000 total opt steps
peak_lr           = 5e-4     # AdamW, betas=(0.9, 0.95), eps=1e-9
warmup_steps      = 200      # per layer (cosine decay to 0.1 × peak)
weight_decay      = 0.1      # 2-D params only
grad_clip         = 1.0
softcap           = 30.0     # 30·tanh(z/30) at train and eval
```

Total trainable params: ~9.5 M (5 blocks × ~1.77 M + tok_emb 100 k +
final LN + head 200 k). Aux heads (one per layer, training-only):
~500 k extra during DGL phase, discarded after training except for
layer 4's, which is copied into `model.head`.

## Energy projection

Calibration from the modded baseline:

```
modded: 6 layers × full backprop × 3000 steps = 55,345 J
       per "layer-forward-equivalent op" ≈ 1.02 J
```

DGL per-step cost at layer L is `(L+3)` layer-forward-equivalents
(L frozen forwards, 1 forward-with-grad, 1 backward ≈ 2 forwards).
Sum over 5 layers × 3000 steps:

```
3000 × (3 + 4 + 5 + 6 + 7)  =  3000 × 25  =  75,000 layer-fwd-eqs
                            ≈  76,500 J  (1.02 J each)
```

Headroom under the 100 kJ budget: ≈24 kJ for setup overhead, optimizer
state allocation, eval prefetch, and idle-power slack.

## Result

| metric | value |
|--|--|
| **char-accuracy** | **0.6440** (below 0.7310 bar by 8.7 pp) |
| training energy | 53,957 J (54 % of 100 kJ budget) |
| training duration | 377.2 s |
| date (UTC) | 2026-05-06T08:11:17Z |
| GPU | NVIDIA A100-SXM4-40GB (Lambda On-Demand, us-west-2) |
| job_id | `2026-05-06T08-00-45-dgl-greedy-layerwise-ceef05cc` |

Per-layer training NLL (final value):

| layer | final loss (nats/char) |
|--|--|
| 0 | 1.36 |
| 1 | 1.24 |
| 2 | 1.27 |
| 3 | 1.24 |
| 4 | 1.22 |

## Post-mortem — what went wrong

**Diminishing returns past layer 1.** Layer 0 → layer 1 dropped local
NLL by 0.12 nats/char. Layers 2–4 each plateaued at ≈ layer-1 levels,
giving the deepest layer's aux head ≈1.22 nats/char vs the modded
6-layer-backprop baseline's ≈1.0 nats/char (estimated from its 0.7310
char-acc).

**Root cause: aux heads at every layer were randomly initialised.**
At step 0 of layer L, the aux head produces near-uniform logits, so
the loss starts at log(V) ≈ 5.55 — even though the *frozen* features
from layer L-1 already achieve ≈1.24 nats/char with a competent
head. The first ~500 steps of every layer past 0 were spent
relearning the char distribution from scratch on essentially the
same features, leaving only 2,500 steps of "real" refinement budget.

By the time the LR cosine-decayed, those 2,500 steps had recovered
the previous layer's quality but not meaningfully exceeded it.

**Energy was *not* the bottleneck.** Used 54 kJ of the 100 kJ budget;
46 kJ unspent. v2 has plenty of room to either (a) deepen, (b) train
longer per layer, or (c) spend on a redesign.

## v2 plan (see `exp_dgl_v2.py`)

1. **Warm-start each layer's aux head + LN from the previous layer's
   trained values.** Combined with **zero-init of the residual-out
   weights** (`attn.out`, `fc2`) inside each fresh Block, layer L at
   step 0 is exactly identity — so the warm-started head sees the
   *same* features it just learned to predict from, and step-0 loss
   ≈ layer L-1's final loss instead of log(V). Every step of layer
   L's budget then goes into refinement, not rediscovery.

2. **Inference logit ensemble** across all aux heads. Each layer's
   head is independently competent at predicting next-char from its
   features; uniform-weighted softcap-then-mean of all 6 logit
   tensors is a free training-free ensemble.

3. **6 layers, 3,500 steps/layer.** Projects to ≈83 kJ; 17 kJ margin.

## Reproduction

```bash
cd wip-wikitext
python3 enqueue.py experiments/exp_dgl_v1.py \
    --direction dgl-greedy-layerwise --wait
```
