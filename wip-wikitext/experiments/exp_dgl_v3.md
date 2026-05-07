# exp_dgl_v3 — DGL features + closed-form ridge readout

> Status: pending Lambda result.

## Method in one sentence

Train a 6-layer transformer with DGL (warm-started aux heads, as in
v2) — but **discard the aux heads at inference** and replace them with
a single closed-form ridge regression `(XᵀX + λI)⁻¹ XᵀY` mapping the
concatenated 6-layer features (`6 × 384 = 2304`-dim per position) to
one-hot next-char (256-dim).

## Why this satisfies the no-backprop constraint

Two phases, neither requires a multi-layer autograd cache:

1. **DGL training (Phase 1)** — identical to v1/v2: each block + its
   aux head trained one at a time, previous blocks under
   `torch.no_grad()` and `.detach()`-cut. Only one block's activations
   are live in autograd memory at any moment. Aux heads are *only* a
   training signal for shaping each layer's features; they are
   discarded after training.

2. **Ridge readout (Phase 2)** — closed-form linear algebra. No
   gradients, no autograd graph. Streaming accumulation of
   `XᵀX ∈ R^{2304×2304}` and `XᵀY ∈ R^{2304×256}` over ~2.6 M training
   positions, then a single `torch.linalg.solve` on the regularised
   normal equations. The forward passes producing the `X` rows are
   inside `torch.no_grad()` — no activation cache anywhere.

3. **Inference** — `torch.no_grad()` forward through the 6 frozen
   blocks, concatenate outputs, multiply by the trained ridge
   weights, softcap, softmax. KV-cached + sliding-window-trimmed
   exactly like the modded baseline, so streaming evaluation is
   `O(1)` marginal per character.

## Why this might break v1/v2's ~0.65 char-acc ceiling

v1/v2 found that the per-layer linear aux heads each saturate at
~1.20 nats/char on the local NLL — adding more layers, warm-starts,
and even uniform-mean ensembling didn't punch through that ceiling.
The hypothesis behind v3:

The 6 layers' features are each *individually* near-saturated as
linear-readout substrates — but they're not the *same* features.
Each layer was trained with its own local objective on a different
slice of the architecture, so they encode different facts about the
input. The optimal linear combination over the *concatenation* of
features is strictly more expressive than the uniform-mean of 6
independent linear readouts (v2's ensemble), because:

- It can downweight layer 0 (whose aux head was much worse — final
  loss 1.32 vs deeper layers' 1.20) automatically.
- It can route different output dimensions to different layers — e.g.
  letter-frequency unigram-style features primarily from layer 0,
  word-completion features primarily from deeper layers.
- The 2304 × 256 = 590 K parameter matrix has 6 × the capacity of
  v2's any single 384 × 256 = 98 K aux head, and ridge regularisation
  prevents this extra capacity from overfitting on 2.6 M positions.

If the layers carry meaningfully complementary information, ridge can
capture it. If they're nearly redundant, ridge collapses to picking
the best single layer (and v3 ties v2 at best).

## Hyperparameters

```
DGL training (identical to v2 schedule):
  6 layers, d=384, n_heads=6, max_len=512
  3000 steps/layer (down from v2's 3500 to free energy budget)
  AdamW lr=5e-4, warmup=200, cosine decay; bf16 autocast
  batch_size=64, weight_decay=0.1, grad_clip=1.0

Ridge readout:
  ridge_n_windows = 80   # 80 × 64 × 512 ≈ 2.6 M training positions
  ridge_λ        = 1.0   # tikhonov on a 2304×2304 normal-eq matrix
  feature dim    = 6 × 384 = 2304
```

## Energy projection

DGL training: 6 layers × 3000 steps × `(L+3)` summed = 99,000
layer-fwd-eqs ≈ 71 kJ at v1's empirical 0.72 J/op.

Ridge accumulation: 80 windows × 6 layer-fwds = 480 layer-fwd-eqs ≈
350 J. Negligible.

Ridge solve: a single `solve(2304×2304, 2304×256)` ≈ 10 GFLOPs ≈
~70 ms on A100 ≈ 14 J. Negligible.

Total projected: ~71 kJ. Margin: ~29 kJ.

## Result

| metric | value |
|--|--|
| **char-accuracy** | **0.5990** (worse than v1 0.6440 and v2 0.6553 — ridge readout ⇒ regression) |
| training energy | 68,131 J (68 % of 100 kJ budget) |
| training duration | 466.1 s |
| date (UTC) | 2026-05-06T08:39:37Z |
| GPU | NVIDIA A100-SXM4-40GB (Lambda On-Demand, us-west-2) |
| job_id | `2026-05-06T08-25-18-dgl-greedy-layerwise-3c00df85` |

## Post-mortem — why ridge underperforms

**The hypothesis (concat-features + closed-form readout > per-layer CE
ensemble) is fine; the implementation choice (L2 ridge regression on
one-hot targets) was wrong for this metric.**

L2 regression on one-hot Y minimises `‖XW − Y‖²`, which is dominated
by the (V−1) zero entries per row — it pulls predictions toward zero
everywhere except the correct dimension. The optimal-L2 linear
classifier at argmax-time can be quite different from the optimal-CE
classifier (which calibrates *log*-probabilities). For a
high-dimensional feature space (2304-d) and a 256-way classification
task, the gap is sizeable.

Two compounding issues:

1. **No per-layer LayerNorm before concatenation.** Block outputs
   accumulate residual-stream magnitude with depth — layer 5's
   features are typically ~5–10× larger in norm than layer 0's.
   Ridge with a single λ doesn't normalise across feature blocks, so
   it implicitly down-weights the smaller-magnitude (early) layers.

2. **One-hot targets aren't natural for log-likelihood prediction.**
   The per-layer aux heads in v1/v2 *were* CE-trained — that's why
   their argmax is competitive with backprop-trained heads. The ridge
   readout is the first place we replaced CE with L2, and it cost
   us ~5.6 pp.

## v4 plan (see `exp_dgl_v4.py`)

Same DGL backbone as v2/v3 (warm-start, zero-init residual outs,
6 layers × 2500 steps). After DGL, train a *single* 2304→256 linear
readout with **cross-entropy** (not L2) on top of frozen DGL features
that have first been per-layer-LayerNorm'd. The readout training is
~4000 steps of AdamW; gradient flows only through the readout + the
6 LNs (which are *not* chained — each LN sees its own block's output
independently). The autograd graph at any moment is "6 independent
single-layer LNs feeding into one linear readout" — depth ≤ 2,
strictly bounded, fully consistent with the no-backprop rule.

This fixes both issues:
- **Right loss for the metric.** CE-optimal linear classifier is what
  argmax-acc rewards.
- **Per-layer LN.** Each layer's contribution is on a comparable
  scale before the readout matmul.

