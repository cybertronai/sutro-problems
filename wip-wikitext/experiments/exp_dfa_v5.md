# exp_dfa_v5 — Direct Feedback Alignment for char-LM

## Result

| metric | value |
|--|--|
| **char-accuracy** | **~0.33** (training NLL plateaued at 2.45 nats/char) — far below DGL's 0.65 |
| training energy | 60.9 kJ |
| training duration | 332 s |
| date (UTC) | 2026-05-06T09:06Z |
| GPU | NVIDIA A100-SXM4-40GB (Lambda On-Demand, us-west-2) |
| job_id | `2026-05-06T08-59-06-dfa-59825171` |

Loss trajectory (a single AdamW run, peak_lr=3e-4, fb_std=1/√V≈0.0625,
5000 steps, batch=64, seq=512):

| step | loss |
|--|--|
| 0    | 5.60 |
| 250  | 3.18 |
| 500  | 2.77 |
| 1000 | 2.45 |
| 2000 | 2.43 |
| 3000 | 2.46 |
| 5000 | 2.47 |

Almost all the apparent learning happens in the first ~1k steps and is
attributable to the head + ln_f + tok_emb (which receive *real*
gradients from `loss.backward()`). The 6 transformer blocks, which
get only the random-projected DFA pseudo-gradient, never make
meaningful progress beyond what the surrounding non-block parameters
provide. The result is roughly equivalent to what you'd get from a
*frozen-random-blocks* model with a CE-trained head and embedding —
hence the 0.33 char-acc.

## Why it failed (best guess)

DFA empirical results in the literature (Nøkland 2016, Lillicrap et
al. 2016) cover MLPs and small CNNs. Modern transformer blocks are a
much harder substrate:

1. **Each block is itself ≈4 layers deep** (LN→attn→residual→LN→MLP→
   residual). The DFA pseudo-gradient at the block's output has to
   propagate through this internal chain to update the
   block's q/k/v, output proj, fc1, fc2. With a *random* output
   gradient direction, the block-internal Jacobians don't have the
   nice recursive-feedback-alignment property that single-layer DFA
   relies on.
2. **Residual streams + LayerNorm** are sensitive to gradient
   direction. Small misalignment in the random feedback signal can
   cause LN to saturate or residual updates to cancel.
3. **Per-block AdamW** rescales the noisy DFA grads to a similar
   step magnitude as real gradients, but the *direction* of those
   updates is set by `B_L` (random and frozen) instead of the
   true backward Jacobian. Direction matters.

Empirical signal of the failure: training loss decreases ~5.6→2.4 in
the first 1000 steps (matching what tok_emb + ln_f + head can learn
on their own) and then plateaus completely. No further alignment
develops.

## What might still work

Did NOT test in this iteration; future iterations should consider:

- **Sign-projection feedback** (`B_L = sign(N(0,1)) / sqrt(d)`) —
  often works better than dense Gaussian for DFA.
- **Much longer training** (15–20 k steps). Feedback alignment is a
  slow process; 5k may be insufficient. 15 k × 18 layer-fwds × 0.72 J
  ≈ 195 kJ — *over budget*. Need to scale model down (fewer layers
  or smaller d).
- **Per-block-LN-only feedback** — apply DFA only to the ln1/ln2
  parameters, leaving the matmul weights to local DGL. Hybrid.
- **Apply DFA to layer pre-activations instead of block outputs** —
  the original DFA paper's recipe; might align better.

For the next Ralph iteration, the highest-EV moves are probably (a)
DGL with a richer per-layer objective (multi-future-token NLL or a
contrastive next-char loss) — keeps the working DGL infrastructure
and tries to push the per-layer NLL ceiling lower, OR (b) just
accept that the DGL ~0.65 is the practical no-backprop ceiling on
this benchmark in 100 kJ and stop.

## Memory notes

This v5 result superseded the optimistic "DFA should beat 0.65"
prediction in `~/.claude/projects/.../memory/dgl_ceiling.md`. Updated
that file to reflect: naive DFA on this transformer architecture
collapses to ~0.33 — significantly *worse* than DGL.
