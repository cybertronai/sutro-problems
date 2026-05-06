# exp_dgl_v2 — DGL with warm-started aux heads + inference ensemble

## Result

| metric | value |
|--|--|
| **char-accuracy** | **0.6553** (below 0.7310 bar by 7.6 pp; +1.13 pp over v1) |
| training energy | 79,725 J (80 % of 100 kJ budget) |
| training duration | 531.6 s |
| date (UTC) | 2026-05-06T08:27:13Z |
| GPU | NVIDIA A100-SXM4-40GB (Lambda On-Demand, us-west-2) |
| job_id | `2026-05-06T08-12-54-dgl-greedy-layerwise-741c7d0c` |

Per-layer training NLL (final value):

| layer | final loss (nats/char) | warm-start step-0 loss |
|--|--|--|
| 0 | 1.32 | 5.60 (random init) |
| 1 | 1.21 | 1.29 ✓ |
| 2 | 1.21 | 1.22 ✓ |
| 3 | 1.20 | 1.23 ✓ |
| 4 | 1.20 | 1.20 ✓ |
| 5 | **1.17** | 1.18 ✓ |

## What worked

**Warm-start + zero-init residual outs solved the rediscovery problem from v1.** Layers 1+ now begin at near-prev-layer's loss instead of `log V ≈ 5.55`. Compare step-0 loss: v1 (random init) = 5.55, v2 (warm-start) = 1.18–1.29. The first ~500 training steps per layer that v1 wasted on relearning the char distribution are now spent on actual refinement.

This is visible in v2's deepest-layer NLL (1.17) vs v1's (1.22) — a 0.05 nats/char improvement entirely from the warm-start mechanism.

## What didn't work — the deep-DGL ceiling

Despite the warm-start, **layers 1–5 each plateau at ~1.20 nats/char on their local NLL**. The final layer (5) only edges below to 1.17. The improvement per added layer is essentially gone after layer 1.

Diagnosis: with no inter-layer gradient flow (the DGL constraint), each layer is shaped *only* by its own next-character prediction objective. Once the per-layer aux head saturates the linear-readout-from-d=384-features capacity (roughly 1.20 nats/char on this data), adding more depth doesn't help — the deeper layers learn near-identity transformations because their local objective is already maximally satisfied.

**The inference ensemble (uniform-mean of all 6 aux head logits) actively hurt the metric by ~0.5 pp early in eval** before climbing back. Reason: layer 0's aux head (final loss 1.32) is meaningfully worse than the deeper heads' (1.17–1.21), and uniform averaging drags the ensemble's predictions toward layer 0's. A learned weighting (e.g. ridge regression on concatenated features — see v3) would have higher capacity to favor the better layers.

## Net change vs v1

+1.13 pp char-acc, +25.8 kJ training energy. The warm-start mechanism is real (v2's deepest layer is 0.05 nats/char better than v1's). But the core DGL architecture has a hard ~0.65 char-acc ceiling on this task.

## v3 plan (see `exp_dgl_v3.py`)

Replace the per-layer aux head ensemble with a **single closed-form ridge regression** from the concatenation of all 6 layers' features `(B, T, 6·384)` to one-hot next-char `(B, T, 256)`. The ridge solve is gradient-free (just `(XᵀX + λI)⁻¹ XᵀY`), and the optimal linear combination is strictly more expressive than uniform-mean of independently-trained linear heads. Layer 0's contribution gets down-weighted automatically if it carries less information.
