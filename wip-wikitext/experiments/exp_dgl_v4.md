# exp_dgl_v4 — DGL features + CE-trained linear readout (per-layer LN)

## Result

| metric | value |
|--|--|
| **char-accuracy** | **0.6555** (below 0.7310 bar by 7.6 pp; effectively ≡ v2) |
| training energy | 77,436 J (77 % of 100 kJ budget) |
| training duration | 505.0 s |
| date (UTC) | 2026-05-06T08:53:18Z |
| GPU | NVIDIA A100-SXM4-40GB (Lambda On-Demand, us-west-2) |
| job_id | `2026-05-06T08-39-54-dgl-greedy-layerwise-7af8fe69` |

DGL phase: 6 layers × 2,500 steps each, last-layer NLL 1.21.
Readout phase: 4,000 CE steps on the 2304→256 linear (per-layer-LN'd
features). Final readout NLL: 1.19. Char-acc: 0.6555 — **two basis
points** above v2's mean-ensemble (0.6553).

## Why no breakthrough

Combined with v1, v2, v3 results, this run gives a clear picture:

| variant | readout | NLL | char-acc | Δ vs v1 |
|--|--|--|--|--|
| v1 | last-layer linear head | 1.22 | 0.6440 | — |
| v2 | uniform mean of 6 layer heads | 1.17 | 0.6553 | +1.13 pp |
| v3 | ridge L2 from 2304-d concat features | (≠NLL) | 0.5990 | −4.50 pp |
| v4 | CE-trained linear from 2304-d concat features | 1.19 | 0.6555 | +1.15 pp |

v4 ≈ v2 means: the CE-optimal *linear* function of the 2304-d concat
features carries no more argmax information than averaging six
independently-trained 384-d linear heads. The features across layers
are functionally redundant for next-character prediction.

This is the diagnostic: **DGL with this architecture gives features
that don't carry layer-wise complementary information**. Each layer
re-encodes the same predictive content because each was trained on
the same local NLL signal.

## What this rules out

- More layers, warm-start variations, residual init tweaks (v1→v2): +1.1pp
- Better readouts on the same DGL features (v2→v4): essentially 0pp
- Closed-form readout of the same features (v3): can regress

So the next iteration needs to change *what features the layers
produce*, not how the readout combines them. Specifically: either

1. **DFA** — fixed-random-feedback projects the *output error* back to
   each layer, giving deeper layers a global signal that DGL's local
   NLL cannot. Most likely path to break the ceiling.
2. **Wider DGL** — d=512 / d=768 might lower the per-layer NLL plateau.
   Marginal at best (extrapolating from v1's d=384 → v2's same-d).
3. **Multi-future-token aux objective** — per-layer head predicts the
   *next K* characters, not just the immediate one. Harder local
   objective, more pressure to encode richer context. Unproven.

DFA is the highest-EV bet. Defer ridge / wider variants until DFA is
ruled out.
