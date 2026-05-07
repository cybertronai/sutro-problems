# exp_dgl_v8 — DGL+Muon, 5 layers × d=512 (vs v7's 4 layers)

## Result

| metric | value |
|--|--|
| **char-accuracy** | **0.6785** (≈ tied with v7 — depth at fixed budget did not help) |
| training energy | 89,950 J (90 % of 100 kJ budget) |
| training duration | 520 s |
| date (UTC) | 2026-05-06T09:50:00Z |
| GPU | NVIDIA A100-SXM4-40GB |
| job_id | `2026-05-06T09-37-39-dgl-greedy-layerwise-56f5b2aa` |

Per-layer training NLL (3500 steps each, WSD schedule):

| layer | v8 (5 × 3500) | v7 (4 × 4000) |
|--|--:|--:|
| 0 | 1.27 | 1.24 |
| 1 | 1.18 | 1.17 |
| 2 | 1.11 | 1.18 |
| 3 | 1.11 | **1.12** |
| 4 | 1.13 | (n/a) |

## What this run says

The hypothesis was that adding a 5th layer to v7's recipe (with a
proportional reduction in per-layer step count to fit 100 kJ) would
let the deeper representation capture more next-char structure.

It did not: layer 4's final NLL (1.13) is *higher* than v7's layer 3
(1.12). The gain in mid-network layers (layer 2 dropped 1.18 → 1.11)
is real but doesn't reach the inference layer, which is what
determines char-acc. The shorter per-layer training (3500 vs 4000
steps) cost ~0.01 nats on the deepest layer, exactly cancelling the
depth gain.

**Net char-acc**: 0.6785, statistically tied with v7's 0.6789.

## Position in the chain

| variant | config | char-acc | Δ |
|--|--|--:|--:|
| v7 | 4 × d=512 × 4000 steps + Muon | 0.6789 | — |
| **v8** | **5 × d=512 × 3500 steps + Muon** | **0.6785** | **−0.04 (tied)** |

Established that **at this energy budget, the depth/steps tradeoff
favours v7's 4 × 4000 over v8's 5 × 3500**. Subsequent variants
stayed at 4 layers.
