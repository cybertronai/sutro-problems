# exp_dgl_v9 — DGL+Muon backbone + CE-trained linear readout

## Result

| metric | value |
|--|--|
| **char-accuracy** | **0.6803** (best so far; below 0.7310 by 5.1 pp; +0.14 pp over v7) |
| training energy | 79,891 J (80 % of 100 kJ budget) |
| training duration | 480.9 s |
| date (UTC) | 2026-05-06T10:05:43Z |
| GPU | NVIDIA A100-SXM4-40GB (Lambda On-Demand, us-west-2) |
| job_id | `2026-05-06T09-53-53-dgl-greedy-layerwise-4d209a01` |

Per-layer DGL+Muon training NLL (3500 steps each, WSD schedule):

| layer | final NLL |
|--|--|
| 0 | 1.26 |
| 1 | 1.18 |
| 2 | 1.14 |
| 3 | 1.14 |
| readout (CE-trained, 3000 steps) | 1.16 |

## What this run says

v9 combines v7's best-known recipe (4×d=512 with Muon for hidden
2-D matmuls + AdamW for the rest, WSD schedule) with a v4-style
post-DGL CE-trained linear readout from per-layer-LN'd concatenated
features. Net gain over v7's last-layer-aux-head readout: **+0.14 pp**.

Three prior experiments (v3 ridge, v4 CE on plain DGL, v9 CE on
DGL+Muon) consistently show that *adding a readout on top of DGL
features* contributes only ~0.1–0.2 pp char-acc. The features
across layers are nearly redundant for next-character prediction
even after Muon's full-rank updates.

## Sequence of stacked improvements

| variant | config | char-acc | Δ |
|--|--|--:|--:|
| v1 | plain DGL 5×384 AdamW | 0.6440 | — |
| v2 | + warm-start aux heads | 0.6553 | +1.13 |
| v6 | + d=512 (4 layers) | 0.6638 | +0.85 |
| v7 | + Muon optimizer | 0.6789 | +1.51 |
| **v9** | **+ CE readout from concat features** | **0.6803** | **+0.14** |

Each component contributes; Muon is the largest single win
(+1.5 pp), warm-start is meaningful (+1.1 pp), wider features
add modest gain (+0.85 pp), CE readout is essentially noise
(+0.14 pp).

## Why this hits a 0.68 ceiling

The cumulative pattern across 9 attempts: under DGL with local
next-character NLL, every architectural / optimization knob we've
tried contributes diminishing-returns improvements that asymptote
near char-acc 0.68 (NLL ~1.10–1.15). The fundamental gap to the
modded backprop baseline (0.7310, NLL ~1.05) appears to need
something that breaks DGL's *local* signal — either:

- A method providing actual end-to-end signal (DFA worked badly in
  v5 but with proper tuning could be revisited), OR
- A multi-future-token / contrastive local objective that pushes
  per-layer features beyond the linear-readout-NLL plateau, OR
- Substantially more energy than 100 kJ (which is the leaderboard
  constraint).

Within 100 kJ and pure local-NLL DGL, **0.6803 looks like the
practical ceiling** on this 6-layer-modded-architecture task.
