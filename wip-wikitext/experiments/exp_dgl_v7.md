# exp_dgl_v7 — DGL + Muon optimizer for hidden 2-D matmul weights

## Result

| metric | value |
|--|--|
| **char-accuracy** | **0.6789** (+1.51 pp over v6 — biggest single jump in the chain) |
| training energy | 74,218 J (74 % of 100 kJ budget) |
| training duration | 458 s |
| date (UTC) | 2026-05-06T09:37:00Z |
| GPU | NVIDIA A100-SXM4-40GB |
| job_id | `2026-05-06T09-25-35-dgl-greedy-layerwise-7867fc5c` |

Per-layer DGL training NLL (4000 steps each, WSD schedule):

| layer | final NLL |
|--|--:|
| 0 | 1.24 |
| 1 | 1.17 |
| 2 | 1.18 |
| 3 | **1.12** |

## What this run says

Up through v6 (4 × d=512, AdamW everywhere, warm-start aux heads),
the per-layer NLL ceiling sat at ~1.18 nats/char regardless of width
or depth. v6's deepest layer (1.18) was *no better* than its second
layer (1.21) — the local-NLL signal was saturating at a limit
unrelated to feature capacity.

v7 adds the **Muon optimizer** (Newton-Schulz5 orthogonalization,
lr=0.02, momentum=0.95, Nesterov) for the hidden 2-D matmul weights
in each block — `attn.qkv`, `attn.proj`, `fc1`, `fc2`. AdamW
(lr=2e-3) keeps everything else: token embedding, LayerNorms, aux
head. WSD schedule (250-step warmup, 80 % stable, 20 % linear decay
to 0).

Layer 3's NLL drops from v6's 1.18 to 1.12 — the first time any
DGL variant in this chain produced a layer below the 1.18 plateau.
Char-acc jumps from 0.6638 to 0.6789 (+1.5 pp) — the largest
single-knob improvement in the entire 12-experiment series.

**Why Muon and not AdamW**: each per-layer DGL update receives
correlated gradient directions (the local NLL at one layer always
points toward the same kind of feature change). AdamW's elementwise
step ends up applying low-rank effective updates over many steps,
because the gradient is dominated by the same singular vectors. NS5
orthogonalization makes every update full-rank in the matrix sense,
extracting the maximum information per step out of the local
gradient. This matters most at the deepest layer, which sees the
most-refined input distribution and benefits most from full-rank
updates.

## Position in the chain

| variant | config | char-acc | Δ |
|--|--|--:|--:|
| v6 | 4 × d=512 + warm-start, AdamW only | 0.6638 | — |
| **v7** | **+ Muon for hidden 2-D matmuls** | **0.6789** | **+1.51** |

This established the recipe (4 × d=512 + Muon + AdamW + WSD +
warm-start) that all subsequent v8/v9/v10/v11/v12 variants build on.
