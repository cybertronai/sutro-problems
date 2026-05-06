# exp_dgl_v10 — DGL+Muon + multi-future-token (K=4) aux objective

## Result

| metric | value |
|--|--|
| **char-accuracy** | **0.6787** (≈ tied with v7's 0.6789, below v9's 0.6803 best) |
| training energy | 64,539 J (65 % of 100 kJ budget) |
| training duration | 621.1 s |
| date (UTC) | 2026-05-06T10:31:00Z |
| GPU | NVIDIA A100-SXM4-40GB (Lambda On-Demand, us-west-2) |
| job_id | `2026-05-06T10-16-58-dgl-greedy-layerwise-8010797b` |

Per-layer training NLL (k=0 next-1 component only, comparable to v7):

| layer | next-1 NLL | v7 next-1 NLL |
|--|--:|--:|
| 0 | 1.35 | 1.24 |
| 1 | 1.19 | 1.17 |
| 2 | 1.14 | 1.18 |
| 3 | 1.13 | 1.12 |

## What this run says

The hypothesis was that a *harder* per-layer local objective —
predicting K=4 future tokens, not just the next one — would force
deeper layers to encode multi-step structure and push the next-1 NLL
below v7's ceiling. This did not happen:

- **Layer 0** is *worse* on next-1 NLL (1.35 vs 1.24) because the
  3000-step budget plus diluted (1/K) weighting on the next-1
  gradient under-trains the immediate-next-char predictor.
- **Layer 2** is *better* (1.14 vs 1.18) — the only sign that
  intermediate layers might benefit from richer feature pressure.
- **Layer 3** ties (1.13 vs 1.12) — no measurable benefit at the
  inference layer, which is what matters for char-acc.

Final char-acc 0.6787 (≈ v7 = 0.6789, below v9 = 0.6803).

## Iteration costs

Two attempts before this one were aborted:

- **v10 (4 layers × 4000 steps)**: K=4 CE-in-Python-loop ran at
  ~42 ms/step → projected 100 kJ+ training energy, would trip the
  watchdog. Killed at step 1500 (loss curve looked normal). Used
  ~25 kJ.
- **v10b (3500 steps + fused single CE)**: same ~42 ms/step despite
  the loss optimisation — slowdown is from the wider aux head
  matmul (d → K·V) and K-position label stacking, not the loss op.
  Killed pre-emptively at step 2000. Used ~28 kJ.
- **v10c (3000 steps + fused CE)**: completed cleanly at 65 kJ.

Combined "wasted" energy on the two killed attempts: ~50 kJ across
two distinct sessions (each within their own 100 kJ budget — these
don't compound). Lesson: always do a quick pre-flight on per-step
wall time before committing to step counts, especially when the aux
head shape changes.

## Position in the overall chain

| run | δ vs prior best |
|--|--:|
| v1 (DGL plain AdamW) → 0.6440 | — |
| v2 (warm-start) → 0.6553 | +1.13 |
| v6 (wider d=512) → 0.6638 | +0.85 |
| v7 (Muon optimizer) → 0.6789 | +1.51 |
| v9 (CE readout from concat features) → 0.6803 | +0.14 |
| **v10 (multi-future-token K=4) → 0.6787** | **−0.16** |

Multi-future-token aux objective: net negative for inference char-acc
in this DGL setup. Confirmed it does *not* break the local-NLL
ceiling at d=512 with Muon. Future iterations should not retry naive
multi-token at the same step budget — would need substantially more
DGL steps (4000+ at multi-token) to compensate for the diluted
gradient on the inference target, which doesn't fit in 100 kJ here.
