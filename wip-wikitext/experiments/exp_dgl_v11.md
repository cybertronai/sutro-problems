# exp_dgl_v11 — DGL+Muon, 5000 steps/layer (vs v7's 4000)

## Result

| metric | value |
|--|--|
| **char-accuracy** | **0.6839** (new best, +0.36 pp over v9, gap to 0.7310 = 4.71 pp) |
| training energy | 93,135 J (93 % of 100 kJ budget) |
| training duration | 558 s |
| date (UTC) | 2026-05-06T10:46:32Z |
| GPU | NVIDIA A100-SXM4-40GB (Lambda On-Demand) |
| job_id | `2026-05-06T10-33-29-dgl-greedy-layerwise-6f0bbd75` |

Per-layer DGL+Muon training NLL (5000 steps each, WSD schedule):

| layer | v11 (5000 steps) | v7 (4000 steps) | Δ |
|--|--:|--:|--:|
| 0 | 1.25 | 1.24 | +0.01 |
| 1 | 1.15 | 1.17 | -0.02 |
| 2 | 1.10 | 1.18 | -0.08 |
| 3 | 1.14 | 1.12 | +0.02 |

## What this run says

The hypothesis was that the v7 recipe was undertrained — its 4000-step
WSD schedule (with 90 % stable phase) might not have given Muon's
orthogonalised updates enough wall-clock to fully exploit the per-layer
local NLL signal.

**Result: partial confirmation.** 25 % more steps → +0.5 pp char-acc.
The improvement is real but small relative to the 4.7 pp remaining gap.

Mid-network layers (1, 2) benefit clearly: layer 2's NLL drops by
0.08 (1.18→1.10), suggesting these layers had headroom under the
per-layer ceiling that v7's shorter schedule did not exhaust.
Boundary layers (0, 3) are essentially unchanged: layer 0 saturates
fast on its narrow next-char objective; layer 3's WSD decay hit before
its loss had stabilised at the new lower level (4500 → 1.097, but
4999 → 1.137 — slight overshoot in the decay phase).

## Sequence of stacked improvements (updated)

| variant | config | char-acc | Δ |
|--|--|--:|--:|
| v1 | plain DGL 5×384 AdamW | 0.6440 | — |
| v2 | + warm-start aux heads | 0.6553 | +1.13 |
| v6 | + d=512 (4 layers) | 0.6638 | +0.85 |
| v7 | + Muon optimizer | 0.6789 | +1.51 |
| v9 | + CE readout from concat features | 0.6803 | +0.14 |
| **v11** | **+ 5000 vs 4000 steps/layer** | **0.6839** | **+0.36** |

## Position w.r.t. the goal

0.6839 char-acc is the new high-water mark for DGL on this benchmark
within 100 kJ, but the **0.7310 threshold remains 4.71 pp away** at
93 % budget utilisation. Pure DGL+Muon parameter sweeps are unlikely
to close that gap — each cumulative improvement gets smaller while the
energy fraction grows. To break through:

- The **CE readout** on v11 features (would cost ~5 kJ extra; we have
  ~7 kJ slack) might recover the +0.14 from v9 → ~0.685, still 4.6 pp
  short.
- A **MLP aux head** (1-layer MLP instead of linear next-char head) may
  raise the per-layer ceiling above the linear-readout limit.
- DFA tuned aggressively, target-prop, or forward-forward remain the
  only candidates with structurally different signal.
