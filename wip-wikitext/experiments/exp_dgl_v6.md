# exp_dgl_v6 — wider DGL (4 layers × d=512) with warm-start

## Result

| metric | value |
|--|--|
| **char-accuracy** | **0.6638** (best DGL result so far; below 0.7310 bar by 6.7 pp; +0.85 pp vs v2) |
| training energy | 72,033 J (72 % of 100 kJ budget) |
| training duration | 449.0 s |
| date (UTC) | 2026-05-06T09:21:43Z |
| GPU | NVIDIA A100-SXM4-40GB (Lambda On-Demand, us-west-2) |
| job_id | `2026-05-06T09-10-33-dgl-greedy-layerwise-8abf3f38` |

Per-layer training NLL:

| layer | final NLL | warm-start step-0 |
|--|--|--|
| 0 | 1.27 | 5.67 (random init — first layer) |
| 1 | 1.21 | 1.27 ✓ |
| 2 | 1.21 | 1.21 ✓ |
| 3 | 1.18 | 1.19 ✓ |

## Comparison

| run | layers × d | n_steps | char-acc | NLL_last | energy |
|--|--|--|--|--|--|
| v1 | 5 × 384 | 3000 | 0.6440 | 1.22 | 54 kJ |
| v2 | 6 × 384 | 3500 (warm-start) | 0.6553 | 1.17 | 80 kJ |
| v4 | 6 × 384 + CE readout | 2500+4000 | 0.6555 | 1.19 | 77 kJ |
| **v6** | **4 × 512** | **4000 (warm-start)** | **0.6638** | **1.18** | **72 kJ** |

Compared to v2 (same warm-start recipe at d=384), v6 achieves the same
last-layer NLL (1.17–1.18) but the char-acc is +0.85 pp. The
improvement is small but real: wider features at each layer translate
to slightly better next-character argmax accuracy at the same
information-theoretic NLL — likely because the *distribution* over
the character vocabulary is sharper / more peaked when the readout
sees a higher-dimensional feature.

Layer-0 final NLL did improve materially from v2's 1.32 to v6's 1.27
— wider really does increase per-layer linear-readout capacity. But
the per-layer ceiling at deeper layers held at ~1.20, confirming
that **the DGL ceiling is intrinsic to the local-NLL training
signal**, not the model size: any single layer's linear readout from
its own features tops out around 1.18–1.21 nats/char regardless of
whether those features are 384-d or 512-d.

## Energy headroom

72 kJ used → 28 kJ left. With this margin, possible next variants:
- 5 × 512 × 3000 ≈ 96 kJ (depth + width)
- 4 × 640 × 3000 ≈ 90 kJ (more width)
- 4 × 512 × 4000 + CE readout ≈ 92 kJ (width + better readout)

None of these are likely to close 6.7 pp on their own based on the
diminishing-returns pattern across v1→v2→v4→v6. To actually beat
0.7310 in 100 kJ probably requires either:

1. A *different training paradigm* with end-to-end signal
   (DFA tuned much more carefully than v5's naive run; or
   target-prop variants), OR
2. A *much faster optimizer per step* (Muon, which the modded
   baseline used at lr=0.02 — 40× higher than the AdamW lrs used
   here — to achieve the bar in 3000 backprop steps). Implementing
   Muon for our blocks would give DGL more update capacity per step.
