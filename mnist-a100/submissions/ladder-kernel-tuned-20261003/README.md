# Kernel-tuned sampled Ladder — difficulty 3, 2026-10-03

An 800-step variant of SethTS’s accepted [sampled Ladder](https://github.com/cybertronai/sutro-problems/blob/aa6c51e6ad661089cbf71a39c95c577c216b627f/mnist-a100/submissions/ladder-sampled-20260929/README.md), with tuned forward tiling and decoder/encoder backward launches. The architecture, objective, loss-proportional sampler, optimizer, TF32 policy and per-call learned-state reset follow that learner. **Three fresh sandboxed Modal A100-80GB runs of unchanged scorer 1.2.0 passed; the result is their median.**

**Submission for maintainer review.**

| Difficulty | Band | Steps | Mix | ms/call | mJ/call above idle | MNIST error, 33 calls | Published best | File |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 2.70% | 800 | 0.9 | **1,157.138** | **151,403.970** | 2.5461% | 1,443.3 ms; 201,792 mJ | [ladder_d3.py](ladder_d3.py) |

Time is 19.83% lower and energy 24.97% lower than the published difficulty-3 row at upstream revision `aa6c51e6ad661089cbf71a39c95c577c216b627f`. This is a comparison of separate canonical runs, not a paired replay of the published baseline. The proposed leaderboard row reports the three-run median.

## Changes to the code

- Training budget: 900 → 800 steps. This inherited the separately qualified 800-step local variant; it is disclosed rather than attributed to kernel tuning.
- Forward fused kernels: 4,096-element tiles instead of 2,048.
- Decoder backward: parameter-gradient stores follow their required intermediates, shortening register lifetimes. Production batch uses four columns at widths ≥250, two at width 60, one at width 10, all eight warps.
- Encoder backward: at batch 1,000, widths ≥250 use 16 warps/four columns; the small layer uses eight warps/one column. Ragged batches retain the previous geometry.
- The entry point is `classify`; attribution is retained. No architecture, loss, sampler, learning-rate schedule, optimizer or prediction change beyond the disclosed training-step reduction.

Using supplied test features for unlabelled reconstruction is the same transductive method as the accepted Ladder; no test labels are available to the sandboxed learner. Every call initializes weights, Adam state, sampling scores/schedule and counters before 800 CUDA-graph replays. Compile/capture caches are reused across the allowed foreign warmup, with learned state reset.

## Runs

| Run | GPU / power limit | Score ms/call | Energy mJ/call | MNIST correct | Mean error | Worst error | Holdout accuracy |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | NVIDIA A100-SXM4-80GB, 500 W | 1,157.137595 | 151,759.932 | 107,336/110,000 | 2.4218% | 2.62% | kmnist: 96.1650% |
| 2 | NVIDIA A100-SXM4-80GB, 400 W | 1,156.458740 | 151,403.970 | 107,180/110,000 | 2.5636% | 2.77% | kmnist: 96.0325% |
| 3 | NVIDIA A100-SXM4-80GB, 400 W | 1,160.505341 | 149,399.707 | 107,082/110,000 | 2.6527% | 2.81% | fashion: 87.5425% |

Three distinct boards and single-use official containers; UUIDs, timing calls, energy/control windows and telemetry references are preserved in [raw official evidence](evidence/official/). Each run includes 11 MNIST and four foreign calls, with a fresh sandboxed worker and foreign warmup for every timed call. The canonical `judge` and `energy_summary` were independently reconstructed with matching results in [verification.log](evidence/official/verification.log).

## How the settings were chosen

Kernel tuning used paired development draws with all mathematical formulas preserved, followed by full-call measurements and gradient comparisons. Default/cuBLAS/cuBLASLt preference changes, transposed weight storage, wider tiles with eight warps, and encoder-only statistics caching gave no useful accepted speedup and were rejected. The final encoder launch was selected from an unchanged-kernel warp/column sweep; it was confirmed on both SXM4 and PCIe hardware with execution order reversed.

Development-only encoder comparison, eight paired draws on four pool splits and three fits per draw/variant:

| Metric | Prior decoder-tuned variant | Final encoder-tuned variant |
| --- | ---: | ---: |
| Median CUDA ms, 24 fits each | 1188.596 | 1155.406 |
| Mean error | 2.4829% | 2.4458% |

These development medians are not the ranked score. Whole-loss/per-row-CE/all-26-parameter-gradient comparisons were recorded after timing, not enforced as a premeasurement rejection gate; the final candidate passed both comparisons, maximum relative L2 discrepancy 0.000161857213. Individual fused-kernel outputs/input/parameter gradients passed rows 1,000/257 and widths 1,000/500/250/60/10 at atol 3e-5/rtol 5e-4; maximum relative L2 discrepancy 1.44024014e-7.

Raw development measurements and source snapshots: [comparison 1](evidence/development/encoder-tuned-profile-1.json), [comparison 2](evidence/development/encoder-tuned-profile-2.json), [warp sweep](evidence/development/encoder-warp-sweep-1.json).

## Profiling

The frozen source’s final development five-step graph trace:

| Group / kernel family | Five-step µs | Scaled 800-step ms | Trace share |
| --- | ---: | ---: | ---: |
| GEMMs and split-K reductions | 3085.120 | 493.619 | 42.31% |
| _encoder_forward | 934.140 | 149.462 | 12.81% |
| _encoder_backward | 908.773 | 145.404 | 12.46% |
| _decoder_backward | 875.676 | 140.108 | 12.01% |
| _decoder_forward | 318.735 | 50.998 | 4.37% |

The 800-step column scales cumulative GPU kernel durations; it is not a separate full-call wall stage. GEMMs remain the largest aggregate cost; encoder forward is the largest individual custom kernel family. Final development stages were reset 1.945 ms, training 1,154.972 ms, and calibration/prediction 2.566 ms. Training dominates time and diagnostic stage energy. Actual occupancy, bandwidth and per-kernel energy were not measured. Official ranked timing and energy come only from the canonical records above, without inserting a profiler into the scorer.

## Validation

- Source passes the canonical import/source gate at 20,194 bytes, 286 bytes below the limit. Every official record matches the packaged source hash and entry point.
- Canonical sandbox, output, mean/worst MNIST error, foreign accuracy, timing dispersion and independent parent-clock gates pass in all three runs. Canonical energy/control/telemetry checks also pass.
- Source and local reset review found no label leakage, hardcoded trained prior, timing manipulation or learned-state carryover mechanism. The additional canonical runs exercise the actual sandboxed GPU path. Maintainer reproduction and acceptance remain separate.
- Development energy values are diagnostic and never used in the official energy column.

## Reproduce

From `mnist-a100/`, with Modal configured; this launches paid A100 work:

```sh
python run_modal.py submissions/ladder-kernel-tuned-20261003/ladder_d3.py:classify \
  --difficulty 3 --runs 3 --json /tmp/ladder-kernel-tuned-new
```

Reconstruct the packaged saved records locally, without GPU work:

```sh
python submissions/ladder-kernel-tuned-20261003/verify_records.py \
  submissions/ladder-kernel-tuned-20261003/evidence/official \
  --source submissions/ladder-kernel-tuned-20261003/ladder_d3.py
```

The qualification used the unchanged official `remote_score`, with only three maximum concurrent containers and a 600-second per-container timeout. Account checking and app cleanup surrounded the run. The source was frozen before execution. The local verifier additionally checks distinct UUIDs; that extra check is not a canonical rejection rule.

## SHA-256 and provenance

- `ladder_d3.py`: `5052f86e9cdf6086783bd64c90a31fcb86b3d19d1d0f27ef8f21457923c713e1`.
- Scorer 1.2.0: `1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064`.
- Runner: `1180b843cf6c32414774da90c835a8b7298d2db8719989237a793ca1c62a10ab`.
- SethTS’s accepted sampled-Ladder source: `0c11521f8ef34e4356ac335160fc1537f16e6805dbbeba9870646614a869452c`.
- Earlier 800-step source: `ecf2c2588947f9d58382dcab1fbb019222c6816f351be89400ac13b4b6cc185a`; its separate official results are not attached to this source.

Manifest and exact-source source/scorer/runner hashes: [manifest.json](evidence/official/manifest.json). Aggregates: [summary.json](evidence/official/summary.json).

## Independent follow-up review and readiness

[Independent qualification review](evidence/INDEPENDENT_QUALIFICATION_REVIEW.md): **GO for submission** after source, manifest, all three canonical judge/energy records, aggregates and report checks. The review is local saved-record reconstruction, not an additional GPU run or maintainer acceptance. All task apps are stopped and the final Modal container list was empty.
