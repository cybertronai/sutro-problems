# D5 kernel-taught Ladder: width 768, 510 updates

This derives directly from [@spartanhaden’s kernel-taught Ladder D5](../ladder-k2-d5-20261019/README.md), retaining its teacher, reconstruction objective, tangent jitter, hard-query sampling, fused Triton combinator backward and guard. Credits remain with the original entry and its sources: @SethTS/Rasmus et al. for Ladder, @islamborghini for kernel-PCG and batch/step work, @likelytobelaura for tangent jitter, and @sigkillme0 for grouped graph replays. This is an architecture/learning-budget change, not a new kernel or learning algorithm.

## Changes

The student changes from **60–1024–512–256–10** to **60–768–512–256–10**. Adam updates decrease from 600 to **510**, batch stays **2500**, and peak LR changes from .011 to **.012970588235294119**, approximately retaining the integrated LR budget. The original constant-then-linear schedule remains: constant for the first two-thirds, linearly decaying afterward. The existing backward kernel receives a width-768 configuration `(64,8,4,2,32)`; its body is unchanged. Grouping stays 30 steps per replay, so all 510 updates execute without truncation.

The two-stage kernel teacher, pseudo-label refit, BN/ReLU Ladder, noise .25, tangent jitter, reconstruction, optimizer implementation and prediction guard are unchanged. Neural activations are BF16; master parameters and Adam states stay FP32 as in the parent. There is one student, no final neighbor lookup and no newly added augmentation. All per-call metric fitting, teacher fitting, student reset/training and prediction remain inside the measured warm call. Input-dependent learned state is reset before reuse; code, graphs and scratch can persist.

## Three independent author canonical sandbox evaluations

| Run | Ranked ms | MNIST error | Above-idle energy mJ |
|---|---:|---:|---:|
| 1 | 508.222125 | 1.758182% | 96301.060554 |
| 2 | 507.518126 | 1.768182% | 84459.817560 |
| 3 | 506.010506 | 1.813636% | 86870.448204 |

**All three pass** the unchanged canonical sandbox judge, including all four foreign hold-out calls in every run. Median ranked time is **507.518126 ms**, median above-idle energy **86870.448204 mJ**, and pooled MNIST error **1.780000%** (324126/330000 correct). Three distinct A100-SXM4 GPU UUIDs were recorded. All original outcomes are included; there were no agent-initiated replacements or qualification reruns, and no infrastructure preemption is recorded in the execution log.

The listed parent reviewer medians are 633.529424 ms / 107885.521303 mJ. This author group is approximately **19.9% faster** and **19.5% lower energy**, while pooled error is slightly higher (1.78% versus 1.75%). These groups use different data draws and GPU hosts, so this is not a paired accuracy or hardware comparison. Independent reviewer replication remains outstanding. This submission does not edit the leaderboard or claim an accepted record.

## Evidence and reproduction

The exact **20452-byte** source has SHA256 `5681b8eafe2310c416b69336cb96ad2fe96c38d9d850818d23602aab9e252667`. Packaging was checked against the development program by AST, excluding updated module documentation. [Manifest](evidence/manifest.json), [all three raw records](evidence/official/) and [verification](evidence/official/verification.json) are included. Saved-record verification independently rebuilds the canonical judge, score, summary, source identity and all energy windows; every check passes. Scorer and runner hashes match the tracked versions. This audit is not a second execution group.

```bash
python run_modal.py submissions/ladder-width768-d5-20261019/submission.py:ladder_kernel --difficulty 5 --runs 3
cd submissions/ladder-width768-d5-20261019
python verify_records.py evidence/official submission.py --difficulty 5 --function ladder_kernel --record-filename classical-width768-s510-d5-20261019.py --output evidence/official/verification.json
```

Included development data are warm complete-call measurements, not canonical qualification. Eleven fresh paired inputs, each with three stochastic retrainings, measured the width-768/510 variant at **1.803636% / 506.665915 ms**, versus the 1024-wide/450-update development control at **1.838182% / 484.882431 ms**. The shorter control failed one of three later canonical runs by eight predictions; its entire original failed group is retained separately in [evidence/prior-fast450](evidence/prior-fast450/). It is a distinct source/configuration and has no qualifying median. No failed result was silently replaced by this candidate.

Additional summaries retain unsuccessful or exploratory optimizer, batch and readout comparisons. They are not part of this submission. The current classifier uses the original head/guard; no nearest-neighbor readout, native fused Adam or larger-batch variant is included.
