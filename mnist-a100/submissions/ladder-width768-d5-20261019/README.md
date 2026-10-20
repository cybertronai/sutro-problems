# D5 kernel-taught Ladder: width 768, 480 fused Adam updates

## Independent reviewer reproduction — October 19, 2026, later pass

The first frozen reviewer group passed **3/3 fresh canonical sandboxed Modal A100-80GB runs**. Median ranked time is **492.842637 ms/call**; median above-idle energy is **77,089.864599 mJ/call**. Pooled MNIST error is **1.792424%** (324,085/330,000 correct). Each run independently passes every accuracy, holdout and timing gate; pooled accuracy is descriptive.

| Reviewer run | MNIST error | Margin, labels | Ranked ms | Above-idle mJ |
| - | -: | -: | -: | -: |
| 1 | 1.807273% | +102 | 492.842637 | 71983.093171 |
| 2 | 1.697273% | +223 | 492.085754 | 77089.864599 |
| 3 | 1.872727% | +30 | 495.066059 | 80849.184445 |

The submitted source is unchanged, SHA-256 `e8d330a1a81b139f5b89290325b6f7a04123f47dc3c84af354b141f96a059b53`; scorer and stock runner are unchanged. All 45 timed calls completed. No outcome was replaced or retried. [Raw records and executable verifier](evidence/reviewer-20261019-pass3/), [reconstruction](evidence/reviewer-20261019-pass3/verification.json) and [source review](evidence/reviewer-20261019-pass3/source-review.json) are retained. Optional energy is assessed separately from score qualification.

This is @sjbaebae's budget/kernel optimization of @spartanhaden's kernel-taught Ladder. Credits to Haden, @SethTS/Rasmus et al., @islamborghini, @likelytobelaura and @sigkillme0 remain in the submission. Teacher fitting, 480 student updates and both predictions execute every call; the label-symmetric guard selects already-computed predictions. Earlier 510-step and failed fast450 groups remain separate from this exact-source group. Development arithmetic reconstructs; unpublished variant source/history claims are not independently authenticated. The author manifest's pending wording is historical.

The leaderboard uses these independent medians; original author results below remain separate. Published instructions require three official runs and their median; this reviewer reproduction supplies evidence and does not introduce a permanent source ban or an additional published submission rule.

Using Python 3.13, from this submission directory: `python evidence/reviewer-20261019-pass3/verify_records.py evidence/reviewer-20261019-pass3 submission.py --difficulty 5 --function ladder_kernel --output /tmp/pr135-verification.json`. This audits saved records without executing the learner.

## Original author report (as submitted)

Pending-review and record claims below describe the author submission; the dated reviewer section gives accepted status.


This derives directly from [@spartanhaden’s kernel-taught Ladder D5](../ladder-k2-d5-20261019/README.md), retaining its kernel teacher, reconstruction, tangent jitter, hard-query sampling, fused Triton combinator backward and guard. Original credits remain: @SethTS/Rasmus et al. for Ladder, @islamborghini for kernel-PCG and batch/step work, @likelytobelaura for tangent jitter, and @sigkillme0 for grouped graph replays. This is an architecture/learning-budget/optimizer-implementation change, not a new learning algorithm.

## Current frozen candidate

The student changes from **60–1024–512–256–10** to **60–768–512–256–10**, and Adam updates decrease from 600 to **480**. Batch remains **2500**; LR is **.01378125**, approximately retaining the integrated LR budget. The original constant-first-two-thirds, linear-tail schedule and Adam betas (.8, .999) remain. Adam uses the native fused implementation rather than compiled foreach, still through the compiled optimizer wrapper. Floating-point updates may differ; bitwise equivalence is not claimed. Grouping remains 30 steps per graph replay, so exactly 480 updates execute.

The existing backward kernel receives a width-768 configuration `(64,8,4,2,32)`; its body is unchanged. The teacher’s two metric updates and pseudo-label refit, BN/ReLU Ladder, Gaussian noise .25, tangent jitter, reconstruction and guard are unchanged. Neural activations are BF16; master parameters and Adam states remain FP32. There is one student and no final neighbor lookup. Test features are used without labels for teacher pseudo-labelling, the training pool and reconstruction, as in the parent.

All metric/teacher fitting, student reset/training and prediction occur in each measured warm call. Input-dependent learned state is reset before reuse; only code, graphs and scratch persist. Fused Adam alone was not faster in development: the retained candidate couples it with fewer updates. The complete-call gain is not attributed solely to optimizer fusion.

## Three independent author canonical sandbox evaluations

| Run | Ranked ms | MNIST error | Above-idle energy mJ |
|---|---:|---:|---:|
| 1 | 491.963506 | 1.790000% | 83887.252169 |
| 2 | 492.013229 | 1.726364% | 84000.853066 |
| 3 | 492.937883 | 1.810000% | 86014.630982 |

**All three pass** the unchanged canonical sandbox judge, including all four foreign hold-out calls in every run. Median ranked latency is **492.013229 ms**, median above-idle energy **84000.853066 mJ**, and pooled MNIST error **1.775455%** (324141/330000 correct). Each of the three ranked times is below 500 ms. Three distinct A100-SXM4 GPU UUIDs were recorded. All original outcomes are included; there were no agent-initiated qualification replacements, and no infrastructure preemption is recorded in the execution log.

The listed parent reviewer medians are 633.529424 ms / 107885.521303 mJ. This author group is approximately **22.3% faster** and **22.1% lower energy**; pooled error is slightly higher (1.775455% versus 1.75%). These are different data draws and GPU hosts, not a paired comparison. Independent reviewer replication is outstanding. The leaderboard is not edited and no accepted record is claimed.

## Evidence and reproduction

Current exact source: **20436 bytes**, SHA256 `e8d330a1a81b139f5b89290325b6f7a04123f47dc3c84af354b141f96a059b53`. Packaging matches the development program by AST except updated module prose. [Manifest](evidence/manifest.json), [all current raw records](evidence/official/) and [verification](evidence/official/verification.json) are included. Saved-record verification independently rebuilds the judge, score, summary, source identity and every energy window; all checks pass. Scorer and runner hashes match the tracked versions. This audit is not a second execution group.

```bash
python run_modal.py submissions/ladder-width768-d5-20261019/submission.py:ladder_kernel --difficulty 5 --runs 3
cd submissions/ladder-width768-d5-20261019
python verify_records.py evidence/official submission.py --difficulty 5 --function ladder_kernel --record-filename classical-width768-fused480-d5-20261019.py --output evidence/official/verification.json
```

## Prior candidates and development

This PR initially contained a distinct 510-update compiled-foreach candidate with a qualifying **507.518126 ms** median and **1.780000%** pooled error. Its exact source, manifest, all three passing raw records and re-audit are retained in [evidence/prior-width768-s510](evidence/prior-width768-s510/). No records were pooled across candidates to choose a median. Its original commit remains in PR history. Re-audit it with:

```bash
python verify_records.py evidence/prior-width768-s510 evidence/prior-width768-s510/submission.py --difficulty 5 --function ladder_kernel --record-filename classical-width768-s510-d5-20261019.py --output evidence/prior-width768-s510/verification.json
```

An earlier, different 1024-wide/450-update candidate failed one of three canonical runs by eight predictions. Its complete original group remains separately in [evidence/prior-fast450](evidence/prior-fast450/); it has no qualifying median. No failed outcome was replaced by a passing run.

Development measurements are warm complete calls, not canonical qualification. Eleven fresh paired inputs with three stochastic retrainings per input measured compiled-foreach/510 at **1.823939% / 509.653650 ms**, native-fused/480 at **1.842424% / 494.013724 ms**, and native-fused/510 at **1.829697% / 525.629177 ms**. Fewer updates reduced runtime with slightly worse development accuracy; the exact final-source canonical group above independently qualifies. Full development records are included. Other batch/readout summaries are exploratory and are not part of this classifier.
