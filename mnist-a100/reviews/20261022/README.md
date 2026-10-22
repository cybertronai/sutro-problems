# Sutro pull-request review — October 22, 2026

This pass merged **#136 (D2), #137 (D3), #138 (D4) and #139 (D5)**, all by @spartanhaden, as new latency records. All **12/12 returned logical reviewer runs** pass canonical sandboxed Modal A100-80GB qualification, covering **180 saved timed calls**. Seven older PRs remain open with unchanged evidence.

| PR | Band | Reviewer passes | Median ms | Median above-idle mJ | Accuracy margin, labels | Latency reduction | Records |
| - | - | -: | -: | -: | - | -: | - |
| [#136](https://github.com/cybertronai/sutro-problems/pull/136) | D2 | 3/3 | 2.924269 | 172.382541 | +366, +274, +318 | 8.2% | [evidence](../../submissions/fisher-woodbury-d2-20261020/evidence/reviewer-20261022/) |
| [#137](https://github.com/cybertronai/sutro-problems/pull/137) | D3 | 3/3 | 3.510784 | 358.758434 | +157, +210, +217 | 23.0% | [evidence](../../submissions/fisher-hknn-d3-20261020/evidence/reviewer-20261022/) |
| [#138](https://github.com/cybertronai/sutro-problems/pull/138) | D4 | 3/3 | 38.695741 | 6664.381444 | +223, +94, +285 | 50.6% | [evidence](../../submissions/hybrid-fisher-triton-220-d4-20261020/evidence/reviewer-20261022/) |
| [#139](https://github.com/cybertronai/sutro-problems/pull/139) | D5 | 3/3 | 406.915422 | 67812.879400 | +120, +152, +226 | 17.4% | [evidence](../../submissions/ladder-label-topdown-450-d5-20261020/evidence/reviewer-20261022/) |

Latency reductions use the previous full-precision record medians: D2 `3.184644651675189`, D3 `4.558382597836581`, D4 `78.33383761669307` and D5 `492.84263749556106` ms. The leaderboard uses the new reviewer medians, not author timing claims. Each complete run independently passes accuracy, holdout, timing, dispersion and sandbox gates; pooled accuracy cannot repair a failed run. All 12 returned runs also have valid optional energy, with three distinct GPU UUIDs per group. Energy is above-idle GPU board energy, excludes host/system energy and initial compilation, and is assessed separately from score.

## Source, protocol and provenance

The exact reviewed heads, source hashes, merge SHAs/times, raw-record hashes and full-precision results are in [summary.json](summary.json). Each linked evidence package retains unchanged learner identity, source/reset/ownership and attribution reviews, source/launch manifests, all three raw returned records, independent reconstruction, an executable saved-record verifier and a SHA-256 inventory. The published scorer and stock runner remain byte-identical to frozen main `b374daa1a59cf0e3ce9a6462ecdf2f143ac5b5fc`; no learner was repaired during qualification. All four packaged verifiers were rerun successfully without executing the learners.

**#138 had a Modal platform preemption and automatic restart.** No maintainer-launched replacement invocation was submitted, and no returned failure was discarded. All three returned records and the controller log are retained. The interrupted container returned no score record or per-call data, so its stage, outcome and mapping to a logical run are unknown. The 12 runs and 180 calls above count returned records; they do not imply twelve uninterrupted physical attempts or count every call that may have executed before preemption. The other three groups have no observed restart. All four reviewer apps stopped.

#136 replaces the D2 preconditioner with standard-form Woodbury algebra and FP16 products. #137 combines Fisher-PCG, a rank-48 metric/natural-gradient step and an HKNN readout on compacted low-margin rows. #138 distils two Fisher-PCG teachers into three tangent-jittered students using 220 updates and a Triton training step. #139 uses a width-768, 450-update kernel-taught Ladder with labelled-row top-down targets, a per-call noise bank and hidden-space HKNN. Reviews checked current-input refitting, parameter/optimizer/sampler resets, stream/graph ownership, bounded reads/writes and branchless guards where applicable; no static blocker remained.

Individual sources and reports retain the relevant @sigkillme0 Fisher/compression and labelled-top-down lineage, @islamborghini kernel-PCG, @likelytobelaura tangent work, @sjbaebae cross-teaching/readout and width-768 work, Haden's hybrid/Ladder/HKNN work, and the cited Ladder/local-hull ancestors. Numerical/algorithm changes are not claimed to be bitwise equivalent or isolated-kernel causal improvements. Each author's separate exact-source three-run group reconstructs, but supplied development counts, pass probabilities, sanitizer assertions and unsubmitted variant histories were not independently reproduced. Those claims remain separate from accepted qualification. These D2-D5 results establish neither D6 qualification nor a global accuracy ceiling.

## Older open PRs

[Fresh GitHub metadata](older-pr-recheck.json) confirms all seven older heads, bodies and file metadata are unchanged, with no new discussion, review or inline comments. Their unchanged sources were not rerun.

| PR | Reason it remains open |
| - | - |
| [#110](https://github.com/cybertronai/sutro-problems/pull/110) | Explicitly non-record experiment; repository policy keeps it on a separate branch. Its requested A100 diagnostic was completed previously. |
| [#114](https://github.com/cybertronai/sutro-problems/pull/114) | Earlier reviewer group passed 1/3; two complete D4 runs missed accuracy by 64 and 4 predictions. |
| [#119](https://github.com/cybertronai/sutro-problems/pull/119) | Author 3/3 supported; reviewer 1/3, with two later untimed warmups exceeding 90 seconds after seven saved calls. Code allows 180 seconds first and 90 later, while documentation says 180; stalled phase remains unidentified. Hold for reproduction diagnosis. |
| [#122](https://github.com/cybertronai/sutro-problems/pull/122) | Earlier reviewer group passed 1/3; two D5 accuracy misses of 28 and 74 predictions. |
| [#126](https://github.com/cybertronai/sutro-problems/pull/126) | Reviewer 1/3 with D4 accuracy misses of 26 and 20. Author records identify different unsupplied source bytes; upstream credit remains unresolved. |
| [#127](https://github.com/cybertronai/sutro-problems/pull/127) | Reviewer 1/3 with D5 accuracy misses of 116 and 57; source attribution remains unresolved. |
| [#133](https://github.com/cybertronai/sutro-problems/pull/133) | Author 3/3 supported; reviewer 2/3. The third complete run missed D2 accuracy by exactly one prediction; no qualifying reviewer-group median. |

The published instructions specify three official runs and their median; they do not expressly require another maintainer batch or permanent rejection after a failed replication. These holds preserve conflicting evidence and unresolved diagnoses. In particular, [#133's complete group](../20261019-pass3/pr133/) remains 2/3 with a one-prediction miss, despite its separate passing author group. A substantive diagnosis or changed-source submission can be reviewed on its merits; no failed unchanged-source result was replaced to manufacture a passing group.

Previous evidence: [October 19 later pass](../20261019-pass3/), [October 18 later pass](../20261018-pass2/). Machine-readable current results: [summary.json](summary.json).
