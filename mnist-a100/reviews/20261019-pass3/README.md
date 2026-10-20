# Sutro pull-request review — October 19, 2026, later pass

This pass merged **#132 (D4), #134 (D3) and #135 (D5)** as new latency records after **9/9 fresh canonical sandboxed Modal A100-80GB passes**. **#133 (D2) remains open**: its third complete reviewer run missed the accuracy gate by one prediction. Its author’s separate three-run group passes and is retained; no reviewer outcome was replaced or retried.

| PR | Band | Reviewer passes | Median ms | Median above-idle mJ | Accuracy margin, labels | Decision |
| - | - | -: | -: | -: | - | - |
| [#132](https://github.com/cybertronai/sutro-problems/pull/132) | D4 | 3/3 | 78.333838 | 12845.449793 | +113, +118, +48 | Merged; new latency record; [evidence](../../submissions/hybrid-tangent-250-d4-20261019/evidence/reviewer-20261019-pass3/) |
| [#133](https://github.com/cybertronai/sutro-problems/pull/133) | D2 | 2/3 | — | — | +108, +85, -1 | Held: run 3 missed by one prediction; [evidence](../../reviews/20261019-pass3/pr133/) |
| [#134](https://github.com/cybertronai/sutro-problems/pull/134) | D3 | 3/3 | 4.558383 | 639.405457 | +216, +147, +268 | Merged; new latency record; [evidence](../../submissions/native-hull-cg-d3-20261019/evidence/reviewer-20261019-pass3/) |
| [#135](https://github.com/cybertronai/sutro-problems/pull/135) | D5 | 3/3 | 492.842637 | 77089.864599 | +102, +223, +30 | Merged; new latency record; [evidence](../../submissions/ladder-width768-d5-20261019/evidence/reviewer-20261019-pass3/) |

All **12 fresh runs / 180 timed calls** are retained, including the failed complete #133 run. The leaderboard medians use the nine passing runs from the three accepted groups; #133's two passing runs do not establish a group median. Every run has 11 MNIST and four foreign-holdout calls and must independently satisfy all canonical gates; pooled accuracy cannot repair a failed run. Optional energy is assessed separately from score. #134 improves latency but uses more measured energy than the prior D3 record; no energy-improvement claim is made.

## Source and protocol review

The exact reviewed heads and source hashes, raw calls, energy windows, source reviews, attribution checks, launch plans and executable saved-record verifiers are linked in each row. The published scorer and stock runner are byte-identical to frozen main `5f40e0e93af6ed5b57d74e81c5121d7cd8af2548`; no learner was repaired during qualification. The first fixed group per new exact source ran with default stock resources; zero replacement runs were launched. Author qualification groups, prior distinct-source groups and warm development comparisons remain separate.

All four submissions by @sjbaebae retain substantial upstream credits. #132 changes Haden’s hybrid training budget and target-refresh schedule. #133 reduces Fisher metric/compression budgets in Haden’s D2 port of sigkillme0’s method. #134 combines that Fisher-PCG family with Haden’s HKNN and a native numerical CG solve. #135 narrows Haden’s kernel-taught Ladder, adjusts its budget and uses native fused Adam, retaining the teacher and guard. Sources credit Haden, sigkillme0, Islam, Laura and other relevant ancestors in their individual reports.

Published instructions specify three official Modal runs and their median; they do not expressly require another maintainer batch or permanent rejection after a failed replication. The #133 hold records conflicting accuracy evidence: author 3/3 passes versus reviewer 2/3, with a one-label miss. A complete changed-source submission or a substantive diagnosis can be reviewed on its merits. No unchanged-source failed run was discarded to manufacture a passing group.

## Older open PRs

The six older PR heads, source/file lists, bodies, discussions, reviews and inline comments are unchanged. [Fresh GitHub metadata](older-pr-recheck.json) preserves this check. Their previous outcomes were retained without rerunning unchanged source.

| PR | Reason it remains open |
| - | - |
| [#110](https://github.com/cybertronai/sutro-problems/pull/110) | Explicitly non-record experiment; repository policy keeps it on a separate branch. Its requested A100 diagnostic was completed previously. |
| [#114](https://github.com/cybertronai/sutro-problems/pull/114) | Earlier reviewer group passed 1/3; two complete runs missed D4 accuracy by 64 and 4 predictions. |
| [#119](https://github.com/cybertronai/sutro-problems/pull/119) | Author 3/3 supported; reviewer 1/3, with two later untimed warmups exceeding 90 seconds after seven saved calls. Code allows 180 seconds first, 90 later, while documentation says 180; the stalled phase remains unidentified. Hold for reproduction diagnosis, not an unambiguous published-rule violation. |
| [#122](https://github.com/cybertronai/sutro-problems/pull/122) | Earlier reviewer group passed 1/3; two D5 accuracy misses of 28 and 74 predictions. |
| [#126](https://github.com/cybertronai/sutro-problems/pull/126) | Reviewer 1/3, D4 accuracy misses of 26 and 20; author records identify different unsupplied source bytes, and upstream credit remains unresolved. It also now has a merge conflict. |
| [#127](https://github.com/cybertronai/sutro-problems/pull/127) | Reviewer 1/3, D5 accuracy misses of 116 and 57; source attribution remains unresolved. It also now has a merge conflict. |

Previous evidence: [October 18 later pass](../20261018-pass2/), [October 19 earlier pass](../20261019/). The current results are machine-readable in [summary.json](summary.json).
