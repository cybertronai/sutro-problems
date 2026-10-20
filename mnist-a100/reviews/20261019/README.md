# Pull-request review — October 19, 2026

Merged Haden Wasserbaech's **#131, #130 and #129**, establishing new D2, D4 and D5 records. All **nine fresh canonical sandboxed Modal A100-80GB runs passed**, completing **135 timed calls** with valid optional above-idle energy measurements. No source was modified and no reviewer outcome was retried, discarded or replaced.

| PR | Band | Reviewer passes | Median ms/call | Median mJ/call | Pooled MNIST error | Evidence |
| - | - | - | -: | -: | -: | - |
| [#131](https://github.com/cybertronai/sutro-problems/pull/131) | D2 | 3/3 | 3.184645 | 207.682235 | 3.087576% | [source, report and raw evidence](../../submissions/fisher-pcg-d2-20261019/README.md) |
| [#130](https://github.com/cybertronai/sutro-problems/pull/130) | D4 | 3/3 | 97.047379 | 14,754.131810 | 2.083333% | [source, report and raw evidence](../../submissions/hybrid-tangent-d4-20261019/README.md) |
| [#129](https://github.com/cybertronai/sutro-problems/pull/129) | D5 | 3/3 | 633.529424 | 107,885.521303 | 1.750000% | [source, report and raw evidence](../../submissions/ladder-k2-d5-20261019/README.md) |

Leaderboard values use these independent reviewer medians. Each of the three runs for each entry independently passed its accuracy, holdout and timing gates; pooled error is descriptive. Each group recorded three distinct GPU UUIDs. Energy is an optional column beside score qualification, and its validity is assessed separately. The learner source, canonical scorer and stock launcher hashes match the frozen review snapshots. [Canonical integrity](canonical-integrity.json) · [Initial measurement plan](measurement-plan.json).

All three exact-source author groups reconstruct separately as 3/3 passes, with documented working-file renames and intact stdout/GPU identifiers. The reports preserve upstream credits and retain the author's original reports. Development-call counts, probability estimates, historical chronology and individual optimization comparisons are author claims outside this review's reproduction scope.

Source review found no remaining static blocker. D5 received a second targeted review of the branchless teacher fallback, all 600 training updates, reset, graph ownership and custom-backward write bounds. D4's buffer-lifetime question was resolved by the strong compiled-bound-method cycle frozen while its owner is still locally rooted; the [lifetime audit](../../submissions/hybrid-tangent-d4-20261019/evidence/reviewer-20261019/object-lifetime-audit.json) contains exact Torch 2.12 source references and a Python 3.13 ownership control. No learner edit or extra GPU diagnostic was used to resolve it.

The six older open PRs are unchanged, with identical source/heads/descriptions and no new comments or reviews resolving their holds: **#114/#122/#126/#127** retain accuracy-reproduction failures; **#119** retains two later warmup failures (90 seconds implemented versus 180 seconds advertised, unresolved phase); **#110** remains explicitly non-record and its requested A100 diagnostic is already complete. Their sources were not rerun. [Detailed current recheck](older-pr-recheck.json).

Published instructions specify three official Modal runs and their median. They do not expressly impose an additional maintainer batch or a permanent ban after a failed batch. Existing holds reflect contradictory observed reproduction evidence, while the advertised/implemented warmup limits remain distinguished.

[Machine-readable results, exact reviewed heads and merge commits](summary.json) · [Final GitHub status](final-github-status.json).

[Independent final publication check](independent-final-publication-check.json) validates parent metrics, exact raw/source copies, evidence digests, verifiers, credits, links and winner markers.
