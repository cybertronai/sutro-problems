# PR #133 — held after a one-prediction accuracy miss

The pinned source at `62b0b1ca6daed88399a14949be59dac24f49cf9b` is retained as [submission.py](submission.py), SHA-256 `65a14f4024d8af73ea3b7c088aeec44591715095b5cf2f76bcaf3a5c1d693a91`. This is @sjbaebae’s budget optimization of Haden’s Fisher-PCG D2; source and README preserve Haden, sigkillme0 and Islam’s credits. Static review found no blocker.

The initial fixed reviewer group passed **2/3** complete stock canonical sandboxed Modal A100-80GB runs. All 45 timed calls completed. D2 accuracy margins were **+108, +85, −1 predictions**. Run 3 returned 106,259/110,000 correct; the gate needs 106,260 (3.400909% error versus 3.40%). There is no qualifying reviewer median; pooled 3.341818% error cannot repair the individual failure. No outcome was retried or replaced.

The author’s separate exact-source three-run group passes and reconstructs to 3.029522 ms / 408.781601 mJ. The hold reflects conflicting accuracy evidence, rather than invalid author arithmetic or a permanent ban on these source bytes. Published instructions specify three official runs and their median and do not expressly require a second maintainer group.

Energy is optional: runs 1/2 have valid measurements and recorded GPU UUIDs, while failed run 3 has no energy block. This group has two recorded energy UUIDs and no qualifying three-run energy median. The hold is the accuracy failure.

[Verification](verification.json), [source review](source-review.json), [decision](review-decision.json), [measurement plan](measurement-plan.json) and all three raw records are retained here. With Python 3.13 from this directory, `python verify_records.py . submission.py --difficulty 2 --function kc_d2 --output /tmp/pr133-verification.json` reconstructs the evidence and exits 1, correctly reproducing the held result without executing the learner. [Original PR](https://github.com/cybertronai/sutro-problems/pull/133).
