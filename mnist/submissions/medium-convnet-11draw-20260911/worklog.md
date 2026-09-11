## Execution record

This record summarizes reproducible actions and saved evidence; it is not a
transcript of internal reasoning. Experiment timestamps are UTC on 2026-09-11.

- Changed the medium mean-accuracy target to exactly 98%, requiring
  64,680 / 66,000 correct. Fourteen evaluator tests passed. Pushed the rule
  change to `main` as `8daa351`; its Pages deployment succeeded.
- Froze the learner, 11 dataset seeds, member seeds, source hashes, and input
  protocol at 05:28:50. Independently checked all 11 index sets and all
  22 resized image arrays before GPU training.
- Ran 33 fresh 71-epoch ConvNet fits, using at most two A100 containers. All
  ensemble predictions and diagnostic member outputs were frozen at 05:34:31.
  No test-label vectors had been derived.
- Froze the separate evaluator and evidence at 05:34:56, then opened labels
  for scoring at 05:35:00. Obtained 64,776 / 66,000, or 98.1% ± 0.1 pp across
  the 11 draws. A separate implementation reproduced every count, mean, and
  sample SD. The accuracy requirement passed by 96 correct predictions.
- Reviewed the exact v4 instruction set and memory rules. The frozen native
  learner does not yet have a complete v4 lowering; theoretical metrics remain
  unavailable. The separate feasibility audit records specific limitations.
- Froze the complete-task A100 measurement protocol at 05:38:24, after the
  accuracy gate. It specifies one full warmup and three measured fresh tasks,
  paired idle NVML measurements, unchanged source, and exact output/state checks.
  All trials passed exact checks. Medians were 5.9 × 10⁴ ms and 1.7 × 10⁶ mJ
  idle-adjusted GPU energy. The independent raw-measurement audit passed.
  The final raw measurements and audit are retained with the report.

Commands, run configuration, and all scientific evidence needed for reproduction
are published with the submission. Raw local inputs and execution logs are not
copied to Pages. This conversation export is a snapshot; the final PR and Pages
deployment are verified separately when publication finishes.

- [Accuracy run on Modal](https://modal.com/apps/yaroslavvb/main/ap-glciYzFFhrmErGOoUR4cOG)
- [Complete-task measurement run on Modal](https://modal.com/apps/yaroslavvb/main/ap-Z0OqxSrCwNovdLMdEmTGIK)
