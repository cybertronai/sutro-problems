# Independent qualification review

Date: 2026-10-03. Reviewed package: `mnist-a100/submissions/ladder-kernel-tuned-20261003/`.

**Verdict: GO — this exact source and its local package are ready to submit for an established difficulty-3 record using the documented three-run median practice. No unmet canonical qualification gate was found.** This is readiness for submission, not maintainer acceptance or publication.

The earlier independent source review concluded that the implementation was legitimate but lacked exact-source official qualification. That historical audit is preserved unchanged. The three new passing canonical runs close that qualification gap; no source change was required.

## Exact source and canonical provenance

- Packaged `ladder_d3.py:classify` is byte-identical to the previously audited `ladder_d3_encoder_tuned.py`: **20,194 bytes**, SHA-256 **`5052f86e9cdf6086783bd64c90a31fcb86b3d19d1d0f27ef8f21457923c713e1`**.
- `official/manifest.json` embeds these exact source bytes and identifies difficulty 3, three runs and app `ap-C0ANEPJzLHOHSyqOG9ej1t`. Every official record independently matches source hash, file name and function.
- Source gate and Python syntax pass. The only advisory flag is proximity to the 20,480-byte size limit, with 286 bytes remaining.
- Manifest scorer hash `1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064` and runner hash `1180b843cf6c32414774da90c835a8b7298d2db8719989237a793ca1c62a10ab` match local canonical files and `origin/main` at `aa6c51e6ad661089cbf71a39c95c577c216b627f`. Every record has canonical scorer version 1.2.0.
- The inspected launcher reuses canonical `run_modal.remote_score`, whose container declaration is single-use. Its wrapper changes only concurrency to three and container timeout to 600 seconds, retaining canonical method/scoring behavior. Saved results identify three distinct A100-SXM4-80GB GPU UUIDs, supporting separate hardware execution. The JSONs do not contain independent Modal container IDs; container lifecycle is evidenced by the launcher contract and supplied execution provenance rather than a separate platform audit.

Source-legitimacy findings remain applicable: attributed neural Ladder; supplied test features used only for the disclosed unlabelled objective; no test-label access, embedded learned prior, timer manipulation, special-case holdout shortcut or learned-state carryover mechanism found. The earlier CPU test exercised the actual tuned `initialise()` against corrupted weights/Adam/sampler state. New official runs additionally exercise the actual sandboxed GPU path.

## Independent reconstruction of all three runs

I ran both the existing local verifier and the package's `verify_records.py`, and additionally rebuilt facts directly from every raw call and energy window with canonical `mnist.judge` and `mnist.energy_summary`. Results agree exactly with saved ranked and energy values.

| Run | Ranked ms | Energy mJ | MNIST correct / 110000 | Mean error | Worst timed MNIST error | Holdout accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1157.137595 | 151759.932481 | 107336 | 2.421818% | 2.62% | KMNIST 96.1650% |
| 2 | 1156.458740 | 151403.970125 | 107180 | 2.563636% | 2.77% | KMNIST 96.0325% |
| 3 | 1160.505341 | 149399.706870 | 107082 | 2.652727% | 2.81% | Fashion-MNIST 87.5425% |

**Median score: 1157.1375954367898 ms. Median energy: 151403.97012476073 mJ.** Aggregate MNIST error is **2.5460606060606072%**, from 321598/330000 correct over 33 timed MNIST calls.

All three runs return zero, report `sandboxed: true`, have no disqualification problems, and contain exactly 11 MNIST plus four calls from the declared foreign holdout. Every call has 10000 scored examples and finite timing fields below the 60-second timed-call limit.

Canonical gates reconstructed successfully:

- Difficulty-3 mean accuracy: at least 107030/110000 MNIST predictions correct per run. All three exceed this threshold.
- Per-draw MNIST accuracy: at least 9580/10000 correct. The worst timed draw is 2.81% error, below the 4.20% ceiling.
- Foreign holdout accuracy: at least 6000/40000 correct. Both observed holdout types comfortably pass.
- Each dataset's maximum/minimum event times meet the 2×median+2 ms / median÷2−2 ms dispersion gates.
- Per-call and trimmed per-dataset independent-parent-clock integrity gates pass. Parent time, protocol overhead and CUDA events reconstruct the exact official ranked score. In these records, no scorer-clock floor raises the score. Run 1 is ranked by its MNIST mean; runs 2 and 3 are ranked by their holdout means, as required.
- All energy summaries rebuild without problems, including exact reference matmul product, plausible telemetry, idle/context gates, energy-window draw accuracy, energy-window timing consistency and positive net energy after empty-control subtraction.

Telemetry references are **7.586125–7.962779 J/TFLOP at 18.100462–18.349342 TFLOP/s**, within the canonical A100 bands. Every idle window records zero maximum GPU utilization and one CUDA context. Method windows have 18 calls each over approximately 20.95–20.97 seconds. Canonical references, control windows, idle windows and actual method-window measurements are retained, unlike the earlier diagnostic energy summaries.

The distinct GPU UUIDs are `GPU-e43b903c-5679-c491-84f5-1099087057ce`, `GPU-9c5cdfd3-d60c-d719-75c0-86646a012593` and `GPU-348a3f5e-55d4-bff5-903d-6b6c52390660`. Distinct physical boards are useful evidence and an additional verifier check, not a canonical scorer rejection rule. Current upstream also permits explicitly provisional rows after one verification run; this package satisfies the stronger documented three-run median practice.

## Package/report consistency

The package README is present and was reviewed. `official/summary.json`'s per-run values and aggregate score, energy, accuracy, worst draw, improvement percentages and minimum accuracy margin were independently recalculated from raw records and match.

The published-baseline comparisons reproduce **19.826952% lower time** versus 1443.3 ms and **24.970281% lower energy** versus 201792 mJ. The README identifies these as comparisons between separate canonical runs rather than paired experiments. It distinguishes the inherited 900-to-800-step reduction from the launch optimizations and preserves SethTS/recipe attribution.

The README correctly separates official timing/energy from paired development screens, identifies whole-loss gradient comparisons as recorded after timing, discloses fixed-seed/GPU nondeterminism, and states that no submission, acceptance or leaderboard change has occurred. Development snapshots/records are byte-identical to their original audited files. The packaged verifier is a byte-identical copy of the previously inspected verifier and successfully resolves the repository's canonical scorer from its new location.

**Accuracy caveat:** run 3 is only **0.047272727 percentage points** below the mean-error limit: 107082 correct versus the required 107030, a margin of 52 predictions. Passing these three fresh runs does not guarantee future random draws pass. The README explicitly discloses this; it does not constitute an unmet gate in the saved qualification.

The source and original manifest/raw records were hash-checked again after the package documentation was read and remain unchanged. The historical prequalification report and existing evidence were not edited. This review creates only this new follow-up report.

## Limits and final readiness decision

This was an independent local reconstruction/source/package review, not a fourth GPU run or a maintainer-controlled reproduction. I launched no paid work. Saved records and their embedded hashes establish internal evidence consistency; they are not cryptographically authenticated attestations of the remote runtime. Individual import/warmup timings and container IDs are not separately retained, so successful lifecycle/output claims rely on the unchanged canonical runner/scorer and successful complete records. Compute-cost estimates were not audited against settled billing and are not qualification gates.

The exact-source canonical runtime and energy qualification gaps identified in the earlier report are now closed by three new passing runs. **GO for submission of this local established-record package; maintainer reproduction, acceptance and any publication remain separate actions.**
