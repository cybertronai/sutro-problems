# D4/D5 GPU co-design: author canonical results and ongoing D5 research

## Independent reviewer reproduction — October 25, 2026

The exact D4 and latest width-480 D5 sources passed their first fixed independent groups, **3/3 fresh canonical sandboxed Modal A100-80GB runs each**. Scorer, runner, source bytes, CPU request and environment defaults were unchanged. All six returned records and launch logs are retained; no maintainer-launched failed-run replacement or tuning was performed. Infrastructure events, if any, are disclosed in the logs. The historical width-512 D5 remains author evidence and was not entered as a new reviewer record.

### D4: three fresh reviewer runs passed

Median ranked time: **25.264627 ms/call**. Median above-idle board energy: **3,346.874018 mJ/call**. Pooled MNIST error: **2.186970%**, descriptive; each complete run separately passed all canonical accuracy, holdout and timing gates.

| Reviewer run | MNIST error | Margin, labels | Ranked ms | Above-idle mJ |
| - | -: | -: | -: | -: |
| 1 | 2.215455% | +93 | 25.264627 | 3346.874018 |
| 2 | 2.163636% | +150 | 26.246056 | 3464.944016 |
| 3 | 2.181818% | +130 | 23.721816 | 2930.405801 |

Exact source SHA-256: `a2c32468ecb4e02957ef218b3d39419a711b55baaca6be98fb1645a6909bdc6a`. [Retained records and reconstruction](evidence/reviewer-20261025/d4/).

Saved-record check from this directory: `python evidence/reviewer-20261025/verify_records.py evidence/reviewer-20261025/d4 d4/fisher-triton-width128-teacher1-split4-rolled-d4-20261022.py --difficulty 4 --function hybrid_d4 --output /tmp/pr140-d4-verification.json`.

### D5: three fresh reviewer runs passed

Median ranked time: **273.889134 ms/call**. Median above-idle board energy: **40,390.436435 mJ/call**. Pooled MNIST error: **1.686970%**, descriptive; each complete run separately passed all canonical accuracy, holdout and timing gates.

| Reviewer run | MNIST error | Margin, labels | Ranked ms | Above-idle mJ |
| - | -: | -: | -: | -: |
| 1 | 1.636364% | +290 | 240.515970 | 36880.254882 |
| 2 | 1.710909% | +208 | 273.889134 | 41203.652224 |
| 3 | 1.713636% | +205 | 276.102590 | 40390.436435 |

Exact source SHA-256: `a245b278c9f425ec951942caf13e4a14b7123717a428bc9862b13491f42fdc29`. [Retained records and reconstruction](evidence/reviewer-20261025/d5_244/).

Saved-record check from this directory: `python evidence/reviewer-20261025/verify_records.py evidence/reviewer-20261025/d5_244 d5_244/ladder-w480-s315-population-tb20-dec50-lr720-cut50-ph2-initboth-d5-20261024.py --difficulty 5 --function ladder_kernel --output /tmp/pr140-d5_244-verification.json`.

Energy is optional and separate from score qualification. It measures GPU-board energy above idle per call, excluding initial compilation and host/system energy. PCIe/SXM and draw variation prevent attributing cross-group speedups to a single change. Author profiling and numerical-parity claims retain the limited scope of their supplied evidence, detailed in the source reviews; they are not new reviewer measurements. [Independent audit](evidence/reviewer-20261025/actual-final-independent-audit.json), [source and provenance reviews](evidence/reviewer-20261025/), and the unchanged canonical reconstruction code are retained.

## Original author report (as submitted)

Pending-review statements below describe the submission date; the dated reviewer section above records accepted status.


The three candidates below each passed all three initial author canonical A100 runs. Independent reviewer qualification is outstanding. None of the run groups measured energy (`--no-energy`), so no energy claim or leaderboard edit is included.

| Candidate | Canonical times (ms) | Median (ms) | Pooled MNIST error | Minimum accuracy margin |
| --- | --- | --- | --- | --- |
| D4, width128 Fisher teacher and split4 rolled student kernel | 22.571541 / 25.705793 / 22.805198 | 22.805198 | 2.216667% | 55 labels |
| D5 historical, width512 label-top-down Ladder, 420 updates | 339.562220 / 340.038926 / 345.591545 | 340.038926 | 1.811515% | 31 labels |
| D5, width480 initialized Ladder, 315 updates | 241.059039 / 275.808218 / 243.411033 | **243.411033** | **1.758485%** | **94 labels** |

Each group contains all three initial outcomes, 11 MNIST and four foreign-holdout calls per run, through the unchanged sandbox and scorer. D4 used two PCIe and one SXM A100. The historical D5 used three SXM A100s; the new 243.411ms group used two PCIe and one SXM. One Function in the new group was preempted and automatically restarted by Modal with the same input. The full launch log and infrastructure history are retained; the unchanged runner returned three completed logical run records. No completed failing result was replaced. Host/draw differences mean the complete cross-group gain cannot be attributed to one kernel change.

## Implementations and provenance

D4 builds on spartanhaden's Fisher-PCG/tangent-student work, with a narrower student, one teacher and split4 scheduling. The final row-kernel change replaces static unrolling with rolled loops. Nsight measurements reduced registers from 255 to 208 with no spills; achieved occupancy remained about 10.7%. The isolated scheduling gain was approximately 1.8% in paired warm full-call tests. This does not explain the entire improvement against the listed 38.7ms reviewed result.

D5 builds on the label-top-down Ladder lineage of Rasmus, SethTS, spartanhaden, sigkillme0, islamborghini, likelytobelaura and Vincent/Bengio. It uses a width512 student, 420 native fused Adam updates, per-call fitting and noise generation, and hidden-space local-hull correction. Credits remain in each standalone source. The listed reviewed D5 best was 406.9ms when this bundle was prepared; this author's new 243.411ms group awaits independent review.

Source SHA256 and sizes are pinned in `manifest.json`. The original source basenames are retained so saved records can be matched without changing bytes.

## Verify and reproduce

From this directory, run `python verify_records.py` in the benchmark Python environment. It validates source limits and hashes, frozen scorer/runner hashes, all three initial record sets, call counts, independent judge verdicts and medians. It does not perform new GPU runs.

From `mnist-a100`, run the unchanged runner with the appropriate source/function:

```sh
python run_modal.py submissions/codesign-d4-d5-20261024/d4/fisher-triton-width128-teacher1-split4-rolled-d4-20261022.py:hybrid_d4 --difficulty 4 --runs 3 --no-energy --json results/d4-review
python run_modal.py submissions/codesign-d4-d5-20261024/d5/ladder-label-topdown-width512-s420-d5-20261022.py:ladder_kernel --difficulty 5 --runs 3 --no-energy --json results/d5-review
```

## D5 244ms result and retained experimental history

`d5_244/` is the new qualifying author candidate: 20439 bytes, SHA256 a245b278c9f425ec951942caf13e4a14b7123717a428bc9862b13491f42fdc29. It combines width480, 315 updates, training-population prediction statistics, ghost BN, encoded sampling, parallel combinator-gradient reductions, transpose-initialized decoders, near-identity denoisers, teacher cutoff and a conditional-decoder posterior plus hidden hull. All learned state is reset per call. The ≤244ms median goal is met under the author canonical protocol, with every completed run passing accuracy; independent review and energy measurement remain outstanding. The SXM score of 275.808ms remains included and shows that the target is a median, not a per-device upper bound.

The redundant 285-update canonical batch and 265/275-update development batch were stopped after this goal was reached. Their incomplete status and logs are retained under `research/`; no qualification is claimed for them. The completed 285/295/315 development comparison is also retained.

`research/` retains both initial three-run integrated-decoder groups. Each passed only 2/3. The parallel-phase2 variant's passing scores were 240.500910 and 240.706130ms, but its failed run had 1.951818% error. There is no qualifying group median. These files must not be used to claim the 244ms goal or a leaderboard record.

The parallel-phase2 combinator backward replaces serial partial-gradient accumulation with masked row reductions, preserving all 17 parameter gradients plus three batch-normalization sums. Complete gradient gates passed on L4 and A100, with about 3ms saved per full call in paired development measurements. This changes floating-point reduction order.

A newer 20439-byte source combines this reduction with transpose-initialized decoders and near-identity denoisers, inspired by sigkillme0's PR127 (commit 19d2a3faba642814aee9c12ed1a118a927b796d2). Original RNG draws are retained; all learned state is reset per call. Fresh development cohorts 12/13/14 had errors 1.806364%, 1.609091%, 1.783636% and warm full-call medians 238.031, 237.789, 238.543ms. Its initial logical canonical batch has now completed 3/3; the standalone source is promoted in `d5_244/`. The development timings themselves do not qualify it.

An initial reference/packed comparison differed in trained parameters and predictions; both versions separately reset reproducibly. A follow-up matched initial parameters, trained parameters and predictions bit for bit. Both results are retained; the earlier cross-process discrepancy remains unexplained. EMA .98 gave mixed accuracy and .995 consistently hurt, adding about 1–2ms; EMA is excluded from the selected candidates.

Readable transform modules and raw evidence are included for review. Research helpers are development code, while each included standalone candidate is self-contained. No scorer, runner, difficulty threshold or leaderboard file is changed.

To reproduce the new result, run from `mnist-a100`:

```sh
python run_modal.py submissions/codesign-d4-d5-20261024/d5_244/ladder-w480-s315-population-tb20-dec50-lr720-cut50-ph2-initboth-d5-20261024.py:ladder_kernel --difficulty 5 --runs 3 --no-energy --json results/d5-244-review
```
