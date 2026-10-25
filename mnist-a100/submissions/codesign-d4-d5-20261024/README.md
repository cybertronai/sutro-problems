# D4/D5 GPU co-design: author canonical results and ongoing D5 research

The two candidates below each passed all three initial author canonical A100 runs. Independent reviewer qualification is outstanding. Neither run group measured energy (`--no-energy`), so no energy claim or leaderboard edit is included.

| Candidate | Canonical times (ms) | Median (ms) | Pooled MNIST error | Minimum accuracy margin |
| --- | --- | --- | --- | --- |
| D4, width128 Fisher teacher and split4 rolled student kernel | 22.571541 / 25.705793 / 22.805198 | 22.805198 | 2.216667% | 55 labels |
| D5, width512 label-top-down Ladder, 420 updates | 339.562220 / 340.038926 / 345.591545 | 340.038926 | 1.811515% | 31 labels |

Each group contains all three initial outcomes, 11 MNIST and four foreign-holdout calls per run, through the unchanged sandbox and scorer. D4 used two PCIe and one SXM A100; D5 used three SXM A100s. Host/draw differences mean the complete cross-group gain cannot be attributed to one kernel change.

## Implementations and provenance

D4 builds on spartanhaden's Fisher-PCG/tangent-student work, with a narrower student, one teacher and split4 scheduling. The final row-kernel change replaces static unrolling with rolled loops. Nsight measurements reduced registers from 255 to 208 with no spills; achieved occupancy remained about 10.7%. The isolated scheduling gain was approximately 1.8% in paired warm full-call tests. This does not explain the entire improvement against the listed 38.7ms reviewed result.

D5 builds on the label-top-down Ladder lineage of Rasmus, SethTS, spartanhaden, sigkillme0, islamborghini, likelytobelaura and Vincent/Bengio. It uses a width512 student, 420 native fused Adam updates, per-call fitting and noise generation, and hidden-space local-hull correction. Credits remain in each standalone source. The listed reviewed D5 best was 406.9ms when this bundle was prepared; this author's 340.04ms group awaits independent review.

Source SHA256 and sizes are pinned in `manifest.json`. The original source basenames are retained so saved records can be matched without changing bytes.

## Verify and reproduce

From this directory, run `python verify_records.py` in the benchmark Python environment. It validates source limits and hashes, frozen scorer/runner hashes, all three initial record sets, call counts, independent judge verdicts and medians. It does not perform new GPU runs.

From `mnist-a100`, run the unchanged runner with the appropriate source/function:

```sh
python run_modal.py submissions/codesign-d4-d5-20261024/d4/fisher-triton-width128-teacher1-split4-rolled-d4-20261022.py:hybrid_d4 --difficulty 4 --runs 3 --no-energy --json results/d4-review
python run_modal.py submissions/codesign-d4-d5-20261024/d5/ladder-label-topdown-width512-s420-d5-20261022.py:ladder_kernel --difficulty 5 --runs 3 --no-energy --json results/d5-review
```

## Experimental D5 work: not a qualifying 244ms result

`research/` retains both initial three-run integrated-decoder groups. Each passed only 2/3. The parallel-phase2 variant's passing scores were 240.500910 and 240.706130ms, but its failed run had 1.951818% error. There is no qualifying group median. These files must not be used to claim the 244ms goal or a leaderboard record.

The parallel-phase2 combinator backward replaces serial partial-gradient accumulation with masked row reductions, preserving all 17 parameter gradients plus three batch-normalization sums. Complete gradient gates passed on L4 and A100, with about 3ms saved per full call in paired development measurements. This changes floating-point reduction order.

A newer 20439-byte source combines this reduction with transpose-initialized decoders and near-identity denoisers, inspired by sigkillme0's PR127 (commit 19d2a3faba642814aee9c12ed1a118a927b796d2). Original RNG draws are retained; all learned state is reset per call. Fresh development cohorts 12/13/14 had errors 1.806364%, 1.609091%, 1.783636% and warm full-call medians 238.031, 237.789, 238.543ms. Its initial canonical batch is running at bundle creation; development timings do not qualify it.

An initial reference/packed comparison differed in trained parameters and predictions; both versions separately reset reproducibly. A follow-up matched initial parameters, trained parameters and predictions bit for bit. Both results are retained; the earlier cross-process discrepancy remains unexplained. EMA .98 gave mixed accuracy and .995 consistently hurt, adding about 1–2ms; EMA is excluded from the selected candidates.

Readable transform modules and raw evidence are included for review. Research helpers are development code, while each included standalone candidate is self-contained. No scorer, runner, difficulty threshold or leaderboard file is changed.
