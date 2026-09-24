# Checks that only an A100 can run

Everything in `tests/test_eval.py` and every red-team re-run behind harness
1.1.0 was done on the CPU dry-run path (`MNIST_EVAL_DEVICE=cpu`). The CUDA
branch of `eval.py` -- `torch.cuda.Event`, the 256 MB L2 flush, the two
`synchronize` barriers around it -- has never executed. Run this list once on
one A100 before a band opens, and record the numbers next to each row.
Sections A1-A3 are what to run; sections R1-R3, after the rule, are what the
first pass actually returned.

    python run_modal.py --band <band> --submission <file> --mode <mode> \
        --seed <secret> --output results/gpu-<name>.json

## A0. What harness 1.2.0 changed

Draws are now released as `(N, 60)` linear features, `z = Q W (x - mu)`, not
`(N, 1, 9, 9)` pixels, so every accuracy recorded below was measured on a
release nobody competes on any more. The staged tensors are 2.4 MB per draw
instead of 3.2 MB, which moves `stage_calibration` and therefore the staging
tripwire; re-measure it. On the CPU path, one 10,000-example draw's map costs
about 60 ms to fit and apply, outside every timed window. Re-run A1 and record
both the new accuracies and the new calibration numbers; the accuracy
expectations in A1 are pixel-release numbers. Measured on CPU, mean over five
10,000/10,000 draws of case seed 101, pixels -> release: `ncm_baseline`
80.1% -> 85.0%, `pca_qda` 95.5% -> 90.3%, `mlp512` 96.4% -> 95.7%, `cg_pair`
98.0% -> 95.8%; on two released Fashion hold-out draws, `ncm_baseline` 77.6%,
`pca_qda` 75.4%, `mlp512` 86.5%, `cg_pair` 86.5%. `submissions/mlp512.py`
needed an input-scale fix for the release before any of that held (it collapsed
onto one class on 2 of 11 released draws without it), so treat its recorded
verdicts as superseded.

## A1. The harness still works

| # | Run | Expect |
| --- | --- | --- |
| 1.1 | `submissions/ncm_baseline.py`, 12% band, `--mode test` | pass on the test-mode slack, ~80% |
| 1.2 | `submissions/ncm_baseline.py`, 12% band, `--mode leaderboard` | fail the accuracy gate (it is the control that must not pass) |
| 1.3 | `submissions/pca_qda.py`, 5% band, `--mode leaderboard` | pass; note `mean`, `wall_mean`, `ipc_calibration`, `stage_calibration` |
| 1.4 | `submissions/cg_pair.py`, 2% band, `--mode leaderboard` | pass; upstream reports ~260 ms per call |
| 1.5 | `submissions/mlp512.py`, 5% band, `--mode leaderboard` | pass. It failed the hold-out at 47.8% in R1 on the pixel release; on the 1.2.0 release, after the input-scale fix in that file, CPU hold-out draws give 86.2-86.9%, so this row is expected to flip -- record what it actually does |
| 1.6 | any, `--mode profile` | a base64 profiler table with CUDA rows |

Record `ipc_calibration` and `stage_calibration` from 1.3. They set how much
slack the timing gate has: the parent-clock bound is
`device >= 0.25 * (parent_round_trip - ipc) - 0.5 ms`, so if `ipc` on the host
is larger than a few milliseconds the bound goes slack for fast entries, and
`PARENT_CLOCK_FRACTION` or the transport needs revisiting. Also confirm the
first timed call is not an outlier against the dispersion gate
(`worst <= 2 * median + 2 ms`) once CUDA graphs and the allocator are involved;
if a legitimate entry trips it, raise `dispersion_x10` in `bands.json`.

## A2. Red-team files that must still be caught

Since harness 1.1.1 the module-level inertness check (rule 2) rejects any file
that runs code at import time, which is 17 of the 29 files in `redteam/` --
including every `timer-patch*` variant. They are now caught *before* the
submission is imported, so re-running them exercises the static check, not the
runtime defence they were written for. To re-test a runtime defence, move the
entry's module-level setup into `custom_kernel` first; the verdicts recorded in
R2 were taken before the static check existed.

| # | File | Mode | Expected verdict |
| --- | --- | --- | --- |
| 2.1 | `redteam/timer-patch.py` | benchmark, 12% | passes, but at the honest time (~= `pca_qda`); the `torch.cuda.Event` subclass must not change `benchmark.0.mean` |
| 2.2 | `redteam/timer-patch_evalglobals.py` | benchmark, 12% | same: honest time, and the L2 flush must still run |
| 2.3 | `redteam/timer-patch_calibration.py` | benchmark, 12% | same |
| 2.4 | `redteam/timer-patch_sidestream.py` | benchmark, 12% | **the one that is genuinely untested.** It forces `enable_timing=False`, returns a constant 0.01 ms from `elapsed_time`, replaces `torch.cuda.synchronize` with a default-stream-only synchronize, and issues the learner on a private side stream whose join is deferred. The harness captures `Event` and `synchronize` before import, so the expected result is a `timing implausible` failure (exit 112). If it passes, stop and fix before opening a band. |
| 2.4b | `redteam/timer-patch_cellsteal.py` | benchmark, 12% | fail with "timing implausible ... under 0.25 of the ... the parent measured". On CUDA the same trick can also steal the captured `Event` and `synchronize` cells; confirm the parent bound still fires. |
| 2.5 | `redteam/lazy-output.py` | benchmark, 5% | honest time; the hijack of `Tensor.copy_` must not reach staging |
| 2.6 | `redteam/lazy-output_postevent.py` | test, 12% | fail (exit 112) |
| 2.7 | `redteam/replay-cache.py` | leaderboard, 12% | fail on the per-draw floor; confirm the int64 pixel hash is bit-stable across calls on CUDA after the 256 MB flush, i.e. that the exploit fails because the table is empty, not because hashing is noisy |
| 2.8 | `redteam/replay-cache_weights.py` | leaderboard, 5% | fail on the dispersion gate |
| 2.9 | `redteam/threshold-gaming.py` | benchmark, 5% | honest time (no warm-up training budget) |
| A2.10 | `redteam/threshold-gaming_margin11.py` | benchmark, 12%, `--case bench_draws=11` | fail on the per-draw floor |
| 2.11 | `redteam/threshold-gaming_clock.py` | benchmark, 12%, `REDTEAM_CLOCK_SCALE=0.001` | honest time |
| 2.12 | `redteam/label-lookup_fallback.py` | test, 2% | rejected before import: over the 20,480-byte cap |
| 2.13 | `redteam/label-lookup_netprobe.py` | test, 12% | with `block_network=True`, **all five** channels blocked, including `subprocess curl`. On CPU here four of five were blocked; the fifth is what the container block is for. |
| 2.14 | `redteam/io-and-process.py` | leaderboard, 12% | fail. Then read `/tmp/redteam-io-recon.json` and check `proc_environ_readable` and `parent_POPCORN_SEED`: on Linux `/proc` exists, and this is the first real test of the re-exec that keeps the secret seed out of the parent's environment. `idx_gz_found` must be empty on a hosted container. |
| 2.15 | `redteam/threshold-gaming_edge.py` with `REDTEAM_EDGE=fd` | test, 12% | fail with "did not return within N s" after the watchdog kills the process group -- not a mode timeout |

## A3. Things to measure while the GPU is warm

* Per-call duration spread of each reference submission over 11 draws, to set
  `draw_slack_bp` (currently 150 bp) and `dispersion_x10` (currently 20) from
  data rather than from CPU dry runs.
* Fashion-MNIST hold-out accuracy of each reference submission over several
  draws, to set `holdout_min_bp` (currently 3000 bp = 70%) with a real margin.
  If a legitimate entry lands near 70%, lower it; the number is a policy choice.
  Measured on the A100 in R1/R3, on the pixel release: pca-qda 78.0%, cg-pair
  88.2%, mlp512 47.8%. On CPU released hold-out draws those become 75.4%, 86.5%
  and 86.5%, with the `ncm_baseline` control at 77.6% -- i.e. the ordering the
  floor relies on may be gone. Also record how separable a hold-out call is from
  a ranked one *from the release alone* (A0 aside: the class-mean scatter
  spectrum is rotation-invariant and separates the pools), since that is what
  decides whether the floor binds an adaptive entry at all.
* Whether `torch==2.12.0` resolves against the `nvidia/cuda:13.3.0` base in
  `run_modal.py`; `TORCH_PIN` is the single place to step back.
* The cost of the 256 MB L2 flush per call, so it can be quoted in the README.

---

# Results of the first A100 pass (2026-09-22, harness 1.1.0)

25 containers, one at a time, Modal `gpu="A100"`, secret seed 20260922 unless
noted. Every JSON is in `results/gpu-*.json`. Environment reported by the
harness: `torch 2.12.0+cu130`, `CUDA 13.0`, numpy 2.5.3, Python 3.13.0,
`Linux-4.19.0-gvisor-x86_64`, capability 8.0. `TORCH_PIN` resolved against the
`nvidia/cuda:13.3.0-devel-ubuntu24.04` base with no change; image build is ~3 s
after the first layer cache.

**`gpu="A100"` is not one board.** 23 containers were `A100-SXM4-40GB` and 2
were `A100-SXM4-80GB`. The same pca-qda computation ran 4.311 ms on the 80 GB
board against 4.504-4.547 ms on the 40 GB board: 4.5% faster, about 20x the
run-to-run spread. A ranked board must pin the variant or record it per entry.

## R1. The harness still works

| # | Run | Result |
| --- | --- | --- |
| 1.1 | ncm, 12%, test | **pass** after the `probe_call` fix; 79.84%, 1.090 ms. Before the fix it failed "timing implausible on call 0" (see DESIGN.md). |
| 1.2 | ncm, 12%, leaderboard | **fail**, exit 112: `draw 0 scored 7965 of 10000, below the per-draw floor 8650`. The per-draw floor fires before the aggregate gate. |
| 1.3 | pca-qda, 5%, leaderboard | **pass**. mean 4.5475 ms, std 9.6 us (0.21%), best 4.5343, worst 4.5670, 13 runs. wall_mean 6.2739, child 5.0422, ipc_calibration 1.3785, stage_calibration 731.8. accuracy 95.35%, hold-out 78.01%, holdout_ratio 1.00. |
| 1.4 | cg-pair, 2%, leaderboard | **fail on accuracy**: 29351/30000 = 97.837%, needs 29400. 269.980 ms/call. See finding below. |
| 1.4b | cg-pair, 3%, leaderboard | **pass**. mean 270.177 ms, std 99 us (0.037%), 97.93% over 11 draws, hold-out 88.23%. |
| 1.5 | mlp512, 5%, leaderboard | **fail on the hold-out**: 9558/20000 Fashion, per-draw [8548, 1010]. MNIST accuracy 96.42% passes. 26.533 s/call. See finding below. |
| 1.6 | pca-qda, profile | **pass**; real sm80 rows (`cutlass_80_simt_sgemm`, `gemvNSP_kernel`, `memcpy32_post`). |

`ipc_calibration` after the fix: 0.747-1.436 ms across all runs, i.e. the
parent-clock bound has under 1.5 ms of slack, not the ~10 ms it had before.
Dispersion of legitimate entries is far inside the 2.0x gate: pca-qda
worst/median 1.004, cg-pair 1.001, mlp512 1.024. The first timed call is not an
outlier once the flush buffer is allocated during calibration.

Repeatability, pca-qda 5% leaderboard, three secret seeds:

| seed | mean ms | within-run std | accuracy | hold-out |
| --- | --- | --- | --- | --- |
| 20260922 | 4.5475 | 0.0096 | 95.350% | 78.01% |
| 20260923 | 4.5349 | 0.0158 | 95.481% | 77.82% |
| 777 | 4.5043 | 0.0104 | 95.677% | 77.40% |

Spread across containers 0.043 ms = 0.95% of the mean; within a run 0.2-0.35%.
Accuracy varies 0.33 points across seeds, hold-out 0.6 points.

## R2. Red-team files on real CUDA

| # | File | Mode | Verdict on the A100 |
| --- | --- | --- | --- |
| 2.1 | `timer-patch.py` | bench 12% | **neutralized**: 4.311 ms, the honest time (80 GB board). accuracy 95.17%, identical to honest. |
| 2.2 | `timer-patch_evalglobals.py` | bench 12% | **neutralized**: 4.523 ms. |
| 2.3 | `timer-patch_calibration.py` | bench 12% | **neutralized**: 4.545 ms. |
| 2.4 | `timer-patch_sidestream.py` | bench 12% | **neutralized**: 23.967 ms, i.e. 5.3x *slower* than honest. `CUDA_EVENT_DISABLE_TIMING`, the constant 0.01 ms `elapsed_time` and the default-stream-only `synchronize` all failed to reach the evaluator, which holds its own `Event` and `synchronize`. The side stream is drained by the captured barrier and the exploit pays for it. This was the one check no CPU run could make. |
| 2.4b | `timer-patch_cellsteal.py` | bench 12% | **caught**, exit 112: `device time 4.523 ms exceeds the child's wall clock 0.000 ms`. On CUDA the `gc` walk reaches the captured `perf_counter` cell but not the `Event`, so it is the upper bound that fires rather than the parent bound it trips on CPU. |
| 2.5 | `lazy-output.py` | bench 5% | **neutralized**: 24.683 ms against 4.5 ms honest. |
| 2.6 | `lazy-output_postevent.py` | test 12% | **caught**, exit 112: `device time 0.132 ms is less than half the child's wall clock 0.608 ms`. The sub-millisecond relative form of the bound added in 1.1.0 is what fires. |
| 2.7 | `replay-cache.py` | lb 12% | **caught**, exit 112: `draw 0 scored 7965 of 10000, below the per-draw floor 8650`. It fails on call 0, before any cache entry exists, so this run does **not** settle whether the int64 pixel hash is bit-stable across calls on CUDA -- still open. |
| 2.8 | `replay-cache_weights.py` | lb 5% | **caught**: `call durations differ by 20.7x (worst 247.557 ms, median 11.978 ms)`. The GPU separates the trained call from the replayed ones far more sharply than the CPU did (7.7x). |
| 2.9 | `threshold-gaming.py` | bench 5% | **caught**: `call durations differ by 10.2x (worst 354.899 ms, median 34.931 ms)`. On CPU this one merely came out honest; on the GPU the dispersion gate rejects it outright. |
| 2.12 | `label-lookup_fallback.py` | test 2% | **caught pre-import**: `submission.py is 330012 bytes, over the 20480-byte limit; entries may not carry embedded data (rule 2)`. |
| 2.13 | `label-lookup_netprobe.py` | test 12% | **all five channels blocked**, including `subprocess curl` (`returned non-zero exit status 6`, name resolution). `block_network=True` closes the one channel the in-process guard cannot. |
| 2.14 | `io-and-process_gpurecon.py` | lb 12% | **caught** (per-draw floor), and the recon is the headline: `proc_environ_readable: true` -- `/proc/<ppid>/environ` *is* readable under gVisor -- yet `parent_POPCORN_SEED: null` and `parent_MNIST_POOL_CACHE: null`. The re-exec scrub holds on real Linux. `idx_gz_found: []`, `cases_files: {}`, cwd is only `submission.py` + `task.py`, and `eval` / `utils` / `mnist_data` are all `ModuleNotFoundError`. Only `parent_POPCORN_FD: "12"` leaks, which is a number, not a writable fd for the child. |
| 2.15 | `threshold-gaming_edge_gpufd.py` | test 12% | **caught**: `the submission did not return from 'untimed' within 150 s`, exit 112, 157 s wall -- the watchdog, not the 300 s mode timeout. It forged `check: pass` on 14 inherited fds (`[8, 10, 18, 20, 21, 40, 43, 45, 47, 49, 51, 53, 55, 57]`) and none of them was the result pipe; the final result carries `check = fail` and nothing else. |

Nothing escaped. Two exploits that merely came out honest on the CPU are now
rejected outright on the GPU (2.8 at 20.7x, 2.9 at 10.2x), because real device
timing separates a trained call from a replayed one much more cleanly.

## R3. Measurements and the two things that need a decision

**The 256 MB L2 flush is cheap; its first allocation is not.** Steady-state,
flush + two barriers + the pipe round trip together are the whole
`ipc_calibration`, 0.75-1.44 ms, of which the pipe is most. The first
allocation of the buffer cost ~10 ms and used to land on timed call 0; it now
lands in calibration. Quote "about 1 ms of protocol overhead per call, outside
the timed window" in the README.

**`draw_slack_bp` and `dispersion_x10` can stay where they are.** Per-call
duration spread over 13 calls is 0.21-0.35% for pca-qda, 0.037% for cg-pair,
1.7% for mlp512 -- nowhere near the 2.0x gate. Per-draw accuracy scatter over
11 draws is +-0.35 points for pca-qda and +-0.3 for mlp512; cg-pair's worst
draw is 0.4 points under its mean. The 150 bp floor has roughly 4 sigma of
margin for every reference entry.

### Finding A: the 2% band sits exactly on cg-pair's capability

cg-pair scores 97.837% over the 3 benchmark draws and 97.932% over the 11
ranked draws at seed 20260922, against a 98.000% requirement. It misses. The
GPU per-draw counts, `[9784, 9812, 9755, ...]`, reproduce the CPU dry run's
`[9785, 9811, 9754]` to within one prediction, so this is the entry's real
accuracy, not a porting artefact -- and upstream's self-reported 98.12% is
about 0.2 points optimistic against this protocol. Because every test half of a
run is drawn from the same 30k universe, the 11 draws are correlated: a run has
roughly the precision of one 30k test set, not of 110k, so re-running at
another seed will move the number by a few tenths of a point rather than
averaging the miss away. Either the 2% band opens knowing its only known
qualifier does not clear it, or the band moves.

### Finding B: the hold-out disqualifies an honest learner

mlp512 passes MNIST at 96.42% over 11 draws and then fails the hold-out at
47.79%, per-draw `[8548, 1010]`. Repeated at seed 777 with `draws=3`: 96.67% on
MNIST, hold-out 47.37%, per-draw `[8463, 1010]`. One Fashion draw is classified
at 84-85%, the other collapses to 1010 correct out of 10000 -- chance, and the
same 1010 at both seeds, i.e. the network deterministically collapses onto a
single class. That is divergence, not memorization: mlp512 is fixed-schedule
SGD (lr 0.1, squared error, `x*4-0.5`) tuned on sparse 9x9 digits, and
Fashion's much denser images blow the step size up. So the hold-out, as
specified, rejects a genuine learner whose hyper-parameters are tuned for the
task it was given -- which is most strong entries. D6 assumed a learner
generalizes to foreign data of the same shape; a learner with a hand-tuned
constant learning rate does not have to. Options, none of them free: score the
hold-out on the *better* of its draws rather than in aggregate; normalize the
hold-out inputs to MNIST's pixel statistics; drop `holdout_min_bp` to around
1500 bp (a lookup-table entry scores ~10% there, so the separation survives); or
require the hold-out only to beat chance by a wide margin instead of reaching a
fixed accuracy. The pca-qda (78%) and cg-pair (88%) numbers show the gate is
comfortable for closed-form learners; it is iterative entries that are exposed.

**Update for harness 1.2.0 (not yet re-run on a GPU).** Every number in this
finding is from the pixel release. Under the linear release the hold-out draw is
whitened by its own Fashion statistics, and `submissions/mlp512.py` needed its
input scale fixed for the release anyway (unscaled, it collapsed onto one class
on 2 of 11 released MNIST draws at N = 10,000). With that fix, CPU measurements
at 10,000/10,000 give mlp512 86.2% and 86.9% on two released Fashion hold-out
draws -- it clears the floor -- while `pca_qda` drops to 74.9-75.9% and
`ncm_baseline`, which meets no band, reaches 77.2-78.0%. The gate may now be
closer to rejecting nobody and admitting the control than to rejecting an
honest learner. Re-run R1 before acting on any of the options above.

**Done 2026-09-24 (release, benchmark mode):** `submissions/cg_pair.py` on `mnist-medium-5pct` with `release_dims: 60`, 3 draws, A100-SXM4-40GB: 28,775/30,000 correct (95.92%), durations 253.29 / 253.16 / 253.46 ms, `check: pass`, `results/gpu-release-cg-pair-5pct-benchmark.json`. Leaderboard mode with the Fashion hold-outs is still to be run on the GPU.
