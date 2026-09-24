# MNIST-medium time leaderboard: design report

Harness `sutro-mnist-medium-time/1.2.0`, 2026-09-23. Reader: Yaroslav.
Section 10 lists the changes an independent audit of 1.1.0 produced and
section 11 the linear release added in 1.2.0; every number measured on hardware
below was taken with 1.1.0 and is unaffected by them, except where those
sections say otherwise. In particular every accuracy in this report was measured
on the pixel release; the 1.2.0 release changes what a learner sees.
Source of every number below: `gpumode/results/gpu-*.json` (one A100 on Modal,
25 sequential containers, 1923 s = 32.1 min of GPU wall time) and
`gpumode/results/cpu-*.json` (CPU dry runs). Code: `gpumode/eval.py`,
`gpumode/utils.py`, `gpumode/task.py`, `gpumode/run_modal.py`.

## 1. Bottom line

The harness works. It builds, runs, ranks, and rejects. Five red-team agents
broke version 1.0.0 in five independent ways; 1.1.0 closes all five, and on real
CUDA nothing escaped, including the one exploit class (side stream plus
disabled-timing events) that no CPU run could test.

One blocker was found and fixed on the first A100 run: the calibration probe
skipped the CUDA preamble, so every submission faster than about 3 ms was
disqualified as "timing implausible". See section 7.

### Measured entries, leaderboard mode, secret seed 20260922

| Submission | Band | Verdict | Mean ms | Std ms | Accuracy, 11 draws | Hold-out | Device | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `submissions/pca_qda.py` | 5% | pass | 4.548 | 0.0096 | 95.35% (104,885/110,000, needs 104,500) | 78.01% | A100-SXM4-40GB | `results/gpu-02-pca-qda-5pct-leaderboard.json` |
| `submissions/cg_pair.py` | 3% | pass | 270.177 | 0.0993 | 97.93% (107,725/110,000, needs 106,700) | 88.23% | A100-SXM4-40GB | `results/gpu-03b-cg-pair-3pct-leaderboard.json` |
| `submissions/cg_pair.py` | 2% | **fail**, accuracy gate | 269.980 | 0.195 | 97.837% (29,351/30,000, needs 29,400) | not reached | A100-SXM4-40GB | `results/gpu-03-cg-pair-2pct-leaderboard.json` |
| `submissions/mlp512.py` | 5% | **fail**, hold-out | 26,533 | 439 | 96.42% (106,057/110,000, passes) | 47.79% (needs 70%) | A100-SXM4-40GB | `results/gpu-04-mlp512-5pct-leaderboard.json` |
| `submissions/ncm_baseline.py` | 12% | **fail**, per-draw floor (control) | 1.09 (test mode) | n/a | 79.84% on the test draw | not reached | A100-SXM4-40GB | `results/gpu-05-ncm-12pct-leaderboard-control.json` |

The two failures are informative, not broken. `cg_pair` genuinely misses 2%
under this protocol; `mlp512` genuinely fails the learning check. Both are
open decisions in section 6.

### Repeatability, `pca_qda` on the 5% band, three secret seeds, three containers

| Secret seed | Mean ms | Within-run std | Accuracy | Hold-out | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 20260922 | 4.5475 | 0.21% | 95.350% | 78.01% | `results/gpu-02-pca-qda-5pct-leaderboard.json` |
| 20260923 | 4.5349 | 0.35% | 95.481% | 77.82% | `results/gpu-06a-pca-qda-5pct-seed20260923.json` |
| 777 | 4.5043 | 0.23% | 95.677% | 77.40% | `results/gpu-06b-pca-qda-5pct-seed777.json` |

Cross-container spread 0.043 ms = 0.95% of the mean. Within-run spread
0.21 to 0.35%. Two entries whose times differ by more than about 1% can be
ranked; below that they are a tie. **Except** across board variants: the same
computation ran 4.311 ms on an A100-SXM4-80GB
(`results/gpu-07-03-timer-patch-12pct.json`) against 4.504 to 4.547 ms on five
separate 40 GB containers, i.e. 4.5% faster, roughly 20x the within-run spread.
Modal's `gpu="A100"` is not one board. Section 6.

### Red-team summary, on CUDA

25 GPU runs, 12 of them adversarial. Nothing escaped. Two gates are strictly
stronger on the GPU than the CPU dry run suggested (dispersion at 20.7x rather
than 7.7x; `threshold-gaming.py` rejected outright rather than merely honest).
Full table in section 3.

## 2. Design choices

Each entry: Decision / Why / Precedent / Rejected.

### D1. The ranked value is mean CUDA-event time per complete training-and-prediction call

**Decision.** `benchmark.i.mean` in nanoseconds over `draws` fresh secret draws
(11 ranked, 3 in benchmark mode), plus `median`, `std`, `err`, `best`, `worst`,
`durations`, `accuracy`, `correct`, `required`, `wall_mean` (parent clock),
`child_wall_mean`, `stage_mean`, `ipc_calibration`, `stage_calibration`,
`loop_wall`, `holdout_accuracy`. `ranking_by: last`, one benchmark case per
band. Every timed call is also an accuracy draw: time and accuracy come from
the same 11 calls.

**Why.** The Sutro numbers before this harness were self-reported and off by
45x and 59x in two cases, and energy and time were measured on draw 0 only
while accuracy used 11 draws. Ranking on the same calls that are scored removes
both classes of error at once.

**Precedent.** KernelBot's `benchmark.i.*` key set and `ranking_by` semantics
(`/tmp/kb/src/libkernelbot/run_eval.py`), so the board renders with no changes
on Mark's side.

**Rejected.** Best-of-N (rewards a lucky call and an unstable learner);
wall-clock only (kernelbot#295: the $100K AMD 2025 challenge ranked on CPU
wall-clock with a hot L2 for its whole run); energy (no NVML on the KernelBot
path, and Seth's PR #87 machinery needs SIGSTOP and root-ish access that a
hosted runner will not give).

### D2. Per-call protocol: fresh draw, staged untimed, L2 flushed, CUDA events, full synchronize, parent-side scoring

**Decision.** `stage` (separate round trip, untimed), `synchronize`,
256 MB L2 flush, `synchronize`, `start.record()`, `custom_kernel`,
`end.record()`, `synchronize`, then validate (plain `torch.Tensor`, shape
`(Q,)`, integer dtype, values in 0..9), copy to host, score in the parent.

**Why.** The 2025 AMD challenge's timing loop reused inputs and left the L2 hot,
which is worth an order of magnitude on a small tensor problem
(kernelbot#295, reference-kernels#142). 256 MB rather than the AMD harness's
`clear_l2_cache_large` 64 GiB: the A100's L2 is 40 MB.

**Precedent.** `/tmp/rk/problems/amd_202602/eval.py`, the 2026 rewrite of that
loop.

**Rejected.** `torch.cuda.synchronize` on the default stream only (a side stream
escapes it; see D4 and `redteam/timer-patch_sidestream.py`); invariant-only
correctness checks (reference-kernels#148, which admit a cache).

### D3. Exactly one untimed warm-up call, on a foreign dataset

**Decision.** One unmeasured call before the timed loop. As of 1.1.0 that call
is a Fashion-MNIST draw of identical shape.

**Why.** Compilation, autotuning and CUDA graph capture should be free, as in
modded-nanogpt. But the red team turned the free call into a free *training*
budget: `threshold-gaming.py` fitted on the warm-up draw and replayed, for a
2.71x discount, and `replay-cache_weights.py` got 9.4x the same way. A foreign
warm-up keeps the compile free and makes the fitted model useless.

**Precedent.** modded-nanogpt's untimed compile; reference-kernels#142 on warm-up
input reuse arming replay paths.

**Rejected.** No warm-up at all (the first call then measures `torch.compile`,
which is not what anyone wants to rank); several warm-ups (more free budget).

### D4. Trust boundary: spawned child, parent holds everything, four clocks

**Decision.** The submission is imported only in a spawned child, on an explicit
`load` command, after the protocol has been calibrated in a process that has no
submission in it. The child captures `time.perf_counter`, `torch.cuda.Event`,
`torch.cuda.synchronize`, the flush and every tensor method used for staging and
readback into locals **before** the submission exists. Four clocks bracket each
call: CUDA events in the child, the child's `perf_counter`, the parent's
`perf_counter` around the call round trip, and the parent's around the staging
round trip. `utils.timing_plausible` bounds the device clock from above by the
child's wall clock and from below by both the child's wall clock (relative below
1 ms) and `0.25 * (parent_round_trip - calibrated_overhead) - 0.5 ms`.

**Why.** In-process timer monkeypatching put a fake 0.000 at #1 on a live board
(reference-kernels#162). Version 1.0.0 of this harness had the same hole in a
subtler form: its three checks compared two attacker-controlled numbers to each
other, so scaling both by 1e-4 passed. That was worth 8,550x
(`redteam/timer-patch.py`, before-fix evidence in `redteam/evidence/`).

**Precedent.** reference-kernels#162; Seth's PR #87 trust boundary (parent holds
the pool, labels, seeds and permutations, child never sees test labels), which
this keeps.

**Rejected.** Trusting CUDA events alone (`CUDA_EVENT_DISABLE_TIMING` is a
KernelGuard rule family for a reason); a parent that never touches CUDA (it
imports torch to resolve the device, which costs ~300 MB RSS and is worth it).

### D5. The labels never touch disk while a submission is alive

**Decision.** The parent downloads MNIST and Fashion-MNIST (md5-verified in
`mnist_data.SOURCES`), loads them into RAM and deletes the files, then spawns
the child. On Modal the files are baked into the image and consumed
(`MNIST_POOL_CONSUME=1`). `MNIST_POOL_CACHE` exists for offline dry runs only.
The evaluator re-execs itself once with `POPCORN_SEED` and `MNIST_POOL_CACHE`
removed from its environment, receiving them on an inherited pipe.

**Why.** `os.environ.pop` does not rewrite `/proc/<pid>/environ`. The io lens
proved the child can read the parent's original environ; with the secret seed
plus `combine` plus the public case seed, every draw, permutation and hold-out
of the run is reconstructible. Confirmed on the container:
`proc_environ_readable=true` under gVisor, and `parent_POPCORN_SEED=null`
(`results/gpu-07-10-io-and-process-12pct.json`).

**Precedent.** The Sutro-specific hole: medium draws from the public 60k MNIST
*train* split, so the labels are published and a disk copy is a free win.

**Rejected.** Relying on `os.environ.pop` (does not work); leaving the cache env
var set on a hosted run (an operator convenience that hands the pool over).

### D6. A learning check on foreign data, timed and ranked

**Decision.** `holdout_draws` (2) Fashion-MNIST calls of identical shape at
positions only the evaluator knows, interleaved with the ranked calls, **timed
and ranked like every other call**, scored in aggregate against
`holdout_min_bp` (3000 bp = 70%).

**Why.** MLPerf's open-division rule: the implementation must not encode
information about the dataset's content or a successful model's state. A
memorizer cannot answer an unseen dataset. Making the hold-out ranked removes
the free compute slot the red team used: 1.0.0's untimed hold-out let an entry
keep a slow honest learner for unknown draws at zero ranking cost.

**Precedent.** MLPerf 2.4 learning policy; reference-kernels#148 on
invariant-only checks.

**Rejected.** An untimed hold-out (a free slot, exploited); a 50% floor (nearest
class mean scores 66.5% on Fashion 9x9 and cleared it). Measured on the A100,
on the 1.1.x **pixel** release: `pca_qda` 78.0%, `cg_pair` 88.2%, `mlp512`
**47.8%**. The last one is an honest learner that fails; see section 6. Under
the 1.2.0 release the hold-out draw is whitened by its own Fashion statistics
and these numbers move (section 11, open decision 2); they have not been
re-measured on a GPU.

**What it does not do.** A hold-out call is cheap to *recognize* from the
release alone -- the eigenvalues of the released class-mean scatter are
invariant under the secret rotation and separate the two pools cleanly -- so the
floor binds a non-adaptive memorizer only. Section 11, residual risk 2.

### D7. Every threshold is a case field, and case fields come from one JSON file

**Decision.** `size`, `train`, `test`, `error_bp`, `draws`, `bench_draws`,
`seed`, `holdout`, `holdout_draws`, `holdout_min_bp`, `max_call_ms`,
`warmup_max_call_ms`, `draw_slack_bp`, `dispersion_x10`, `max_source_bytes`,
`max_literal_bytes`. All documented in `task.py:TestSpec`, all defaulted in
`bands.json`, all reaching KernelBot only through generated `task.yml` case
lines. `make_bands.py` regenerates every band folder plus `sutro.yaml`.

**Why.** You said the thresholds will move. A band change must not be a code
change, and a hosted competition cannot be edited mid-flight
(kernelbot#295 again: the harness could not be fixed once the $100K challenge
was live).

**Precedent.** reference-kernels `task.yml` `tests:` / `benchmarks:` case lines.

**Rejected.** Constants in `eval.py`; a band per branch.

### D8. Four modes with KernelBot's exit codes

**Decision.** `test` (1 draw, format plus a loose gate at `error_bp + 1000`),
`benchmark` (`bench_draws` = 3, no hold-out), `leaderboard` (full `draws` plus
hold-out), `profile` (base64 `torch.profiler` table). Exit 0 / 112 fail / 111 no
`POPCORN_FD` / 113 bad cases file. Timeouts 300 / 600 / 1200 s.

**Why.** KernelBot runs test then benchmark then leaderboard as three fresh
processes and stops at the first failure; matching that exactly means Mark
changes nothing.

**Precedent.** `/tmp/kb/src/libkernelbot/run_eval.py`, `run_pytorch_script`.

**Rejected.** A single mode (participants then cannot smoke-test cheaply);
the 180 s default timeouts (too short: `mlp512`'s leaderboard step took 382 s).

### D9. A CPU dry-run path that runs the identical protocol

**Decision.** `MNIST_EVAL_DEVICE=cpu` or no CUDA gives the same code path with
`perf_counter` in place of events and no L2 flush. `run_modal.py --local`
exercises the whole KernelBot pipeline without Modal.

**Why.** Everything in this project was developed and red-teamed on a Mac with
no GPU. 28 unit tests and 40-odd exploit runs happened before a single A100
minute was spent.

**Precedent.** None in GPU MODE; this is a local requirement.

**Rejected.** GPU-only development. Note the cost of the choice: the one blocker
of the GPU stage was precisely a CUDA path that the CPU dry run treats as a
no-op (section 7).

### D10. Versions in the result keys

**Decision.** `system.harness`, `system.torch`, `system.cuda`, `system.device`,
`system.device_count`, `system.capability`, `system.numpy`, `system.python`,
`system.platform`, `system.driver`.

**Why.** reference-kernels#143: participants cannot see the evaluator version,
so a silent change invalidates records nobody can audit. Rule 7 of the README
depends on this key existing. `system.driver` is read through
`torch.cuda.driver_version()` with `torch._C._cuda_getDriverVersion()` as the
fallback, after `torch.cuda.init()`; if both fail the key carries the exception
instead of a bare `unknown`.

**Precedent.** #143, and #164 / #140 / #23 on buggy references becoming the spec.

**Rejected.** Nothing, except that `system.driver` logs `unknown`: torch exposes
no driver string. Cosmetic gap against D10, listed in section 7.

## 3. Red-team results

Five lenses, 35 exploit files in `redteam/`, all re-run behind 1.1.0 and then on
the A100. "Before" is harness 1.0.0.

| Lens | Exploit | Before (1.0.0) | After, on the A100 | Residual risk |
| --- | --- | --- | --- | --- |
| timer-patch | `redteam/timer-patch.py` | 0.008 ms ranked, 8,550x, pass | 4.311 ms, honest (`gpu-07-03-timer-patch-12pct.json`) | a 2-4x lie still passes |
| timer-patch | `redteam/timer-patch_evalglobals.py` | 0.011 ms, pass | 4.523 ms, honest | same |
| timer-patch | `redteam/timer-patch_calibration.py` | 8,710 ns, pass | 4.545 ms, honest | same |
| timer-patch | `redteam/timer-patch_sidestream.py` | inert on CPU, untested | **defeated**: 23.967 ms, 5.3x *slower* than honest, because the captured barrier drains its side stream | none found |
| timer-patch | `redteam/timer-patch_cellsteal.py` | n/a (written to test the fix) | caught: "device time 4.523 ms exceeds the child's wall clock 0.000 ms" | catches by the upper bound on CUDA, the lower bound on CPU |
| lazy-output | `redteam/lazy-output.py` (`Tensor.copy_` staging hijack) | 0.0022 ms, 32,000x, pass | 24.683 ms, 5.4x slower than honest | half the work can still legally sit past the end event |
| lazy-output | `redteam/lazy-output_postevent.py` | caught | caught: "0.132 ms is less than half the child's wall clock 0.608 ms" (the sub-ms relative form added in 1.1.0; the old 0.5 ms slack would have let it through) | none |
| lazy-output | `redteam/lazy-output_subclass.py` | caught | caught (plain-tensor check) | none |
| replay-cache | `redteam/replay-cache.py` | 12%, 5% and 2% bands passed with a 79% payload | caught: "draw 0 scored 7965 of 10000, below the per-draw floor 8650" | fails before any cache entry exists, so CUDA hash bit-stability is still unconfirmed |
| replay-cache | `redteam/replay-cache_weights.py` | 9.4x discount, pass | caught: "call durations differ by 20.7x (worst 247.557 ms, median 11.978 ms)"; on CPU the same exploit only reached 7.7x | gate is a heuristic; a flat-profile cheat with a cheap fallback needs D3's foreign warm-up to be sound |
| label-lookup | `redteam/label-lookup_fallback.py` | 2% band pass at 100.0000% accuracy, 517x | caught before the child is spawned: "submission.py is 330012 bytes, over the 20480-byte limit" | a <20 KB quantized net still scores ~98% on the public pool |
| label-lookup | `redteam/label-lookup_probe.py` | 2% band pass at 100%, hold-out 80.2% (higher than the honest entry) | same pre-import rejection | the size cap is the only thing stopping it; a smaller net is not stopped |
| label-lookup | `redteam/label-lookup_netprobe.py` | 3 of 5 egress channels reached the real MNIST label file during a passing run | **all five blocked**: urllib, `socket`, `_socket`, `importlib.reload(socket)`, and `subprocess curl` (exit 6, name resolution) | the last one is closed by Modal `block_network=True`, not by the harness |
| io-and-process | `redteam/io-and-process_gpurecon.py` | 12% band pass at 100% by finding MNIST on disk | caught; recon confirms `proc_environ_readable=true` yet `parent_POPCORN_SEED=null`, `idx_gz_found=[]`, `eval`/`utils`/`mnist_data` all `ModuleNotFoundError`, cwd is `[submission.py, task.py]` | not a sandbox: same uid, writable filesystem |
| io-and-process | `redteam/io-and-process_fdwrite.py` | caught (fd already dead) | caught | none |
| threshold-gaming | `redteam/threshold-gaming.py` (untimed warm-up as a training budget) | 2.71x discount, pass | caught: "call durations differ by 10.2x (worst 354.899 ms, median 34.931 ms)"; on CPU it merely came out honest | none found |
| threshold-gaming | `redteam/threshold-gaming_margin11.py`, `_margin.py` | 1.73x by running a good learner on 7 of 11 draws | caught by the per-draw floor (CPU; not re-run on GPU, `--case` env forwarding) | floor slack of 150 bp is a policy choice |
| threshold-gaming | `redteam/threshold-gaming_edge_gpufd.py` (fd scribbling) | **wedged the evaluator for 300 s with no result at all** | caught in 157 s: "the submission did not return from 'untimed' within 150 s"; 14 forged `check: pass` writes on inherited fds, none of them the result pipe | 143 s of margin under the 300 s test timeout, and only because `warmup_max_call_ms` was lowered to 120 s |
| threshold-gaming | `redteam/threshold-gaming_forge.py` | caught | caught (CPU) | none |

The full before-state evidence is in `redteam/evidence/` and
`redteam/logs/`; the post-fix CPU re-runs are `results/after-fix-*.json` and the
GPU verdicts are `results/gpu-07-*.json`.

## 4. What is parameterized, and how to change a threshold

Everything a band can vary lives in **`bands.json`** and nowhere else.
`defaults{}` applies to every band; any key can be overridden per band.

```bash
$EDITOR bands.json
python make_bands.py          # rewrites mnist-medium-*/task.yml and sutro.yaml
python make_bands.py --check  # CI: fails if anything is stale
```

`make_bands.py` also regenerates `bands.md`, the band table inside `README.md`
(spliced between the `GENERATED` and `END GENERATED` markers) and one
`<band>/submission.py` per band carrying that band's `#!POPCORN leaderboard`
line, so `--check` covers all of them and a downloaded template cannot post to
the wrong board. A unit test asserts the table and `bands.json` agree.

**To add the 15% entry band:** append to `bands[]`

```json
{ "name": "mnist-medium-15pct", "error_bp": 1500, "label": "15%" }
```

**To add the 1.6% top band:**

```json
{ "name": "mnist-medium-1p6pct", "error_bp": 160, "label": "1.6%" }
```

then `python make_bands.py`. A new folder `mnist-medium-1p6pct/` appears with a
`task.yml` and a `README.md`, and `sutro.yaml` gains the problem entry. Nothing
in `eval.py` changes. Note for 1.6%: the best measured entry here is 97.93%, so
that band currently has no qualifier at all (section 6).

**To change the gates rather than the bands:** `draw_slack_bp` (per-draw floor
under the aggregate rule, 150 bp), `dispersion_x10` (ten times the allowed
worst/median ratio, 20), `holdout_min_bp` (3000), `holdout_draws` (2),
`max_call_ms` (60000), `warmup_max_call_ms` (120000), `max_source_bytes`
(20480), `max_literal_bytes` (4096), `draws` (11), `bench_draws` (3). Same
edit-and-regenerate loop.

## 5. Hosting: what Mark needs

**Files to copy** into `reference-kernels/problems/sutro_mnist/` (the
`directory:` fields in `sutro.yaml` already say `sutro_mnist/<band>`):

| From `gpumode/` | Role |
| --- | --- |
| `eval.py`, `utils.py`, `task.py`, `mnist_data.py`, `reference.py`, `submission.py` | the shared problem files each `task.yml` lists |
| `mnist-medium-*/task.yml` | one problem per band, generated |
| `mnist-medium-*/README.md` | per-band blurb, generated |
| `mnist-medium-*/submission.py` | the per-band starter template, generated (its `#!POPCORN leaderboard` line names its own band) |
| `sutro.yaml` | the competition file: deadline `2026-12-31 23:59`, five problems, `gpus: [A100]` each |
| `README.md`, `assets/mnist-medium-task.png` | what participants read |
| `bands.json`, `make_bands.py` | so a band can be moved without editing code |

`submissions/`, `redteam/`, `results/`, `tests/`, `run_modal.py` and
`DESIGN.md` are ours, not the hosted problem, but `redteam/` is worth handing
over as a regression suite.

**GPU string.** `A100`, exactly as `consts.ModalGPU.A100`. See the board-variant
problem in section 6 before accepting that as final.

**Timeouts** (already in each `task.yml`): `test_timeout: 300`,
`benchmark_timeout: 600`, `ranked_timeout: 1200`. The 180 s defaults are not
enough: `mlp512`'s test step alone took 72 s and its leaderboard step 382 s.

**A100-minutes per submission**, measured end to end including container start:

| Entry class | Measured | Source |
| --- | ---: | --- |
| fast entry (4.5 ms/call), full test + benchmark + leaderboard | 49 s | `gpu-02` |
| mid entry (270 ms/call), same | 42 s | `gpu-03b` |
| slow entry (26.5 s/call), same | 587 s | `gpu-04` |
| a rejected entry | 19 to 39 s | `gpu-05`, `gpu-07-*` |
| a hanging entry, killed by the watchdog | 169 s | `gpu-07-11` |

Budget about one A100-minute per ordinary ranked submission and ten for the
slowest band-legal entry. The whole 25-run validation session was 32.1 minutes.

**Artifact sizes.** The result channel is about 1.5 KB of `key: value` lines per
mode, well inside the 64 KiB pipe buffer that KernelBot only drains after
`eval.py` exits. Do not add raw record dumps without switching to a
concurrently drained pipe. The `profile` mode report is a base64 profiler table,
about 6 KB (`results/gpu-08-pca-qda-profile.json`).

**Image.** `nvidia/cuda:13.3.0-devel-ubuntu24.04` + `add_python="3.13"` +
`torch==2.12.0` + `numpy~=2.3`, which resolved as `torch 2.12.0+cu130` with no
change needed. Build is about 3 s warm. `block_network=True` is required: it is
the only thing that closes `subprocess curl`.

**What this harness cannot give him.**

* **Energy.** No NVML, no SIGSTOP idle subtraction, nothing from Seth's PR #87
  survived. Time only.
* **Per-board normalization.** There is no correction factor between the 40 GB
  and 80 GB A100. The harness records `system.device`; it does not rank across
  variants, and it cannot.
* **Secret-seed rotation.** KernelBot's `secret_seed` is a leaderboard column
  defaulted at creation, so README rule 6 (rerun the top three on a fresh seed)
  has to happen outside the hosted service.
* **Static screening.** KernelBot runs its own KernelGuard precheck
  (`enforce_submission_precheck`) before the evaluator, so replay, hardcoded
  shapes and trivial-work screening is hosting behaviour we inherit rather than
  a gap. This harness adds only the size caps and the module-level inertness
  check; a human read of the top of each band is still worth it.

## 6. Open decisions for Yaroslav

1. **40 GB vs 80 GB A100.** The same computation is 4.5% faster on the 80 GB
   board, about 5x the cross-container noise and 20x the within-run noise. As it
   stands a leaderboard ranks entries partly by which board they landed on.
   Options: pin `A100-40GB` in `sutro.yaml` (Modal accepts the variant strings,
   but KernelBot's `ModalGPU.A100` is the bare one, so this needs Mark);
   or record `system.device` and refuse to rank across variants; or accept a
   4.5% noise floor. Worth raising with Mark directly, since it is a property
   of the KernelBot runner, not of this harness.
2. **The hold-out floor rejects an honest learner.** `mlp512` scores 96.4% on
   MNIST over 11 draws and then fails at 47.79% on Fashion, with per-draw
   `[8548, 1010]` at seed 20260922 and `[8463, 1010]` at seed 777. The same
   1010 both times is chance for a 10k draw: the network deterministically
   diverges, because its hand-tuned constant learning rate (0.1, squared error,
   `x*4-0.5`) is tuned for sparse 9x9 digits and Fashion's denser images blow
   the step size up. D6 assumed a real learner generalizes to foreign data of
   the same shape; a learner with hand-tuned constants does not have to, and
   the strong entries are exactly the ones with hand-tuned constants. Choices:
   score the hold-out on its *best* draw rather than in aggregate; normalize
   hold-out inputs to MNIST's pixel statistics; drop `holdout_min_bp` to about
   1500 (a lookup-table entry still scores ~10%, so separation survives); or
   require only a wide margin over chance. Measured floors to calibrate
   against: `pca_qda` 77.4 to 78.0%, `cg_pair` 88.2%, `mlp512` 47.8%, nearest
   class mean 66.5% (CPU). **All of those are pixel-release numbers, and the
   1.2.0 release moves them**, because the hold-out draw is whitened by its own
   Fashion statistics before the entry sees it: on CPU, two released Fashion
   hold-out draws of 10,000/10,000 give `pca_qda` 74.9-75.9%, `ncm_baseline`
   77.2-78.0% and `mlp512` 86.2-86.9% (the last one only after its input scale
   was fixed for the release; see `submissions/mlp512.py`). So the specific
   failure this decision is about may already be gone, while the margin for
   `pca_qda` is now about 5 points. Re-measure on an A100 before touching
   `holdout_min_bp`.
3. **The 2% band has no qualifier.** `cg_pair` scores 97.837% over the
   benchmark draws and 97.932% over 11 ranked draws against a 98.000%
   requirement, and the GPU per-draw counts reproduce the CPU dry run to within
   one prediction, so upstream's self-reported 98.12% is about 0.2 points
   optimistic under this protocol. Because every test half of a run comes from
   the same 30k universe, a run has roughly the precision of one 30k test set,
   not of 110k, so re-seeding moves the number by tenths of a point rather than
   averaging the miss away. Either open 2% knowing its only known entry misses,
   or move the band. The 1.6% band under discussion is further still.
4. **Mean or median as the ranked value.** Currently mean, matching KernelBot.
   Measured within-run std is 0.2 to 0.4% for every honest entry, so it makes
   almost no difference today; median is more robust if a band ever admits an
   entry with a bimodal call.
5. **Should compile and graph capture stay untimed?** D3 says yes, and it costs
   a foreign-dataset warm-up call to keep safe. The alternative, timing the
   first call, would rank `torch.compile` rather than the learner and would make
   CUDA-graph entries look terrible. Worth a deliberate answer since the answer
   is what the board actually measures.
6. **Test mode on the public seed.** `test_seed: 101` and `benchmark_seed: 202`
   are public and combined with the secret, so nobody can precompute a draw, but
   a participant can rehearse against a fixed public case repeatedly. That is
   intended (cheap smoke tests) but it also means `benchmark` mode, which runs
   no hold-out, is a rehearsal surface for a memorizer. Consider whether
   `benchmark` should also run one hold-out call.
7. **Deadline.** `bands.json` says `2026-12-31 23:59`, a placeholder.
8. **Timeout headroom at the slow end.** `max_call_ms` is 60 s while
   `ranked_timeout` is 1200 s, and `mlp512` at 26.5 s/call already uses 382 s of
   it. An entry 3x slower times out before the per-call limit ever fires. Either
   lower `max_call_ms` to about 40 s or raise `ranked_timeout`, before opening a
   band to slow entries.

## 7. Findings and surprises from the GPU runs

**The blocker, fixed.** The first GPU run failed on the harness's own honest
baseline: `timing implausible on call 0: device time 1.004 ms is under 0.25 of
the 11.203 ms the parent measured for this call (round trip 12.507 ms minus
1.304 ms of calibrated overhead)`. `probe_call`, documented as "a timed call
with no submission in it", skipped the `synchronize / flush_l2 / synchronize`
preamble that opens every timed call. That sequence is a no-op on the CPU dry
run, so the omission was invisible to all 28 unit tests and 40-odd CPU exploit
runs. On an A100 it costs about 10 ms, almost all of it the *first* allocation
of the 256 MB flush buffer, which then landed on timed call 0. `ipc_ms`
under-measured the protocol by that amount, the budget was inflated by the same
amount, and every submission faster than about 3 ms was disqualified. Since a
leaderboard run starts with `test`, no fast entry could have been scored at all.
`probe_call` now runs the same preamble. The timed window and the ranked value
are untouched; 28/28 unit tests still pass. This was the only scoring-path edit
made during the GPU stage.

**`gpu="A100"` is two different boards.** 23 of 25 containers were
A100-SXM4-40GB (1555 GB/s HBM2), 2 were A100-SXM4-80GB (2039 GB/s HBM2e). The
identical pca-qda computation: 4.311 ms on the 80 GB board against 4.504,
4.523, 4.535, 4.545 and 4.547 ms on five separate 40 GB containers.

**Two gates are stronger on the GPU than on CPU.** `replay-cache_weights` was
caught at 20.7x dispersion (7.7x on CPU); `threshold-gaming` was rejected
outright at 10.2x, where on CPU it had merely come out honest. Real device
timing separates a trained call from a replayed one much more sharply.

**The genuinely untested exploit was fully neutralized.**
`redteam/timer-patch_sidestream.py` forces `enable_timing=False`, returns a
constant 0.01 ms from `elapsed_time`, replaces `torch.cuda.synchronize` with a
default-stream-only synchronize, and issues the learner on a private side
stream with a deferred join. None of it reached the evaluator, which holds its
own pre-import `Event` and `synchronize`. It ranked at 23.967 ms, 5.3x *slower*
than its honest 4.5 ms, because the captured barrier drains its side stream.
The pre-import capture is what does the work.

**Two verdicts differ from what `tests/GPU_CHECKS.md` predicted, in detail
though not in outcome.** (a) `timer-patch_cellsteal` is caught by the upper
bound ("device time 4.523 ms exceeds the child's wall clock 0.000 ms") rather
than the parent-clock bound that catches it on CPU: on CUDA the `gc` walk
reaches the captured `perf_counter` cell but not the captured `Event`, so the
child's wall clock reads zero while the device clock stays honest. (b)
`ncm_baseline`'s leaderboard control fails on the per-draw floor rather than
the aggregate accuracy gate, because the floor is evaluated per call and fires
at draw 0. Both are correct rejections; the expected strings in the checklist
should be updated.

**The Linux-only defences hold.** `/proc/<ppid>/environ` *is* readable under
gVisor, so the threat was real, and yet `parent_POPCORN_SEED=null` and
`parent_MNIST_POOL_CACHE=null`: the re-exec scrub works on real Linux, which
was the single biggest untested assumption. Also confirmed: `idx_gz_found=[]`,
`cases_files={}`, `cwd_listing=['__pycache__','submission.py','task.py']`, and
`eval`/`utils`/`mnist_data` all `ModuleNotFoundError`. Only
`parent_POPCORN_FD='12'` leaks, which is a number and not a writable fd.

**All five egress channels are blocked on the real container**, against four of
five on CPU. `subprocess curl` returns exit 6, could not resolve host, i.e.
Modal's `block_network=True` closes the channel the in-process guard cannot.

**Capacity waits: none.** Zero capacity, queue or resource errors across all 25
runs; the exponential-backoff path in `run_on_modal` never fired. Every
container started within a few seconds of `app.run()`.

**Things against expectation.** `pca_qda` came in at 4.55 ms, not the 3 to 4 ms
the brief expected. `cg_pair` came in at 270 ms and 97.93%, against ~260 ms and
98.12% upstream.

**What did not run.** `mlp512` on the 8% band (it fails on the hold-out, not on
accuracy, so the result is identical and it costs ~10 min of GPU time);
`redteam/threshold-gaming_clock.py` on the GPU, because it is driven by an
environment variable and `run_modal.py`'s `remote_evaluate` used to pass
`env_extra={}` (`--case` overrides *were* forwarded all along, so
`threshold-gaming_margin11.py --case bench_draws=11` could have run; both were
verified on CPU). `run_modal.py` now has an `--env K=V` passthrough. The
replay-cache GPU sub-question is still open: that exploit fails at draw 0,
before any cache entry exists, so this run does **not** confirm that the int64
pixel hash is bit-stable across calls on CUDA after the 256 MB flush
(`tests/GPU_CHECKS.md` 2.7 asks for that, and it needs a purpose-built probe).

**Cosmetic.** `system.driver` logs `unknown`: torch exposes no driver string,
so that D10 field is empty. And `tests/test_web.py` errors with
`ModuleNotFoundError: fastapi`, pre-existing, from a parallel agent's
`gpumode/web/`; `tests/test_eval.py` is 28/28.

## Harness changes from red-teaming (2026-09-22, harness 1.0.0 -> 1.1.0)

Five agents attacked harness 1.0.0; their submissions are in `redteam/` and the
escapes they found are listed with the fix that closed each one. Every fix was
re-run against the exploit that motivated it, on the CPU dry-run path.

**Clocks (lens: timer-patch, threshold-gaming escape 3, lazy-output)**

1. The child now captures `time.perf_counter`, `torch.cuda.Event`,
   `torch.cuda.synchronize`, the L2 flush and every tensor method used to stage
   inputs and read outputs into locals *before* the submission is imported, and
   uses only those. Rebinding them -- in the stdlib, in `torch`, or by rewriting
   the evaluator module's `__globals__` -- no longer reaches anything the
   harness measures. Closes `timer-patch.py`, `timer-patch_evalglobals.py`,
   `timer-patch_calibration.py`, `threshold-gaming_clock.py` and the
   `Tensor.copy_` staging hijack in `lazy-output.py`.
2. The submission is imported on a `load` command rather than at process start,
   so the evaluator can calibrate the protocol's own cost (`probe` and `stage`
   round trips) in a process that has no submission in it. This is what makes a
   poisoned calibration impossible (`timer-patch_calibration.py` slowed exactly
   the probe's signature to make the budget negative).
3. `utils.timing_plausible` gained a lower bound against the parent's clock:
   `device_ms >= 0.25 * (parent_round_trip - calibrated_overhead) - 0.5`. The
   staging round trip, which carries 6.5 MB, is a separate message with its own
   ceiling, so the transfer cost does not loosen the bound on the call. The old
   `0.5 * child_wall - 0.5` bound became relative below a millisecond.
4. `benchmark.i.wall_mean` is now the parent's clock, as D1 always said it was;
   the child's is logged separately as `child_wall_mean`, alongside
   `stage_mean`, `ipc_calibration`, `stage_calibration`, `loop_wall` and the
   per-call `durations` list.

**Replay and carry-over (lens: replay-cache, threshold-gaming escape 1)**

5. The pool is cut in half once per run from the secret seed, and training
   halves are drawn from one half and test halves from the other. Every draw
   used to re-split the same 60,000 rows, so 1 - (5/6)^k of the test images of
   draw k had already arrived, labelled, in an earlier training half: a hash
   table reached 98% with a 79% payload. Closes `replay-cache.py` and
   `replay-cache_persist.py`.
6. The warm-up call is now a Fashion-MNIST draw of identical shape. Compilation,
   autotuning and graph capture stay free; a model fitted during the warm-up is
   about the wrong dataset. Closes the free training budget used by
   `threshold-gaming.py` (2.7x) and `replay-cache_weights.py` (9.4x).
7. A dispersion gate fails a run whose slowest ranked call exceeds
   `dispersion_x10/10` times the median plus 2 ms. Training on call 1 and
   reusing it for calls 2..11 is exactly this shape.

**Accuracy gaming (lens: threshold-gaming escape 2)**

8. A per-draw floor (`draw_slack_bp`, 150 bp) sits under the aggregate rule.
   Mixing an accurate learner on 7 of 11 draws with a cheap one on the rest used
   to clear the mean by 435 of 110,000; now the cheap draws fail on their own.
   150 bp rather than the 100 bp the red team proposed: the measured per-draw
   scatter of `cg_pair` at the 2% band is about +/- 30 of 10,000, and 100 bp
   left under two standard deviations of headroom.

**Memorization (lens: label-lookup, io-and-process)**

9. `submission.py` is capped at `max_source_bytes` (20,480) with no literal over
   `max_literal_bytes` (4,096), checked by `ast` before the child is spawned.
   Rule 2 was an honour rule; it is now machine-checked. Closes all four
   table-carrying and pretrained-net variants of `label-lookup*.py` (141-330 KB).
10. Hold-out calls are `holdout_draws` (2) Fashion-MNIST draws at secret
    positions, timed and ranked like every other call, and scored in aggregate
    against `holdout_min_bp`, raised from 5000 bp (50%) to 3000 bp (70%) so a
    nearest-class-mean alibi (66.5%) no longer clears it. They are no longer a
    free compute slot.

**Process and I/O (lens: io-and-process)**

11. The submission runs in a private directory containing only `submission.py`
    and `task.py`; the harness directory is removed from `sys.path` and the
    harness modules are removed from `sys.modules`, so `eval`, `utils`,
    `mnist_data`, `combine`, the draw constants and the cases file (with its
    public seed) are all unreachable. A fresh directory per phase, deleted after.
12. The evaluator re-execs itself once with `POPCORN_SEED` and
    `MNIST_POOL_CACHE` removed from its environment, receiving them on an
    inherited pipe. `os.environ.pop` does not rewrite `/proc/<pid>/environ`, so
    without this a child could read the parent's secret seed and reconstruct
    every draw, permutation and hold-out of the run.
13. `install_network_guard` installs an audit hook (permanent, C-level) in
    addition to the monkeypatches. `_socket.socket().connect` and
    `importlib.reload(socket)` both escaped the monkeypatches and were verified
    to fetch the real MNIST label file; both are now refused. The hook also
    denies opening files whose names look like datasets. `subprocess curl` still
    escapes -- it is a fresh interpreter -- so `run_modal.py` sets
    `block_network=True` and the pools are baked into the image, loaded into RAM
    and deleted before any submission process exists.
14. The child calls `setsid`, and the parent kills its process group, so a
    detached process or thread cannot outlive the evaluation.

**Availability**

15. `Child.call` waits with a deadline (`max_call_ms` + 30 s for timed calls,
    `warmup_max_call_ms` + 30 s for the warm-up) enforced by a watchdog that
    kills the process group. `poll` alone is not enough: a submission that
    writes on low file descriptors corrupts the pipe, after which `poll` reports
    data while `recv` blocks forever on a message that never arrives. That
    variant (`threshold-gaming_edge.py REDTEAM_EDGE=fd`) used to hang the
    evaluator until KernelBot's mode timeout killed it with no result at all;
    on the A100 it failed in 157 s with "the submission did not return from
    'untimed' within 150 s" (`warmup_max_call_ms` 120 s plus 30 s), inside the
    300 s mode timeout. Since 1.1.1 every per-command deadline is additionally
    clamped to what is left of the mode's budget (section 10).

**Parameters added to `task.yml` / `bands.json`**

`holdout_draws` (2), `draw_slack_bp` (150), `dispersion_x10` (20),
`warmup_max_call_ms` (120000), `max_source_bytes` (20480),
`max_literal_bytes` (4096); `holdout_min_bp` default 5000 -> 3000. All still
live in `bands.json` and reach KernelBot only through generated `task.yml` case
lines.

### Two consequences worth knowing about

*Accuracy is measured on a smaller universe.* Because the test halves of a run
all come from the same 30,000-image half of the pool, the 11 ranked draws
overlap heavily and are no longer independent samples: the run-level accuracy
estimate has roughly the precision of one 30,000-example test set, not of
110,000. Band margins should be measured across several secret seeds, not one.
In the CPU re-run under seed 20260922, `cg_pair` scored 97.83% over three draws
against the 2% band's 98.00% -- upstream reports 98.12%, so that band sits
exactly at this entry's capability and needs a decision (move the band, or
accept that the entry sometimes misses).

*The hold-out calls are in the ranked mean.* A submission's published time is
now the mean over 11 MNIST calls and 2 Fashion calls. That is deliberate -- an
untimed hold-out is a free compute slot -- but it means times from harness
1.0.0 are not comparable with times from 1.1.0, and rule 7 (re-score standing
records when the evaluator changes) applies.

### What the re-run showed

| Exploit | Before | After |
| --- | --- | --- |
| `timer-patch.py` | 0.008 ms ranked, 8,550x | 67.2 ms, honest; the lie has no effect |
| `timer-patch_evalglobals.py` | 0.011 ms ranked | 59.8 ms, honest |
| `timer-patch_calibration.py` | 8,710 ns ranked | 62.9 ms, honest |
| `lazy-output.py` | 0.0022 ms, 32,000x | 58.2 ms, honest |
| `lazy-output_postevent.py` | caught (timing) | caught (per-draw floor; the patched `min` never runs) |
| `lazy-output_subclass.py` | caught | caught |
| `replay-cache.py` | 12% band pass at 94.9% | fail: draw 0 at 7,965 of 10,000, below the floor |
| `replay-cache_persist.py` | 2% and 5% bands pass | fail: same |
| `replay-cache_weights.py` | 9.4x discount, pass | fail: durations differ by 7.7x |
| `replay-cache_naive.py` | caught (accuracy) | caught (timing, on call 0) |
| `label-lookup{,_fallback,_pretrained,_probe}.py` | 2 of 4 escaped at 100% | all four rejected before import: over the size cap |
| `label-lookup_netprobe.py` | 3 of 5 channels reached the internet | 4 of 5 blocked; `subprocess curl` needs the container block |
| `io-and-process.py` | 12% band pass at 100% by disk lookup | fail: harness unimportable, dataset files denied, falls back to 79% |
| `threshold-gaming.py` | 2.71x from the untimed warm-up | 60.9 ms, honest |
| `threshold-gaming_margin11.py` | 1.73x by mixing 7 good draws in 11 | fail: draw 7 scored 8,002, floor 8,650 |
| `threshold-gaming_margin.py` | pass at 91.3% | fail: draw 2 below the per-draw floor |
| `threshold-gaming_clock.py` | 1,021x | 62.5 ms, honest |
| `threshold-gaming_slowcall.py` | caught after the fact | caught, and a hanging call is now killed |
| `threshold-gaming_forge.py` | caught (fd already dead) | caught |
| `timer-patch_cellsteal.py` (new, written to test the fix) | n/a | caught: it does overwrite the evaluator's captured clock through `gc`, and the parent's clock rejects the call -- "device time 0.003 ms is under 0.25 of the 29.015 ms the parent measured" |

GPU-only checks that this pass could not run are listed in
[`tests/GPU_CHECKS.md`](tests/GPU_CHECKS.md).

## Harness change from the first A100 run (2026-09-22, harness 1.1.0)

One change to `eval.py` was needed before anything could run on real hardware,
and it is the only scoring-path edit made during the GPU stage.

**`probe_call` now runs the same CUDA preamble as `timed_call`.** The
calibration probe is documented as "a timed call with no submission in it", but
it skipped the `synchronize / flush_l2 / synchronize` sequence that opens every
timed call. On the CPU dry-run path that sequence is a no-op, so the omission
was invisible. On an A100 it is not: the two barriers plus the *first*
allocation of the 256 MB L2 flush buffer cost about 10 ms, and all of it sits
inside the parent's round trip of a timed call while sitting outside the child's
wall clock. `ipc_ms` therefore under-measured the protocol by roughly 10 ms, the
budget `parent_wall - ipc` was inflated by the same amount, and the
parent-clock bound `device >= 0.25 * budget - 0.5 ms` rejected honest
millisecond kernels:

    timing implausible on call 0: device time 1.004 ms is under 0.25 of the
    11.203 ms the parent measured for this call (round trip 12.507 ms minus
    1.304 ms of calibrated overhead)

That was the nearest-class-mean baseline, in `test` mode, on the 12% band --
i.e. every submission faster than ~3 ms was disqualified, and since a
leaderboard run begins with `test`, no fast entry could have been scored at all.
Running the preamble in the probe fixes both halves: the flush buffer is
allocated during calibration rather than on timed call 0, and `ipc_ms` measures
what a timed call actually costs outside the kernel. The timed window itself is
untouched, so the ranked number means exactly what it meant before. After the
change `ipc_calibration` lands at 0.75-1.44 ms and the parent's clock agrees
with the device clock to within 1.3 ms on every reference submission.

Two red-team variants were added so that two GPU-only checks could run without
environment forwarding into the container:
`redteam/io-and-process_gpurecon.py` (identical, plus a stderr dump of the recon
dictionary) and `redteam/threshold-gaming_edge_gpufd.py` (`REDTEAM_EDGE`
defaults to `fd`).

## 10. Changes from the 1.1.0 audit (harness 1.1.1)

An independent audit of 1.1.0 found one defect that made the published number
non-secret and two hosting-path gaps that no GPU run had exercised. All three
are fixed here; the rest of the pass was documentation drift.

**1. The ranked run now always has a secret seed.** `main()` read
`POPCORN_SEED` and, when it was absent, silently fell back to the *public* case
seed in `task.yml`. That is the hosted path, not a hypothetical: KernelBot's
participant-visible run is submitted with `seed=None`
(`backend.py:submit_leaderboard`), `run_eval.py` sets `POPCORN_SEED` only when a
seed is given, and the leaderboard ranks on the non-secret run
(`leaderboard_db.py`, `AND NOT r.secret`). Every draw, every per-draw label
permutation and every hold-out position of the published run would therefore
have been reproducible offline from the public `task.yml`, the public `eval.py`
and the public 60k MNIST split. `eval.py` now draws its own secret with
`os.urandom(8)` when the host supplies none, and logs `system.seed_source`
(`popcorn` or `random`) so an operator can see which happened. Failing hard was
the stricter alternative but it would break `popcorn submit --mode test`.

**2. `profile` mode now meets the NVIDIA contract.** reference-kernels'
`docs/ncu-profiling.md` is explicit that accepting `profile` and running
`torch.profiler` is not enough: the runner wraps the evaluator in
`ncu --nvtx --nvtx-include 'custom_kernel/'` with `POPCORN_NCU=1` and fails the
run if no report comes out. When `POPCORN_NCU=1`, the child now makes one call
inside `torch.cuda.nvtx.range("custom_kernel")` with the synchronize inside the
range and validation outside, and torch.profiler is not started (the two
profilers compete). Every mode logs `benchmark.{i}.status`. This path has still
never run under a real `ncu`: `run_modal.py` does not invoke it.

**3. Module level must be inert.** KernelBot compiles a Python submission by
*running* it once (`python3 submission.py`) in the work directory, before
`eval.py` starts -- outside the network guard, the private submission
directory, the `sys.path` scrub and the re-exec. Module-level code could
therefore fetch the public MNIST labels, or simply `import mnist_data`, stash
the table under a name the dataset-open guard does not match, and read it back
from inside `custom_kernel`. `check_submission_source` now rejects any
module-level statement other than an import, a `def`, a `class`, a constant
assignment (any expression built only from literals and operators) and a short
allowlist of `torch` configuration calls and `torch.*` flag assignments. All
four reference entries pass unchanged; 17 of the 29 red-team files do not, so
`tests/GPU_CHECKS.md` now says their recorded verdicts predate this check.
Worth raising with Mark: whether the compile step can be skipped for `lang: py`
problems that do not use `load_inline`.

**4. Per-command deadlines are budgeted against the mode timeout.** The
startup, calibration, `load` and warm-up deadlines were fixed and independent,
so a submission could burn 149 s in `load` and 149 s in the warm-up and run the
300 s `test` timeout out, leaving KernelBot with a bare `TIMEOUT` and no
`check` line -- the exact failure change #15 was written to prevent (the real
board had only 143 s of margin). `task.yml` now carries `test_timeout`,
`benchmark_timeout` and `ranked_timeout` as case fields as well, each mode takes
a wall deadline on entry, and `Child.call` clamps every per-command deadline to
what is left of it minus a 30 s reserve. The harness therefore always gets to
emit `check: fail`.

**5. Generated files that used to drift.** `make_bands.py` now also rewrites the
band table inside `README.md` and one `<band>/submission.py` per band, so
`--check` catches a stale README (it did not before) and a participant who
downloads the 12% template no longer posts to the 5% board.

**6. Smaller corrections.** The per-draw floor is skipped when a mode runs a
single draw, where it is redundant with the aggregate rule and named the wrong
cause; `run_modal.py` gained `--env K=V` and now runs KernelBot's compile step
(`python3 submission.py`) before the evaluator, so the hosted sequence is
exercised; `utils.clear_l2_cache` and `mnist_data.source_permutations` were dead
copies of logic that lives in `eval.py` and are deleted; `system.driver` is read
properly; the README's CPU smoke-test command used a band the baseline cannot
clear; two "current board" rows quoted numbers from a different band's result
and now say "not run"; `tests/test_web.py` skips without `fastapi`; and
`bands.json` no longer says the hold-out floor is CPU-only calibration.
## 11. Linear release (harness 1.2.0)

Same form as section 2: Decision / Why / Precedent / Rejected.

### D11. A draw is released as `z = Q W (x - mu)`, under a secret per-draw map

**Decision.** A new case field, `release_dims`, set to 60 in `bands.json` and
therefore in every generated `task.yml`. `make_draw` fits one map on the draw's
own training rows -- `mu` their mean, `W` exact PCA whitening onto the top 60
principal directions of their covariance (no variance floor, `eps = 0`,
eigenvector signs pinned by the largest-magnitude entry) -- draws a Haar-random
60x60 `Q` from `np.random.default_rng([seed, RELEASE_SALT])`, and applies
`Q W (x - mu)` to *both* halves in float64, handing back float32 `(n, 60)`. The
map is never returned, logged, written or sent to the child; it goes out of
scope inside `make_draw`. Warm-up and Fashion-MNIST hold-out draws get the same
treatment with their own fitted maps, so every draw in a run has the same shape
and a different basis. `release_dims: 0` reproduces the 1.1.1 pixel release bit
for bit, and is what a local dry run with `--case release_dims=0` uses; every
band, and therefore every dry run that does not override it, runs at 60. A
top-60 eigenvalue at or below 1e-10 raises rather than dividing by zero, and
`eval.check_case` (mirrored by `make_bands.check_settings`) rejects a
`release_dims` outside `0..size^2`, or one a draw's `train` cannot support, when
the case file is read -- so an organizer's typo exits 113 (unreadable cases)
instead of being reported to the entrant as a failed validation.

**Why.** The 9x9 pixel grid was the cheapest thing left to smuggle in: an
entrant could train a CNN offline on the public split and ship the feature
extractor, and nothing in the harness distinguishes that from learning. The
rotation-obfuscation study of 2026-09-23
(`mnist/experiments/rotation-obfuscation-20260923/`) measured every variant of
the fix on three draw seeds:

* **Rotation alone leaks the covariance.** With `W = I`, one eigendecomposition
  of the released covariance plus a moment fit recovered pixels to 7% RMS.
  Whitening deletes that channel by construction: the released covariance is the
  identity whatever the pixels were.
* **A protocol-constant map is worthless.** A fixed published map -- including
  the fixed secret *pixel permutation* that was the previous proposal -- is
  undone exactly by a 5.5-7.2 KB int8 inverse, which fits inside the
  20,480-byte cap with room for a classifier. The map has to be redrawn per
  draw, from the run's secret seed, which is why `Q` is keyed on the draw seed
  and not on a constant.
* **A whitening floor leaks the border.** ZCA with `eps` keeps the 17 dead
  border pixels as a recognizable bottom subspace. Exact whitening of the top
  60 directions does not, which is why `eps = 0` and `k = 60`.
* **One map for both halves.** Fitting train and test separately would make the
  two halves incomparable and cost honest learners accuracy for nothing; the
  study measured it and it buys no security, because the attacker can whiten
  either half themselves.
* **A blind entrant cannot get the lattice back.** Sparse non-negative unmixing
  plus a QAP lattice search recovered about 20 of 64 live pixels, with 4-5x
  chance edge precision, and a CNN trained on the recovered lattice was no
  better than an MLP on the released features. The cheap route is closed.
* **The cost to honest entrants is small.** Under a point for dense learners in
  the rotation study; kernel methods pay more because the metric changes (exact
  whitening cost an arc-cosine kernel ridge about 1.2 pp at N = 10,000 in
  `mnist/experiments/pmnist-medium-cutoffs-20260923/`).

**Precedent.** `common.whitener(x, 'pca', eps=0.0, k)` and `common.haar(dim,
seed)` of the rotation-obfuscation study, ported line for line (the same
ascending-eigenvalue `eigh`, the same sign fix, the same QR-with-R-diagonal-sign
Haar construction), so the harness and the study measure the same release.
Random orthogonal obfuscation of released features is standard in
privacy-preserving ML; whitening-before-rotation is the standard fix for the
covariance leak.

**Rejected.** A secret pixel permutation (a constant map; a 5.5 KB inverse
undoes it, and a minimum-volume cone fit on pixel non-negativity recovers most
pixel axes without any external data); rotation without whitening (7% RMS pixel
recovery); ZCA with a variance floor (leaks the dead-border subspace); fitting
train and test separately (no security, real accuracy cost); releasing all 81
directions (the bottom 17 are numerically dead, so whitening them is a division
by noise).

**Residual risks this does not close.**

1. **An informed attacker with public MNIST.** Class-conditional second moments
   survive whitening. Aligning them recovered pixels to 10-13% RMS on whitened
   releases, and an offline CNN on the recovered pixels scored 98.4-98.8% --
   above anything an honest in-competition learner reached. What stands against
   it: the 20,480-byte source cap (the attack needs ~19 KB of class statistics
   *plus* a network), the Fashion-MNIST hold-out draws at a 70% floor (a
   *non-adaptive* MNIST-specific smuggled model fails them, and they are timed
   and ranked so there is no free slot), and rule 8, which forbids the attempt
   outright. Only the first of those binds an adaptive entrant: a hold-out call
   is cheap to recognize from the release alone (next risk), so a smuggler can
   fall back to an honest learner on it.
2. **Hold-out calls are detectable, so the floor stops only a non-adaptive
   smuggler.** The red team's own moment residual separates the domains 11.5x
   (`redteam/release/results.json`, `summary.holdout_detectability`), and no
   external data is needed for the same result: the class-mean scatter matrix
   transforms as `B -> Q B Q^T`, so its eigenvalues pass through the secret
   rotation unchanged. Measured on this harness at 10,000/10,000,
   `release_dims: 60`, four draws per pool: the top eigenvalue is 0.778-0.784 on
   MNIST against 0.922-0.947 on Fashion, non-overlapping. The untimed warm-up
   call is itself a labelled Fashion draw (`run_case` draws it from the hold-out
   pool so that nothing fitted before the timed loop is about MNIST), which also
   hands every run a free reference value. Moving the warm-up to a third dataset
   would remove the reference but not the separation, and it would cost the
   property that makes the warm-up useless for MNIST; a hold-out that is
   genuinely indistinguishable needs a second pool from the *same* distribution,
   i.e. a private split. Recorded rather than fixed: the size cap is what the
   informed attack actually runs into.
3. **Exact-source re-identification.** If released rows can be matched to the
   public pool, a least-squares fit on the matched pairs recovers the entire
   map. Whitening removes the norm cue that made this trivial (99.9% of rows in
   the study), but the images are still public. Only a private pool closes it.
4. **The bands were calibrated on pixels.** The published thresholds come from
   pixel-release measurements, and the release is not accuracy-neutral for the
   entries we have. CPU, mean over five 10,000/10,000 draws of case seed 101,
   pixels -> release: `cg_pair` 98.0% -> 95.8%, `mlp512` 96.4% -> 95.7%,
   `pca_qda` 95.5% -> 90.3%, `ncm_baseline` 80.1% -> **85.0%** (whitening turns
   its Euclidean metric into something closer to a Mahalanobis one). Per-entry
   ranges and the hold-out columns are in `submissions/README.md`; five draws
   rather than one because a single draw hid a divergence in `mlp512` that two
   of eleven draws showed. A linear control confirms the release
   itself is faithful -- ridge regression scores 84.2% on pixels and 84.0% on
   the release, as an affine-equivariant learner must -- so what is being lost
   is pixel-specific structure: the arcsine transform, the RBF metric on pixel
   distances, and an isotropic covariance prior that was implicitly damping the
   low-variance pixel directions whitening now amplifies. On these numbers the
   2% and 3% bands have no known entry. The dense ceiling on the release is
   being re-measured; if it moves, `bands.json` changes and rule 7 re-scores
   every standing record.
