# MNIST energy leaderboard on KernelBot (prototype)

**Status: a proposal to react to, not a finished competition.** It answers the
README's first open question ("Can we measure A100 energy correctly?") in the
setting a GPU MODE competition would run in: submissions measured by the
organizers on [KernelBot](https://github.com/gpu-mode/kernelbot), not
self-reported.

It runs on KernelBot without changes to KernelBot (two would help: see open
questions 5 and 6). A problem ships its own `eval.py`, and KernelBot ranks
whatever that script logs as `benchmark.0.mean`. Here that value is
**GPU board energy above idle, in nanojoules, per complete
training-and-prediction call**, so the leaderboard ranks by joules (its time
column shows 1 s for 1 J).

## Layout

| File | Role |
| --- | --- |
| `medium_5pct/task.yml` | One leaderboard: MNIST-medium, 5% mean-error band. Other bands and tiers are copies with different case values. |
| `eval.py` | Trusted evaluator. Holds test labels, seeds and label permutations; reads NVML; freezes the submission during idle windows; logs results. Never imports the submission or touches CUDA. |
| `worker.py` | GPU worker processes: one imports the submission, one ("harness") runs the telemetry reference and the input-staging control. |
| `nvml.py` | ctypes NVML binding (KernelBot's image has no `pynvml`). |
| `../code/energy.py` | Shared protocol, analysis and gates ([protocol notes](../doc/energy.md)); shipped to KernelBot as a task file. |
| `../code/data.py` | Verified MNIST download and box-area resize; shipped as `mnist_data.py`. |
| `reference.py`, `template.py`, `task.py` | Nearest-class-mean baseline, participant template, types. |
| `submissions/pca_qda.py` | Juraj's PCA-QDA learner, unchanged, wrapped in a captured CUDA graph. |
| `kernelbot_run.py` | Runs a problem through KernelBot's own `run_config`: locally (CPU, fake NVML: plumbing only) or on Modal with KernelBot's runner image on an A100-80GB. |

## Submission interface

```python
def custom_kernel(data):
    train_x, train_y, test_x = data   # (N,1,H,W) float32 in [0,1], (N,) int64, (Q,1,H,W) float32, all CUDA
    return predicted_labels            # (Q,) integer tensor
```

Train from scratch on every call. The three input tensors are the same objects
on every call (only their contents change), so a learner can capture a CUDA
graph on its first call. The first call is warm-up and is not measured.

## What a ranked submission goes through

KernelBot runs `eval.py` three times.

1. **test**: one draw; output format and a loose accuracy check.
2. **benchmark**: 11 secret-seeded draws; the exact band threshold on
   `sum(correct) / 110,000`, as in the repository README; reports time per call.
3. **leaderboard**: the same accuracy gate on 11 fresh draws, then the energy
   protocol:
   - FP32 4096² matmul reference; **abort if telemetry is implausible** (a host
     problem, reported with its own exit code so the submitter can resubmit)
   - task windows: the learner runs back to back on 8 staged draws for ~20 s
   - idle-only sham; **harness control** (the same per-call input staging with no
     learner, subtracted from the task); a second reference
   - every window bracketed by settled, measured idle
   - fresh-draw and staged-output accuracy spot checks before and after

The ranked value is the median task round, harness-subtracted, with the
power-integral meter (the counter is recorded as a cross-check). Per-window
figures are always logged. The raw record (power trace, intervals, gates),
which anyone can recompute with `python -m mnist.code.energy analyze`, is
written to `energy-record.json` and logged only when it fits KernelBot's
result pipe (open question 6).

## Design decisions made for this prototype

**Trust boundary.** The evaluator process is trusted; the submission's worker is
not. The worker only ever receives training images, training labels and test
images. Test labels, draw seeds and the label permutation stay in the
evaluator, which scores predictions itself. (KernelBot's example evaluator
checks correctness inside the submission's process.)

**The submission cannot run while idle is measured.** A submission that does work
while idle is being measured raises the baseline and lowers its apparent
energy. The evaluator freezes the submission's process with SIGSTOP whenever
idle, reference, sham or control energy is measured. In the simulated test
(`mnist/code/tests/test_energy.py`), a submission that works only between its
own calls hides more than 30% of its energy without the freeze and none with it.
A child process escapes SIGSTOP, but a second CUDA context shows up in the
compute-process check and aborts the run.

**Power is sampled from a separate process.** A sampler thread inside the
measured process is starved by the GIL. In simulation it read 0.63× the true
energy at 50% duty cycle; the energy counter read 1.00×. The two existing
PCA-QDA and small-QDA audits both use threads.

**Telemetry is checked before it is trusted.** The matmul reference must land in
6–11 J per 10¹² FLOPs at 15–23 TFLOP/s. Two healthy A100-SXM4-40GB hosts gave
8.49 and 8.36; the faulty host behind the original 3.8 mJ PCA-QDA claim would
read about 0.07. The bands came from 40GB boards; the four A100-80GB boards
measured here gave 8.1–8.8, inside them.

**Harness overhead is measured, not ignored.** Staging each draw into the fixed
input tensors costs energy that a small learner could not otherwise escape.
The control window measures it without the learner and subtracts it.

**Labels are permuted per draw.** Each draw's class labels pass through a secret
permutation, so hard-coded weights predict the wrong classes. This stops only
the most naive cheat; see below.

## Open questions for the competition

1. **Hardware.** KernelBot's Modal A100 is the 80GB board, and every current MNIST
   energy number is from SXM4-40GB. Re-measure on 80GB, or ask GPU MODE for 40GB?
2. **Cost per submission.** The full protocol takes about 5 minutes of A100 time,
   against about 2 for KernelBot's timing benchmarks. Idle windows can shrink
   for learners with large energy per call; tiny learners need long windows
   before subtracting idle can be trusted.
3. **Cheating with public data.** The test images come from the public 60k pool.
   Label permutation defeats hard-coded weights, but not a submission that
   downloads MNIST and matches test images to their public labels, or one that
   ships pretrained features and learns only a relabeling. Candidate defenses:
   - a **held-out check dataset**: also train and test on a draw from a
     different dataset at the same resolution (e.g. Fashion-MNIST) and require
     a sane accuracy floor, which lookup-based submissions fail;
   - blocking network access in the runner;
   - code review of top entries (KernelBot already has a top-three review flow).
4. **Memoizing staged draws.** Energy windows cycle 8 staged draws, so a learner
   could cache outputs after the first cycle. Staged-output spot checks confirm
   answers are right, not recomputed. Options: many more staged draws (GPU
   memory), fresh draws generated on the GPU per call (adds overhead the
   control must subtract), or treat caching as a review issue.
5. **Ranking display.** The leaderboard formats scores as time. A score-unit field
   in KernelBot would fix this.
6. **Keeping the raw record.** KernelBot reads the result pipe only after
   `eval.py` exits, so everything logged must fit in 64 KiB. A 5-minute record is
   ~70 KB compressed, so the evaluator logs it only when it is under 32 KB and
   otherwise logs a per-window summary. Publishing full records (for anyone to
   recompute a score) needs artifact storage in KernelBot.
7. **Raw joules or reference-normalized energy?** On four A100-80GB boards,
   PCA-QDA's raw energy spans 9% but its ratio to the same board's matmul
   reference spans 2.3% (see "random board" below). If that holds across
   workload types, dividing by the reference makes rankings independent of
   which board a submission lands on.
8. **Which accuracy bands become leaderboards,** and whether small/large join.

## Running it

```sh
git clone https://github.com/gpu-mode/kernelbot /path/to/kernelbot
export KERNELBOT_SRC=/path/to/kernelbot/src

# Plumbing dry run on a laptop (torch, numpy, pyyaml): CPU tensors, fake NVML
python mnist/kernelbot/kernelbot_run.py --local --submission reference.py \
    --set error_bp=3000 train=2000 test=2000 draws=2 rounds=1 active_s=2 idle_s=1 settle_s=1 reference_s=1 staged=2

# Modal, KernelBot's runner image, A100-80GB. Needs Python 3.13 locally (the image's
# version, required for serialized functions), pyyaml and modal.
KERNELBOT_MODAL_APP=sutro-mnist-kernelbot KERNELBOT_PCH_VOLUME=sutro-kernelbot-pch \
python mnist/kernelbot/kernelbot_run.py --submission submissions/pca_qda.py --output results/pca-qda.json

# Recompute any saved record
python -m mnist.code.energy analyze mnist/kernelbot/results/pca-qda-a100-80gb-full1.json.gz
```

The dry run exercises every mode, window, freeze and log key; its gates fail as
expected, because a constant fake sensor measures zero net energy.

## Results (2026-09-16)

PCA-QDA through KernelBot's `run_config` and runner image on Modal, one
NVIDIA A100-SXM4-80GB per run (400 W limit). Every run passed every gate. All
values recompute from the records in `results/`.

| Run | Board | VBIOS | Idle W | Reference J/TFLOP (start, end) | TFLOP/s | Raw mJ/call | Harness mJ | **Net mJ/call** | ms/call | Raw ÷ reference |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke 1 | …3f5e81 | 92.00.94.00.04 | 70.0 | 8.69, 8.82 | 18.7 | 191.7 | 0.03* | **191.7** | 3.52 | 21.9 |
| smoke 2 | …ee004e | 92.00.45.00.0E | 71.2 | 8.10, 8.09 | 18.6 | 175.2 | 1.55 | **173.7** | 3.56 | 21.6 |
| full 1 | …c84699 | 92.00.45.00.05 | 70.5 | 8.52, 8.47 | 18.7 | 186.3 | 1.93 | **184.4** | 3.55 | 21.9 |
| full 2 | …88c0e6 | 92.00.45.00.0E | 69.1 | 8.20, 8.15 | 18.8 | 180.6 | 1.77 | **178.9** | 3.49 | 22.1 |

Smoke runs use one 5 s task window; full runs use the default protocol (three
20 s task windows, 11-draw accuracy gate). \*Smoke 1's control window was sized
by call count and lasted 72 ms, too short to measure; controls now run as long
as task windows.

- **Repeatability within a run is tight.** Full 1's three rounds spread 0.3%;
  its two meters agree within 0.2%; its idle-only sham reads 0.6% of task energy.
- **Board-to-board spread is larger.** Raw energy spans 175–192 mJ (9%) across
  four boards, but the ratio to each board's own matmul reference spans 21.6–22.1
  (2.3%).
- **Comparison with 40GB.** Yaroslav's SXM4-40GB rerun measured 174 mJ; Juraj's
  healthy 40GB hosts 177 and 188 mJ. The 80GB boards' reference
  (8.1–8.8 J/TFLOP) overlaps the 40GB hosts' (8.36, 8.49), so the telemetry band
  appears to carry over.
- **Accuracy.** 105,231 / 110,000 (95.66%) on 11 secret-seeded draws, against the
  5% band's 104,500; the submission's own frozen evaluation was 95.57%.

## Can energy be measured when Modal places each run on a random board?

Yaroslav's question: "I think what's more important is finding all the snags. Is
it even possible to measure energy if the modal deploys you at random places?"

**On any one board, yes, precisely. Across boards, raw joules move about 9%, and a
same-run reference removes most of that for PCA-QDA; whether that correction
holds for other kinds of workload is the main untested snag.**

The four runs above landed on four different A100-SXM4-80GB boards:

| Board VBIOS | Idle W | Mean temperature in task windows | Matmul reference J/TFLOP | PCA-QDA raw mJ/call | Raw ÷ reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 92.00.94.00.04 | 70.0 | 35 °C | 8.75 | 191.7 | 21.9 |
| 92.00.45.00.05 | 70.5 | 37 °C | 8.50 | 186.3 | 21.9 |
| 92.00.45.00.0E | 69.1 | 38 °C | 8.17 | 180.6 | 22.1 |
| 92.00.45.00.0E | 71.2 | 41 °C | 8.10 | 175.2 | 21.6 |

- **Within a placement:** three task rounds agree within 0.3%, the two meters
  within 0.2%, and the idle-only sham reads 0.6% of task energy.
- **Between placements, the boards themselves differ.** The matmul reference
  varies 8%, in the same order as the task's 9%. Idle power (69–71 W), driver
  (580.95.05) and performance state (P0) were identical, and temperature did
  not explain it: the warmest board was the most efficient. The only visible
  difference is the VBIOS; the two boards sharing 92.00.45.00.0E were the two
  most efficient.
- **The same-run reference cancels most of it.** Dividing task energy by the
  board's own reference narrows the spread from 9% to 2.3%.
- **Broken or unusual hosts are detectable.** The Vast.ai 40GB hosts behind the
  earlier audits had 38–58 W idle, 320 W vs 400 W power limits, and one broken
  sensor (≈0.07 J/TFLOP). The reference gate rejects the broken one before any
  task is measured. Four Modal placements are too few to estimate how often
  such hosts occur.

### Snags found so far

| Snag | Where it bit | Handling |
| --- | --- | --- |
| Broken power telemetry on a host | Original 3.8 mJ PCA-QDA claim | Matmul reference gate; abort as a host failure |
| Idle power differs by host (38–71 W) | All hosts | Report energy above paired, settled idle |
| Board efficiency differs by placement (~8–9%) | Modal A100-80GB | Same-run reference recorded; normalization proposed |
| Sampler thread starved by the GIL | Simulation: 0.63× true energy | Sample from a separate process |
| Energy counter too coarse for short windows | 72 ms control window read −3.7 mJ/call | Every window lasts ≥ 5–20 s |
| Submission working while idle is measured | Simulation: >30% energy hidden | SIGSTOP the submission outside its windows |
| Input staging energy | 1.6–1.9 mJ/call (~1%) | Control window, subtracted |
| SIGSTOP + process groups killed the container | Modal, exit 129, re-queued 8× | Workers in their own session |
| Result pipe holds 64 KiB, read only after exit | Hung 22 min on a 70 KB record | Log records only when small |
| Container reuse between submissions | KernelBot GPU functions | Pre-existing GPU processes counted; stale records deleted |

### Snags not yet tested

1. **Does reference normalization hold for other workload types?** The matmul
   is pure compute. A kernel-launch-dominated learner (small-QDA takes 17 µs)
   or a memory-bound one may not track board efficiency the same way.
2. **Frequency of bad hosts** on Modal's fleet.
3. **Bursty workloads.** NVML power is reportedly an average over only part of
   each update period; saturated 20 s windows average this out, uneven
   learners may not.
4. **Absolute accuracy.** Both meters use the board's own sensors; nothing is
   checked against a wall or shunt meter.
5. **Which SKU Modal assigns.** "A100-80GB" gave SXM4-80GB four times out of four;
   nothing guarantees it.

### Planned next experiment

About 12 fresh placements, each measuring three workloads with very different
profiles (PCA-QDA, a kernel-launch-dominated small-tier learner, and a
memory-bound learner), with short matmul references interleaved between task
windows instead of only at the start and end. Questions it answers: the raw
and reference-normalized spread per workload type; whether per-round
bracketing references beat one reference per run; how much the interleaving
perturbs task measurements (heat from the matmul carrying into the next task
window); and the added wall-clock time. Roughly 40 A100-minutes.

## Known issues

- One simulated-GPU unit test failed intermittently (2 of ~14 runs, both right
  after a CPU-heavy local dry run) and did not reproduce under deliberate load.
  The failing assertion was not captured.
