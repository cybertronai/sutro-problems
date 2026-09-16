# QDA: MNIST-small, 67% accuracy target

Submission date: September 15, 2026 (UTC). Contributors:
[@jurajselep](https://github.com/jurajselep)

Closed-form quadratic discriminant analysis (QDA) fits one Gaussian per class
and predicts all 1,000 test labels after a single pass over 1,000 training
examples. The fresh evaluation achieves **7,474 / 11,000 = 67.95% ±
1.64 pp**, exceeding the required 7,370 correct. There is no iterative training,
learner seed, stopping rule, or transfer of learned state between datasets.

The complete serialized spatial-grid computation costs **0.00089 mJ** and
**10 ms**, with **47,088 bytes** of scratch memory. These are approximately
208× less model energy and 174× less model time than the published NR-K8 Adam
entry. That entry labels its grid score provisional. The small accuracy
increase is not a claim of statistically established superiority.

## Accuracy and provenance

Each draw uses `Generator(PCG64(seed)).permutation(60000)`, taking the first
1,000 original MNIST training rows for training and the next 1,000 for testing.
The subsets are disjoint within each draw; independently sampled draws can
share examples. Images are divided by 255 in FP32 and resized from 28×28 to
3×3 with the repository's exact fractional box-area weights
([`data.py`](../../code/data.py)) in a recorded accumulation order
(`run.resize_recorded`: increasing order, float64 product and sum before each
FP32 accumulation), which reproduces all original and fresh input hashes on the
source server and independent verification host; the repository's float32 matmul `area_resize` rounds
differently on some BLAS builds. The CPU and GPU learner inputs are `4*x - 0.5`.
Test labels are used only for evaluation and are never kernel inputs.

### Fresh evaluation (qualifying result)

Because the original protocol's timestamp postdates its evaluation (below), the
submitted learner was re-evaluated on eleven new draws under
[`protocol_fresh.json`](protocol_fresh.json): the learner is the unchanged
`reference.py` (its hash is recorded in the protocol), the seeds
20262101–20262111 are declared by the authors as previously unused.
The repository records the protocol commit first, followed by the draw manifest
(prepared 2026-09-15T17:57:05Z), the prediction freeze (2026-09-15T17:57:06Z) and the
evaluation (2026-09-15T17:57:06Z), each committed in that order. The freeze/scoring
code hashes all predictions before taking evaluation-label slices. Protocol and
learner hashes are checked against the manifests. This reproduces the fresh
workflow; local timestamps do not independently prove absence of prior evaluation.

| Draw | Dataset seed | Correct / total | Accuracy |
| ---: | ---: | ---: | ---: |
| 0 | 20262101 | 690 / 1,000 | 69.0% |
| 1 | 20262102 | 691 / 1,000 | 69.1% |
| 2 | 20262103 | 690 / 1,000 | 69.0% |
| 3 | 20262104 | 705 / 1,000 | 70.5% |
| 4 | 20262105 | 670 / 1,000 | 67.0% |
| 5 | 20262106 | 656 / 1,000 | 65.6% |
| 6 | 20262107 | 688 / 1,000 | 68.8% |
| 7 | 20262108 | 649 / 1,000 | 64.9% |
| 8 | 20262109 | 673 / 1,000 | 67.3% |
| 9 | 20262110 | 683 / 1,000 | 68.3% |
| 10 | 20262111 | 679 / 1,000 | 67.9% |

**7,474 / 11,000 = 67.95%**, sample standard deviation
1.64 pp (`ddof=1`); the exact count exceeds the 7,370 threshold by
104. Evidence: [`evidence/fresh/accuracy/`](evidence/fresh/accuracy/) and
[`results/cpu_verification_fresh.json`](results/cpu_verification_fresh.json)
(`SUTRO_PROTOCOL=protocol_fresh.json python verify.py --evidence-dir evidence/fresh/accuracy`).

### Beacon-seeded evaluation (in progress)

The fresh evaluation's chronology rests on this repository's commit timestamps.
For a public precommitment, [`protocol_beacon.json`](protocol_beacon.json) fixes
how eleven seeds are derived from the NIST randomness-beacon pulse of
2026-09-16T12:00:00Z (`seed_i = SHA-256(outputValue ‖ ':' ‖ i)[:8]` as an integer),
a value that is unavailable in advance. The protocol and its SHA-256 were
[published on the pull request](https://github.com/cybertronai/sutro-problems/pull/82#issuecomment-5686480922)
at 2026-09-15T19:05:04Z, before the pulse. The evaluation remains pending.
After the pulse, `run.py fetch-beacon` stores it with the beacon's signature and
`prepare`/`freeze`/`score` run under `SUTRO_PROTOCOL=protocol_beacon.json`
into `evidence/beacon/accuracy`; `verify.py` compares the stored pulse with an
official NIST HTTPS response and re-derives the seeds. This checks the complete
pulse object, rather than verifying the signature offline. Public timestamps
must separately establish that this exact protocol was published before the
pulse. No beacon result is reported until the run is complete.

### Original evaluation (retained, timestamp limitation disclosed)

| Draw | Dataset seed | Correct / total | Accuracy |
| ---: | ---: | ---: | ---: |
| 0 | 20261201 | 654 / 1,000 | 65.4% |
| 1 | 20261202 | 681 / 1,000 | 68.1% |
| 2 | 20261203 | 687 / 1,000 | 68.7% |
| 3 | 20261204 | 648 / 1,000 | 64.8% |
| 4 | 20261205 | 631 / 1,000 | 63.1% |
| 5 | 20261206 | 682 / 1,000 | 68.2% |
| 6 | 20261207 | 679 / 1,000 | 67.9% |
| 7 | 20261208 | 706 / 1,000 | 70.6% |
| 8 | 20261209 | 710 / 1,000 | 71.0% |
| 9 | 20261210 | 717 / 1,000 | 71.7% |
| 10 | 20261211 | 670 / 1,000 | 67.0% |

The exact mean is `7465 / 11000 = 0.6786363636363636`; the sample standard
deviation is `2.6796539803760946` percentage points (`ddof=1`). Qualification
uses the integer count, rather than the rounded display percentage.

The imported selection records report that QDA with no shrinkage, class
priors, and a log-determinant term was chosen using pilot seeds
20261101–20261110 (67.59% mean). The [pilot log](evidence/selection/pilot_log.jsonl)
and [QDA ablations](evidence/selection/tune_qda_pilot10.json) are included here.
They are historical evidence; the pilot experiment scripts are not needed to
reproduce the submitted learner.

**Historical timestamp discrepancy:** the unchanged [protocol](protocol.json)
says it was created at 08:40 UTC, while the prediction manifest and evaluation
freeze say 08:19:03 UTC on the same day. These timestamps do not establish that
the protocol was frozen before evaluation. The source claims pilot-only
selection, but that chronology requires author/reviewer confirmation. Fresh
verification establishes reproducibility on the published draws; it does not
constitute a new held-out evaluation. Original files and their hashes are
identified in the [import manifest](evidence/import.json).

## Learning procedure

For each class `c`, with `n_c` training samples and `N = 1000`:

```text
mu_c      = sum(x) / n_c
S_c       = sum(x x^T) / n_c - mu_c mu_c^T
P_c       = inverse(S_c)
kappa_c   = log(n_c / N) - 0.5 * log(det(S_c))
score_c(q)= kappa_c - 0.5 * (q - mu_c)^T P_c (q - mu_c)
label(q)  = first class attaining the maximum score
```

[`reference.py`](reference.py) specifies ordered FP32 arithmetic: sequential
masked statistics, biased covariance, Gauss-Jordan inversion without pivoting,
and a packed upper-triangle quadratic form with doubled off-diagonals. The
logarithm uses 40 halving and 40 doubling compare/select steps into
`[0.75, 1.5]`, followed by `k*ln(2) + 2*atanh((x-1)/(x+1))`, truncated after
`z^15`. Multiplications and additions round separately. The selected data have
nonempty classes and usable covariance pivots; this submission does not add a
fallback for singular covariance or absent classes on arbitrary inputs.

## Grid model

| Grid energy (mJ) | Grid time (ms) | Peak scratch (bytes) | Host time to score (s) |
| ---: | ---: | ---: | ---: |
| 0.00089 | 10 | 47,088 | 0.084 |

The exact [score](grid/grid-score.json) is **889,633,966 fJ / word-node hops**
and **10,316,314 cycles**, equivalent to `0.000889633966 mJ` and
`10.316314 ms`. It uses the existing
[shared scorer](../grid-mlp-scoring-20260912/README.md) without changes and
[spatial-computer pitch 128 / ISA v4 at revision
01a0bd5e0d2564825b0f53dd766f763c82dbc7c0](https://github.com/cybertronai/simplified-dally-model/blob/01a0bd5e0d2564825b0f53dd766f763c82dbc7c0/models/spatial-computer/README.md).

[`spatial_program.py`](spatial_program.py) generates the complete program and
executes its expanded instructions for numerical verification. The generated
4.6 MB JSON is reproducible and excluded from Git. Relative to the CPU
reference, the submitted lowering:

- Consumes unnormalized resized pixels and computes statistics on those values.
- Expands each quadratic form into constant, linear, and quadratic terms,
  sharing 45 query products across all ten classes.
- Stages each training sample near the processor and orders memory regions by
  access density.
- Charges initialization, skipping zeroing only for `k`, `xs`, `prod`,
  `labels`, and `x`, which the scorer proves are written before any read.

These changes alter intermediate FP32 arithmetic. Verification compares the
**emitted labels** with the frozen reference labels; it does not claim bitwise
equality of scores or parameters, or equivalence for all possible inputs.

### Memory, tapes, and schedule

All arithmetic executes on `P(125,0)`. Its 11,522 program words and one staging
word fit within the tile's 12,288-word capacity. Each of 250 bottom-edge tiles
has one reserved staging word, giving `4*(11522 + 250) = 47,088` bytes of
allocated scratch. There are 250 instruction-issuing processors, with one
instruction and one scratch access active at any instant.

The tape contains 9,000 training pixel words, 1,000 raw integer training labels,
and 9,000 test pixel words. The program emits 1,000 raw integer predictions.
Words are striped across the 250 ports in order, with 76 input and four output
words per port. No test labels or pretrained parameters enter the tape or
program. Test queries are streamed after training; tapes are not rewound.

The shared scorer assigns legal scratch cells nearest-first, excluding each
core/router region, and reserves a staging cell 32 hops from each bottom core.
Regions are declared in the generator. It lowers each receive/send to its tape
stage plus a copy to/from the compute processor. Each access finishes before
the next starts; remote requests route horizontally then vertically, with
read responses retracing the route. This schedule has no concurrent traffic.

| Count or cost component | Exact value |
| --- | ---: |
| Executed instructions, including tape lowering | 2,457,208 |
| Scratch reads, including staging and tape | 5,401,610 |
| Scratch writes, including staging and tape | 2,456,208 |
| Normal program read/write hops | 563,558,934 |
| Input-destination/output-source hops | 2,235,032 |
| Tape and staging hops | 323,840,000 |
| Total word-node hops | 889,633,966 |

For legal local cell distance `d`, a scratch access costs `max(50, 2*d)` fJ
and one cycle. At mesh distance `L`, remote accesses add `256*L` fJ; remote
reads take `2*L+1` cycles and writes `L+2`. Bottom tape operations cost 128 fJ
and two cycles. Every distance is at least 32, so the local energy floor is
inactive and energy in fJ equals word-node hops. One cycle is 1 ns. This is
a theoretical score of the specified serialized schedule, not measured GPU
energy or a universal lower bound.

Scope includes initialization, all training, prediction, local and remote
memory accesses, tape input, and output labels. Sampling and image resizing
prepare the workload inputs and are excluded. Host scoring time covers
validation, initialization proof, placement, histograms, costs, and hashing;
it excludes program generation, file I/O, and numerical execution. The
[imported score](evidence/imported-grid-score.json) is preserved; current
metadata corrects the descriptions of normalization and initialization without
changing the program's instructions, placement, or costs.

## A100 measurement

The submitted energy result is a fresh measurement on two separate
NVIDIA A100-SXM4-40GB hosts. The Netherlands host was chosen for the headline
before these runs; Canada is an independent cross-check. Both run the unchanged
two-kernel FP32 PTX learner. All three rounds from each host are retained.

| Host | Idle-adjusted power integral (mJ) | Idle-adjusted counter (mJ) | Gross power integral (mJ) | CUDA time (ms) |
| --- | ---: | ---: | ---: | ---: |
| [Netherlands (headline)](results/energy-netherlands.json) | 0.585680 | 0.589103 | 1.232160 | 0.017047 |
| [Canada (cross-check)](results/energy-canada.json) | 0.602574 | 0.614870 | 1.577335 | 0.016847 |

Values are medians per complete training-and-prediction task. The leaderboard
uses the idle-adjusted sampled-power result. Each round replays original draw 0
1,200,000 times. Before and after each active window, the GPU settles for three
seconds, followed by ten seconds of measured idle. NVML power is polled with a
50 ms sleep between samples and integrated with trapezoids using the actual
timestamps; cumulative NVML energy is recorded too:

`net_mJ/task = (active_J - mean(before_idle_W, after_idle_W) * active_seconds) * 1000 / replays`

The two NVML methods share device telemetry; their agreement is not calibration
against an external power meter. An interleaved 20-second idle-only sham tests
baseline subtraction. A separate ten-second 4096×4096 FP32 matrix multiply
checks the output and requires both NVML methods to show at least 20 W above
idle. Both hosts pass. These controls are excluded from the task energy.
Signed estimates, raw samples, clocks, temperatures, and GPU process checks
are saved in the linked results.

| Host | Net-energy range (mJ) | Sample SD (mJ) | Idle-only residual (mJ/task) | Matrix control / idle (W) |
| --- | ---: | ---: | ---: | ---: |
| Netherlands | 0.585413–0.592533 | 0.004036 | -0.000752 | 199.5 / 38.6 |
| Canada | 0.599833–0.617118 | 0.009290 | -0.000410 | 216.0 / 57.7 |

These diagnostics use sampled-power integration. The idle-only residual is
the signed energy of the 20-second sham divided by 1,200,000 nominal tasks.

Round spread describes these runs, not a confidence interval or total
measurement uncertainty. The two hosts have different power limits and
drivers; their results are not pooled.

Netherlands: driver 595.71.05, 320 W power limit,
`GPU-03d6de3a-ca5b-03ca-727f-698cddfe0c46`.
Canada: driver 570.133.20, 400 W power limit,
`GPU-257d1077-bc61-c5a2-f856-e613780b1a6e`.
Both use Python 3.11.10, PyTorch 2.5.1+cu124, CUDA 12.4, NumPy 2.1.2,
pyptx 0.1.1, and nvidia-ml-py 13.610.43. Raw results bind the source, emitted
PTX, input manifests, NVML library, and GPU UUID with hashes or identifiers.

On each host, all 22,000 original and fresh predictions match the frozen CPU
labels, with 7,465 and 7,474 correct respectively. All 66 poisoned-buffer graph
replay checks pass. The generated PTX hash remains
`38d36146e4303b7acd640ee6c04ba642c751182c4fb9501b9fbccd52333cba58`.

The previous **0.010 mJ** claim is withdrawn: the original host failed
subsequent power-telemetry sanity checks. Its
[seven-round result](results/gpu_verification_benchmark.json) and earlier
measurements remain unchanged as historical evidence and are superseded by
the two results above. This correction changes the measurement, not the learner,
accuracy, frozen predictions, or grid score.

The first PTX kernel uses eight blocks of 512 threads, each accumulating class
statistics over 125 samples. The second uses 32 blocks of 320 threads; each
block combines the partials, inverts all ten covariances using warp shuffles,
and predicts up to 32 queries (eight in the final block). Every replay overwrites
all learned statistics and predictions. GPU reciprocal/logarithm instructions
and reduction order differ from the ordered CPU reference; labels are verified.

Scope is complete training and 1,000 predictions on device-resident normalized
inputs, including host dispatch gaps between graph replays. It excludes
transfers, normalization, allocation, compilation, graph capture, and cold
start. Prior small GPU entries include normalization, so their published
numbers are not an exactly matched GPU speedup comparison.

## Reproduction

The recorded-order resize avoids BLAS-dependent accumulation;
`requirements.txt` pins the versions used. From
the repository root:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r mnist/submissions/small-qda-20260915/requirements.txt
.venv/bin/python -m mnist.code.data --output mnist/data
. .venv/bin/activate
cd mnist/submissions/small-qda-20260915
python verify.py --output results/cpu_verification.json                                   # original draws
SUTRO_PROTOCOL=protocol_fresh.json python verify.py --evidence-dir evidence/fresh/accuracy \
    --output results/cpu_verification_fresh.json                                          # fresh draws
```

The download command prepares the canonical raw source files. The submission
scripts implement the required 1,000/1,000 sampling themselves; they do not use
the generator's legacy tier sizes. Set `SUTRO_RAW` to an existing raw-file
directory or `SUTRO_REPO` to the checkout root if needed.

To reproduce the two-phase workflow without overwriting the original evidence:

```sh
python run.py prepare --evidence-dir generated/reproduction
python run.py freeze --evidence-dir generated/reproduction
python run.py score --evidence-dir generated/reproduction
python verify.py --evidence-dir generated/reproduction --output generated/reproduction-check.json
python spatial_program.py
python run.py gpu-payloads
```

Use a new output directory on another run; existing frozen evidence is protected
against overwriting. The fresh evaluation was produced the same way with
`SUTRO_PROTOCOL=protocol_fresh.json` and `--evidence-dir evidence/fresh/accuracy`,
committing the protocol, the draw manifest, the freeze and the score as four
successive commits. `spatial_program.py` regenerates the grid program and score;
its host runtime and source/software provenance can vary across machines while
exact counts remain unchanged.

Export the fresh payloads in the same CPU environment:

```sh
SUTRO_PROTOCOL=protocol_fresh.json python run.py gpu-payloads \
    --evidence-dir evidence/fresh/accuracy --output-dir generated/payloads-fresh
```

Copy the submission directory, including its evidence manifests and both
payload directories, to an otherwise idle A100 host. In a separate Python 3.11
GPU environment with GCC and CUDA 12.4 driver support:

```sh
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-gpu.txt
nvidia-smi --query-gpu=uuid --format=csv,noheader
python energy_benchmark.py --original-payloads generated/payloads \
    --fresh-payloads generated/payloads-fresh --expected-uuid GPU-REPLACE-WITH-YOUR-UUID \
    --output generated/energy-rerun.json
```

Replace the UUID with the selected device's value. The harness checks both
sets of eleven draws before measuring and rejects another compute process or
a failed positive control. Use a new output filename for each run.
To check correctness without measuring energy:

```sh
python gpu_benchmark_ptx.py generated/payloads generated/gpu-original.json --verify-only
python gpu_benchmark_ptx.py generated/payloads-fresh generated/gpu-fresh.json --verify-only
```

Recompute both saved energy summaries locally in either pinned NumPy
environment; no GPU is needed:

```sh
python energy_verify.py results/energy-netherlands.json
python energy_verify.py results/energy-canada.json
```

The original accuracy evidence and protocol are preserved. New verification
results retain their own source hashes and software versions. The verifier
checks canonical MNIST source hashes, sampling and preprocessing, saved
predictions, regenerated learner parameters and scores, exact accuracy, and
all eleven spatial-program executions. The shared scorer's five existing
tests also pass.

Key evidence: [fresh accuracy](evidence/fresh/accuracy/accuracy.json),
[fresh protocol](protocol_fresh.json), [original accuracy](evidence/accuracy/accuracy.json),
[draw manifest](evidence/accuracy/draw_manifest.json),
[prediction manifest](evidence/accuracy/prediction_manifest.json),
[evaluation freeze](evidence/accuracy/evaluation_freeze.json),
[grid score](grid/grid-score.json), and [submission summary](submission.json).
