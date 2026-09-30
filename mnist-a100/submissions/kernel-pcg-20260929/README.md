# Non-neural metric kernel submission — difficulty 2, 2026-09-29

Contributor: [@islamborghini](https://github.com/islamborghini).

The unchanged [classifier](kernel_pcg.py) passed three sandboxed Modal
A100-SXM4-80GB runs of scorer 1.2.0. Median ranked time: **38.319672 ms**;
median energy above idle: **8,070.955853 mJ per call**.

## Runs

| Run | Ranked ms | MNIST correct / 110,000 | MNIST error | Hold-out accuracy | Energy, mJ |
| --- | ---: | ---: | ---: | --- | ---: |
| [1](evidence/run-1.json) | 38.319672 | 106,728 | 2.974545% | KMNIST 96.0375% | 8,031.205938 |
| [2](evidence/run-2.json) | 39.769809 | 106,810 | 2.900000% | KMNIST 95.9925% | 8,070.955853 |
| [3](evidence/run-3.json) | 37.734545 | 106,783 | 2.924545% | Fashion 88.1475% | 8,219.682094 |

Each run contains 11 MNIST and four foreign hold-out calls, each in a fresh
sandboxed process after a foreign-data warmup. All recorded accuracy, worst-draw,
timing-dispersion, host-clock and energy checks pass. The weakest timed MNIST
draw achieved 96.89% accuracy, above the required 95.10%; each run's mean also
exceeds the required 96.60%. Three distinct GPU UUIDs are recorded in the evidence.

Compared with the previous difficulty-2 entry (252.1 ms / 14,436 mJ), this is
6.58x faster and 44.1% less measured energy. This is a comparison of leaderboard
measurements, **not a controlled same-board experiment**: the previous entry used
PCIe A100-80GB boards; these runs used SXM4-80GB boards with recorded 500 W limits.
Both are covered by the official A100-80GB workflow.

## Same-board comparison

Both unchanged methods were also scored three times each on one PCIe A100-80GB
at an unchanged 300 W limit. All six runs passed. Median ranked time was
253.356 ms for the previous ensemble and 35.589 ms for this method; median
energy was 14,897 mJ and 7,664 mJ respectively: **7.12x faster and 48.55% less
energy on the same board**. The run order was fixed before outcomes, and the
scorer supplied independent fresh draws. These are supplementary results, not
replacements for the three-board medians above.

[Comparison report and all six records](same-board.md).

## Method and attribution

Every call normalizes each feature vector, fits RBF kernel ridge regression to
the supplied labels, computes a supervised 60-by-60 distance metric from analytic
kernel-score gradients on 512 training examples, and refits once. Each fit uses
16 conjugate-gradient steps with a rank-256 Nyström/Woodbury preconditioner.
The matrix product still uses the full kernel. A custom Triton split-reduction
product uses TF32x3 tensor-core arithmetic with FP32 accumulation and a fused
reduction/ridge term.

This is a kernel machine with learned metrics, not a trained neural network.
There are no pretrained coefficients, original pixels, external examples,
test labels, data-file reads, or learned state reused across calls. Test rows
are used only for per-row normalization and prediction. Production code does
not read clocks, events, NVML or scorer state. Every call uses the same schedule.
JIT compilation caches executable code only; the benchmark explicitly allows
compilation during the foreign warmup.

The metric update follows the average-gradient-outer-product feature-learning
idea in [Radhakrishnan et al., Recursive Feature Machines](https://arxiv.org/abs/2212.13881).
The optimization work was guided by reducing work and data movement, following
[Bill Dally's energy-efficient AI hardware principles](https://aha.stanford.edu/sites/g/files/sbiybj20066/files/media/file/aha-retreat-2023_dally_keynote_en_eff_ai_hw_0.pdf).
Implementation and experimentation were AI-assisted, including DeepSeek v4.1
Flash consultations through OpenCode. Hyperparameters were selected on development
draws; the source was frozen before the three scored runs reported here.

## Scope and limitations

The 16-step fit is deliberately approximate, not a converged exact KRR solve.
A separate diagnostic measured relative residual about 0.158. Early stopping
is an accuracy/runtime tradeoff; qualification uses the actual classification
errors above. The included synthetic self-check tests direct-solve agreement,
ragged matrix-product tiles and analytic gradients, but does not prove exact
convergence on every production draw. Cholesky has no fallback for an unfavorable
future draw. No such failure occurred in these three scored runs.

Time covers training plus prediction within the benchmark call, not downloads,
data preprocessing, process startup or first-ever compilation. Energy follows
the [official above-idle protocol](../../energy/README.md): it excludes host
energy and the first call, and subtracts idle and empty-round-trip costs.
The energy reference products and all telemetry plausibility checks pass.
The ~31 ms warm-process diagnostic is not substituted for the ranked time.

## Reproduce

From `mnist-a100/`, with the repository's ordinary Modal setup:

```sh
python run_modal.py submissions/kernel-pcg-20260929/kernel_pcg.py:classify --difficulty 2 --runs 3 --json /tmp/kernel-pcg-new-runs
```

Use a new output directory to preserve prior evidence. The stock runner creates
an ephemeral app; no personal account or experiment helper is required. Confirm
that its app has stopped after completion or cancellation. A fresh run is needed
to reproduce GPU measurements; the following inexpensive check only verifies
source identity and reconstructs all nine saved results, without launching Modal:

```sh
python submissions/kernel-pcg-20260929/verify_evidence.py
```

## Provenance

Repository/scorer revision used for comparison:
`6af17e1bf8527237bd31db1bfcbb368c0b21135b`.

- Candidate SHA-256 (8,074 bytes):
  `fc0ddef0d3dba8f00fc44f9cd576548c2a5184c5cb88741767d33d6f10b38012`.
- Current scorer SHA-256, byte-identical to that upstream revision:
  `1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064`.
- Stock runner SHA-256:
  `1180b843cf6c32414774da90c835a8b7298d2db8719989237a793ca1c62a10ab`.

The evidence files are byte-for-byte copies of the three original runner
outputs, not reconstructed or filtered records. Each embeds the candidate hash,
scorer version, stdout, stderr, all timed calls and all energy windows. The
original launch used a local context wrapper solely to verify the requested
Modal account and stop the app; the scoring function, image and sandbox were
unchanged. That wrapper and runner modification are excluded from this submission.
The original evaluation app was `ap-3fziCrNsh6iR74ZPcHhzyT`, as recorded in the
operator's launch transcript; all its tasks were stopped after evaluation.

**Evidence limit:** the original runner did not record the remote scorer hash,
revision or app ID inside each JSON result. Current byte equality and replaying
the scorer establish compatibility and internal consistency, not cryptographic
authentication of the historical remote scorer. The app reference above is
operator-reported, not independently embedded run metadata. No independent
maintainer GPU rerun or upstream acceptance is claimed. A separate skeptical
source/evidence review found no disqualifying exploit; it did not rerun GPU jobs.

The subsequent same-board comparison does record the scorer hash inside the
remote container, along with timestamps, source snapshots and hardware metadata.

## Files

- [Standalone classifier](kernel_pcg.py)
- [Evidence verifier](verify_evidence.py)
- Original scored runs: [1](evidence/run-1.json), [2](evidence/run-2.json), [3](evidence/run-3.json)
- [Same-board report](same-board.md) and [source/run manifest](evidence/same-board/manifest.json)
