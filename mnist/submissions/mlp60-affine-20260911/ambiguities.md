# Ambiguities and problems: the 60% submission

The fixed network scores **374/600 correct (62%)**, above the current **60%**
requirement. This establishes classification qualification on the supplied
dataset, subject to the benchmark decisions below.

## Selection after the public study

The 300-epoch configuration was one of five configurations frozen before the
feasibility study's test evaluation. Seed 101 was the first of its three
predeclared seeds. However, choosing this configuration as the submission for
the newly requested 60% target happened after all study test results were
public. It is not a fresh blind selection. The other fixed 300-epoch seeds
scored 371/600 and 377/600; neither is substituted for the chosen first seed.
No new hyperparameter or seed search was performed for this submission.

The gap above the threshold is 14 correct predictions. It is a pass on this
fixed split, not a claim that unseen datasets or arbitrary seeds will pass.

## Compact IL is a proposal

`sutro-affine-v4/0.1` compresses fixed loops and affine addresses while charging
every expanded primitive operation. It is submission-owned code, not an
officially accepted benchmark format. The benchmark needs to decide whether
this representation and its restricted verifier are acceptable.

The full program contains 618,842,313 primitive instructions and was not
numerically interpreted instruction by instruction. Its static counts match
the frozen study exactly. Small expanded MLP programs test the lowering's
semantics; a separate full 300-epoch ordered FP32 replay verifies this learner's
parameters, scores, and predictions. Those checks provide different evidence
and do not amount to a full expanded-program proof.

## Arithmetic, tape, placement, and area

The submission inherits the original attempt's declared binary32 arithmetic,
rounding after each multiply and add, unfused operations, free tape operations,
and fixed half-diamond placement at the pinned v4 model revision. The ISA's
relationship to this concrete FP32 interpretation remains a benchmark policy
question. The compact language adds no free matrix operation.

Reported area is **0.020 mm²**: occupied scratch-cell area under one square
micrometre per word. It is not a fabricated chip area, A100 die area, enclosing
rectangle, or GPU allocator footprint. The unit conversion changes no allocation
or physical assumptions. Tape, instruction storage, and host/device buffers
outside the modeled scratch memory are excluded.

Seed-only initial parameter constants are generated before execution and reset
within each task. This treats the fixed random initialization as program
constants; it excludes running a random-number generator from both model and
GPU costs. The benchmark should explicitly decide this convention.

## GPU scope and energy uncertainty

CUDA graphs measure repeated, GPU-resident complete training-and-prediction
tasks. Each replay recomputes the learned parameters. Transfers, allocations,
compilation, graph capture, verification, cold start, and host CPU energy are
excluded. This is steady-state task throughput, not application launch latency.

The idle-adjusted energy subtracts the mean of idle board power measured before
and after each active interval. GPU clocks, temperature, and telemetry sampling
affect this baseline. Raw energy, both one-sided baseline alternatives, and all
three trials are reported alongside the primary result; no negative value is
clipped. These measurements are not a laboratory power-meter calibration.

The first A100 run passed numerical checks but contained a CUDA-time conversion
error. It is retained as invalid evidence and contributes no headline metric.
The corrected run adds a CUDA/wall-time consistency check, extends active
intervals, and allows idle power to settle. See `gpu_run_history.txt` for both
attempts. The algorithm, seed, and training budget did not change.

The A100 implementation was written manually from the same update equations.
It is not emitted by an automatic IL-to-PTX compiler. Its kernels preserve
ordered FP32 arithmetic, and complete learned-parameter and score bits are
checked against the CPU implementation before and after timing.

## Scoring runtime excludes semantic verification

The fast score includes IL validation, placement, address histograms, exact
integer cost sums, and canonical program hashing. It excludes JSON loading,
numerical training, accuracy evaluation, and file output. CPU learning and
the independent verification suite have separate timings in the report.
Calling the static score a complete submission-verification runtime would be
incorrect.

## Evidence and provenance

Raw JSON retains exact measurements and native model units. Human tables use
two significant figures, execution time in ms, energy in mJ, area in mm², and
scoring time in s. Threshold decisions use exact counts, not rounded percentages.
The human-readable session is a snapshot ending at its stated export cutoff;
it does not include later publishing steps or hidden reasoning.

The medium and large targets are unrelated to this small-tier submission.
This run establishes no new measurement for either tier.

- [Results and reproduction](index.html)
- [CPU verification evidence](verification.json)
- [GPU measurement history](gpu_run_history.txt)
- [Original study protocol](../accuracy-il-20260911/protocol.html)
- [Original study ambiguities](../accuracy-il-20260911/ambiguities.html)
