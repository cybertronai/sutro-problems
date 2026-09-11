# Ambiguities and measurement limits

This document separates interpretation and measurement limits from the result
table. The accompanying submission report records the selected algorithm,
accuracy, complete model scores, A100 measurements, and reproduction commands.

## What counts as a successful attempt

The latest request defines five medium error targets: **10%, 8%, 6%, 4%, 2%**.
This submission attempts **4% error**, using eleven draws of **10,000 training
and 10,000 query examples** at 9×9 resolution. The former 6,000/6,000 protocol
and intermediate 90%, 97%, and 98% requests are superseded for this attempt.
Historical reports retain their original scope and are not directly comparable.
The inclusive 4% threshold requires 105,600 / 110,000 correct; the result is
107,534 correct. It does not meet the strictest 2% level (107,800 required).
The decision uses exact counts. Sample SD uses the eleven dataset accuracies
and denominator ten; it is in percentage points, not a standard error.

Training and query rows are disjoint within each draw. Independently drawn
datasets may overlap because all are sampled from the same original 60,000
training examples. Consequently the eleven numbers describe this resampling
protocol, not eleven entirely separate source populations. Architecture
development used a previously available training subset; overlap between that
subset and later random query rows is possible. No fitted weights transfer
between draws, and no per-draw query labels are provided to the learner.

This is an open-data evaluation, not a hidden-label test. The data-preparation
process reads the canonical label file to extract permitted training labels,
but never creates query-label arrays before all predictions are frozen. The
learner containers receive only the permitted three arrays. Development choices
were made on a prior 4,800/1,200 training/validation split, which may overlap the
later query rows. Thus the evidence supports a frozen algorithm on fresh random
draws, but does not prove architecture selection was independent of every query
example ever seen in development. A stricter interpretation forbidding even
development overlap would require a separate development pool or an external
hidden evaluation. No validation or model selection occurs within formal draws.

The dataset CLI retains its historical default for reproducibility; current
medium runs must explicitly select `medium-error-targets-v1`. The original
competition sketch is illustrative; current counts and targets are those in
the written rules, not old annotations in the sketch.

Dataset preparation used NumPy 2.4.6 on macOS; the pinned GPU container used
NumPy 2.2.6. Published input hashes detect any reproduction difference. The
FP32 matrix multiplications in box-area resizing may depend on BLAS/platform,
so matching package versions alone is not a universal bitwise portability
guarantee. No such mismatch was observed in the retained checks.

## Numerical meaning of v4 instructions

The pinned v4 specification defines 32-bit words but does not completely define
floating-point rounding, subnormal handling, comparison encodings, or reduction
order. This submission declares FP32 round-to-nearest-even arithmetic, separate
multiply and add operations, explicit ordered reductions, and raw-word
comparison flags. Its A100 lowering is checked against an independent CPU
reference, with emitted PTX retained for inspection. These are submission
conventions rather than new official benchmark rules.

The training probability calculation is a declared repeated-squaring
approximation. It is part of the submitted algorithm. Its arithmetic and
temporary storage must be scored; it is not an unpriced call to an exponential
or softmax library. The algorithm is evaluated again after translation, because
native library reductions and random schedules can produce different trained
weights and predictions.

## Intermediate language and scoring

The compact language is a generator for a finite, statically addressed v4
instruction stream. Loops and seed-generated lookup tables compress the program
description. They are not new unpriced machine instructions. Each table value
used as an immediate expands to a charged `set`; each data access expands to the
corresponding priced scratch-memory access. Repeated source operands are charged
each time. The scorer aggregates exact access multiplicities rather than
interpreting every individual dynamic instruction.

Random initialization, permutations, and augmentation geometry depend only on
published seeds and fixed dimensions. They can therefore be generated when the
program is compiled. No image, training label, fitted weight, or query result
may be used to create those literals. Pixel interpolation, normalization,
training, parameter updates, and inference remain part of the scored program.
This interpretation assumes the model's uncharged instruction supply can carry
a large generated straight-line program. The report should not be read as a
measurement of instruction-storage size, decoder cost, or compilation energy.

The scorer is a submission-supplied research implementation, not an official
acceptance service. Its evidence includes expanded small-program execution and
independent per-address cost comparisons. Those checks provide concrete
evidence but do not constitute a formal proof of the implementation.

## Area and execution boundaries

Area means occupied 1 μm² scratch cells in the declared physical placement,
reported in mm². It excludes processor area, tape, instruction storage,
interconnect overhead, and a fabricated-chip floorplan. It must not be equated
with an A100's physical die area or its tensor allocation size. The A100 compiler
may keep values in registers and use parallel kernels while preserving the
declared numerical computation.

The model includes fresh learning and every query prediction. Tape operations
are free under the pinned model. GPU measurements use already resident inputs
and include fresh learned-state reset, full training, inference, ensemble
arithmetic, and launch overhead. Host/device input transfer, allocation,
compilation, graph capture, and audit serialization are reported outside that
boundary. Time to score is the host runtime of theoretical cost computation,
not learner training time and not compilation time.

## Energy and display units

Model time and A100 time are both shown in **ms**; both energies are shown in
**mJ**; area is shown in **mm²**; time to score is shown in **s**. This is a common
display scale, not a claim that latency and energy have the same physical
dimension. Conversions are 10⁹ ps per ms, 10¹² fJ per mJ, and 10⁶ μm² per mm².
Cost tables use two significant figures, while exact values remain in JSON.

NVML measures integrated GPU-board energy. Idle adjustment subtracts the mean
of paired before/after idle powers over the active counter interval. Temperature,
clocks, telemetry quantization, and baseline drift affect this estimate. The
measurement record retains raw counters, gross energy, paired baselines, and
before-only/after-only sensitivity. Host CPU energy is excluded. Repeated runs
can reuse GPU caches; measured performance is not a cold-start service latency.
