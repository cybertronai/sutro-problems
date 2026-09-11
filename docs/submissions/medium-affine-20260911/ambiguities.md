# Ambiguities and problems: MNIST-medium

> **Accuracy scope:** The current small/medium requirement is mean ± sample SD over 11 independently resampled datasets. This report preserves a historical single-dataset evaluation; its per-draw threshold checks do not establish that aggregate. [Current accuracy protocol](https://github.com/cybertronai/sutro-problems/blob/main/mnist/instructions.md#accuracy-over-11-random-datasets).

## Two per-dataset diagnostic thresholds

The measured result is **5,755/6,000 correct (96%)**, short of the requested
goal by 125 correct predictions and the repository threshold by 134.

The user requested 98%, allowing a lower-accuracy attempt to be reported. This
means 5,880 correct predictions out of 6,000. For this historical single-dataset check,
98.14% rounds upward to 5,889 required correct predictions. Meeting the
first threshold does not establish the second. A result below a threshold is a
measured attempt, not a qualifying submission at that threshold. Exact counts
drive both checks; rounded display percentages do not.

The historical 98.14% CNN result used 10,000 training examples and the official
10,000-example test split. This submission uses canonical competition-v2
6,000/6,000 disjoint samples from the original MNIST training split. The
historical result is context for the target, not a comparable measurement on
this dataset. No historical learned weights are used.

## Selection scope and uncertainty

The predeclared search evaluates 12 width/rate trajectories and up to five
checkpoints each: widths 128, 256, and 512; rates 0.01, 0.03, 0.1, and 0.2;
epochs 25, 50, 100, 200, and 300. Nonfinite trajectories may stop early. A
single non-stratified random split uses 4,800 fit rows and 1,200 validation rows
from the supplied training set, with split seed 20260913 and initialization
seed 11. The final configuration uses initialization seed 101.

Selection seeks at least 1,176 validation hits, then minimizes epochs × width,
width, and learning rate. If no candidate meets that threshold, it maximizes
validation hits before applying those tie-breaks. Epochs × width is a coarse
cost proxy, not the physical score. The configuration is frozen before test
evaluation and no further search is performed afterward. This procedure does
not establish stability across initialization seeds, validation splits, or new
datasets, nor an optimum among other architectures and training algorithms.

The plan and freeze were saved locally before final test evaluation. The public
report was published afterward; this is recorded session chronology, not an
external preregistration.

## Compact scoring remains a proposal

The submission uses `sutro-affine-v4/0.1`, a submission-owned compact format
whose generic scorer is unchanged from the earlier small study. Its bounded
loops and affine operands mechanically determine every primitive and
per-address access multiplicity. It introduces no free matrix operation or
user-supplied cost certificate. The benchmark still needs to accept this
representation and its restricted verifier.

The complete medium program is not numerically expanded instruction by
instruction. Small expanded programs compare parameters, predictions, costs,
and per-address counts; a legacy small regression compares the canonical
program hash and exact costs. Those checks support the generalized lowering
without proving full semantic equivalence for every possible input.

The separate verifier checks ordered reductions on multiple matrix shapes, one
complete epoch on the full training set, and final inference. A fresh complete
learner run checks all retained parameter, score, and prediction bits, including
an archive containing only the three allowed inputs. The full training run uses
the same optimized CPU reduction implementation; it is not an independent
ordered replay of every epoch. The GPU checks complete learned parameters and
all scores and predictions against host-only CPU oracles. These checks do not
amount to a numerical replay of all expanded IL instructions.

## Arithmetic, tape, constants, and area

The submission inherits the earlier attempt's declared binary32 rounding,
unfused multiply/add operations, free tape operations, and fixed half-diamond
placement under the pinned v4 model. How the ISA specifies this exact FP32
interpretation remains a benchmark policy question.

Every scratch allocation is explicitly initialized. Ordinary scratch accesses
are charged; tape initialization retains the declared free-tape convention. Training data
remain resident; one test query is received and processed at a time after
training. This avoids increasing the unchanged one-million-word scorer guard.
The initialized-read verifier retains its original restricted rules and
resource limits. A successful static score is not a general language-safety or
numerical-correctness proof.

Occupied scratch area assumes one square micrometre per word and is displayed
in mm² by dividing by one million. It excludes tape storage, instruction
storage, and host/device buffers outside modeled scratch memory. It is not
the enclosing rectangle, fabricated chip area, A100 die area, or GPU allocator
footprint. The GPU uses a parallel layout with all queries buffered, so its
allocation should not be inferred from the model's occupied area.

Seed-only initial parameter words are generated before execution and reset
within every task; biases are zero. This treats fixed random initialization as
program constants. Running PCG64 is excluded from the theoretical and GPU
costs. No learned weights are constants. The benchmark should decide this
initialization convention explicitly.

## GPU measurement boundaries

One complete GPU task is an initialization-graph replay, E one-epoch-graph
replays, and an inference-graph replay. It is E + 2 actual replays, not one
replay containing all epochs. Initialization transforms inputs, constructs
targets, and restores initial parameters, so every task includes fresh
training. Steady-state measurements exclude transfers, allocation,
compilation, graph capture, validation, cold start, and host CPU energy.

The GPU implementation is manually written from the learner equations; it is
not produced by an automatic IL-to-PTX compiler. CPU parameters, scores, and
predictions are verification oracles retained in host memory. They are never
copied to the GPU or used to initialize its learner. Bitwise comparisons on
the selected input support the implementation; they are not a proof for all
possible inputs.

Energy is measured through NVML and subtracts the mean of idle board-power
measurements surrounding each active interval. Raw energy and the two
one-sided baseline alternatives remain relevant because telemetry cadence,
temperature, and clock state affect this subtraction. Negative estimates are
not clipped. These are steady-state board-energy estimates, not calibrated
laboratory measurements or whole-system energy. The reported theoretical and
measured energy share units while retaining different physical assumptions
and measurement boundaries.

## Scoring time and evidence

Time to score covers restricted IL validation, placement, address histograms,
exact integer cost sums, and canonical program hashing. Program construction,
JSON loading, numerical learning, accuracy evaluation, and file output are
excluded. It therefore cannot be read as the time to verify an entire
submission. Numerical checks and CPU learning have separate scopes.

Human results use two significant figures: execution time in ms, energy in
mJ, area in mm², and static scoring time in s. Raw evidence retains exact
native units and measurements. Exact counts, target percentages, seeds,
configuration values, and hashes are preserved for reproduction rather than
rounded as display metrics.

The session export is a snapshot through its stated cutoff, and excludes
hidden reasoning, system instructions, and tool payloads. Later publication
steps may therefore be absent from that snapshot.

- [Results and reproduction](index.html)
- [Predeclared search](predeclared_plan.json)
- [CPU verification evidence](verification.json)
- [Generalized IL tests](model-ir-validation.json)
