# MNIST-medium: validated spatial-grid CG pair

The frozen spatial implementation independently meets the **2% error target**:
**107,917 / 110,000 correct**, or **98.11% ± 0.14 percentage points**
across 11 datasets. Its mean error is **1.8936364%**. All saved scores and
checked intermediate arrays are finite.

The GPU and spatial implementations have separate accuracy results:

| Implementation | Accuracy, mean ± sample SD | Mean error | Qualification |
|---|---:|---:|---:|
| A100, independently reproduced on two hosts | 98.12% ± 0.15 pp | 1.88% | 107,932 / 110,000 |
| Spatial grid, ordered FP32 implementation | 98.11% ± 0.14 pp | 1.8936364% | 107,917 / 110,000 |

Both pass the exact requirement of at least **107,800 correct**. The leaderboard identifies each implementation’s accuracy separately.

## Arithmetic and qualification

The architecture and parameters remain fixed: 512 random 3×3 convolutional
filters, ReLU and 3×3 average pooling, centered feature ridge plus RBF kernel
ridge, 300 Jacobi-preconditioned CG iterations per member, and standardized
scores summed before the first strict maximum.

The grid uses explicit FP32 primitives. Each reduction follows ascending
source indices with round-to-nearest, ties-to-even; FMA, reassociation and
subnormal flushing are disabled. Square roots use six Newton steps, arcsine
uses a ninth-degree polynomial with complementary-angle reduction, and the
exponential uses a quartic polynomial followed by eight squarings. Kernel
distances accumulate the 81 squared pixel differences directly. These
operations differ numerically from the A100 tensor implementation, including
the square root used in score standardization. Pixel clamping and kernel
Jacobi preconditioning are explicit in the lowering.

The grid source and parameters were frozen before qualification. Every draw
uses fresh training on the same canonical 10,000-train/10,000-query split as
the A100 evidence, with seeds 2026091600–2026091610. The prediction runner never
receives query labels. All 110,000 predictions were saved before an independent
verifier reconstructed the labels and checked counts, hashes and argmaxes.

## Numerical validation

A generic compiled executor translates each affine IL instruction directly
and was checked against the original Python interpreter, including instruction
edge cases. The faster C++ specialization matches all 25 compared numerical
regions plus the output tape bitwise across three reduced shapes, including the final
CG state. One synthetic fixture also confirms identical nonfinite CG behavior;
both other fixtures and every full qualification draw are finite.

An independent **full-size** execution (10,000 training examples, 10,000
queries, 512 filters, 300 iterations) also matches all 10,000 predictions and
all 25 checked final floating-point regions: **316,113,456 words**, all finite.
It uses the first frozen draw and reads no query labels. The
[full conformance report](evidence/conformance/full-draw.json) binds the exact
program, source, input, reference-result and numerical-region hashes.

The specialization parallelizes independent work and caches repeated
transforms while preserving every reduction's numerical order. Its mean CPU
execution time, **3.7 seconds**, is validation time; the spatial runtime is
computed separately from the declared grid schedule.

## Spatial energy and runtime scope

| Modeled quantity | Result |
| --- | ---: |
| Energy | **1,500 mJ** |
| Runtime | **1.2 × 10⁷ ms** (about 3.5 hours) |
| Peak scratch | **1,264,557,504 bytes** |

The exact counts are 1,533,903,647,390,828 word-node hops and
12,423,113,281,675 model cycles. The full score retains unrounded energy,
runtime and each cost component.
Computing the score took **510 s** on server-v80 (Ryzen 9 9950X3D2,
Python 3.11.15, NumPy 2.1.2), with 25,408,872,448 bytes peak host RAM.
Numerical executors were built with GCC 13.3.0; the specialization used 16 threads.
Displayed energy and runtime use two significant figures.

The score covers a complete fresh training-and-prediction run, including
scratch initialization, filter and constant initialization, target encoding,
input receipt, both fits, prediction and output. Dataset selection and 9×9
area resizing supply the input tape and are outside this scope.

- **Model:** spatial-computer pitch128 / ISA v4, revision
  `01a0bd5e0d2564825b0f53dd766f763c82dbc7c0`.
- **Placement:** 316,139,126 scratch words plus 250 reserved tape-stage words,
  totaling **1,264,557,504 bytes** across 25,728 memory tiles; fixed addresses
  map to distinct legal cells.
- **Tape:** 810,000 training-pixel words, 10,000 label words and 810,000 query-pixel
  words enter; 10,000 prediction words leave. Word k uses port k modulo 250.
- **Schedule:** arithmetic executes on P(125,0); bottom processors handle tape
  traffic. Across 250 instruction-issuing processors, at most one instruction
  or access is active at a time. Blocking accesses and stage copies are charged.
- **Energy:** the standard charges scratch/tape movement and exact word-node
  hops. Arithmetic, instruction fetch, idle/leakage power and off-chip host
  storage or transport are outside its energy model.

These are exact modeled costs for the declared globally serialized schedule.
The scorer wrapper discards operand histograms after counting them to reduce
host memory use. Its reduced checks match every invariant field of the shared
scorer; it preserves validation, placement, tape handling and all cost terms.

## Reproduce and verify

Run from the repository root with Python 3.11 and NumPy 2.1.2. Only a fresh
numerical execution needs a C++17 compiler with OpenMP (`g++`); verification of
saved predictions needs no compiler or GPU. `OPENBLAS_CORETYPE=Haswell` pins
area-resize arithmetic to the A100 input hashes on x86 hosts.

```bash
SUB=mnist/submissions/medium-cg-pair-20260916
GRID="$SUB/grid"
SHARED="$PWD/mnist/submissions/grid-mlp-scoring-20260912"
export PYTHONPATH="$PWD:$SHARED"
export OPENBLAS_CORETYPE=Haswell
python3.11 -m venv .venv-grid
.venv-grid/bin/pip install -r "$GRID/requirements.txt"
PY="$PWD/.venv-grid/bin/python"
RAW=/tmp/mnist-raw
"$PY" - <<'PYDATA'
from pathlib import Path
from mnist.code import data
for kind in ("train_images", "train_labels"):
    data.download_source(Path("/tmp/mnist-raw"), *data.SOURCES[kind])
PYDATA
"$PY" "$GRID/verify.py" --raw-dir "$RAW"
(cd "$SUB" && sha256sum -c SHA256SUMS)
```

For fresh predictions, build the specialization, validate its reduced
numerics against the independent interpreter, and run all eleven draws.
The runner inputs contain training labels only; the verification step reads
query labels after every prediction has been saved.

```bash
RUN=/tmp/cg-grid-run
mkdir -p "$RUN"
"$PY" "$GRID/spatial_program.py" --output "$RUN/program.json"
"$PY" "$GRID/prepare_data.py" --raw-dir "$RAW" \
  --a100-record "$SUB/evidence/a100/croatia.json.gz" --output-dir "$RUN/inputs"
"$PY" "$GRID/executor/executor.py"
"$PY" "$GRID/il_executor/validate.py" reference "$RUN/validation" \
  --affine-dir "$SHARED" --grid-dir "$GRID"
"$PY" "$GRID/il_executor/validate.py" compare "$RUN/validation" --affine-dir "$SHARED"
"$PY" "$GRID/executor/parity.py" "$RUN/validation/grid" \
  --affine-dir "$SHARED" --output "$RUN/parity.json"
"$PY" "$GRID/evaluate.py" --inputs-dir "$RUN/inputs" --output-dir "$RUN/evaluation" \
  --executor-dir "$GRID/executor" --protocol "$GRID/protocol.json" \
  --program "$RUN/program.json" --parity-report "$RUN/parity.json" --threads 16
"$PY" "$GRID/verify_accuracy.py" --raw-dir "$RAW" --results-dir "$RUN/evaluation"
```

To reproduce the independent full-size conformance check after fresh evaluation:

```bash
"$PY" "$GRID/il_executor/verify_full_draw.py" "$RUN" --affine-dir "$SHARED" \
  --output-dir "$RUN/full-conformance"
```

The full static scorer needs a host with **40 GB available RAM**. Its large
host arrays are separate from the modeled scratch memory. It validates
instruction legality and initialization, places every address, and recomputes
all access counts, tape traffic, hop energy and blocking latency:

```bash
"$PY" "$GRID/scorer/score_program.py" "$RUN/program.json" \
  --shared-scorer "$SHARED/score.py" --output "$RUN/grid-score.json"
```

Scores are compressed losslessly as `.npy.gz`; original decoded file hashes
remain in the frozen draw records. The verifier reads them directly.

Evidence: [accuracy](evidence/accuracy.json), [prediction freeze](evidence/prediction_freeze.json),
[run manifest](evidence/run_manifest.json), [numerical checks](evidence/conformance/),
[complete modeled score](evidence/grid-score.json), and [verification](evidence/verification.json).
