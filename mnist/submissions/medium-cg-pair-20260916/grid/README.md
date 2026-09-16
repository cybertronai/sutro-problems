# MNIST-medium: spatial-grid CG pair

The frozen grid implementation passes the **2% error target** independently:
**107,917 / 110,000 correct**, or **98.11% ± 0.14 percentage points** across
11 datasets (**1.8936364% mean error**). All saved scores and checked
intermediate arrays are finite. The [submission summary](../README.md) gives
the separately qualified A100 result and shared dataset protocol.

## Results and scope

| Modeled quantity | Result |
| --- | ---: |
| Energy | **1,500 mJ** |
| Runtime | **1.2 × 10⁷ ms** (about 3.5 hours) |
| Peak scratch | **1,264,557,504 bytes** |

Energy and runtime are shown to two significant figures. Exact costs are
**1,533.903647390828 mJ** and **12,423,113.281675 ms**, from
1,533,903,647,390,828 word-node hops and 12,423,113,281,675 model cycles.
The repository's **unmodified shared scorer** computes these costs directly
from the frozen program, with its default histogram cache.

The program covers fresh training and prediction, including initialization,
input/output tape traffic and target encoding. Dataset selection and 9×9
resizing are excluded. The **globally serialized schedule** permits at most
one instruction or access at a time. Energy follows the model's scratch/tape
movement and communication costs; arithmetic, instruction fetch, idle/leakage
power and off-chip host storage or transport are outside its energy model.

## Verify saved evidence

After the [submission setup](../README.md#verify-saved-evidence), run from the
repository root:

```bash
python "$SUB/verify.py" --all --raw-dir /tmp/mnist-raw
```

This checks the checksum manifest and saved A100/grid evidence. Quick grid
checks audit recorded totals and scorer provenance; the full scorer recomputes
per-address access counts and placement. Fresh grid scoring is a separate **128 GB RAM,
about 11-minute** operation and does not require rerunning numerical draws.

<details>
<summary>Model placement, tape and instruction accounting</summary>

- **Model:** spatial-computer pitch128 / ISA v4, revision
  `01a0bd5e0d2564825b0f53dd766f763c82dbc7c0`.
- **Placement:** 316,139,126 scratch words plus 250 reserved tape-stage words,
  totaling **1,264,557,504 bytes** across 25,728 memory tiles; fixed addresses
  map to distinct legal cells.
- **Tape:** 810,000 training-pixel words, 10,000 label words and 810,000 query-pixel
  words enter; 10,000 prediction words leave. Word k uses port k modulo 250.
- **Schedule:** arithmetic executes on P(125,0); bottom processors handle tape
  traffic. There are 250 instruction-issuing processors, with no concurrent
  instructions or accesses. Blocking accesses and stage copies are charged.

Initialization includes scratch, filters and constants. Both model fits,
score standardization, prediction and output are included. The exact score
retains every cost component, address placement and memory occupancy.

</details>

<details>
<summary>Grid arithmetic, qualification and numerical conformance</summary>

The grid uses explicit FP32 primitives. Each reduction follows ascending
source indices with round-to-nearest, ties-to-even; FMA, reassociation and
subnormal flushing are disabled. Square roots use six Newton steps, arcsine
uses a ninth-degree polynomial with complementary-angle reduction, and the
exponential uses a quartic polynomial followed by eight squarings. Kernel
distances accumulate the 81 squared pixel differences directly. These
operations differ numerically from the A100 tensor implementation, including
the square root used in score standardization. Pixel clamping and kernel
Jacobi preconditioning are explicit in the lowering.

Source and parameters were frozen before qualification. Each draw trains a
fresh learner on the same canonical 10,000-train/10,000-query split as the A100
evidence, with seeds 2026091600–2026091610. The prediction runner never receives
query labels. All 110,000 predictions were saved before an independent verifier
reconstructed labels and checked counts, hashes and argmaxes. The exact
qualification threshold is **107,800 correct**.

A generic compiled executor translates each affine IL instruction directly
and was checked against the original Python interpreter, including instruction
edge cases. The faster C++ specialization matches all 25 compared numerical
regions plus the output tape bitwise across three reduced shapes, including
final CG state. One synthetic fixture also confirms identical nonfinite CG
behavior; both other fixtures and every full qualification draw are finite.

An independent **full-size** execution (10,000 training examples, 10,000
queries, 512 filters, 300 iterations) matches all 10,000 predictions and all
25 checked final floating-point regions: **316,113,456 words**, all finite.
It uses the first frozen draw and reads no query labels. The
[full conformance report](evidence/conformance/full-draw.json) binds program,
source, input, reference-result and numerical-region hashes.

The specialization parallelizes independent work and caches repeated transforms
while preserving every reduction's numerical order. Built with GCC 13.3.0 and
run with 16 threads, its mean CPU execution time was **3.7 seconds**. This is
validation time; spatial runtime comes from the modeled schedule above.

</details>

## Reproduce

Use Python 3.11 and NumPy 2.1.2 from the
[submission setup](../README.md#verify-saved-evidence). Run commands from the
repository root. The three workflows below are independent except that full
numerical conformance uses the fresh predictions from the preceding workflow.

<details>
<summary>Recompute the full spatial score: 128 GB RAM, about 11 minutes</summary>

This reads the saved frozen program directly. It needs neither MNIST data nor
fresh predictions. The scorer validates instruction legality and initialization,
places every address, and recomputes access counts, tape traffic, hop energy
and blocking latency. Its large host arrays are separate from the modeled
scratch memory: the recorded run took **675.73 seconds** with
**89,219,280,896 bytes** peak process RAM.

The shared scorer's CLI generates MLPs. Call its existing
[`score.score(document)`](../../grid-mlp-scoring-20260912/score.py) function for
the frozen CG program; the scorer and dependencies are unmodified:

```bash
SUB=mnist/submissions/medium-cg-pair-20260916
SHARED="$PWD/mnist/submissions/grid-mlp-scoring-20260912"
PYTHONPATH="$SHARED" python - "$SUB/grid/evidence/program.json.gz" /tmp/cg-grid-score.json <<'PYSCORE'
import gzip, json, sys
from pathlib import Path
from score import score

with gzip.open(sys.argv[1], "rt") as source:
    program = json.load(source)
result = score(program)
Path(sys.argv[2]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(result["energy_mj"], result["time_ms"])
PYSCORE
```

</details>

<details>
<summary>Reproduce all eleven numerical qualification draws</summary>

This also requires a C++17 compiler with OpenMP (`g++`). Build the specialization,
validate its reduced numerics against the independent interpreter, and execute
all eleven draws. Runner inputs contain training labels only; verification
reads query labels after every prediction has been saved.

```bash
SUB=mnist/submissions/medium-cg-pair-20260916
GRID="$SUB/grid"
SHARED="$PWD/mnist/submissions/grid-mlp-scoring-20260912"
export PYTHONPATH="$PWD:$SHARED"
export OPENBLAS_CORETYPE=Haswell
RAW=/tmp/mnist-raw
RUN=/tmp/cg-grid-run
mkdir -p "$RUN"
python "$GRID/spatial_program.py" --output "$RUN/program.json"
python "$GRID/prepare_data.py" --raw-dir "$RAW" \
  --a100-record "$SUB/evidence/a100/croatia.json.gz" --output-dir "$RUN/inputs"
python "$GRID/executor/executor.py"
python "$GRID/il_executor/validate.py" reference "$RUN/validation" \
  --affine-dir "$SHARED" --grid-dir "$GRID"
python "$GRID/il_executor/validate.py" compare "$RUN/validation" --affine-dir "$SHARED"
python "$GRID/executor/parity.py" "$RUN/validation/grid" \
  --affine-dir "$SHARED" --output "$RUN/parity.json"
python "$GRID/evaluate.py" --inputs-dir "$RUN/inputs" --output-dir "$RUN/evaluation" \
  --executor-dir "$GRID/executor" --protocol "$GRID/protocol.json" \
  --program "$RUN/program.json" --parity-report "$RUN/parity.json" --threads 16
python "$GRID/verify_accuracy.py" --raw-dir "$RAW" --results-dir "$RUN/evaluation"
```

</details>

<details>
<summary>Reproduce independent full-size numerical conformance</summary>

After the fresh numerical evaluation above, compare the first complete draw
against the generic instruction-by-instruction executor:

```bash
python "$GRID/il_executor/verify_full_draw.py" "$RUN" --affine-dir "$SHARED" \
  --output-dir "$RUN/full-conformance"
```

This verifies numerical behavior, separately from the full static cost scorer.

</details>

Saved predictions and scores are losslessly packed in `evidence/outputs.npz`.
Original decoded file hashes remain in the frozen draw records; the verifier
reads the archive directly. Fresh runs write individual `.npy` files.

Evidence: [accuracy](evidence/accuracy.json), [prediction freeze](evidence/prediction_freeze.json),
[run manifest](evidence/run_manifest.json), [numerical checks](evidence/conformance/),
[complete modeled score](evidence/grid-score.json), [scoring provenance](evidence/score-run.json),
and [verification](evidence/verification.json).
