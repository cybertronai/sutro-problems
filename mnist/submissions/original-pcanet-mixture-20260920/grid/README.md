# PCANet K100, k8: complete spatial-grid result

The grid implementation trains on all 60,000 official MNIST training images
and predicts all 10,000 test images, scoring **9,911 / 10,000 (99.11%)**.
It meets the MNIST-original target of at most 1% test error. The complete
training-and-prediction program costs **23,000 mJ** and **1.8 × 10⁸ ms** in
the specified spatial model. These are theoretical data-movement costs;
physical GPU-board measurements remain in the [A100 report](../README.md).

September 21, 2026. Contributor: [@jurajselep](https://github.com/jurajselep).
Prepared with AI assistance.

| Quantity | Result |
| --- | ---: |
| Correct / total | 9,911 / 10,000 |
| Accuracy | 99.11% |
| Grid energy (mJ, two significant figures) | 23,000 |
| Grid elapsed time (ms, two significant figures) | 1.8 × 10⁸ |
| Word-node hops / energy in fJ | 23,317,800,433,846,862 |
| Cycles at 1 ns per cycle | 181,126,315,554,952 |
| Executed instructions, including tape lowering | 4,315,182,031,106 |
| Peak allocated scratch (bytes) | 1,253,845,632 |
| Instruction-issuing processors | 250 |
| Arithmetic processors / simultaneous instructions | 1 / 1 |
| Host time to score (s, two significant figures) | 200 |

The exact values in [placed-score.json](placed-score.json) are
**23,317.800433846864 mJ**, **181,126,315.554952 ms**, and
**195.99277520296164 s** to score. The modeled runtime is approximately
50.3 hours because the submitted schedule serializes every access. It is a
valid conservative schedule, with no claim of optimal placement or runtime.
CPU numerical validation took 549.633 s; this is distinct from both modeled
grid time and host time to compute the grid score.

## Learner and numerical qualification

The port preserves the submitted nine-block PCANet architecture: 7×7 kernels,
8 first-stage filters, 5 second-stage filters, nine 14×14 blocks at stride 7,
2,304 features, K=100 PCA dimensions, and eight mixtures per class. It uses
the same filter-training samples, eight k-means passes, eight EM updates and
final refit, covariance blending 0.5, class shrinkage 0.05, and ridge 0.0001.
Input images retain official row order and are FP32 pixels divided by 255.
Every run learns the filter banks, feature PCA and class mixtures afresh.

The grid uses ordered FP32 covariance and reductions, orthonormal subspace
iteration (128 rounds for filters, 256 for feature PCA), Gauss–Jordan inverses,
and explicit log/exp/sqrt primitives. The A100 uses library eigensolvers,
FP64 filter covariance and CUDA reductions. The grid's **99.11%** is therefore
reported separately from the A100's **99.06%**: **14 of 10,000 predictions
differ**. The same high-level hyperparameters do not imply bitwise equivalence.

Per-class initial mixture selection reproduces the first eight entries of
CPU `torch.randperm` with seeds 900–909. The ISA implementation uses public
MT19937 seed bits, exact binary remainder and a sparse Fisher–Yates swap map;
masked gathers select class-relative rows. No training-derived constants,
pretrained state or test labels enter the program. Empty clusters retain their
previous centroid unless their count exceeds one. Responsibility epsilon is
1e-9; the explicit exponential saturates at a delta of 128, where FP32 exp
underflows, and uses ten squarings. Single-member bag score standardization
is omitted.

[frozen-protocol.json](frozen-protocol.json) records source-based corrections
and source hashes before any full-run test accuracy was read. Hyperparameters
were not tuned against this grid result. The inherited model's historical
test-set feedback remains a limitation; the official test split is not a new
held-out evaluation. Frozen program metadata says “experimental; numeric
qualification required” because it predates execution. The subsequent
[qualification report](verification.json) records **150 passed checks**,
including exact canonical input tape, source/program hashes, all 58 finite
scratch regions, output labels and class-score argmax.

Additional executable checks cover:

- A reduced version of the corrected program: all 22,244 memory words agree
  bitwise between the serial C++ executor, parallel C++ executor and Python
  ISA interpreter, on dense and sparse/imbalanced fixtures.
- 140 seed/class-size cases matching actual PyTorch 2.5.1 CPU `randperm`, in
  [rng-verification.json](rng-verification.json).
- Exact reduced-program cost parity between the bounded-memory scorer and
  the unchanged shared scorer, in [scorer-parity.json](conformance/scorer-parity.json).
- Independent saved-prediction accuracy and source/evidence checks, in
  [saved-verification.json](saved-verification.json).

The CPU executor parallelizes independent images and classes using private
host scratch, preserves each FP32 instruction's order and disables contraction,
fast math and reassociation. These host copies and threads accelerate validation;
they are not part of the separately scored serialized grid schedule.

## Model, placement and accounting

The score pins spatial-model revision
`01a0bd5e0d2564825b0f53dd766f763c82dbc7c0`:
[spatial-computer specification](https://github.com/cybertronai/simplified-dally-model/blob/01a0bd5e0d2564825b0f53dd766f763c82dbc7c0/models/spatial-computer/README.md)
and [instruction set v4](https://github.com/cybertronai/simplified-dally-model/blob/01a0bd5e0d2564825b0f53dd766f763c82dbc7c0/instruction-sets/v4/README.md).
Words are 32 bits; floating arithmetic is FP32 and predicted labels are uint32.
One word-node hop costs 1 fJ, and one cycle is 1 ns.

Regions are ordered by descending exact ordinary-read/write accesses per word,
with region name breaking ties. This placement depends only on the program,
not on inputs or predictions. [The placement record](placed-program.json.placement.json)
retains both region orders and densities. Logical instructions, region names,
sizes and constants are identical before and after placement; the verifier
checks this fixed address bijection. Numeric execution uses the original
layout and cost scoring uses the placed layout.

Addresses fill legal scratch cells in tiles ordered by Manhattan distance from
arithmetic processor P(125,0), then row and column. Cells within each tile are
ordered by local distance and coordinates. Every bottom-row tile reserves one
tape-stage cell at local (64,31). The allocation uses 25,510 memory tiles,
313,461,158 program words and 250 stage words, within the 384,000,000-word
capacity. [shared/score.py](shared/score.py) defines the exact injective mapping.

Normal instructions run on P(125,0), reading their sources then writing their
destination. Every access completes before the next begins. Remote requests
route horizontally then vertically, with responses retracing the path. This
global serialization eliminates network contention. For mesh distance L and
local cell distance d, each scratch access costs
`256*L + max(50, 2*d)` fJ. Local access takes one cycle; remote reads take
`2*L+1` cycles and remote writes `L+2` cycles. Tape accesses and explicit copies
between stage cells and program memory are charged separately.

The input tape contains 47,040,000 training pixel words, 7,840,000 query pixel
words and 60,000 training labels: **54,940,000 words**, striped cyclically over
250 ports. The output tape contains 10,000 uint32 labels, 40 per port. The
scorer lowers each receive/send into tape-stage access and an explicit copy.
It prices initialization, filter learning, all feature extraction, PCA,
mixture fitting and prediction. It excludes IDX download/decoding and pixel
normalization before tape input. Arithmetic energy, instruction fetch,
leakage and host/platform energy are outside the spatial model's cost rules.

Exact affine access counts, integer dot products and tape costs produce the
reported hops and cycles; no A100 accuracy or partial frontend/head cost is
substituted. The bounded-memory wrapper vectorizes source-free initialization,
avoids cached histograms and checks integer overflow. Scoring time includes
schema/address/initialization checks, placement, histograms, cost accumulation,
port counts and hashing. It excludes generation, JSON/file I/O, region-density
selection and numerical execution.

## Reproduce

Use Python 3.11, GCC 13 with OpenMP, and a 64 GB RAM machine for full scoring.
No GPU or cloud account is needed. The measured host used Python 3.11.15,
NumPy 2.4.6, GCC 13.3.0 and an AMD Ryzen 9 9950X3D2; details are in
[source-provenance.json](source-provenance.json) and
[build.json](full-execution/build.json). Portable checks also pass with
NumPy 2.1.2. Full compilation plus execution took 12m58s on that host, with
approximately 13.64 GiB peak host RSS. Allow several GB of output disk space.

From this `grid` directory, verify the archived result without a GPU or C++
compiler (canonical MNIST files download automatically if absent):

```sh
python -m pip install -r requirements.txt
python verify_saved.py --raw ../raw
```

Recompute the complete theoretical grid score:

```sh
mkdir -p generated
python score_candidate.py placed-program.json.gz generated/grid-score.json
```

Run fresh training and prediction, then check the full memory dump and exact
input tape without overwriting archived evidence:

```sh
python parallel_compiler.py program.json.gz generated/run --raw ../raw --data-module ..
python verify_fresh.py --execution-dir generated/run --raw ../raw --output generated/verification.json --a100-predictions ../evidence/rerun/predictions-k100.npy
```

Check executor/interpreter and seeded-rank conformance, and scorer parity:

```sh
python check_conformance.py --output generated/conformance
python score_candidate.py conformance/program.json.gz generated/reduced-score.json --compare-original
```

Optionally install PyTorch 2.5.1 CPU and add `--torch-seeds` to the conformance
command to repeat the external permutation comparison. Without it, the helper
checks the actual ISA against an independent integer Fisher–Yates reference.

Regenerate the exact frozen logical program and deterministic region placement:

```sh
python build_seeded.py --output generated/program.json.gz
cmp program.json.gz generated/program.json.gz
python place_candidate.py program.json.gz generated/placed-program.json.gz
cmp placed-program.json.gz generated/placed-program.json.gz
```

The package includes the generators, frozen programs, scorer, executors,
prediction and class-score arrays, hashes of every final region, small complete
conformance fixtures and raw logs. The 1.25 GB full memory dump, full input tape,
generated C++ and compiled library are regenerated by the commands above.
The lightweight saved-evidence checker does not rerun learning or recompute
histograms; recorded finite checks for non-score regions rely on the archived
full-run verification. Imported sources are pinned in the provenance record;
the four files in `shared/` retain their original shared implementation bytes.
