# Model scoring method and declared conventions

This submission implements the **single-core-with-tape** model and its **v4**
instruction set as documented at upstream commit
`26abcca402de647381d31286d42dfbb7a001763d`. `score_v4.py` is a submission-owned
interpreter and generator, not an official benchmark scorer. Its score is exact
for the published scratch-access formulas and the numerical, serialization, and
area conventions stated here.

## Algorithm and instruction stream

For each test row in input order, the program scans all 600 training rows in
training order. A candidate distance starts at binary32 zero. For pixels 0
through 8 in row-major order, separate `sub`, `mul`, and `add` instructions
compute `(test_pixel - training_pixel)^2` and add it to the accumulator. Each
arithmetic instruction rounds to IEEE 754 binary32, round-to-nearest ties-to-even;
there is no fused multiply-add and no reduction reassociation. All canonical
pixels are finite and in `[0, 1]`.

The first candidate initializes the best distance and label using two `copy`
instructions. Each later candidate uses a strict `cmp lt` and two `select`
instructions. Because replacement requires strictly smaller distance, exact
distance ties retain the lowest training-row index. Both value operands of
every `select` are read and charged, including the unselected operand. The
algorithm has no fitted parameters and performs no training computation beyond
receiving and retaining the supplied training data.

v4 has no branch, loop, indirect-load, or register operations. Python loops in
`program()` are code-generation loops only: they expand into an entirely
straight-line, data-independent instruction stream. The optional compressed
`program.v4.gz` contains every actual v4 instruction and can be replayed by
`--replay-ir`. No executed control instructions, implicit index arithmetic, or
scratch copies are omitted. Instruction-fetch, program-store, and arithmetic
unit costs are not defined by this model; no separate charges are added for
them.

## Input and output tape convention

The benchmark specifies NPZ arrays but does **not** specify a canonical 32-bit
word tape serialization. This submission chooses:

1. 5,400 training pixels: 600 rows × 9 row-major pixels, preserving float32 bits.
2. 600 training labels: one unsigned 32-bit integer word per label, exactly
   representing the supplied int64 labels in the range 0 through 9.
3. 5,400 test pixels: 600 rows × 9 row-major pixels, preserving float32 bits.

This consumes all **11,400 words**, or 45,600 bytes. Output consists of **600
unsigned 32-bit label words**, in test-row order, or 2,400 bytes. Binary artifact
files use little-endian byte order. Endianness affects serialization only: each
machine operand is a 32-bit word.

`input_tape()` accesses only the `train_images`, `train_labels`, and
`test_images` NPZ entries. It does not access test labels, source indices, other
tiers, or extra examples. NPZ decompression and tape serialization are host-side
benchmark preparation, outside the modeled run. After tape preparation, all
modeled dataset input is explicit `recv`; all modeled output is explicit
`send`.

The model attaches tape to the processor, explicitly excludes tape transport
and the associated scratch write/read from time and energy, and specifies no
separate tape-buffer or tape-storage area charge. The complete input tape and
the output tape are not counted as scratch. During execution only one test row
is stored in scratch; the output word resides in the already-counted best-label
cell when `send` appends it to the output tape.

## Scratch and placement

The program allocates 6,014 32-bit scratch words, with a fixed one-to-one
mapping from addresses to permitted grid coordinates:

| Addresses | Purpose | Words |
| --- | --- | ---: |
| 1–9 | Current test image | 9 |
| 10 | Difference, overwritten by its square | 1 |
| 11 | Candidate distance accumulator | 1 |
| 12 | Best distance | 1 |
| 13 | Best training label and output source | 1 |
| 14 | Comparison result | 1 |
| 15–5,414 | Training pixels | 5,400 |
| 5,415–6,014 | Training labels | 600 |

Address 1 is placed at `(0, 1)`. For each hop distance `h = 1, 2, ...`, enumerate
`x = -(h-1), ..., h-1` and `y = h - abs(x)`, assigning consecutive addresses
until all allocated words are placed. All 14 frequently accessed temporary
words are within four hops. The farthest allocated word is 78 hops away.
`placement.csv` lists every coordinate and its charged read/write counts.

All training words are retained while test images are streamed. Each training
word has exactly 600 charged reads; the initial `recv` is uncharged. The
interpreter verifies read-before-write validity, fixed-address bounds, unique
coordinates, complete input consumption, and the exact output length.

The reported area is **0.0060 mm²**, rounded to two significant figures.
The exact internal convention assigns one square micrometer to each occupied
scratch word on the 1 µm grid; divide the native µm² total by 10⁶ for the mm²
display. Raw integer area totals are unchanged. Peak allocated
scratch is **6,014 words = 24,056 bytes**; all 6,014 locations become initialized.
The smallest axis-aligned rectangle around this particular placement, including
unit-cell width, is **0.012 mm²** (rounded). The occupied-cell convention is explicit
because neither the benchmark nor model defines whether leaderboard area
should include empty sites, the enclosing rectangle, processor, tapes, program
store, or routing. This is not a physical die-area estimate.

## Exact accounting and independent checks

For every charged read at hop distance `h`, the interpreter adds
`max(50, 2*h)` fJ and `max(50, 0.8*h)` ps. For every charged write it adds
`max(50, 2*h)` fJ and `max(50, 0.4*h)` ps. It charges each source occurrence
separately before overwriting the destination. Repeated operands and
source/destination aliases do not remove charges.

All energy accumulation uses integer femtojoules; time uses integer ticks of
0.2 ps. `recv` and `send` are tracked and executed, but add zero energy and
time, including their associated scratch accesses. There is no final implicit
read. The processor executes all other accesses sequentially with no overlap.

| Opcode | Count |
| --- | ---: |
| `recv` | 11,400 |
| `send` | 600 |
| `set` | 360,000 |
| `sub` | 3,240,000 |
| `mul` | 3,240,000 |
| `add` | 3,240,000 |
| `copy` | 1,200 |
| `cmp` | 359,400 |
| `select` | 718,800 |
| **Total** | **11,171,400** |

For each query, the first training row costs 86 scratch accesses and each of
the remaining 599 rows costs 93. Hence there are
`600 * (86 + 599 * 93) = 33,475,800` charged accesses. Exactly 3,600,000 are
training-word reads; the remaining 29,875,800 are hot-word reads/writes at the
50-unit floors. Independently summing the persistent-word read distances and
this hot-access count produces exactly the same full-program totals:

- **Model energy: 1,875,974,400 fJ = 1.8759744 µJ.**
- **Model time: 1,682,197,200 ps = 1.6821972 ms.**

`model-score.json` records the final-source measured time to score: **34.8991800621 seconds** on an Intel Core i9-9880H host reporting macOS 26.6.2, x86_64 Python 3.11.13 and NumPy 2.4.6. This is one retained full-program execution, not a warmed steady-state model runtime. Its timer surrounds
machine construction plus complete instruction generation, initialization
checks, instruction execution, and access-cost accumulation. It excludes file
loading, initial placement generation, independent cross-checks, artifact
writes, and optional textual IR emission. Replay measurements additionally
include text parsing and decompression; that distinction is recorded in JSON.

The full exported 220,146,130-byte text stream (11,171,400 instructions) was also parsed and replayed in **111.1639540470 seconds**, producing identical predictions, read/write counts, peak scratch, time, and energy. Its expanded-text SHA256 is `f6f657cc159160050e73d1df8ae169b313b795f3d1b1a45a72ad4f757383ef2b`; see `trace-manifest.json` and `validation.json`. The large compressed IR can be regenerated locally from the published source.

The full interpreted output is compared bit-for-bit with a separate NumPy
implementation that accumulates the nine squared differences sequentially in
float32 and uses first-index `argmin`. It also matches the root submission's
independently saved predictions. Self-tests cover the published 450 fJ / 300 ps
example in integer-arithmetic mode, free distant tape I/O, uninitialized source
rejection (including unselected operands), and equal-distance tie handling.

## Reproduction

From a checkout containing the canonical `competition-v2` MNIST dataset, with
NumPy installed:

```bash
python score_v4.py --self-test --data /path/to/mnist/data/small.npz \
  --output model-results --emit-ir model-results/program.v4.gz
python score_v4.py --data /path/to/mnist/data/small.npz \
  --replay-ir model-results/program.v4.gz --output replay-results
```

The generator reads no machine-learned weights or hidden predictions. Only the
dimensions, arithmetic instructions, fixed placement, and tape contract are
compiled into the program. The exported IR is large when expanded because v4
does not provide loops, but is reproducibly generated from the compact source.

## Remaining interpretation limits

The v4 documentation describes 32-bit words and generic `add/sub/mul/cmp`, but
does not specify numeric types, signedness, floating-point rounding,
denormals, overflow, or NaN behavior. Its integer-like hexadecimal example does
not establish the float32 semantics required here. Accordingly, these are exact
scores for a clearly declared FP32 interpretation of v4; benchmark maintainers
should confirm it before treating the entry as unconditionally comparable.

The model results exclude physical tape work, processor arithmetic costs,
instruction fetch, and physical device effects by construction. They should
not be compared directly to GPU energy as if both represented the same
end-to-end hardware measurement boundary. GPU results and idle adjustment are
reported separately by the submission.

The upstream documentation and instruction set are available at the pinned
revision:

- [Single-core-with-tape model](https://github.com/cybertronai/simplified-dally-model/blob/26abcca402de647381d31286d42dfbb7a001763d/models/single-core-with-tape/README.md)
- [Instruction set v4](https://github.com/cybertronai/simplified-dally-model/blob/26abcca402de647381d31286d42dfbb7a001763d/instruction-sets/v4/README.md)
