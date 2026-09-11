# Compact affine v4 prototype

This experiment proposes a small intermediate language for programs whose complete instruction stream is fixed before the input values are known. It stores nested bounded loops, affine scratch addresses, and ordinary v4 instructions. The loops compress the description; expanding them produces a straight-line program. They do not add a new machine instruction or grant a free matrix multiplication, training update, or algorithm certificate.

This is a submission-owned research prototype, pinned to the same model revision and arithmetic conventions as the existing 1NN attempt. Acceptance of this representation as an official submission format still requires a benchmark decision.

## Language

A JSON document declares ordered, non-overlapping scratch regions, a fixed placement rule, and a program body. Regions occupy consecutive word addresses beginning at 1. Each word occupies a unique cell in the existing half-diamond placement, filling increasing Manhattan-distance shells and then increasing x within a shell.

An address is a named region plus an affine offset. For example, `ref("weights", 0, feature=32, hidden=1)` denotes `weights[32*feature + hidden]`. Coefficients and offsets are signed integer literals; the verifier checks the complete finite range. Loop counts and starts, and affine address coefficients, are compile-time integer constants. There are no data-dependent loop bounds, branches, address indirection, arbitrary expressions, Python evaluation, or externally supplied score certificates.

A short example in the supplied builder API is:

```python
make_program(
    [("x", 9), ("weights", 9), ("acc", 1), ("product", 1)],
    [loop("i", 9, [ins("recv", ref("x", i=1))]),
     loop("i", 9, [ins("recv", ref("weights", i=1))]),
     ins("set", ref("acc"), 0),
     loop("i", 9, [
         ins("mul", ref("product"), ref("x", i=1), ref("weights", i=1)),
         ins("add", ref("acc"), ref("acc"), ref("product")),
     ]),
     ins("send", ref("acc"))]
)
```

The two loops above read a dot product's input and weights from the tape, and the third loop contains nine separate multiplies and adds. Any training or parameter initialization must likewise be represented by ordinary priced instructions. The JSON form of `ref` is `{"region":"x","offset":0,"coefficients":{"i":1}}`; a loop is `{"loop":"i","start":0,"count":9,"body":[...]}`. A primitive is, for example, `{"op":"add","dst":...,"src":[..., ...]}`. `set` uses a literal unsigned 32-bit `imm` field instead of `src`.

The supported leaves are `set`, `recv`, `send`, `copy`, `add`, `sub`, `mul`, `cmp`, and `select`. `cmp` supports an optional literal predicate `lt`, `le`, `eq`, `ne`, `gt`, or `ge`, defaulting to `lt`. The declared arithmetic convention is binary32 with round-to-nearest, ties-to-even after each arithmetic instruction, no fused multiply-add, FP32 comparisons, and raw-word copies/selections. The scoring pass does not execute that arithmetic; the semantic verifier is a separate component. `expand` emits executable tuple instructions; non-default comparisons use `cmp_eq`, etc., as explicit predicate-bearing comparison tuples for the semantic runner.

## Exact counting without instruction-by-instruction scoring

For every primitive leaf, the scorer knows its enclosing loop domains. Their Cartesian-product size gives the exact dynamic instruction multiplicity. Each source and destination is processed independently, preserving repeated reads when source addresses alias one another or the destination. All three sources of a `select` are read and charged, including the unchosen value.

For an affine operand, each referenced loop contributes an arithmetic-progression histogram of addresses. Discrete convolution combines these histograms, including overlapping addresses and negative strides. A sliding-window implementation performs the convolution without enumerating the Cartesian product. Loops absent from the address merely multiply its access counts. The resulting vectors specify exact read/write multiplicities at each concrete scratch address; no fitted coefficient or learner-specific cost formula is accepted as input.

With `h = |x| + y` for each cell, the aggregate score is:

- Energy: `(reads + writes) * max(50, 2*h)` fJ.
- Read time: `reads * max(250, 4*h)` ticks, where one tick is 0.2 ps.
- Write time: `writes * max(250, 2*h)` ticks.
- Area: one square micrometre per declared scratch word.

These quantities are summed using exact integers. `recv` and `send` count toward program/tape length but incur no memory-access cost under the pinned v4 tape convention. `recv` initializes its destination; `send` must have an initialized source. Scalar operations and fixed placement can therefore be compared with the existing interpreter's read/write vectors directly.

Runtime scales with static program size and the spans of its affine address distributions, rather than the number of dynamic instructions. Increasing a training epoch loop from 100 to 1,000 has little effect on static scoring time when that loop changes no addresses. The training component of its instruction counts and costs still increases tenfold. This improvement does not make numerical training, algorithm validation, or test accuracy evaluation free.

## Program validity

The parser checks opcodes, operand arity, immediate ranges, unique regions, bound and unshadowed loop variables, and every operand's minimum and maximum reachable address. The region layout gives each allocated word one unique permitted physical location. Signed-int64 histogram overflow is prevented by rejecting programs whose conservative access bound exceeds the prototype limit before aggregation. Prototype allocation is restricted to one million scratch words, comfortably inside the model's physical placement bounds.

A separate definite-write check follows source-before-destination order and rejects any uninitialized source, including unchosen `select` operands. Initialized words remain initialized. Once all allocated words have been written, the already-completed bounds check suffices for all remaining accesses. A loop whose variable appears in no descendant address needs only its first iteration for this proof: subsequent iterations repeat the identical address sequence with a superset of initialized words. Address-dependent loops are otherwise traversed for initialization checking. A two-million-node proof budget causes explicit rejection rather than assuming validity. Explicitly zeroing work buffers is allowed and is charged; the scorer never adds or ignores such operations on its own.

This is a sound, restricted verifier, not a complete theorem prover. Valid programs whose initialization proof exceeds the work budget may be rejected. There is no deallocation, mutable placement, dynamic control, indirect access, or dynamic storage reuse. Programs can still reuse fixed scratch cells, with the ordinary source/destination rules.

The score reports input/output tape lengths but does not establish that the program implements the claimed learner, lacks a hardcoded answer, obeys a benchmark's training/test separation, or produces accurate predictions. The experiment performs those checks separately. A practical official evaluator would need both cost validation and semantic/accuracy verification, along with an agreed tape and arithmetic contract.

## Validation and timing

`test_il.py` compares the complete baseline's opcode totals, scores, and every one of its 6,014 per-address read/write counts against the prior executed submission. It also compares small expanded 1NN programs instruction-for-instruction and numerically against the existing v4 interpreter; checks 50 independently enumerated affine programs with negative strides, collisions, source aliases, and selections; and exercises free tape operations, distance floors, initialization, bounds, and overflow rejection. A compact two-batch, two-epoch MLP also matches independently expanded per-address counts. All 11 tests pass; results are recorded in `il-validation.json`. A separate one-time full expansion hashes byte-for-byte identically to the original 220 MB v4 text; this identity check is recorded in `il-expansion-validation.json`.

The compact 1NN has 17 primitive leaf nodes and 24 total nodes. It describes the same 11,171,400 instructions and 33,475,800 charged accesses as the original submission. Its exact scores are unchanged. The human-facing report rounds display values; machine-readable score artifacts preserve exact integers for verification.

Static scoring, including schema, bounds and initialization checking, placement, histograms, integer cost sums, and canonical hashing, takes roughly 2.0 × 10¹⁰ ps on this host. The earlier interpreter took roughly 3.5 × 10¹³ ps while also executing every FP32 instruction and checking its inputs and outputs. These timings have different scopes: the compact result demonstrates inexpensive cost evaluation, not a measured end-to-end speedup of complete submission verification. Program loading and file output are excluded from both reported static timings. Repeated measurements and runtime metadata are recorded in `il-benchmark.json`. The MLP scaling experiment in `il-scaling.json` describes 100 million to 41 billion dynamic instructions at widths 16, 32, and 64 and epoch counts 100, 1,000, and 10,000. Median static scoring time stays about 8.0 × 10¹⁰–1.1 × 10¹¹ ps across those cases. These are scoring workloads, not additional accuracy measurements; no accuracy is claimed for these parameter combinations.

Run with Python and NumPy:

```sh
python nn_il.py --output 1nn.il.json
python il.py 1nn.il.json --output 1nn.il.score.json
python test_il.py
```

For small programs, `python il.py example.il.json --expand expanded.jsonl` materializes every leaf for an independent evaluator. Expansion remains proportional to dynamic instructions and is deliberately absent from the fast static scoring path.
