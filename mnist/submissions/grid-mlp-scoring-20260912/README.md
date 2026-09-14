# Exact serialized spatial-computer scores for the two MLP submissions

This directory supplies reproducible grid energy and time for the current
MNIST-small 60% and MNIST-medium 96% submissions. It lowers their complete
ordered-FP32 training and prediction to instruction set v4 and scores a legal
**globally serialized schedule**. It is a conservative baseline: just one access
is outstanding across the machine, including tape staging and remote accesses.
It makes no claim about the best attainable parallel grid performance.

| Configuration | Grid energy (mJ) | Grid time (ms) | Allocated scratch (bytes) | Time to score (s) |
| --- | ---: | ---: | ---: | ---: |
| Small: 9→32→10, 300 epochs, batch 25, 1,000 train / 1,000 test | 0.24 | 3.1 × 10³ | 91,116 | 0.14 |
| Medium: 81→512→10, 200 epochs, batch 25, 10,000 train / 10,000 test | 4.4 × 10² | 3.7 × 10⁶ | 3,973,260 | 5.1 |

These are costs for **one full training-and-prediction dataset**, not the sum
of 11 evaluation datasets. The trace and placement do not depend on the input
values. Changing the declared learner seed changes initialization literal bits
without changing the number or address of any access, so all 11 runs of a fixed
configuration have these same grid costs. The linked submission reports contain
the separate accuracy and measured A100 evidence. No A100 measurement is used
to infer a grid score.

## Model, representation, and scope

The machine is
[spatial-computer pitch 128 / ISA v4 at revision 01a0bd5e0d2564825b0f53dd766f763c82dbc7c0](https://github.com/cybertronai/simplified-dally-model/blob/01a0bd5e0d2564825b0f53dd766f763c82dbc7c0/models/spatial-computer/README.md).
This revision was verified from the repository's `main` on September 12, 2026.
It has 250 × 125 processor tiles, each with 12,288 addressable 32-bit scratch
words; bottom-edge tapes are charged and are striped over 250 ports.

The program includes scratch initialization, seed-only initial parameter
literals, receiving every input, pixel normalization `x * 4 - 0.5`, one-hot
targets, every minibatch of training, every prediction, and all output labels.
The supplied inputs are already 3 × 3 or 9 × 9 FP32 images; dataset sampling and
downsampling define the workload inputs and are outside this training and
prediction scope. No trained values, test labels, or input-dependent immediates
enter the generated program.

Multiplication, addition and subtraction round separately to IEEE FP32 with
round-to-nearest, ties-to-even. There is no fused multiply-add. Reductions run
in ascending index order. ReLU and first-index argmax use strict comparisons.
Labels and output class indices are raw uint32 words. Label equality and class
constants agree for the complete allowed range 0–9. Conditional selection copies
one raw word; both candidate source operands are read and charged. The model
charges data movement rather than arithmetic or instruction fetch.

`model_ir.py` generalizes the historical affine MLP generator. `affine.py`
retains its bounded affine-loop validation, definite-initialization proof and
exact address-use histograms, while replacing the former single-core revision
and placement contract. `score.py` supplies the new physical placement, tape
lowering and serialized cycle/energy accounting. Historical free-tape,
single-core energy/time or area scores are not reused.

## Fixed tapes and complete input consumption

The **same workload tape layout** is fixed for both configurations before
scoring any solution:

1. All training image pixels, row-major FP32 bit patterns.
2. All training labels, one raw uint32 word per label.
3. All test image pixels, row-major FP32 bit patterns.

Global input word `k` is input tape word `floor(k/250)` at port `k mod 250`.
There is no packing of multiple pixels or labels into one word. Output word
`q` is the raw uint32 predicted label for test image `q`, on output port
`q mod 250` at index `floor(q/250)`. No test labels are on any input tape.
Small consumes 19,000 words and emits 1,000 words. Medium consumes 1,630,000
words and emits 10,000 words. The exact per-port counts are in each score JSON.

Training images, labels and one-hot targets remain in scratch across epochs;
tapes are never rewound or used as memory. Test queries stream through one
reused `D`-word scratch buffer after training, with one output emitted per query.

## Legal processor and memory placement

All arithmetic executes on `P(125,0)`, whose core is at `(64,64)`. Each bottom
processor `P(i,0)` also issues its own tape instructions. There are 250
instruction-issuing processors, with at most one executing at any instant.
Other owning tiles service remote scratch requests without running arithmetic.

Within a tile a cell `(u,v)` means coordinate
`(-16000 + 128*i + u, 128*j + 1 + v)`. The core is at local `(64,63)`.
Cells with both `32 ≤ u ≤ 95` and `32 ≤ v ≤ 95` are excluded. Sort all remaining
cells by `(abs(u-64)+abs(v-63), u, v)`. Reserve the first cell, `(64,31)`, as the
stage in every bottom-row tile; it is 32 node hops from its core.

Order tiles by `(abs(i-125)+j, j, i)`. Fill the ordered legal cells of each tile
with consecutive program addresses, skipping each bottom stage. Region order
and address counts are declared in the program JSON. This is a fixed injective
mapping with no free migration or replicas. No scratch access uses the reserved
core/router square. `placement()` reconstructs the complete map; its coordinate
arrays are hashed in the score result.

Small has 22,529 program words plus 250 stages, 22,779 words total. Medium has
993,065 program words plus 250 stages, 993,315 words total. Every used tile stays
within 12,288 words. Small uses 250 memory tiles including mostly empty tape
stage tiles; medium uses 314. All program words are explicitly zeroed before
use, then initial constants are set and charged. A stage is written by `recv`
or `copy` before its first read. The reported peak allocated bytes include all
stages, even when a particular stage is temporarily unused.

## Complete schedule and charge calculation

Expand the affine loops in their declared iteration order. Execute one
instruction at a time, completing each source access in listed order followed
by its destination write before issuing the next instruction. A repeated source
is read again. Lower the original tape operations as follows:

```text
kth recv dst:
    P(k mod 250,0): recv stage[k mod 250]
    P(125,0):       copy dst, stage[k mod 250]

qth send src:
    P(125,0):       copy stage[q mod 250], src
    P(q mod 250,0): send stage[q mod 250]
```

This preserves each fixed tape's order and delivers each input word exactly
once. Every copy source and destination incurs an access charge. The schedule
does not encode data in dependencies: all loop bounds, processor choices,
addresses and starts depend only on configuration and earlier completion times.

For cell distance `d` from its owner and mesh distance `L` from `P(125,0)`:

| Access | Energy (fJ) | Cycles |
| --- | ---: | ---: |
| Local read or write | `max(50,2d)` | 1 |
| Remote read | `256L + max(50,2d)` | `2L+1` |
| Remote write | `256L + max(50,2d)` | `L+2` |
| Bottom `recv` or `send` to a stage at distance 32 | 128 | 2 |

A read request routes horizontally and then vertically, takes one local scratch
cycle, and returns along the reverse path. A remote write sends address and data
as consecutive words along the same path, then performs its local write.
If a path has `L` links and begins at boundary `t`, a read request uses link `r`
at `t+r`, local service at `t+L`, and return link `r` at `t+L+1+r`. A write uses
path link `r` for its address at `t+r` and data at `t+r+1`, followed by owner
service at `t+L+1`. Next-link transfers begin at the arrival boundary.

Global serialization ensures that other accesses cannot contend for a scratch
slot or directed link. The address/data write pipeline fits one word per
incoming link plus the owner's pending request slot; a forwarding FIFO may
dequeue and accept its replacement on the same boundary. Nothing uses an
unbounded queue. Memory effects and output append complete before any dependent
operation begins. The fixed schedule therefore has no races or deadlocks.

`scheduled_instructions()` lazily emits the concrete per-processor v4 trace.
`scheduled_accesses()` emits its exact access start and finish cycles, addresses
and coordinates. A complete schedule can be reconstructed from these generators
without storing billions of events. The cost scorer instead sums exact affine
address histograms. Since all accesses serialize, summing their elapsed cycles
is exactly the run's elapsed time; it is not a sum of concurrent processor times.
No uncontended-latency assumption is applied to competing traffic.

Energy is the sum of all accesses and tape movement. With this legal placement,
all local distances are at least 32, so the 50 fJ floor is inactive. Consequently
the integer fJ score also equals the total word-node hops at 1 fJ per hop. One
cycle is 1 ns. Divide integer energy by `10^12` for mJ and cycles by `10^6` for
ms. No old single-core area conversion is used.

Exact small score: **242,990,182,992 fJ; 3,073,508,456 cycles**.
Exact medium score: **437,180,228,679,904 fJ; 3,740,014,858,820 cycles**.

## Verification and reproduction

The five checked-in tests pass. They exercise published access examples, legal
placement/capacity, tape wrap across 250 ports and remote memory, and rejection
of uninitialized sources. A complete tiny MLP is executed instruction by
instruction and matches an independent ordered NumPy learner bitwise on every
final parameter word and every prediction. A separate expanded wire-event
reference checks scratch-slot and directed-link capacity and reproduces the
scorer's exact energy and cycles. Full-size CPU/A100 numerical checks are in the
two submission reports; the grid scorer itself does not run the full numeric
training computation.

From the repository root, with Python and `requirements.txt` installed:

```bash
python mnist/submissions/grid-mlp-scoring-20260912/test_score.py
python mnist/submissions/grid-mlp-scoring-20260912/score.py \
  --features 9 --width 32 --epochs 300 --batch 25 \
  --n-train 1000 --n-test 1000 --learning-rate .2 --seed 101 \
  --output mnist/submissions/grid-mlp-scoring-20260912/small60
python mnist/submissions/grid-mlp-scoring-20260912/score.py \
  --features 81 --width 512 --epochs 200 --batch 25 \
  --n-train 10000 --n-test 10000 --learning-rate .1 --seed 101 \
  --output mnist/submissions/grid-mlp-scoring-20260912/medium96
```

Alternatively prefix each script command with
`uvx --with numpy==2.2.6 python`. The saved score includes software versions,
source and program hashes, exact component totals, initialization proof counts,
memory capacity and host scoring duration. A score covers program validation,
placement, histograms, exact integer totals and hashes; it excludes JSON/file
I/O, generating the program, and numerical accuracy execution.

- [Small submission and accuracy/A100 evidence](../small60-grid-20260912/README.md)
- [Medium submission and accuracy/A100 evidence](../medium96-grid-20260912/README.md)
- [Small exact score](small60/grid-score.json) · [Small generated program](small60/program.spatial.json)
- [Medium exact score](medium96/grid-score.json) · [Medium generated program](medium96/program.spatial.json)
- [Verification results](test-results.json) · [Verifier source](test_score.py)
- [Scorer and schedule generator](score.py) · [MLP generator](model_ir.py) · [Affine validator](affine.py)

## Adam optimizer path

`score.py` accepts `--optimizer adam --nr-iterations K` (default sgd, K=12).
The Adam path uses the same affine/distance pricing with `m`/`v` state, per-step
bias correction and a Newton-Raphson square root built from existing ops
(`div`, `add`, `mul`). The SGD path is unchanged and reproduces the original
scores exactly. Example:

```sh
uv run --with numpy==2.2.6 python score.py --features 9 --width 32 \
  --epochs 100 --batch 25 --n-train 1000 --n-test 1000 --learning-rate 0.2 \
  --seed 101 --optimizer adam --nr-iterations 8 \
  --output small-adam-nr-k8-20260912
```
