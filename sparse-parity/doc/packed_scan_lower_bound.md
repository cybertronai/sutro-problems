# Packed sparse-parity scan: lower bounds and construction

This note deliberately separates four claims that are easy to conflate:

1. a lower bound for **any** valid MASK32 circuit;
2. the unavoidable number of simultaneously resident input cells;
3. a lower bound for assigning addresses to a **fixed emitted trace**; and
4. a lower bound for the coefficient walk used by the bounded-weight scan.

Only (2), (3), and the cap-5 instance of (4) are tight here. This PR does
**not** prove that 409,001 is globally optimal among arbitrary sparse-parity
circuits.

## 1. Universal circuit lower bounds are clear but extremely weak

There are

\[
M={32 \choose 5}=201,376
\]

possible secrets. Let a secret be uniform and let a circuit recover it with
probability `p`. Fano's inequality implies that the observed input values must
carry at least

\[
\log_2 M-h_2(1-p)-(1-p)\log_2(M-1)
\]

bits of mutual information. Every distinct observed input cell is binary, so
this yields:

| recovery target | information lower bound | distinct binary observations |
|---:|---:|---:|
| 20% | 2.802 bits | 3 |
| 40% | 6.077 bits | 7 |
| 60% | 9.601 bits | 10 |
| 80% | 13.374 bits | 14 |
| 100% | 17.620 bits | 18 |

This is universal but not a useful energy lower bound. It ignores that the
independent `X` bits carry information about the secret only through their
joint relation with the labels. It also says nothing about the straight-line
work needed to decode that information.

The repository's companion analysis states the stronger but still elementary
worst-case bound `Omega(n*m)` for exact recovery on **all** identifiable
instances: an adversary can place a distinguishing bit in an unread input
cell. At MASK32 size that means 594 inspected input cells. It does not transfer
automatically to the 20--80% distributional bands or to a finite dev-suite
record. If every input is read once, its cheapest distinct placement costs

\[
\sum_{a=1}^{594}\lceil\sqrt a\rceil=9,950.
\]

For exact recovery, the 32 output coordinates must be independently
representable; their final reads alone have the weak residence-cost floor

\[
\sum_{a=1}^{32}\lceil\sqrt a\rceil=137.
\]

Combining such elementary facts still leaves several orders of magnitude of
slack. A useful arbitrary-IR lower bound—one that accounts for GF(2) decoding,
straight-line branch elimination, and spatial read cost—is not present in the
repository and remains the main theoretical gap.

## 2. Exact 594-cell input-residence lower bound

The first IR line declares `18*32+18=594` input cells, and the evaluator requires
those addresses to be distinct. Before the first operation, all 594 values are
therefore live simultaneously. No semantics-preserving register allocation can
use fewer than 594 physical cells in the elimination phase.

The packed generator's SSA interval allocation reaches exactly **594 prefix
slots**. Bridge-cycle scratch is taken from a dead prefix slot, so every final
submitted IR uses exactly the address set `1..594`—no 595th cell is introduced.
This is a tight storage lower bound, not by itself an energy lower bound.

## 3. Exact address lower bound for the emitted fixed trace

For a fixed straight-line trace, let `f_i` be the number of reads of logical
cell `i`. This includes implicit destination reads in two-operand instructions
such as `xor d,s`, bridge reads, and final output reads. Static energy is

\[
\sum_i f_i\lceil\sqrt{a_i}\rceil,
\]

where the positive addresses `a_i` are distinct. By the rearrangement
inequality, the exact optimum is obtained by sorting cells by decreasing
`f_i` and assigning addresses `1,2,...` in that order.

The generator first performs phase-local SSA liveness allocation, allowing the
same physical slots to be reused after values die. It then transfers only the
187 values live across the RREF-to-walk boundary with a cycle-safe parallel
copy. Finally, it globally sorts the **actual emitted trace** by read frequency.
Applying that final sorter again is byte-identical and leaves cap-5 cost at
**409,001**. Thus the address assignment is optimal for this emitted trace.

This still does not make the algorithm globally optimal: another trace may
perform fewer reads.

## 4. Exact transition lower bound for the cap-5 walk

For a rank-18 MASK32 instance, the affine solution space has `32-18=14` free
coordinates. A weight-5 secret has at most five ones among those coordinates,
so exhaustive full-rank recovery needs only

\[
B=\{z\in\{0,1\}^{14}: |z|\le 5\}.
\]

It contains

\[
|B|=\sum_{j=0}^{5}{14\choose j}=3,473
\]

states. Split it by parity:

\[
E={14\choose0}+{14\choose2}+{14\choose4}=1,093,
\]

\[
O={14\choose1}+{14\choose3}+{14\choose5}=2,380.
\]

Take any walk starting at zero and visiting every state in `B`; repeated states
and long Hamming-distance jumps are allowed. Expand every distance-`d` jump
into `d` unit hypercube edges. A length-`C` unit-edge walk starting at an even
vertex has at most `ceil(C/2)` odd positions. Visiting 2,380 distinct odd
vertices requires

\[
C\ge 2O-1=4,759.
\]

Therefore **4,759 coefficient-bit flips is a lower bound for every zero-start
walk covering `B`**, not merely Hamiltonian paths.

The construction filters the 14-bit binary-reflected Gray code to masks of
weight at most five. Exhaustive finite verification shows that it visits all
3,473 allowed masks exactly once and has total Hamming transition cost exactly
**4,759**. The bound is attained. The repository's previous lexicographic
weight schedule used 8,959 flips.

## 5. Packed-column elimination

Each of the 33 augmented `[X | y]` columns is represented by three 6-bit cells,
one for each group of six training rows. Six bits are used instead of eight so
all packed masks remain nonnegative under the signed 8-bit division semantics.

For input column `c`:

1. `eligible = column & ~used` is formed in each chunk;
2. `eligible & -eligible` isolates the first available pivot row;
3. the three selected chunks form one 18-row one-hot pivot mask;
4. the current column becomes that one-hot mask when a pivot exists; and
5. every later augmented column and `y` is updated by one pivot-bit test and
   three bytewise XORs.

Earlier pivot columns are skipped. They are already unit columns, and the new
pivot row was unused, so its entry in every earlier pivot column is zero. The
corresponding row operation would be an identity on those columns.

Free columns receive streamed ranks `0..13`; pivot columns receive sentinel 63.
After elimination, `y` is the zero-free-variable solution in packed pivot-row
coordinates, and each free column is its packed affine basis vector. The walk
therefore updates only three cells per coefficient flip, rather than a 32-cell
candidate mask.

## 6. Exact packed weight predicates

At coefficient weight `t`, the pivot-row part must have weight `5-t`:

| coefficient weight `t` | states | required pivot weight |
|---:|---:|---:|
| 0 | 1 | 5 |
| 1 | 14 | 4 |
| 2 | 91 | 3 |
| 3 | 364 | 2 |
| 4 | 1,001 | 1 |
| 5 | 2,002 | 0 |

A general popcount is wasteful for most states. Weight zero is an OR-reduction
plus one comparison. Weight one uses a power-of-two test on the aligned OR and
a zero-majority test, excluding two chunks that occupy the same bit position.
Weights two and three use exact predicates derived from per-position parity
and majority. Only the 15 states targeting pivot weights four or five use the
general packed popcount.

For chunk bits `a,b,c`, the total population is

\[
\operatorname{popcount}(a\oplus b\oplus c)
+2\operatorname{popcount}(\operatorname{majority}(a,b,c)).
\]

## 7. Correctness scope

For a rank-18 training matrix, there are exactly 14 free coordinates. The cap-5
walk enumerates every possible restriction of a weight-5 secret to those
coordinates. Every visited coefficient mask defines a solution of `Xs=y`; an
affine solution whose reconstructed total weight is five must be the benchmark's
unique weight-5 secret.

The submitted cap-5 IR recovered all 1,024 deterministic dev instances and all
4,096 instances in two independent final-sized suites. The scalar reference
interpreter also matched on a sampled prefix, and optimized-vs-raw equivalence
was tested on arbitrary binary inputs.

As with the repository's previous full scan, a rank-deficient training matrix
can have more than 14 free coordinates. This circuit records the first 14 and
may miss a secret using a later free coordinate. The repository documentation
estimates that event near `2^-14`. The leaderboard's 100% claim is therefore a
measured dev-tier result plus an exact full-rank proof—not a claim over every
rank-deficient input.

## 8. Reproduction

From `sparse-parity/` after applying the package:

```bash
python3 -m pytest test_mask_sparse_parity.py test_packed_sparse_parity.py -v
python3 verify_packed_scan.py
python3 verify_packed_scan.py --recorded
python3 verify_packed_scan.py --fresh 2
```
