# RFC: The Scheduled Dally Language

Status: draft · Author: Andy Zhang (sutro agents) · September 2026 ·
Discussion: Sutro group chat, 2026-09-02/03

## 1. Why a new language

The sparse-parity and matmul competitions currently take submissions as
straight-line instruction traces over the simplified Bill Dally model
(reads priced at `ceil(sqrt(addr))`, writes and arithmetic free). That
format works at 4x4 and holds to a few hundred thousand lines, but the
competitions are heading somewhere it cannot follow:

- An 8192 x 8192 x 8192 matmul is ~1.7e12 operations. As raw trace
  text, one line per operation, that is roughly **27 TB**. Traces are
  dead at realistic scale.
- A single forward epoch of MNIST on a SOTA model is the same order
  (~1e12 ops). Same wall.
- Earlier Python-script submissions with instrumentation allowed too
  many ways to cheat; unrestricted hosts are not sandboxable.
- Existing languages leave memory placement to the compiler, but under
  the Dally model placement *is the algorithm* - the whole score is
  where data lives relative to the processor.

The request (Yaroslav, 2026-09-02): a language that covers both matmul
and MNIST, is immune to cheating (interpreted in a sandbox), and makes
memory placement an explicit part of program design.

The Scheduled Dally Language (SDL) is that language. Programs are
**hierarchical loop schedules with explicit placement**, scored in
closed form without simulating the trace, and executed (when execution
is needed at all) by the trusted Rust interpreter
[dally-eval](https://github.com/cybertronai/dally-eval).

## 2. Design constraints

1. **No escape vectors.** A restricted AST, deterministically
   evaluated. No I/O, no host calls, no reflection, no unbounded
   allocation. The reference interpreter is the only execution
   semantics, and it is bit-exact and open source.
2. **Memory placement is syntax.** Every value's grid address is
   either explicit or derived from an explicit placement rule. The
   compiler (such as it is) never chooses addresses.
3. **O(1) scoring at any scale.** The score of a 1.7e12-op program is
   computed algebraically from the schedule structure, in closed form.
   No submission ever costs the judge more than milliseconds.
4. **Physical fidelity.** The address hierarchy mirrors 2D silicon
   distance: low addresses are register-adjacent, mid addresses
   core-local, high addresses die-far. One unit of movement = 1 fJ
   (the anchor validated against nvidia-smi within an order of
   magnitude on a Ciresan-scale MNIST network).

## 3. Language architecture

SDL is two levels, explicitly composed:

```
program     := macro | micro
macro       := loop-nest over tiles, body = placement-block + (macro | micro)
micro       := straight-line trace over the 13 existing Dally ops
```

### 3.1 Micro level: the base case

The micro level is exactly today's IR: `set, copy, not, abs, and, or,
xor, add, sub, mul, div, cmp, select` over 8-bit cells at positive
integer addresses. 4x4 and 8x8 matmul kernels, the current sparse-parity
record circuits - all of these are already valid SDL micro programs.
Nothing changes for existing submitters.

### 3.2 Macro level: affine loop nests with placement

```
tile A[i, j] at (AX + i, AY + j)          ; explicit placement grid
tile B[k, j] at (BX + k, BY + j)
tile C[i, j] at (CX + i, CY + j)

for i in 0..N/T:
  for j in 0..N/T:
    stage aT[., .] from A[i*T:(i+1)*T, .] at (SA + .., SB + ..)
    stage bT[., .] from B[., j*T:(j+1)*T] at (SC + .., SD + ..)
    call micro_4x4(aT, bT, cT)
    accumulate C[i*T:(i+1)*T, j*T:(j+1)*T] += cT
    release aT, bT
```

The placement primitives:

- **`tile X[...] at (x, y)`** - declares a named region and pins its
  grid origin. Addresses inside a tile are affine in the tile
  coordinates.
- **`stage tmp from src at (x, y)`** - copies a tile region to a new
  placement (a data-movement block; its cost is charged exactly as the
  equivalent micro `copy` sequence).
- **`place`** - binds a scratch tile to a low-address window.
- **`release`** - ends a tile's lifetime; the address window becomes
  reusable (the same discipline as the dead-cell recycling that took
  the 4x4 schedule search from 1633 to 1309).
- **`call micro_k(...)`** - inlines a straight-line kernel at bound
  placements.

Only affine index expressions appear in loop bounds and tile
coordinates - this is what keeps scoring closed-form.

### 3.3 Recursive tiling

Macro bodies may themselves be macro schedules over sub-tiles
(frustum/tiled-recursive matmul, FFT butterflies, convolutions as
im2col + matmul trees):

```
schedule matmul(N):
  if N <= 8: call micro_8x8
  else:
    tile A,B,C into quadrants at explicit origins
    call matmul(N/2) x8 on quadrant pairs (recursive)
```

Spatially this mirrors the physical hierarchy: die-level movement for
the top-level quadrant shuffle, core-local for intra-tile staging,
register-adjacent for the micro kernel. The fractal locality is the
point - a good SDL program is a good VLSI floorplan.

## 4. Closed-form scoring

The score is the total read cost of the fully expanded trace. The
expansion is never materialized; it is evaluated symbolically.

### 4.1 Cost of an affine loop nest

For a loop nest with iteration space `I = I_1 x ... x I_d` and body
whose per-iteration read multiset is `R(it)` where every read address
is affine `a(it) = a_0 + sum c_k it_k`:

```
Total = sum_{it in I}  sum_{a in R(it)}  ceil(sqrt(a(it)))
```

Each distinct affine address form is summed over the box `I` in O(1)
via the antiderivative of `ceil(sqrt(x))` over arithmetic progressions
(a degree-3/2 polynomial envelope over the progression; exact, using
the same integer-sqrt identity `ceil(sqrt(n)) = isqrt(n-1)+1` that the
current scorer uses). A loop nest with `m` distinct read forms costs
O(m) to score regardless of |I|. An 8192^3 matmul schedule has on the
order of 10 distinct forms.

### 4.2 Cost of a micro kernel

A straight-line block's cost is the existing static scorer - sum of
`ceil(sqrt(addr))` over operand reads - already implemented and
bit-exact in dally-eval.

### 4.3 Recurrence for recursive tiling

Let `S(n)` be the optimal (or submitted) score for a problem of size
`n` under a recursive schedule with base case at `b`, per-level
movement `M(n)` (the staging/quadrant copies), and branching factor
`f`:

```
S(n) = f * S(n/2) + M(n),   S(b) = micro score
```

`M(n)` is itself an affine-nest cost (section 4.1). The judge
evaluates the recurrence symbolically to the base case. Total judge
time: O(depth x forms) - microseconds for any feasible program.

### 4.4 Cross-validation

For every program small enough to expand (up to ~1M ops), the
closed-form score must equal the dally-eval static score of the
expanded trace, bit-exact. This equality is a property test in the
reference implementation, run continuously - the packed sparse-parity
circuits (15k-709k ops) and every matmul record are the calibration
corpus. The two scorers can never silently diverge.

## 5. Sandbox and execution semantics

- **Deterministic AST.** SDL has no clocks, no randomness, no
  allocation beyond declared tiles, no control flow beyond counted
  affine loops and `if` on program values. Every program terminates
  (loop bounds are literal or tile-derived).
- **One interpreter.** The Rust dally-eval engine is the reference:
  bit-exact, open source, no unsafe code in the scoring path. A
  submission is judged by running the analyzer; when execution is
  needed (correctness spot-checks), the interpreter runs it with
  bounded cells and steps.
- **No cheating surface.** You cannot hide work outside the model
  because there is no outside: no FFI, no host types, no I/O. Data
  movement you do is movement you pay for, by construction.

## 6. Examples

### 6.1 Micro base case (4x4, the current 681-record style)

```
micro matmul4(a: tile 4x4 at A, b: tile 4x4 at B) -> c: tile 4x4 at C:
  ; straight-line: 64 mul, 48 add, staging copies
  ; (today's trace format, verbatim - this IS a valid SDL program)
```

### 6.2 Hierarchical 8192x8192 schedule

```
schedule matmul8k(A at 900000, B at 1800000, C at 2700000):
  T = 128
  for i in 0..64:
    for j in 0..64:
      stage aT = A[i*T:(i+1)*T, j*T:(j+1)*T] at 100      ; die -> core-local
      stage bT = B[i*T:(i+1)*T, j*T:(j+1)*T] at 5000     ; die -> core-local
      for k in 0..32:
        place aK = aT[k*4:(k+1)*4, .] at 1               ; core -> registers
        place bK = bT[., k*4:(k+1)*4] at 20
        call matmul4(aK, bK) -> cK at 40
        accumulate cT[., .] += cK
        release aK, bK, cK
      stage C[i*T:(i+1)*T, j*T:(j+1)*T] <- cT at 100000
      release aT, bT, cT
```

Expanded, this is ~1.7e12 operations and ~27 TB of trace text. Scored
in closed form: a few dozen address forms, microseconds of judge time.
The staging structure is the algorithm - competitors tune tile sizes,
staging placements, and release points, exactly the dimensions the 4x4
search already showed carry the win (1633 -> 1309 by recycling alone).

## 7. Relationship to existing work

- **dally-eval** (cybertronai/dally-eval): becomes the reference
  interpreter and static micro scorer; its Rust cost model is the
  semantic anchor.
- **Sparse-parity packed circuits** (the 151k-938k records): the
  calibration corpus for analyzer/interpreter equality.
- **Matmul 4x4 records** (681 and the 474-floor analysis): the micro
  optimization frontier; SDL competitions inherit them directly.
- **s1-dally** (MCTS bridge): program search over SDL ASTs - the
  schedule language gives MCTS a compact, closed-form-scored action
  space instead of raw traces.
- **Spatial-model analysis** (docs/spatial-model-analysis.html): the
  fJ / physical-distance grounding this language operationalizes.

## 8. Open questions

1. Should MNIST data load as a special tile type with fixed placement,
   or as explicit per-value placement statements (compositional but
   verbose)?
2. Accumulation semantics at the macro level: pure functional
   (accumulate into a fresh tile) vs in-place (release/rebind)? The
   4x4 work suggests in-place with release wins.
3. Do we want a `reduce` primitive (log-tree sums priced as nested
   staging), or is recursion the only combiner?
4. Naming and home: SDL in sutro-problems/docs, or its own repo next
   to dally-eval?
