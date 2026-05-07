# Matrix Multiplication

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-05-06
**Problem:** 16×16 matmul
**Cost:** 69,697
**IR:** [`aliased_16x16.ir`](aliased_16x16.ir)
**Method:** `generate_aliased_16x16` (C↔A address aliasing + final-add fusion)

## Idea

Two structure-preserving tricks stacked on the 73,602
[`sa_cache_16x16`](sa_cache_16x16.ir) base (`TI=8, TJ=4`, sA single-cell
cache + sB rank-Tj scratchpad + sC `Ti×Tj` accumulator). Both tricks
target liveness, not arithmetic.

This was an independent rediscovery in a closed-context exploration; both
tricks already appear in earlier records:

* **Dead-input output reuse** is also the core idea of
  [`dead_input_outputs_packed_16x16`](dead_input_outputs_packed_16x16.md)
  (70,053, 2026-05-01).
* **Final-add fusion** is the same instruction-level move as
  [`colmajor_fused_16x16`](colmajor_fused_16x16.md) (68,452, 2026-05-05).
  That submission combines it with column-major super-block order and a
  more compact bulk layout, which is why it wins by ~1,200.

So this entry sits between the two; it's posted for completeness, not as
a new direction.

## Trick 1 — Alias C output cells with A input cells

The scorer requires *input* addresses to be distinct from each other, but
output addresses may coincide with input addresses — only the value at exit
matters. Once `A[i,k]` has been read for the last time, the cell is dead
and can host a `C` value.

Under `(bi, bj, bk)` loop order with `nbi=2, nbj=4`:

* **Last bj-block** (`bj = nbj-1`, `j ∈ {12..15}`): for each row, `A[i, j-12]`
  is fully consumed by the time the corresponding `C[i, j]` is being
  written. **64 C cells** alias onto `A[i, 0..3]`.
* **bi=1, non-last bj** (`j ∈ {0..11}`): by the time `bi=1` starts, all of
  `A[bi=0, k=4..15]` is dead. **96 C cells** alias onto those slots.
* **bi=0, non-last bj** (`j ∈ {0..11}`): these C values are written while
  the corresponding A cells are still being read (for the `bi=1` phase).
  **96 cells** still need fresh bulk slots.

Total: 160 of 256 C cells alias; 96 stay as fresh bulk addresses.

## Trick 2 — Final-add fusion (`bk = N-1`)

Each accumulating step is `mul tmp, sA, sB; add sC, tmp` — two ops, four
reads. At `bk = N-1` the result is final and would normally be copied to
its `C` address with one more `sC` read. The 3-operand `add` form lets us
fuse the final accumulation and the copy-out:

```text
mul tmp, sA, sB                   # standard accumulating step
add sC[ii,jj], tmp

mul tmp, sA, sB                   # bk = N-1: fused
add C(bi*TI+ii, bj*TJ+jj), sC[ii,jj], tmp
```

Saves one `sC` read per `(ii, jj, bi, bj)` = 256 reads at `sC` cost 3-7.

## Path to 69,697

| step                                                | score  | savings |
|-----------------------------------------------------|-------:|--------:|
| `sa_cache_16x16` base (`TI=8, TJ=4`)                | 73,602 |         |
| + C↔A address aliasing (160/256 C cells)            | 70,993 | 2,609   |
| + final-add fusion (3-op `add` at `bk = N-1`)       | 69,697 | 1,296   |

## Layout (greedy by descending read count)

```text
addr 1         sA cache         (4,096 reads × cost 1 = 4,096)
addr 2         tmp              (3,840 reads × cost 2 = 7,680)
addrs 3..6     sB strip         (1,024 reads each)
addrs 7..38    sC accumulator   (120 reads each)
addrs 39..198  A aliased with C (5 reads each: 4 as A + 1 as C exit)
addrs 199..294 A non-aliased    (4 reads each)
addrs 295..550 B input          (2 reads each)
addrs 551..646 C non-aliased    (1 read at exit)
```

`cost(addr) = ⌈√addr⌉` is non-decreasing, so this greedy assignment is
optimal among address permutations for the chosen schedule.

## Region cost breakdown

| region                | cost   | reads | cells |
|-----------------------|-------:|------:|------:|
| sA @ 1                | 4,096  | 4,096 |   1   |
| tmp @ 2               | 7,680  | 3,840 |   1   |
| sB @ 3..6             | 10,240 | 4,096 |   4   |
| sC @ 7..38            | 19,440 | 3,840 |  32   |
| A∪C aliased @ 39..198 | 8,900  | 800   | 160   |
| A only @ 199..294     | 6,208  | 384   |  96   |
| B @ 295..550          | 10,738 | 512   | 256   |
| C only @ 551..646     | 2,395  |  96   |  96   |
| **total**             | **69,697** | 17,664 | **646** |

## Verification

```bash
python3 matmul/submissions/aliased_16x16.py
# aliased_16x16.ir  cost=69,697
```

The generator is self-contained: it builds the op list with symbolic
labels, counts reads per cell, then assigns addresses by descending read
count.
