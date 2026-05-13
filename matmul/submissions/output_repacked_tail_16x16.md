# Matrix Multiplication

**Author:** Codex and Cosmin
**Date:** 2026-05-13
**Problem:** 16x16 matmul
**Cost:** 68,392
**IR:** [`output_repacked_tail_16x16.ir`](output_repacked_tail_16x16.ir)
**Method:** `generate_output_repacked_tail_16x16` (liveness-ordered outputs + output-read-aware packing + scratch tail)

## Idea

This is still ordinary 16x16 matrix multiplication. It does not change the
arithmetic; it only changes where values live in memory. In this problem,
reads from smaller addresses are cheaper, writes are free, and an output is
charged once more when it is read at exit. So the goal is to keep frequently
read values and final outputs at cheap addresses without overwriting an input
before its last use.

The schedule computes the matrix in eight 4x8 output blocks. It uses the same
inner arithmetic as [`colmajor_fused_16x16`](colmajor_fused_16x16.md), where a
non-final block writes each output directly to its final home on the last
accumulation. The improvement is to choose a block order that exposes dead
input cells earlier:

```text
00, 01, 10, 20, 30, 11, 21, 31
```

When a block is the last user of an A row slab or B column slab, those input
cells can safely be reused as output storage. Completed outputs are placed in
dead B cells first, then dead A cells, then fresh spill cells:

| home type | outputs |
|-----------|--------:|
| dead B input cells | 96 |
| dead A input cells | 64 |
| fresh spill cells | 64 |
| final-block sC cells | 28 |
| final scratch cells | 4 |

The B/A cells that also become outputs get one extra exit read, so they are
packed at the front of their input regions before ordinary input cells.

Finally, in the last block `(3,1)`, local output column `jb=7` is computed
last. Its four outputs can stay directly in the cheapest scratch cells:

```text
C[12,15] -> sA0 @ 3
C[13,15] -> sA1 @ 4
C[14,15] -> TMP @ 2
C[15,15] -> SB  @ 1
```

The last four accumulations therefore write the final result in place:

```text
mul 3,3,1
add 3,35,3
mul 4,4,1
add 4,36,4
mul 2,5,1
add 2,37,2
mul 1,6,1
add 1,38,1
```

No later instruction reads the old `SB`, `TMP`, or `sA` values after these
writes; the next reads from addresses 1, 2, 3, and 4 are the exit reads.

## Path to 68,392

| step | score | savings |
|------|------:|--------:|
| `colmajor_fused_16x16` | 68,452 | |
| + liveness order + output-read-aware packing | 68,411 | 41 |
| + final scratch tail for `jb=7` | **68,392** | 19 |

## Region cost breakdown

| region | addrs | reads | cost |
|--------|-------|------:|-----:|
| SB | 1 | 5,057 | 5,057 |
| TMP | 2 | 2,879 | 5,758 |
| sA | 3..6 | 4,100 | 10,248 |
| sC | 7..38 | 3,868 | 19,576 |
| B bulk | 39..294 | 1,120 | 14,255 |
| A bulk | 295..550 | 576 | 11,924 |
| C spill | 551..614 | 64 | 1,574 |
| **total** | | **17,664** | **68,392** |

The total number of paid reads is unchanged from the previous record. The
savings are purely from moving reads onto cheaper addresses and removing the
last four final-block sC output homes.

## Verification

```bash
python matmul/submissions/output_repacked_tail_16x16.py
python -m pytest -q matmul/test_matmul.py
```

Observed locally for the generator:

```text
output_repacked_tail_16x16.ir  cost=68,392
```

The generator also asserts:

- 512 distinct positive input addresses.
- 256 distinct positive output addresses.
- 160 input addresses safely reused as output homes.
- No input address is read after it has been overwritten as an output.
