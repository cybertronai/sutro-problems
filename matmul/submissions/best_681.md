# Matrix Multiplication

**Author:** [@jurajselep](https://github.com/jurajselep)<br>
**Date:** 2026-09-01<br>
**Problem:** 4x4 matmul<br>
**Cost:** 681<br>
**IR:** [`best_681.ir`](best_681.ir)<br>
**Verifier:** [`best_681.py`](best_681.py)<br>
**SHA-256:** `05d1150da06eea9581e51c0617488c49bf52d02c41242920d40e2eb940e854a7`

## Summary

This submission computes ordinary 4x4 matrix multiplication with the classical
64 multiplications and 48 additions. It scores **681**, improving the previous
689 record by **8 read-cost units (1.16%)**.

The improvement comes from the schedule and physical layout. Row 0 precomputes
its four `k=0` products in the column order `(1, 3, 2, 0)`, evacuates `B[0,1]`
before its cheap input cell is reused, and completes its dot products directly
from the remaining A inputs. Rows 1--3 use a `k`-outer schedule: each A scalar
is copied just in time to address 1, used by four multiplications, and then
replaced. Independent final additions are interleaved with nearby work to
shorten output lifetimes.

The program uses addresses 1 through 37. Address 1 serves 76 reads. Thirteen
copies enable scratch-cell reuse and repeated cheap reads of staged A values.

## Physical layout

The first sixteen declared cells contain A in row-major order, followed by the
sixteen row-major B values.

| Value | Addresses |
|---|---|
| `A[0,:]` | `1, 2, 3, 5` |
| `A[1,:]` | `6, 26, 27, 28` |
| `A[2,:]` | `29, 37, 30, 31` |
| `A[3,:]` | `32, 33, 34, 35` |
| `B[0,:]` | `7, 4, 8, 9` |
| `B[1,:]` | `17, 18, 19, 10` |
| `B[2,:]` | `20, 21, 11, 12` |
| `B[3,:]` | `13, 14, 15, 16` |

The sixteen row-major outputs are read from:

```text
36,22,24,23,26,28,29,27,9,5,6,30,2,3,4,1
```

## Cost breakdown

| Instruction | Count | Paid reads | Read cost |
|---|---:|---:|---:|
| `mul` | 64 | 128 | 345 |
| `add` | 48 | 96 | 197 |
| `copy` | 13 | 13 | 72 |
| output exit | -- | 16 | 67 |
| **total** | **125** | **253** | **681** |

## Correctness and static verification

The repository scorer evaluates the program over exact symbolic integer
polynomials. Every one of the sixteen outputs is exactly

```text
C[i,j] = sum(k=0..3) A[i,k] * B[k,j]
```

and no multiplication creates a polynomial above degree two. The verifier
locks the submitted bytes by SHA-256, checks the symbolic score, and checks the
operation and cost breakdown. It has no third-party dependencies:

```bash
python3 matmul/submissions/best_681.py
```

Expected output:

```text
best_681.ir: score=681, sha256=05d1150da06eea9581e51c0617488c49bf52d02c41242920d40e2eb940e854a7
operations: {'add': 48, 'copy': 13, 'mul': 64}
read costs: {'add': 197, 'copy': 72, 'mul': 345, 'output': 67}
```
