# Matrix Multiplication

**Author:** [@jurajselep](https://github.com/jurajselep)<br>
**Date:** 2026-09-04<br>
**Problem:** 4x4 matmul<br>
**Cost:** 678<br>
**IR:** [`best_678.ir`](best_678.ir)<br>
**Verifier:** [`best_678.py`](best_678.py)<br>
**SHA-256:** `1e024f1aefdafc044fd825c2174bc3c3b7293c4845fad46dfa0fb81ef6ad7805`

## Summary

This submission lowers the 4x4 record from 681 to **678**. It copies
`B[0,0]` once so row 0 and rows 1-3 can consume separate lifetimes, then
reschedules and reallocates the resulting DAG. The extra copy costs 2 while
reducing multiplication reads by 3.

The IR uses 64 multiplications, 48 additions, and 14 copies.

| Read source | Cost |
| - | -: |
| `mul` | 353 |
| `add` | 185 |
| `copy` | 73 |
| outputs | 67 |
| **total** | **678** |

## Verification

```bash
python3 matmul/submissions/best_678.py
```
