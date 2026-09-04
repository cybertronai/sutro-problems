# Matrix Multiplication

**Author:** [@jurajselep](https://github.com/jurajselep)<br>
**Date:** 2026-09-04<br>
**Problem:** 4x4 matmul<br>
**Cost:** 675<br>
**IR:** [`best_675.ir`](best_675.ir)<br>
**Verifier:** [`best_675.py`](best_675.py)<br>
**SHA-256:** `761d0401876c4c1fe8f51c0ff7ecf3676b7abca1f56b4e36715d70426d35427e`

## Summary

This submission lowers the 4x4 record from 681 to **675**. Selective copies
split the lifetimes of `B[0,0]`, `B[0,2]`, and `A[0,3]`; dependency-safe
rescheduling and exact lifetime coloring then reduce the total read cost.

The IR uses 64 multiplications, 48 additions, and 16 copies.

| Read source | Cost |
| - | -: |
| `mul` | 341 |
| `add` | 186 |
| `copy` | 80 |
| outputs | 68 |
| **total** | **675** |

## Verification

```bash
python3 matmul/submissions/best_675.py
```
