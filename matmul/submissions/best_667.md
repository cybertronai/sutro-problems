# Matrix Multiplication

**Author:** [@jurajselep](https://github.com/jurajselep)<br>
**Date:** 2026-09-20<br>
**Problem:** 4x4 matmul<br>
**Cost:** 667<br>
**IR:** [`best_667.ir`](best_667.ir)<br>
**Verifier:** [`best_667.py`](best_667.py)<br>
**SHA-256:** `7bf91dafc04584b6ce8f4d4d5e4de917ea426e47d84b413b8a068f78e143d7a3`

## Summary

This submission lowers the 4x4 record from 675 to **667**, an improvement of
8 weighted-read units (1.185%). Changes to contraction order, input captures,
and storage allocation reduce the total read cost while retaining the
classical 64-product formula.

The IR uses 64 multiplications, 48 additions, and 20 copies.

| Read source | Cost |
| - | -: |
| `mul` | 319 |
| `add` | 178 |
| `copy` | 102 |
| outputs | 68 |
| **total** | **667** |

## Verification

Run from the repository root to verify the frozen IR:

```bash
python3 -S matmul/submissions/best_667.py
```
