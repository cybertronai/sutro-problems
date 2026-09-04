# Matrix Multiplication

**Author:** [@jurajselep](https://github.com/jurajselep)<br>
**Date:** 2026-09-04<br>
**Problem:** 4x4 matmul<br>
**Cost:** 679<br>
**IR:** [`best_679.ir`](best_679.ir)<br>
**Verifier:** [`best_679.py`](best_679.py)<br>
**SHA-256:** `8bdb36329929307d26ef6736487870ecb7e17240bf67f1d83fb0517d074a5cd8`

## Summary

This submission lowers the 4x4 record from 681 to **679** by re-associating
independent dot-product additions, rescheduling the resulting DAG, and assigning
addresses from exact value lifetimes. A cold lifetime uses address 37 so hotter
values can remain in cheaper tiers.

The arithmetic remains classical: 64 multiplications, 48 additions, and 13
copies.

| Read source | Cost |
| - | -: |
| `mul` | 356 |
| `add` | 185 |
| `copy` | 71 |
| outputs | 67 |
| **total** | **679** |

## Verification

```bash
python3 matmul/submissions/best_679.py
```

The verifier checks the file hash, exact symbolic score, operation counts, and
read-cost breakdown.
