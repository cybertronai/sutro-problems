# Sparse parity - packed candidate check (medium)

**Author:** [@sjbaebae](https://github.com/sjbaebae)
**Date:** 2026-05-08
**Problem:** sparse parity (n=8, k=3, 8 train / 64 test)
**Cost:** 16,084
**IR:** [`ge_medium_packed.ir`](ge_medium_packed.ir)
**Generator:** [`ge_medium_packed.py`](ge_medium_packed.py)
**Method:** `generate_ge_medium_packed` (pack training columns, check all 56 candidates by equality)

## Algorithm

Decode packs each 8-row training column and the label vector into integer bit masks, then checks all 56 possible 3-bit sparse secrets with packed GF(2) equality.  Candidate indicators are OR-accumulated into an 8-bit secret mask, and predictions use that mask over the 64 test rows.  The IR uses only official v3 instructions and does not specialize to any seed list.

## Cost breakdown

| section | ops |
| --- | ---: |
| total IR body | 1,558 |

This improves the upstream medium GE baseline from 473,046 to 16,084 by replacing row-reduction state with packed column equality checks.
