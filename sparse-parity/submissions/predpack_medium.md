# Sparse parity - pair-XOR reuse (medium)

**Author:** [@sjbaebae](https://github.com/sjbaebae)
**Date:** 2026-05-08
**Problem:** sparse parity (n=8, k=3, 8 train / 64 test)
**Cost:** 15,960
**IR:** [`predpack_medium.ir`](predpack_medium.ir)
**Generator:** [`predpack_medium.py`](predpack_medium.py)
**Method:** `generate_predpack_medium` (packed candidate check with shared pair-XOR reuse)

## Algorithm

This is a generated official v3 IR for arbitrary `score_medium` seeds. It keeps the packed 8-row training column masks from `ge_medium_packed.py`, but groups candidate triples by their first two columns.  The shared two-column XOR is computed once per `(a, b)` pair and reused across all valid third columns `c`, reducing decode work from 56 pair XORs to 21 pair XORs.

Packing the 64 prediction rows into bit masks was tested, but the required scalar output unpacking costs more than it saves under the v3 read model.  The emitted IR therefore uses direct scalar prediction from the recovered secret mask.

## Cost breakdown

| section | ops |
| --- | ---: |
| total IR body | 1,523 |

The pair-XOR grouping reduces decode work relative to `ge_medium_packed.ir`, lowering cost from 16,084 to 15,960.
