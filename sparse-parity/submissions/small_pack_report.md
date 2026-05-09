# Sparse parity - low-address row decoder (small)

**Author:** [@sjbaebae](https://github.com/sjbaebae)
**Date:** 2026-05-08
**Problem:** sparse parity (n=3, k=2, 4 train / 32 test)
**Cost:** 1,932
**IR:** [`small_pack_best.ir`](small_pack_best.ir)
**Generator:** [`small_pack_generator.py`](small_pack_generator.py)
**Method:** `generate_small_pack_best` (low-address row decoder with scheduled output/test aliasing)

## Verification

```bash
PYTHONPATH=sparse-parity python3 sparse-parity/submissions/small_pack_generator.py
```

## Cost breakdown

| artifact | cost | ops | notes |
| --- | ---: | ---: | --- |
| `small_pack_best.ir` | 1,932 | 249 | selected |
| `small_pack_column_gf2.ir` | 2,077 | 240 | packed-column GF(2) comparison |
| `ge_small.ir` | 22,238 | 811 | upstream GF(2) Gaussian elimination baseline |
| `baseline_small.ir` | 6,918 | 302 | upstream baseline |

## Algorithm

The selected IR keeps the row-wise candidate decoder from the train-low family
because its hot scratch and training reads fit well in low addresses.  The main
change is liveness scheduling for test inputs: `X_test` starts at address 20,
inside the output region but after the low training block.  Predictions whose
final output cells are still live as secret/scratch are computed into high
staging cells first, consuming `X_test[0..4]` before direct predictions can
overwrite addresses 20..34.

The packed-column GF(2) variant packs each training column and the label vector
into four-bit masks, then validates each candidate with one equality compare.
That reduces instruction count to 240, but the mask, weight, and indicator cells
live above the packed input block, so their read distances outweigh the saved
decode operations on the small instance.

## Robustness

`small_pack_best.ir` is seed-independent and uses only official v3 operations:
`set`, `xor`, `and`, `or`, and `copy`.
