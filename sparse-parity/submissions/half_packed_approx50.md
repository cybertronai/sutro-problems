# Sparse parity — half-output packed decoder (medium, 50% target)

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-05-09
**Problem:** sparse parity (n=8, k=3, 8 train / 64 test; **50% target**)
**Cost:** 8,723
**IR:** [`half_packed_approx50.ir`](half_packed_approx50.ir)
**Generator:** [`half_packed_approx50.py`](half_packed_approx50.py)
**Method:** packed-column candidate check on the **first 32** outputs only; remaining 32 outputs aliased to a constant 0 cell.

## Algorithm

Reuses Sung Jae's [`ge_medium_packed`](ge_medium_packed.md) decoder
(pack each training column + label vector into 8-bit bitmasks, equality-check all `C(8,3)=56` candidate triples) to recover the secret mask. The prediction phase is the only difference:

- **rows 0–31:** real prediction — `pred[j] = XOR_c (secret_mask[c] AND X_test[j][c])`.
- **rows 32–63:** all aliased to a single `set ZERO, 0` cell — no per-row prediction work.

Each "lazy" output matches the true label with probability ½, so expected accuracy across all 64 rows is `(32·1 + 32·½) / 64 ≈ 0.75`. Across all 8 canonical seeds × 32 random rows, the per-seed accuracy is concentrated around 75% (well above the 50% threshold), so this submission reliably passes [`score_medium_approx50`](../sparse_parity.py).

## Cost vs the strict-100% packed decoder

| variant | scorer | cost | ops |
|---------|--------|-----:|----:|
| `ge_medium_packed.ir` | `score_medium` (100% target) | 16,084 | 1,558 |
| `half_packed_approx50.ir` | `score_medium_approx50` (50% target) | **8,723** | 1,047 |

The savings come from the 32 skipped per-row prediction blocks (each was 1 `and` + 7 `(and, xor)` pairs + 1 `copy` = 16 ops × 32 rows = 512 ops, a few thousand units of read-cost). The decoder phase is unchanged — recovering the secret mask costs the same regardless of how many rows you eventually use it for.

## Robustness note

Designed to **fail** strict `score_medium` (the always-zero outputs miss every row whose true label is 1) and to **pass** `score_medium_approx50` (75% accuracy, well above the 50% threshold). The strict scorer rejects this submission on the first canonical seed.
