# Sparse parity - tuned pair-XOR reuse (medium)

**Author:** [@sjbaebae](https://github.com/sjbaebae)
**Date:** 2026-05-08
**Problem:** sparse parity (n=8, k=3, 8 train / 64 test)
**Cost:** 15,691
**IR:** [`predpack_tuned_medium.ir`](predpack_tuned_medium.ir)
**Generator:** [`predpack_tuned.py`](predpack_tuned.py)
**Method:** `generate_predpack_tuned_medium` (packed candidate check with pair-XOR reuse and address/liveness tuning)

## Algorithm

The medium task has 3 hidden bits among 8 total bits, 8 training examples, and
64 test examples. The IR first packs each training column into an 8-bit mask and
packs the training labels into `ymask`.

For each candidate triple `(a, b, c)`, the decoder checks:

```text
col[a] xor col[b] xor col[c] == ymask
```

If the comparison is true, the candidate contributes to the recovered secret
mask cells. Instead of recomputing the first two XORs for every triple, triples
are grouped by `(a, b)`: `pair = col[a] xor col[b]` is computed once and reused
for every later column `c`.

After the secret mask is recovered, the IR predicts all 64 test rows by gating
the input bits with the recovered mask and XORing the selected bits.

## Tuning

Relative to `predpack_medium.ir`, this version keeps the candidate indicator in
`tmp_gate`, aliases early output cells to dead decode scratch (`w_ind`, `ymask`,
and `cm[0..7]`), and leaves the final prediction in `tmp_gate`. These are
placement/liveness changes only; the algorithm remains a seed-independent
packed-column GF(2) decoder.

Opcode counts:

| Opcode | Count |
| --- | ---: |
| `and` | 512 |
| `cmp` | 56 |
| `copy` | 80 |
| `mul` | 63 |
| `or` | 223 |
| `set` | 63 |
| `xor` | 525 |

## Comparison

| Submission | Cost | IR ops | Notes |
| --- | ---: | ---: | --- |
| `baseline_medium.ir` | 816,251 | 16,457 | Upstream try-each-candidate baseline |
| `ge_medium.ir` | 473,046 | 8,590 | Upstream GF(2) Gaussian elimination baseline |
| `ge_medium_packed.ir` | 16,084 | 1,558 | Packed GF(2) candidate check |
| `predpack_medium.ir` | 15,960 | 1,523 | Pair-XOR reuse |
| `predpack_tuned_medium.ir` | 15,691 | 1,522 | Pair-XOR reuse plus address/liveness tuning |

The IR uses only official v3 operations: `set`, `copy`, `xor`, `and`, `or`,
`mul`, and `cmp`.
