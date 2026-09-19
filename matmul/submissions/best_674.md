# 4×4 matmul: 675 → 674

**Author:** [@jurajselep](https://github.com/jurajselep)<br>
**Date:** 2026-09-19<br>
**Cost:** 674

This submission saves one weighted operand-read unit by changing the reduction
trees, instruction order, and input staging within a 20-instruction window of
the 675 record. Inputs, outputs, and instructions outside that window are
unchanged. Both programs use 64 multiplications, 48 additions, and 16 copies.

The selected window is instructions 17–36, counting instructions from one
and excluding the input/output placement lines. Its cost falls from 91 to 90;
only instructions 17–34 differ in the final IR.

For the first row's third output, let `p_k = A[0,k] * B[k,2]`. Its reduction
changes from `((p_1 + p_2) + p_0) + p_3` to
`((p_1 + p_2) + p_3) + p_0`. The final addition reads addresses 1 and 2 instead
of 1 and 5, saving one unit. The neighboring output's reduction and the staging
of `A[0,3]` also change to make that cheaper placement possible.

| Read source | Previous cost | New cost |
| --- | ---: | ---: |
| Multiplications | 341 | 341 |
| Additions | 186 | 185 |
| Copies | 80 | 80 |
| Outputs | 68 | 68 |
| **Total** | **675** | **674** |

There are still 256 paid reads and the highest address is still 37. Overall,
one tier-3 read becomes a tier-2 read. The benchmark charges `ceil(sqrt(address))`
per operand or output read; these costs are not hardware energy measurements.

The replacement was found by a bounded search over partial-sum subsets, input
replicas, operation order, and storage tiers with fixed surrounding code.
This establishes a verified cost of 674, without claiming global optimality.

## Reproduce and verify

```bash
python3 -S matmul/submissions/best_674.py
python3 -m pytest -q matmul/test_matmul.py matmul/test_schedule_search.py
```

The standard-library constructor replays the winning replacement against the
hash-pinned 675 record and reproduces the submitted IR byte for byte. It does
not rerun the search. The verifier checks the official symbolic scorer for all
16 outputs, hashes, operation counts, and read costs. Verification does not
depend on the archive's vendored scorer or search tools.

IR: [best_674.ir](best_674.ir). Constructor and verifier:
[best_674.py](best_674.py).

SHA-256: `9370c290d54e5ef2edb2ae69986f65c8932e68ca9f409574dd2ee96b7137094b`.
