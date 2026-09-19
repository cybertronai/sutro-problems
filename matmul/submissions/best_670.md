# 4×4 matmul: 675 → 670

**Author:** [@jurajselep](https://github.com/jurajselep)<br>
**Date:** 2026-09-20<br>
**Score:** 670 weighted-read units

The [submitted IR](best_670.ir) computes all 16 outputs of arbitrary 4×4 matrix
multiplication exactly. Its score is five units (0.741%) below the 675 record
on `main`, and four units below the 674 submission in
[PR #91](https://github.com/cybertronai/sutro-problems/pull/91).

| Read source | Record 675 | Starting witness 674 | New program 670 |
| --- | ---: | ---: | ---: |
| Multiplications | 341 | 341 | 321 |
| Additions | 186 | 185 | 185 |
| Copies | 80 | 80 | 95 |
| Output reads | 68 | 68 | 69 |
| **Total** | **675** | **674** | **670** |

The program uses 64 multiplications, 48 additions, and 19 copies. It makes
259 paid reads; its highest address is 37 and peak liveness is 36 residence
versions. The score includes every copy-source read and all final output reads.
It is a benchmark cost, not a measurement of hardware energy.

## How the program changed

Whole-program search changes instruction order, addition trees, input replicas,
and storage allocation. Reversing row 2's contraction groups from `0,1,2,3` to
`3,2,1,0` brings its `k=3` products next to row 1's final `k=3` group. A cheap
replica of `B[3,0]` can then serve both rows across that boundary. The supplied
search history records a 674 → 673 improvement from this search; that initial
stochastic discovery was not rerun during submission verification.

Two semantic block rewrites then produce 673 → 671 → 670. The first replaces
zero-based instruction interval `[18,38)`, reducing its cost from 84 to 82.
The second replaces `[56,72)` in the resulting concrete layout, reducing its
cost from 77 to 76. Rebuilding the supplied C++ semantic engines and replaying
both searches reproduced the final IR byte for byte (3,811 and 2,588 expanded
states respectively).

Three B captures help the final layout. Removing a capture and redirecting its
uses to the original source, with all other instructions and addresses fixed,
gives these independently checked scores:

| Capture removed | Original read tier → replica tier | Score after removal |
| --- | --- | ---: |
| `B[0,1]` | 5 → 1 | 673 |
| `B[3,0]` | 5 → 2 | 671 |
| `B[3,3]` | 5 → 2 | 671 |
| All three | | 675 |

These are ablations of the final 670 layout. They do not describe the savings
from adding the same copies to the earlier 674 program.

## Verification and allocation certificate

Run from the repository root, with no third-party packages or solver:

```bash
python3 -S matmul/submissions/best_670.py
```

The [verifier](best_670.py) checks the frozen artifact's SHA-256, the repository's
official `score_4x4`, independent exact integer-polynomial outputs, operation
counts, read costs, and storage statistics. Symbolic equality establishes
correctness for arbitrary inputs. During review, the supplied verifier also
passed 1,000 random integer matrix pairs and rejected corrupted arithmetic and
output ordering.

The [allocation certificate](best_670.certificate.json) is checked with exact
rational arithmetic. Its lower bound of 670 matches the feasible witness.
It uses 163 residence versions, 94 simultaneous-live cliques, and 2,119 dual
inequalities across 13 storage tiers; the verifier also checks the remaining
higher tiers with zero capacity prices.

This proves optimal allocation only with the arithmetic, instruction order,
copies, source-version bindings, and distinct overlapping residence versions
fixed. Changing any of those choices or coalescing equal values falls outside
the certificate. It does not prove global matrix-multiplication optimality.

## Provenance

The witness and certificate are unchanged from `sutro_4x4_670_search.zip`.
The starting 674 witness matches PR #91. The search archive includes the saved
673 witness, replay engines, intermediate results, and logs; this record
submission contains the frozen result and its standalone verification.
Loading the IR does not reconstruct or rerun the optimizer.

- IR SHA-256: `cd48466797857b35a063de73eb5e222b98b4d3f1056691bbddd5b3c37b73e32a`
- Certificate SHA-256: `6bc4a36557a41217c64c44b1419b2cd5a3a929c88fa9039eb0a856ae91a688dd`
- Source archive SHA-256: `c3e7183767dc80430d368e4097a4401764de9e3f62d38347cf8a95a86e9c5eef`
