# Sparse Parity

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-08-26
**Problem:** Mask sparse parity, n=32 — 100% accuracy target
**Cost:** 43,325,468
**IR:** [`scan_full_mask32.ir`](scan_full_mask32.ir)
**Method:** `generate_scan(16383)` (Gray scan, full walk)

## Idea — GE + null-space Gray scan

Three phases, all branchless straight-line code:

1. **Gaussian elimination.** Full-width GF(2) RREF of the augmented training
   system [X_train | y] with select-based pivoting (no branches, no
   data-dependent addressing).
2. **Null-space extraction.** Read off one particular solution s₀ and the
   null-space basis — dimension G = n − m = 14 with high probability — and
   gather the basis vectors into Gray-variable slots.
3. **Gray walk.** Walk s steps of the reflected Gray code over the 2¹⁴
   solution space w = s₀ + Σ aⱼ·basisⱼ; each step XORs one basis vector into
   the running solution and captures w into the output whenever
   weight(w) = k. Every visited w solves the training system, so a weight-k
   visitor is provably the secret (unique identifiability).

The full walk (s = 16,383 = 2¹⁴ − 1) visits the entire solution space: 100%
recovery on the dev suite at 43,325,468 reads and 1,772,808 instructions —
the only known family that reaches 100% under the 2,000,000-instruction cap.
(Rank-deficient training draws, ~2⁻¹⁴ of instances, can still miss on a
fresh adjudication suite.)

Capping the walk at s < 2¹⁴ − 1 trades recovery for energy along a concave
curve — much better than the naive (s+1)/2¹⁴ line, because the secret's Gray
coefficient vector is low-weight and tends to be visited early (median
capture step ~1,100). The partial-walk submissions
([s=511](scan511_mask32.md), [s=4095](scan4095_mask32.md),
[s=8191](scan8191_mask32.md)) hold the 40/60/80% bands.

## Reproduce

```python
import mask_sparse_parity as mp

ir = mp.generate_scan()       # full walk, = generate_scan(16383)
mp.evaluate_mask(ir)          # → cost 43,325,468, recovery 1.0
```

The circuit construction is documented phase-by-phase in the
`generate_scan` docstring of
[`mask_sparse_parity.py`](../mask_sparse_parity.py).

Further reading: [benchmark report](https://cybertronai.github.io/sutro-problems/docs/).
