# Sparse Parity

**Author:** [@yaroslavvb](https://github.com/yaroslavvb)
**Date:** 2026-08-26
**Problem:** Mask sparse parity, n=32 — 20% accuracy target
**Cost:** 12,042,480
**IR:** [`isd8_mask32.ir`](isd8_mask32.ir)
**Method:** `generate_isd_mask(8)` (ISD restarts)

## Idea — information-set decoding

Run T cheap Gaussian eliminations over GF(2), each on a different rotating
18-column subset (an "information set") of the 32 bit positions. A restart
succeeds when the hidden k = 5 secret positions happen to fall inside the
chosen 18 columns and the restricted system pins them down: the candidate it
produces is accepted only if it has weight k **and** reproduces every
training parity. Because training sets are uniquely identifiable, an accepted
candidate *is* the secret, so the circuit never outputs a wrong mask — it
either recovers the secret or outputs zeros.

Each restart is branchless straight-line code (~36.5k instructions, ~1.5M
reads), and recovery grows roughly linearly in T at first — this is the
cheapest known way to buy low recovery. This submission uses T = 8:
22.5% recovery at 12,042,480 reads on the dev suite.

Diminishing returns set in as the rotating information sets start to overlap:
measured recovery on the dev suite is 5.8% (T=1), 22.5% (T=8), 29.8% (T=16),
and then an exact plateau at 36.2% for every measured T in 32–54 (T=40, 48,
and 54 all score recovery 0.3623; T=54 is the largest setting under the
2,000,000-instruction cap at 1,970,406 ops). The rotation finds no new
secrets past T≈32, so the family cannot reach the 40% band and the
[Gray scan](scan_full_mask32.md) takes over from there. The T ≤ 32 points
are in [doc/mask32_bands.json](../doc/mask32_bands.json); regenerate the
T &gt; 32 points with `mp.evaluate_mask(mp.generate_isd_mask(54))`.

## Reproduce

```python
import mask_sparse_parity as mp

ir = mp.generate_isd_mask(8)
mp.evaluate_mask(ir)          # → cost 12,042,480, recovery 0.2246
```

The generator lives in
[`mask_sparse_parity.py`](../mask_sparse_parity.py); it wraps
`generate_isd(..., mask_output=True)` from
[`scaled_sparse_parity.py`](../scaled_sparse_parity.py), where the ISD
circuit construction is documented in detail.

Further reading: [benchmark report](https://cybertronai.github.io/sutro-problems/docs/).
