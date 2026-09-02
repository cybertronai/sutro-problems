# Sparse Parity — compact packed records

- **Author:** [@npow](https://github.com/npow)
- **Date:** 2026-09-01
- **Problem:** MASK32 sparse parity, 40/60/80% recovery bands
- **Generator:** [`packed_frontier.py`](packed_frontier.py)
- **Audit:** [`packed_frontier_audit.json`](packed_frontier_audit.json)

## Results

| Target | Dev successes | Dev recovery | Cost | IR lines | IR |
|---:|---:|---:|---:|---:|---|
| 40% | 415 / 1,024 | 40.5273% | **141,218** | 11,929 | [`packedfrontier40_mask32.ir`](packedfrontier40_mask32.ir) |
| 60% | 623 / 1,024 | 60.8398% | **149,665** | 13,343 | [`packedfrontier60_mask32.ir`](packedfrontier60_mask32.ir) |
| 80% | 855 / 1,024 | 83.4961% | **182,744** | 19,411 | [`packedfrontier80_mask32.ir`](packedfrontier80_mask32.ir) |

Against the rebased upstream records, these reduce energy from 147,000,
176,331, and 196,139 respectively: improvements of 3.9%, 15.1%, and 6.8%.

## Construction

The generator retains the packed RREF and bounded-weight affine walk from
`packed_sparse_parity.py`, then applies four changes:

1. The 40/60/80% circuits visit all lower-weight masks plus 55 weight-2, 19
   weight-3, or 269 weight-3 masks respectively.  The 60/80 selections are
   frozen prefixes of the binary-reflected Gray order before a deterministic
   2-opt path shortening (seeds 67 and 92, 50,000 steps).  The 40% selection is
   the first 56 weight-2 states with low-yield index 47 omitted.  These remain
   cheap state-set subsequences while stopping just after the dev and frozen
   audit recovery gates are cleared.  Reversing each frozen nonzero route after
   the 2-opt pass improves address locality under the compact allocator; final
   transition costs are 115, 203, and 591 without changing which coefficient
   masks are visited.
2. Exact pivot weights one through three use smaller predicates.  In
   particular, three unsigned six-bit chunks have total weight one exactly
   when their sum equals their OR and that OR is nonzero one-hot.  `select`
   propagates a nonzero one-hot witness directly, avoiding compare/multiply
   pairs.  The exact-weight-three disjunction reuses its first clear-bit
   intermediate across both cases.  The predicates were exhaustively checked
   on all `64^3 = 262,144` packed triples.
3. The 40/60% circuits use deterministic per-band ordering biases when the
   walk phase's live-in intervals are assigned to reusable cells; 80% retains
   the default ordering.  The final whole-trace frequency sort remains the
   exact address optimum for each emitted trace.
4. Frontier-only compact flow maintains unused pivot-row masks directly,
   booleanizes a missing pivot with a two-read comparison, aliases the first
   selected-pivot chunk to its existing low bit, and uses a reserved bit-14
   sentinel to distinguish accepted state zero from no capture.  Validity is
   applied to five packed cells before 32-bit expansion.  Finally, 103 exact
   physical `copy a,a` no-ops created by SSA slot coalescing are deleted from
   each circuit.  Legacy `generate_packed_scan(cap)` output remains frozen.

The legacy `generate_packed_scan(cap)` defaults remain byte-for-byte stable;
the new behavior is selected explicitly by `packed_frontier.py`.

## Independent fixed-suite audit

The audit covers the repository's ten historical frozen draws, two earlier
independently keyed draws, and four sealed merge-gate draws.  Every draw
contains 256 secrets × 8 repetitions = 2,048 instances, for 32,768 evaluated
instances per band.  The JSON freezes each suite hash, integer successes,
denominators, costs, and recovery values.

| Target | Total successes | Minimum draw | Mean | Maximum draw |
|---:|---:|---:|---:|---:|
| 40% | 15,730 / 32,768 | 44.5801% | 48.0042% | 51.9531% |
| 60% | 21,023 / 32,768 | 60.4492% | 64.1571% | 68.1641% |
| 80% | 27,263 / 32,768 | 80.1758% | 83.2001% | 87.7930% |

All sixteen suite hashes and per-draw numerators are recorded in
`packed_frontier_audit.json`.

## Reproduce

```bash
python3 sparse-parity/submissions/packed_frontier.py
python3 sparse-parity/submissions/audit_packed_frontier.py
python3 -m pytest -q sparse-parity/test_packed_frontier.py
```
