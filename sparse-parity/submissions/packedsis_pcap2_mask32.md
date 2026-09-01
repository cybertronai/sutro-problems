# Sparse Parity — packed SIS walk (20% and 60% bands)

**Date:** 2026-08-31
**Problem:** Mask sparse parity, n=32 — 20% and 60% accuracy targets
**Cost:** 151,117 @ 0.3291 dev recovery (20%); 284,049 @ 0.7852 (60%)
**IRs:** [`packedsis_pcap2_mask32.ir`](packedsis_pcap2_mask32.ir) (20%),
[`packedsis_cap3_s13_mask32.ir`](packedsis_cap3_s13_mask32.ir) (60%)
**Generator:** [`packedsis.py`](packedsis.py) → `generate_packed_sis(cap=2, seed=13, g2=8)` / `generate_packed_sis(cap=3, seed=13)`

## Method

Same family as the previous record (`generate_sis_mask`: static information
set + coefficient-weight-ordered null-space walk + weight-k capture), rebuilt
around 8-bit packing:

- **Packed RREF.** The 18×33 augmented matrix [X | y] lives 5 bytes/row
  instead of 33 cells/row, so a row-XOR elimination step costs 5 ops, not 33.
  Pivot search uses truthiness masks (`and` with POW2) instead of bit
  extraction, takes the *last* eligible row as pivot (no first-found
  tracking), and marks rows used during the elimination pass.
- **Row-order walk state.** The walked vector is kept as
  (S-part packed 3 bytes *in pivot-row order*, coefficient vector packed 2
  bytes) — the free part of `s0 + Σ a_j·knob_j` is exactly `a`. This deletes
  the 18×18 pivot-order gather entirely: s0 and the knob vectors read
  straight off RREF table columns, and the scatter to column order happens
  once, after the walk, via 18-way select chains at walk-phase (low)
  addresses.
- **Cheap capture.** The coefficient weight of every visit is known at
  compile time, so only the 3 packed S-bytes need popcounting (SWAR
  nibble fold; the top byte holds 2 bits → 3 ops). Capture is 5 packed
  selects instead of 32.
- **Partial cap-2 walk (`g2=8`).** Visits: empty + all 14 singles + pairs
  within the first 8 knobs (57 visits vs 106 at full cap 2). Full cap 2
  costs +14% and recovers 0.54 — unnecessary for this band.
- **Layout:** custom generic two-phase staging (frequency-sorted addressing
  per phase, walk re-addressed from 1 into the dead RREF range, cycle-safe
  copy bridge) in `packedsis.py:_layout2`.

Cost model note: `div`/`mul` by `POW2[7]` is signed (`-128`); bit *tests*
use `and`-truthiness instead, and bit *placement* normalizes first.

## Numbers

Dev suite (1,024 instances): **cost 151,117, recovery 0.3291**, 12,789 lines.
On ten fixed, hashed 2,048-instance adjudication suites this exact IR scored
5,934/20,480 overall, with per-suite recovery min 0.2568, mean 0.2897, max
0.3169. Exact suite keys, hashes, successes, and denominators are in the
[`packed_records_audit.json`](packed_records_audit.json) artifact, reproduced
by [`audit_packed_records.py`](audit_packed_records.py).

The static information set contains 14 of columns 0–15, so there are exactly
120 possible sets. Exhaustively compiling all 120 found seed 13 (free columns
exclude 7 and 15) to be the unique cheapest layout: 151,117 versus 151,319
for the prior seed-3 record.

Previous record for the band: 1,317,480 @ 0.4453 → **8.7× cheaper**.

## 60% band — full cap-3 walk

[`packedsis_cap3_s13_mask32.ir`](packedsis_cap3_s13_mask32.ir) =
`generate_packed_sis(cap=3, seed=13)`: **284,049 @ 0.7852 dev** (30,522
lines), the current 60%-band record — cheaper than the packedwalk cap-3
entry (290,951 @ 0.6348) and ~18pp higher recovery. The static
information set (and therefore per-instance layout) is seed-dependent for
this generator, so cost varies slightly by seed (~284,049–284,258 across
seeds 0–19); `seed=13` was the cheapest seed found with a comfortable
adjudication margin. The fixed ten-suite audit scores 15,334/20,480 overall;
per-suite recovery min 0.7104, mean 0.7487, max 0.7944, comfortably above
the 0.60 line.

## Rejected: cap-1 20% variant

`generate_packed_sis(cap=1, seed=3)` measures 141,010 @ 0.257 on the dev
suite — cheaper than this submission — but the fixed audit dips to
**0.1509**, below the 0.20 adjudication line. Same failure mode as the
`packedwalk.generate(1, 1)` false lead noted below, so the 20% row stays
at 151,117.

## Reproduce

```python
import sys; sys.path.insert(0, "submissions")
import mask_sparse_parity as mp
from packedsis import generate_packed_sis

ir = generate_packed_sis(cap=2, seed=13, g2=8)
res = mp.evaluate_mask(ir)          # → cost 151117, recovery 0.3291
```

Verified bit-exact against an independent numpy emulation of the algorithm
on all 1,024 dev instances (`packedsis_xcheck.py`, 0 mismatches for cap 1/2/3
at seed 0 and the promoted cap-2/cap-3 configurations at seed 13).

## Note on `doc/mask32_bands.json`'s 20% point

The mechanical sweep in `doc/generate_mask_graph.py` reports its cheapest
20%-clearing point as `packedwalk.generate(1, 1)` (131,620 @ 0.2031 dev) —
cheaper than this submission. **Do not promote it**: the fixed ten-suite
audit gives a minimum recovery of **0.1655**, below the 0.20 line. The
dev-suite recovery alone isn't
sufficient evidence a config clears its band; the bands.json "cheapest"
column is mechanically "cheapest config clearing the *dev* suite" and
does not itself check fresh-draw margin. This submission (g2=8) was
chosen after the fixed audit found per-suite minima of 0.1587, 0.1680,
0.1826, and 0.1982 for g2=0/2/4/6, respectively, while g2=8 held at
0.2568. The exact integer results and denominators are committed in the
audit JSON rather than depending on unrecoverable random suite keys.
