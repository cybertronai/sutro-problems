# Sparse Parity — packed-column scan, cap 5, compacted

**Authors:** [@mikech76](https://github.com/mikech76)
**Date:** 2026-10-04  
**Problem:** MASK32 sparse parity, 100% target  
**Cost:** 382,840
**IR lines:** 58,530 (58,528 body operations)
**IR:** [`packedscan_beat382840_mask32.ir`](packedscan_beat382840_mask32.ir)  
**Generator:** [`generator_beat382840.py`](generator_beat382840.py) → [`../packed_sparse_parity.py`](../packed_sparse_parity.py)  
**Audit:** [`audit_packed_records_beat382840.json`](audit_packed_records_beat382840.json)

## Result

`generate_packed_scan(5, compact_predicates=True, compact_flow=True,
walk_order_bias=1.0)` recovers **100.0000%** of the deterministic 1,024-instance
dev suite (`DEV_SUITE_KEY = "mask-dev"`, 128 secrets × 8 repetitions) at static
read cost **382,840** — a delta of **−9,826 (−2.5024%)** against the previous
100% record, `generate_packed_scan(5)` = 392,666, at identical recovery.

Four independent, final-sized deterministic validation suites (256 secrets × 8
repetitions, 2,048 instances each, all with fixed public suite keys) produced
**100.0000%** recovery on every suite: **8,192 / 8,192** exact 32-cell matches.
No non-secret mask was emitted on any suite. Per-suite keys, `inputs_sha256`,
`targets_sha256`, integer successes and denominators are recorded in the audit
JSON, together with the dev-suite block.

## Construction

This is the existing packed-column scan circuit, not a new algorithm. The 18 rows
of each augmented `[X | y]` column are packed into three nonnegative 6-bit
cells, so branchless elimination updates all rows with three bytewise XORs
instead of an 18×33 bit matrix. Earlier pivot columns are not revisited: once a
column is one-hot, a later unused pivot row is necessarily zero there.

After elimination the affine candidate state is only three packed pivot-row
cells. Coefficient vectors of weight at most `5` are visited in
binary-reflected Gray order after filtering; target pivot weights are captured
with exact bit predicates, zero-weight capture uses an inverted select directly,
weight-one capture uses packed sum-versus-XOR, and the remaining rare targets use
a packed popcount. A two-phase SSA pass removes dead writes and reuses cells by
exact liveness; a final frequency sort is the rearrangement-inequality optimum
for the emitted fixed access trace.

The whole −2.5024% comes from two upstream layout flags being switched on —
`compact_predicates=True` and `compact_flow=True` — which shorten the predicate
and control-flow encodings without changing the circuit's semantics. Upstream
ships both as `False`, i.e. the shipped 80%/100% records were submitted without
them. The selection surface is exactly those two booleans: no threshold,
constant, layout or schedule was introduced or changed.

## Selection surface disclosure

**No parameter was chosen by looking at a held-out or adjudication suite.** The
configuration space searched was `{False, True}²` for `compact_predicates` ×
`compact_flow` (4 configurations), scored on the deterministic dev suite only.
`weight_cap=5` is the upstream default and the 100% target's cap.
`walk_order_bias=1.0` is the upstream default (`packed_sparse_parity.py:704`) and
was left untouched — the generator wrapper exposes it only as a passthrough
argument and never overrides it. There is no seed to tune: the generator is
deterministic and takes no random state. The four audit suites were computed
once, after the configuration was fixed, and are reported as measurement, not
used as a selection signal.

## Rank-deficiency caveat

As with the repository's previous full scan, a rank-deficient training matrix can
have more than 14 free coordinates. This circuit records the first 14 and may
miss a secret using a later free coordinate; the repository documentation
estimates that event near `2^-14`. The 100% claim here is a measured dev-tier
result plus a full-rank proof — **not** a claim over every rank-deficient input.
On the fixed suites above the effect did not appear (8,192 / 8,192 exact), but
those suites are not conditioned on rank deficiency, so they do not bound it.

## Toolchain

Measured with Python 3.14.6, NumPy 2.5.1, Windows 11 (10.0.26100), using the
upstream evaluator at commit
[`aa6c51e`](https://github.com/cybertronai/sutro-problems/blob/aa6c51e6ad661089cbf71a39c95c577c216b627f/sparse-parity/mask_sparse_parity.py)
(`mask_sparse_parity.evaluate_mask`, `engine="vector"`, `SUITE_VERSION =
"mask-sparse-parity-v1"`). The generator is pure Python with no third-party
dependency; the evaluator requires only NumPy. Layout compaction is integer
arithmetic on small integers, so the emitted IR and the cost are expected to be
independent of NumPy version; only recovery timing is machine-dependent. The
audit JSON records the evaluator commit so the numbers are pinned to code.

## Reproduce

Regenerate the IR byte-identically from the upstream generator:

```bash
python submissions/generator_beat382840.py \
    --upstream-dir . \
    --out /tmp/regen.ir \
    --compare submissions/packedscan_beat382840_mask32.ir
```

Score it:

```python
import mask_sparse_parity as mp
import packed_sparse_parity as packed

ir = packed.generate_packed_scan(
    5, compact_predicates=True, compact_flow=True, walk_order_bias=1.0)
result = mp.evaluate_mask(ir)
assert len(ir.splitlines()) == 58530
assert result.cost == 382840
assert result.recovery == 1.0
```

SHA-256 of the stored IR, LF-normalised (the committed file uses CRLF
endings, so the raw-byte digest differs): `91089d94694ff71d8598d4a0333187ca01737a4e6867689c41078b145189ab86`.
Raw-byte SHA-256 as committed: `08757dccc081ed85fb3ada23b6fe95a10f8857335acac0c18dd131146e576280`.