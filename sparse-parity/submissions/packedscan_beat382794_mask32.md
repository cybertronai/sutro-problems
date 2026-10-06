# Sparse Parity — packed-column scan, cap 5, compacted, walk bias 15

**Authors:** [@mikech76](https://github.com/mikech76)
**Date:** 2026-10-05  
**Problem:** MASK32 sparse parity, 100% target  
**Cost:** 382,794
**IR lines:** 58,530 (58,528 body operations)
**IR:** [`packedscan_beat382794_mask32.ir`](packedscan_beat382794_mask32.ir)  
**Generator:** [`generator_beat382794.py`](generator_beat382794.py) → [`../packed_sparse_parity.py`](../packed_sparse_parity.py)  
**Audit:** [`audit_packed_records_beat382794.json`](audit_packed_records_beat382794.json)

## Result

`generate_packed_scan(5, compact_predicates=True, compact_flow=True,
walk_order_bias=15.0)` recovers **100.0000%** of the deterministic 1,024-instance
dev suite (`DEV_SUITE_KEY = "mask-dev"`, 128 secrets × 8 repetitions) at static
read cost **382,794** — **−46 (−0.0120%)** against the current record
`walk_order_bias=1.0` = 382,840, and **−9,872 (−2.5140%)** against the shipped
100% record `generate_packed_scan(5)` = 392,666, at identical dev recovery.

Four independent, final-sized deterministic validation suites (256 secrets × 8
repetitions, 2,048 instances each, four fixed public suite keys not used by any
earlier audit) produced **8,191 / 8,192** exact 32-cell matches. Three of the
four suites are 2,048 / 2,048; the fourth is 2,047 / 2,048. The single miss is
the rank-deficiency case analysed below, and it is **not** specific to this
configuration — the `walk_order_bias=1.0` record and the upstream baseline both
miss the same instance on the same suite.

## What changed relative to the 382,840 record

Only `walk_order_bias`: `1.0` → `15.0`. The two flags
`compact_predicates=True` and `compact_flow=True` were already ON in the
382,840 record.

`walk_order_bias` is a tie-break weight inside `_allocate_phase`
(`packed_sparse_parity.py`), the linear-scan slot allocator used by
`_optimize_two_phase_ir` for the walk phase. It changes the sort key applied to
values that become live at the same program point:

| `walk_order_bias` | sort key |
| - | - |
| `None` | `(-reads, end - start, id)` |
| `1.0` | `((end - start) / (reads + 1), id)` |
| other `b` | `(end - start) / (reads + b)`, then `-reads`, then `id` |

Raising the denominator bias reduces the relative influence of read count and
puts more emphasis on short lifetimes. For this circuit, the resulting address
assignment lowers the measured static read cost.

**This is a layout change, not a circuit change.** The two IRs are provably the
same function:

- **Opcode sequence identical** — all 58,528 body instructions, same opcodes in
  the same order.
- **3,428 of 58,528 body lines (5.857%) differ**, and every difference is in
  the register operands: 3,577 operand slots are reassigned across **117
  distinct slot transpositions**. No instruction is added, removed or reordered.
- **Output matrices identical on 7,168 instances** — the official dev suite plus
  three fresh `suite_key=None` adjudication draws (2,048 each), compared
  cell-for-cell across all 32 output cells. Zero disagreements.

So the walk visits the same 3,473 bounded-weight Gray states with the same
4,759 total Hamming transitions in both configurations. The walk *order* is not
what differs; the *address assignment* along that unchanged walk is. That is
why the cost falls while the recovery is bit-for-bit the same.

The dominant transposition is slot 14 ↔ 15 (1,619 occurrences each direction),
i.e. the two cheapest walk-phase slots swap their most frequent value.

## Rank-deficiency caveat — and a correction to the 382,840 report

On suite `packedscan-382794-pr2-adjudication-20261005-a`, instance 748 is missed.
Diagnosed: that instance's training matrix has **rank(X) = 17**, hence **15 free
coordinates — one more than the 14 the circuit records**. The secret is
`[1, 3, 12, 26, 31]`; the circuit returns the all-zero mask, because the secret
uses a free coordinate it never visited. Parity rows are consistent
(`parity_rows_ok: true`), so this is not a malformed input.

The 382,840 submission report estimated this event near `2^-14` and stated that
the fixed suites "did not bound it". The observed frequency in these suites is
**1 miss in 8,192 final-tier instances ≈ 2^-13**; this finite sample does not
establish a bound on the population failure rate. The miss is reproduced
identically by all three configurations tested (bias 15.0, bias 1.0, and the
upstream baseline `generate_packed_scan(5)`), each failing on the same instance
748. It is a property of the packed-scan family's 14-coordinate recording, not
of the bias parameter.

The 100% claim here is therefore a measured dev-tier result plus a full-rank
proof — **not** a claim over every rank-deficient input, and the `8191/8192`
figure is reported as measured rather than rounded up.

## Selection surface disclosure

The selection surface is exactly three upstream parameters:
`compact_predicates` (bool), `compact_flow` (bool) and `walk_order_bias`
(float). The first two were fixed to `True` by the 382,840 record and are not
re-tuned here. `walk_order_bias` was selected **by static cost alone** —
`mp._compile_ir(ir, OP_CAP)[1]` — swept over a finite grid, with recovery never
used as the objective. The submitted variant keeps the accepted walk-state
order. No held-out or adjudication suite was consulted during selection.

**There is no seed to tune.** `generate_packed_scan` takes no random state and
no `seed` argument; `bounded_weight_gray_states` enumerates Gray codes
deterministically. Verified over 16 runs spanning four `PYTHONHASHSEED` values
(0, 1, 42, 999) and four `random`/`numpy.random` seeds (0, 1, 42, 1337) —
**one single IR sha256**. The seed in this benchmark's vocabulary is the
`suite_key`, i.e. which random set of secrets and training rows a suite
contains; it is a property of the suite, not of the solution.

## Toolchain

Measured with Python 3.14.6, NumPy 2.5.3, Windows 11, using the upstream
evaluator at commit
[`4152e6e`](https://github.com/cybertronai/sutro-problems/blob/4152e6e018513819ba86610489d4cb098ec2ef41/sparse-parity/mask_sparse_parity.py)
(`mask_sparse_parity.evaluate_mask`, `engine="vector"`, `SUITE_VERSION =
"mask-sparse-parity-v1"`). All recovery figures come from
`mp.evaluate_mask`; all cost figures come from `mp._compile_ir`. No metric was
computed by any other route. IR generation uses Python arithmetic; the
generator's evaluator import stack requires NumPy. Layout compaction and slot
allocation use Python integer and floating-point arithmetic, including the
floating-point allocation bias.
The emitted IR and the static cost do not depend on the NumPy version;
recovery evaluation uses NumPy. The audit JSON records the
evaluator commit so the numbers are pinned to code.

Note: the previously reported evaluator commit for the 382,840 audit,
`aa6c51e`, is upstream `main` *before* PR #104. `mask_sparse_parity.py` and
`packed_sparse_parity.py` are byte-identical between `aa6c51e` and `4152e6e`
(verified by `git diff --stat`, empty), so the two records are directly
comparable.

## Reproduce

Regenerate the IR byte-identically from the upstream generator:

```bash
python submissions/generator_beat382794.py \
    --upstream-dir . \
    --out /tmp/regen.ir \
    --compare submissions/packedscan_beat382794_mask32.ir
```

Score it:

```python
import mask_sparse_parity as mp
import packed_sparse_parity as packed

ir = packed.generate_packed_scan(
    5, compact_predicates=True, compact_flow=True, walk_order_bias=15.0)
result = mp.evaluate_mask(ir)
assert len(ir.splitlines()) == 58530
assert result.cost == 382794
assert result.recovery == 1.0
```

SHA-256 of the committed IR: `6bbebaa34ab25cc3edd1a73fc1fd9b655bce03dfccef6ac893c49f4cdf2bd675`.
The committed file uses LF endings, so its raw-byte and LF-normalised digests
are identical.
