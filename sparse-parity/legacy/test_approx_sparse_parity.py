"""Tests for the approximate sparse-parity benchmark. Run with:
``python3 -m pytest test_approx_sparse_parity.py`` or just
``python3 test_approx_sparse_parity.py`` (built-in __main__ runner)."""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest

import approx_sparse_parity as ap
import sparse_parity


R = 2  # smallest legal repetition count -- keeps the tests fast


# ---------------------------------------------------------------------------
# Suite construction
# ---------------------------------------------------------------------------

def test_suite_deterministic():
    """Rebuilding the suite from scratch reproduces it bit-for-bit; a
    different suite key produces different instances."""
    inputs_a, y_a, meta_a = ap.suite(repetitions=R)
    ap._suite_cached.cache_clear()
    inputs_b, y_b, meta_b = ap.suite(repetitions=R)
    assert np.array_equal(inputs_a, inputs_b)
    assert np.array_equal(y_a, y_b)
    assert meta_a == meta_b

    inputs_c, _, _ = ap.suite(repetitions=R, suite_key="other")
    assert not np.array_equal(inputs_a, inputs_c)


def test_suite_stratified_over_all_secrets():
    """Every one of the C(12,3)=220 secrets appears exactly R times."""
    _, _, meta = ap.suite(repetitions=R)
    counts = {}
    for secret, _rep in meta:
        counts[secret] = counts.get(secret, 0) + 1
    assert len(counts) == 220 == ap.n_secrets()
    assert set(counts.values()) == {R}


def test_suite_instances_uniquely_identifiable():
    """Each training set admits exactly one matching k-subset: the secret."""
    spec = ap.APPROX
    inputs, _, meta = ap.suite(repetitions=R)
    n_tr = spec.n_bits * spec.m_train
    for idx in range(0, len(meta), 97):  # spot-check a spread of instances
        row = inputs[idx]
        X_tr = [
            [int(v) for v in row[i * spec.n_bits:(i + 1) * spec.n_bits]]
            for i in range(spec.m_train)
        ]
        y_tr = [int(v) for v in row[n_tr:n_tr + spec.m_train]]
        combs = combinations(range(spec.n_bits), spec.k_secret)
        assert sparse_parity._count_matches(X_tr, y_tr, combs) == 1
        assert sparse_parity._matches_all(X_tr, y_tr, meta[idx][0])


def test_complement_pairing_balances_labels():
    """Repetition 2t+1 holds the bitwise complements of repetition 2t's test
    rows in the same order, so labels flip (k odd) and any constant guess
    scores exactly 50% over the whole suite."""
    spec = ap.APPROX
    inputs, y, meta = ap.suite(repetitions=R)
    n_pre = spec.n_bits * spec.m_train + spec.m_train
    for idx in range(0, len(meta) - 1, 2):
        assert meta[idx][0] == meta[idx + 1][0]
        X_te_even = inputs[idx, n_pre:]
        X_te_odd = inputs[idx + 1, n_pre:]
        assert np.array_equal(X_te_odd, 1 - X_te_even)
        assert np.array_equal(y[idx + 1], 1 - y[idx])
    assert float(y.mean()) == 0.5


def test_full_cube_audit_covers_every_test_row():
    """Full-cube mode partitions all 2**n test rows into blocks: every row
    appears exactly once per secret (checked on a smaller spec)."""
    spec = sparse_parity.Spec(n_bits=8, k_secret=3, m_train=8, m_test=32)
    inputs, _, meta = ap.suite(spec=spec, full_cube=True)
    reps = 256 // 32
    assert len(meta) == math.comb(8, 3) * reps
    n_pre = spec.n_bits * spec.m_train + spec.m_train
    weights = 1 << np.arange(spec.n_bits)
    rows_seen = (
        inputs[:, n_pre:].reshape(len(meta), spec.m_test, spec.n_bits) @ weights
    )
    for s in range(0, len(meta), reps * 19):  # spot-check several secrets
        secret_rows = rows_seen[s - s % reps:s - s % reps + reps].ravel()
        assert sorted(secret_rows.tolist()) == list(range(256))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_full_baseline_is_exact():
    """Searching all 220 candidates and computing all 32 outputs recovers
    every label: advantage exactly 1.0."""
    res = ap.evaluate(ap.generate_approx_baseline(), repetitions=R)
    assert res.raw_accuracy == 1.0
    assert res.advantage == 1.0
    assert res.n_labels == 220 * R * 32


def test_partial_baselines_match_analytic_advantage():
    """With unique identifiability and complement-paired tests, the (q, f)
    baseline's aggregate advantage is exactly (q/220) * (f/32)."""
    for q, f in [(0, 0), (55, 32), (220, 8), (110, 16), (1, 32)]:
        ir = ap.generate_approx_baseline(q, f)
        res = ap.evaluate(ir, repetitions=R)
        assert res.advantage == pytest.approx((q / 220) * (f / 32), abs=1e-12)


def test_mask_baseline_is_exact_and_cheaper():
    """The two-phase mask decoder is label-for-label exact at full settings
    and much cheaper than try-each-candidate (predictions cost O(n) instead
    of O(candidates))."""
    res_mask = ap.evaluate(ap.generate_mask_baseline(), repetitions=R)
    res_naive = ap.evaluate(ap.generate_approx_baseline(), repetitions=R)
    assert res_mask.advantage == 1.0
    assert res_mask.cost < res_naive.cost / 3


def test_mask_baseline_matches_analytic_advantage():
    """The (q, f) mask baseline hits exactly (q/220) * (f/32) too."""
    for q, f in [(0, 16), (55, 32), (220, 8), (110, 16)]:
        ir = ap.generate_mask_baseline(q, f)
        res = ap.evaluate(ir, repetitions=R)
        assert res.advantage == pytest.approx((q / 220) * (f / 32), abs=1e-12)


def test_energy_monotone_in_both_knobs():
    """Static cost strictly grows with candidates searched and with
    outputs computed."""
    costs_q = [
        ap.evaluate(ap.generate_approx_baseline(q, 32), repetitions=R).cost
        for q in (55, 110, 220)
    ]
    assert costs_q == sorted(costs_q) and len(set(costs_q)) == 3
    costs_f = [
        ap.evaluate(ap.generate_approx_baseline(220, f), repetitions=R).cost
        for f in (8, 16, 32)
    ]
    assert costs_f == sorted(costs_f) and len(set(costs_f)) == 3


def test_score_approx_threshold():
    """score_approx returns the cost at or below the measured advantage and
    rejects above it."""
    ir = ap.generate_approx_baseline(110, 16)  # advantage exactly 0.25
    cost = ap.score_approx(ir, 0.25, repetitions=R)
    assert cost == ap.evaluate(ir, repetitions=R).cost
    with pytest.raises(ValueError, match="below target"):
        ap.score_approx(ir, 0.26, repetitions=R)


def test_output_count_mismatch_rejected():
    """An IR with the right inputs but wrong output arity is rejected."""
    spec = ap.APPROX
    n_in = sparse_parity._n_inputs(spec)
    input_addrs = ",".join(str(a) for a in range(1, n_in + 1))
    bad = f"{input_addrs}\n1,2,3,4,5"
    with pytest.raises(ValueError, match="outputs"):
        ap.evaluate(bad, repetitions=R)


# ---------------------------------------------------------------------------
# Vectorized engine vs. reference scalar simulator
# ---------------------------------------------------------------------------

def test_vector_engine_matches_reference_on_suite():
    """Batched numpy execution and the pure-Python simulator agree label
    for label on a small approximate baseline."""
    ir = ap.generate_approx_baseline(10, 4)
    vec = ap.evaluate(ir, repetitions=R, engine="vector")
    ref = ap.evaluate(ir, repetitions=R, engine="reference")
    assert vec == ref


def test_vector_engine_matches_reference_on_all_ops():
    """Every opcode (arithmetic wraparound included) matches the scalar
    simulator on random 8-bit inputs."""
    ir = "\n".join([
        "1,2,3",
        "add 4,1,2",        # wraps at 8 bits
        "sub 5,1,2",
        "mul 6,1,2",
        "xor 7,1,2",
        "and 8,1,2",
        "or 9,1,2",
        "not 10,1",
        "abs 11,5",
        "cmp 12,1,2,lt",
        "cmp 13,1,2,ge",
        "select 14,12,1,2",
        "set 15,200",        # canonicalized to -56
        "copy 16,15",
        "div 17,1,3",
        "copy 18,4",
        "add 18,6",          # two-operand form: dest op= src
        "4,5,6,7,8,9,10,11,12,13,14,15,16,17,18",
    ])
    run, _, _ = ap._compile_vector(ir)
    rng = np.random.default_rng(0)
    a = rng.integers(-128, 128, size=64)
    b = rng.integers(-128, 128, size=64)
    c = rng.integers(1, 128, size=64)  # nonzero divisor
    batch = np.stack([a, b, c], axis=1).astype(np.int16)
    got = run(batch)
    for i in range(64):
        expected, _ = sparse_parity._simulate(ir, [int(a[i]), int(b[i]), int(c[i])])
        assert got[i].tolist() == expected


def test_vector_engine_chunked_matches_unchunked(monkeypatch):
    """The memory-bounding chunked path (used when n_mem x instances would
    exceed the cell cap) produces identical results."""
    ir = ap.generate_mask_baseline(10, 4)
    baseline = ap.evaluate(ir, repetitions=R)
    monkeypatch.setattr(ap, "_MAX_BATCH_CELLS", 1 << 12)  # force many chunks
    assert ap.evaluate(ir, repetitions=R) == baseline


def test_vector_cost_matches_reference_compiler():
    """The batched compiler reports the same static read cost as
    ``sparse_parity._compile_ir``."""
    ir = ap.generate_approx_baseline(20, 8)
    _, vec_cost, _ = ap._compile_vector(ir)
    _, ref_cost, _ = sparse_parity._compile_ir(ir)
    assert vec_cost == ref_cost


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
