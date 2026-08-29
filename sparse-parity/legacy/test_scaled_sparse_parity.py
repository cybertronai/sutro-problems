"""Tests for the scaled (n=32) sparse-parity tier. Run with:
``python3 -m pytest test_scaled_sparse_parity.py``."""
from __future__ import annotations

from itertools import combinations, islice

import numpy as np
import pytest

import approx_sparse_parity as ap
import scaled_sparse_parity as sp
import sparse_parity


SMALL_SUITE = dict(n_secrets=16, repetitions=2)   # fast suites for tests


# ---------------------------------------------------------------------------
# Suite construction
# ---------------------------------------------------------------------------

def test_dev_suite_deterministic_fresh_suite_not():
    inputs_a, y_a, meta_a = sp.scaled_suite(**SMALL_SUITE)
    sp._scaled_suite_cached.cache_clear()
    inputs_b, y_b, meta_b = sp.scaled_suite(**SMALL_SUITE)
    assert np.array_equal(inputs_a, inputs_b) and meta_a == meta_b

    fresh_a = sp.scaled_suite(suite_key=None, **SMALL_SUITE)
    fresh_b = sp.scaled_suite(suite_key=None, **SMALL_SUITE)
    assert not np.array_equal(fresh_a[0], fresh_b[0])


def test_sampled_secrets_distinct_and_stratified():
    _, _, meta = sp.scaled_suite(**SMALL_SUITE)
    counts = {}
    for secret, _rep in meta:
        assert len(set(secret)) == sp.SCALED32.k_secret
        counts[secret] = counts.get(secret, 0) + 1
    assert len(counts) == SMALL_SUITE["n_secrets"]
    assert set(counts.values()) == {SMALL_SUITE["repetitions"]}


def test_complement_pairing_balances_labels():
    spec = sp.SCALED32
    inputs, y, meta = sp.scaled_suite(**SMALL_SUITE)
    n_pre = spec.n_bits * spec.m_train + spec.m_train
    for idx in range(0, len(meta), 2):
        assert np.array_equal(inputs[idx + 1, n_pre:], 1 - inputs[idx, n_pre:])
        assert np.array_equal(y[idx + 1], 1 - y[idx])
    assert float(y.mean()) == 0.5


def test_identifiability_cross_check():
    """The packed-signature uniqueness check agrees with the slow
    exhaustive candidate scan from sparse_parity on a sampled instance."""
    spec = sp.SCALED32
    inputs, _, meta = sp.scaled_suite(**SMALL_SUITE)
    row = inputs[0]
    X_tr = [
        [int(v) for v in row[i * spec.n_bits:(i + 1) * spec.n_bits]]
        for i in range(spec.m_train)
    ]
    n_tr = spec.n_bits * spec.m_train
    y_tr = [int(v) for v in row[n_tr:n_tr + spec.m_train]]
    combs = combinations(range(spec.n_bits), spec.k_secret)
    assert sparse_parity._count_matches(X_tr, y_tr, combs) == 1
    assert sparse_parity._matches_all(X_tr, y_tr, meta[0][0])


# ---------------------------------------------------------------------------
# ISD reference circuit
# ---------------------------------------------------------------------------

def test_isd_all_or_nothing_invariant():
    """On identifiable instances the ISD circuit either recovers the
    secret (all computed outputs correct) or outputs all zeros -- never
    garbage.  Checked on the cheap n=12 suite."""
    ir = sp.generate_isd(2, spec=ap.APPROX)
    run, _, _ = ap._compile_vector(ir)
    inputs, y, _ = ap.suite(repetitions=2)
    out = run(inputs)
    full_ok = (out == y).all(axis=1)
    all_zero = (out == 0).all(axis=1)
    assert bool((full_ok | all_zero).all())
    assert 0 < int(full_ok.sum()) < len(y)


def test_isd_advantage_near_analytic_on_n12():
    """Measured single-restart recovery sits in the right range: below
    the idealized information-set bound (rank-deficient subsets lose some
    cases), well above half of it."""
    ir = sp.generate_isd(1, spec=ap.APPROX)
    res = ap.evaluate(ir, repetitions=8)
    p = sp.isd_success_probability(ap.APPROX)   # 0.2545 idealized
    assert 0.5 * p < res.advantage < p


def test_isd_cost_monotone_in_restarts_and_outputs():
    costs_T = [
        sparse_parity._compile_ir(sp.generate_isd(T), sp.OP_CAP)[1]
        for T in (1, 2, 3)
    ]
    assert costs_T == sorted(costs_T) and len(set(costs_T)) == 3
    costs_f = [
        sparse_parity._compile_ir(sp.generate_isd(1, f), sp.OP_CAP)[1]
        for f in (64, 128, 256)
    ]
    assert costs_f == sorted(costs_f) and len(set(costs_f)) == 3


def test_isd_scaled_smoke():
    """T=1 on the (small) scaled dev suite: positive but small advantage,
    and every instance is all-correct or all-zero."""
    ir = sp.generate_isd(1)
    res = sp.evaluate_scaled(ir, **SMALL_SUITE)
    assert res.n_labels == 16 * 2 * 256
    assert -0.02 <= res.advantage <= 0.15
    run, _, _ = ap._compile_vector(ir, sp.OP_CAP)
    inputs, y, _ = sp.scaled_suite(**SMALL_SUITE)
    out = run(inputs)
    assert bool(((out == y).all(axis=1) | (out == 0).all(axis=1)).all())


def test_constant_zero_is_exactly_chance_on_scaled_suite():
    ir = ap.generate_mask_baseline(0, 0, spec=sp.SCALED32)
    res = sp.evaluate_scaled(ir, **SMALL_SUITE)
    assert res.raw_accuracy == 0.5 and res.advantage == 0.0


# ---------------------------------------------------------------------------
# Instruction cap
# ---------------------------------------------------------------------------

def test_generator_enforces_cap():
    with pytest.raises(ValueError, match="cap"):
        sp.generate_isd(8)   # ~290k ops > 250k


def test_compile_cap_is_parameterized():
    body = "\n".join(f"set {i},1" for i in range(2, 150_005))
    ir = "1\n" + body + "\n2"
    with pytest.raises(ValueError, match="maximum allowed length"):
        sparse_parity._compile_ir(ir)                    # default 100k cap
    _, cost, _ = sparse_parity._compile_ir(ir, 200_000)  # scaled cap passes
    assert cost > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
