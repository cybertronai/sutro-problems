"""Tests for the mask-recovery (test-set-free) sparse-parity tier. Run with
``python3 -m pytest test_mask_sparse_parity.py``."""
from __future__ import annotations

import numpy as np
import pytest

import mask_sparse_parity as mp
from mask_sparse_parity import Spec


SMALL = Spec(n_bits=12, k_secret=3, m_train=8, m_test=0)
SMALL_SUITE = dict(spec=SMALL, n_secrets=32, repetitions=4)


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

def test_dev_suite_deterministic_fresh_not():
    a1, m1, meta1 = mp.mask_suite(**SMALL_SUITE)
    mp._mask_suite_cached.cache_clear()
    a2, m2, meta2 = mp.mask_suite(**SMALL_SUITE)
    assert np.array_equal(a1, a2) and meta1 == meta2

    f1 = mp.mask_suite(suite_key=None, n_secrets=4, repetitions=2, spec=SMALL)
    f2 = mp.mask_suite(suite_key=None, n_secrets=4, repetitions=2, spec=SMALL)
    assert not np.array_equal(f1[0], f2[0])


def test_targets_are_secret_masks_and_instances_identifiable():
    spec = SMALL
    inputs, masks, meta = mp.mask_suite(**SMALL_SUITE)
    assert masks.shape[1] == spec.n_bits
    for idx in range(0, len(meta), 17):
        secret, _rep = meta[idx]
        expect = np.zeros(spec.n_bits, dtype=np.int16)
        expect[list(secret)] = 1
        assert np.array_equal(masks[idx], expect)
    # spot-check identifiability with the slow exhaustive scan
    row = inputs[0]
    X_tr = [
        [int(v) for v in row[i * spec.n_bits:(i + 1) * spec.n_bits]]
        for i in range(spec.m_train)
    ]
    y_tr = [int(v) for v in row[spec.n_bits * spec.m_train:]]
    from itertools import combinations
    n_match = sum(
        all(sum(X_tr[r][c] for c in comb) % 2 == y_tr[r]
            for r in range(spec.m_train))
        for comb in combinations(range(spec.n_bits), spec.k_secret)
    )
    assert n_match == 1


def test_final_default_sample_size(monkeypatch):
    """suite_key=None resolves to the FINAL sample sizes (checked with
    shrunken constants so the test stays fast)."""
    monkeypatch.setattr(mp, "FINAL_SECRETS", 8)
    monkeypatch.setattr(mp, "FINAL_REPS", 2)
    inputs, _, _ = mp.mask_suite(spec=SMALL, suite_key=None)
    assert inputs.shape[0] == 8 * 2


def test_oversampling_secrets_rejected():
    with pytest.raises(ValueError, match="distinct secrets"):
        mp.mask_suite(spec=SMALL, n_secrets=300, repetitions=2,
                      suite_key="oversample-test")


# ---------------------------------------------------------------------------
# Scan family
# ---------------------------------------------------------------------------

def test_scan_full_recovers_nearly_everything():
    res = mp.evaluate_mask(mp.generate_scan(spec=SMALL), **SMALL_SUITE)
    assert res.recovery >= 0.95


def test_scan_zero_steps_is_min_support_ge():
    res = mp.evaluate_mask(mp.generate_scan(0, spec=SMALL), **SMALL_SUITE)
    assert 0.15 < res.recovery < 0.5


def test_scan_monotone_in_steps():
    recs, costs = [], []
    for s in (0, 7, 15):
        r = mp.evaluate_mask(mp.generate_scan(s, spec=SMALL), **SMALL_SUITE)
        recs.append(r.recovery)
        costs.append(r.cost)
    assert recs == sorted(recs)
    assert costs == sorted(costs) and len(set(costs)) == 3


def test_scan_all_or_nothing():
    """Scan output is always exactly the secret mask or exactly zeros --
    the weight-k capture can never admit a non-secret."""
    ir = mp.generate_scan(7, spec=SMALL)
    run, _, _ = mp._compile_vector(ir, mp.OP_CAP)
    inputs, masks, _ = mp.mask_suite(**SMALL_SUITE)
    out = run(inputs)
    exact = (out == masks).all(axis=1)
    zeros = (out == 0).all(axis=1)
    assert bool((exact | zeros).all())
    assert 0 < int(exact.sum()) < len(masks)


# ---------------------------------------------------------------------------
# Other families + scoring
# ---------------------------------------------------------------------------

def test_enum_full_is_perfect_on_small():
    ir = mp.generate_enum_mask(220, spec=SMALL)
    res = mp.evaluate_mask(ir, **SMALL_SUITE)
    assert res.recovery == 1.0


def test_isd_mask_positive_recovery():
    res = mp.evaluate_mask(mp.generate_isd_mask(2, spec=SMALL), **SMALL_SUITE)
    assert 0.2 < res.recovery < 0.8


def test_sis_mask_positive_recovery():
    res = mp.evaluate_mask(mp.generate_sis_mask(1, 2, spec=SMALL), **SMALL_SUITE)
    assert 0.1 < res.recovery < 0.9


def test_sis_mask_cheaper_than_full_scan():
    sis = mp.evaluate_mask(mp.generate_sis_mask(1, 4, spec=SMALL), **SMALL_SUITE)
    scan = mp.evaluate_mask(
        mp.generate_scan(0, walk="weight", weight_cap=4, spec=SMALL),
        **SMALL_SUITE,
    )
    assert sis.cost < scan.cost

def test_output_arity_validated():
    n_in = mp._n_inputs(SMALL)
    bad = ",".join(str(i) for i in range(1, n_in + 1)) + "\n1,2,3"
    with pytest.raises(ValueError, match="outputs"):
        mp.evaluate_mask(bad, **SMALL_SUITE)


def test_cap_enforced_by_generators():
    with pytest.raises(ValueError, match="cap"):
        mp.generate_enum_mask(20_000)   # ~2.5M ops > 2M


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
