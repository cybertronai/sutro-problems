"""Tests for the mask-recovery (test-set-free) sparse-parity tier. Run with
``python3 -m pytest test_mask_sparse_parity.py``."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import mask_sparse_parity as mp
from mask_sparse_parity import Spec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submissions"))
import packedsis  # noqa: E402
import packedwalk  # noqa: E402
import septwalk  # noqa: E402


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

def test_renumber_preserves_recovery_and_reduces_cost():
    ir = mp.generate_scan(0, walk="weight", weight_cap=2, spec=SMALL)
    r0 = mp.evaluate_mask(ir, **SMALL_SUITE)
    r1 = mp.evaluate_mask(mp.renumber_addresses(ir), **SMALL_SUITE)
    assert r0.recovery == r1.recovery
    assert r1.cost < r0.cost

def test_stage_walk_layout_preserves_recovery():
    ir = mp.generate_scan(0, walk="weight", weight_cap=3, spec=SMALL)
    r0 = mp.evaluate_mask(ir, **SMALL_SUITE)
    r1 = mp.evaluate_mask(mp.optimize_layout(ir), **SMALL_SUITE)
    assert r0.recovery == r1.recovery
    assert r1.cost < mp.evaluate_mask(
        mp.renumber_addresses(ir), **SMALL_SUITE).cost

def test_output_arity_validated():
    n_in = mp._n_inputs(SMALL)
    bad = ",".join(str(i) for i in range(1, n_in + 1)) + "\n1,2,3"
    with pytest.raises(ValueError, match="outputs"):
        mp.evaluate_mask(bad, **SMALL_SUITE)


def test_cap_enforced_by_generators():
    with pytest.raises(ValueError, match="cap"):
        mp.generate_enum_mask(20_000)   # ~2.5M ops > 2M


# ---------------------------------------------------------------------------
# Bit-packed families (submissions/): same algorithms as above, cells hold
# multiple bits instead of one. Hardcoded to MASK32 (n=32, m=18, k=5), so
# these run against the full dev suite rather than SMALL.
# ---------------------------------------------------------------------------

def test_packedsis_positive_recovery_and_cheaper_than_reference():
    # Not bit-identical to generate_sis_mask: packedsis takes the *last*
    # eligible pivot row (no first-found tracking) instead of the first,
    # so it's a different, independently-verified algorithm in the same
    # family (checked against its own numpy emulation in
    # submissions/packedsis_xcheck.py), not a drop-in reimplementation.
    ref = mp.evaluate_mask(mp.generate_sis_mask(1, 2, seed=3))
    packed = mp.evaluate_mask(packedsis.generate_packed_sis(cap=2, seed=3))
    assert 0.1 < packed.recovery < 0.9
    assert packed.cost < ref.cost


def test_packedsis_tuned_partial_walk_record():
    """Seed 13 is strictly cheaper than the prior seed-3 20% record."""
    prior = mp.evaluate_mask(packedsis.generate_packed_sis(
        cap=2, seed=3, g2=8))
    tuned = mp.evaluate_mask(packedsis.generate_packed_sis(
        cap=2, seed=13, g2=8))
    assert tuned.cost < prior.cost
    assert tuned.recovery >= 0.20


def test_packedwalk_matches_reference_recovery():
    ref = mp.evaluate_mask(mp.optimize_layout(mp.generate_sis_mask(1, 2)))
    packed = mp.evaluate_mask(packedwalk.generate(1, 2))
    assert packed.recovery == ref.recovery
    assert packed.cost < ref.cost


def test_packedwalk_seed5_record_and_generated_band_data_agree():
    """The historical 40% packed-walk record remains the seed-5 point."""
    seed0 = mp.evaluate_mask(packedwalk.generate(1, 2, seed=0))
    seed5 = mp.evaluate_mask(packedwalk.generate(1, 2, seed=5))
    assert seed5.cost == seed0.cost == 163_378
    assert seed5.recovery > seed0.recovery

    bands_path = Path(__file__).with_name("doc") / "mask32_bands.json"
    bands = json.loads(bands_path.read_text())["bands"]
    band40 = next(b for b in bands if b["target"] == 0.4)
    assert band40["adjudicated_best"]["call"] == "generate_packed_scan(2)"
    assert (band40["runner_up"]["call"] ==
            "packedwalk.generate(1, 2, seed=5)")
    assert band40["runner_up"]["recovery"] == seed5.recovery


def test_packed_record_irs_regenerate_byte_identically():
    records = {
        "packedsis_pcap2_mask32.ir":
            packedsis.generate_packed_sis(cap=2, seed=13, g2=8),
        "packedwalk1_cap2_s5_mask32.ir":
            packedwalk.generate(1, 2, seed=5),
        "packedsis_cap3_s13_mask32.ir":
            packedsis.generate_packed_sis(cap=3, seed=13),
        "septwalk_wcap3_mask32.ir":
            septwalk.generate_staged(weight_cap=3),
        "septwalk_mask32.ir":
            septwalk.generate_staged(weight_cap=5),
    }
    submissions = Path(__file__).with_name("submissions")
    for filename, generated in records.items():
        committed = (submissions / filename).read_text()
        assert generated.rstrip("\n") + "\n" == committed.rstrip("\n") + "\n"


def test_septwalk_full_recovery():
    res = mp.evaluate_mask(septwalk.generate_staged(weight_cap=5))
    assert res.recovery == 1.0
    ref = mp.evaluate_mask(
        mp.optimize_layout(
            mp.generate_scan(0, walk="weight", weight_cap=5)))
    assert res.cost < ref.cost


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
