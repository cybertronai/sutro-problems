"""Regression tests for packed_sparse_parity.py."""
from __future__ import annotations

from collections import Counter
from math import isqrt
from pathlib import Path

import numpy as np
import pytest

import mask_sparse_parity as mp
import packed_sparse_parity as packed


HERE = Path(__file__).resolve().parent
EXPECTED = {
    1: (10_732, 135_348, 0.2001953125),
    2: (13_752, 151_943, 0.5654296875),
    3: (23_325, 200_937, 0.8984375),
    5: (64_073, 409_001, 1.0),
}


def _address_universe_and_counts(ir: str):
    lines = [line.strip() for line in ir.splitlines() if line.strip()]
    inputs = [int(text) for text in lines[0].split(",")]
    outputs = [int(text) for text in lines[-1].split(",")]
    universe = set(inputs) | set(outputs)
    reads: Counter[int] = Counter()
    for line in lines[1:-1]:
        op, args = packed._parse_instruction(line)
        universe.add(int(args[0]))
        for source in packed._source_addresses(op, args):
            universe.add(source)
            reads[source] += 1
    for output in outputs:
        reads[output] += 1
    return inputs, outputs, universe, reads


@pytest.mark.parametrize("cap", [1, 2, 3, 5])
def test_artifact_reproducible_and_dev_score(cap):
    expected_lines, expected_cost, expected_recovery = EXPECTED[cap]
    generated = packed.generate_packed_scan(cap)
    stored = (
        HERE / "submissions" / f"packedscan{cap}_mask32.ir"
    ).read_text().rstrip("\n")
    assert generated == stored
    assert len(generated.splitlines()) == expected_lines

    result = mp.evaluate_mask(generated)
    assert result.cost == expected_cost
    assert result.recovery == expected_recovery


def test_cap5_walk_attains_transition_lower_bound():
    states = packed.bounded_weight_gray_states(14, 5)
    assert states[0] == 0
    assert len(states) == 3_473
    assert len(set(states)) == len(states)
    assert all(packed._bit_count(state) <= 5 for state in states)
    assert packed.transition_cost(states) == 4_759
    assert packed.bounded_weight_transition_lower_bound(14, 5) == 4_759


def test_optimizer_preserves_raw_semantics_on_arbitrary_inputs():
    raw = packed.generate_packed_scan(3, optimize_layout=False)
    optimized = packed.generate_packed_scan(3)
    run_raw, _, n_raw = mp._compile_vector(raw, mp.OP_CAP)
    run_optimized, _, n_optimized = mp._compile_vector(optimized, mp.OP_CAP)
    assert n_raw == n_optimized == 594

    rng = np.random.default_rng(20260901)
    inputs = rng.integers(0, 2, size=(17, 594), dtype=np.int16)
    assert np.array_equal(run_raw(inputs), run_optimized(inputs))


def test_partial_result_is_only_secret_or_zero():
    ir = packed.generate_packed_scan(3)
    run, _, _ = mp._compile_vector(ir, mp.OP_CAP)
    inputs, masks, _ = mp.mask_suite(
        n_secrets=16,
        repetitions=2,
        suite_key="packed-scan-partial-v2",
    )
    output = run(inputs)
    exact = (output == masks).all(axis=1)
    zero = (output == 0).all(axis=1)
    assert bool((exact | zero).all())
    assert 0 < int(exact.sum()) < len(exact)


def test_storage_floor_and_fixed_trace_address_optimum():
    ir = packed.generate_packed_scan(5)
    inputs, outputs, universe, reads = _address_universe_and_counts(ir)
    assert len(inputs) == len(set(inputs)) == 594
    assert len(outputs) == 32
    assert universe == set(range(1, 595))

    # Rearrangement-inequality optimum for this exact frequency multiset.
    frequencies = sorted((reads[address] for address in universe), reverse=True)
    lower_bound = sum(
        frequency * (isqrt(position - 1) + 1)
        for position, frequency in enumerate(frequencies, start=1)
    )
    _, actual_cost, _ = mp._compile_ir(ir, mp.OP_CAP)
    assert actual_cost == lower_bound == 409_001
    assert packed._global_frequency_layout(ir) == ir


def test_mask32_only_contract_and_cap_validation():
    small = mp.Spec(n_bits=12, k_secret=3, m_train=8, m_test=0)
    with pytest.raises(ValueError, match="MASK32 only"):
        packed.generate_packed_scan(3, spec=small)
    with pytest.raises(ValueError, match="weight_cap"):
        packed.generate_packed_scan(6)
