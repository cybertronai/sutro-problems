"""Regression and exact-predicate tests for the packed frontier."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mask_sparse_parity as mp
import packed_sparse_parity as packed
from submissions.audit_packed_frontier import (
    GATE_SUITES,
    SEARCH_SUITES,
    suite_hash,
)
from submissions.packed_frontier import (
    CONFIGS,
    generate_packed_frontier,
    states_for_config,
)


HERE = Path(__file__).resolve().parent
EXPECTED = {
    40: (
        11_929,
        141_218,
        415,
        "3e1a3a376e56e70092a07c8d14941e69df02038f9fc569c19e60ac07f200d0a2",
    ),
    60: (
        13_343,
        149_665,
        623,
        "c0f02ad80d3ffa2f1f6f506ccf5ff5062eb595a2223a6828b2054744754fa606",
    ),
    80: (
        19_411,
        182_744,
        855,
        "ad0517528a0de27b0be976352d9cb6c40b5d889215575af87b74bf13cf789925",
    ),
}


def test_artifacts_reproduce_scores_hashes_and_dev_denominators():
    for target, (lines, cost, successes, digest) in EXPECTED.items():
        generated = generate_packed_frontier(target)
        stored_path = HERE / "submissions" / f"packedfrontier{target}_mask32.ir"
        stored = stored_path.read_text(encoding="utf-8")
        assert generated == stored.rstrip("\n")
        assert len(generated.splitlines()) == lines
        assert hashlib.sha256(stored.encode()).hexdigest() == digest
        assert packed._global_frequency_layout(generated) == generated
        result = mp.evaluate_mask(generated)
        assert result.cost == cost
        assert result.n_instances == 1_024
        assert result.recovery == successes / result.n_instances
        assert not any(
            line.startswith("copy ")
            and len(set(line[5:].split(","))) == 1
            for line in generated.splitlines()
        )


def test_state_subsets_are_frozen_and_nested_by_weight():
    expected_counts = {40: 70, 60: 125, 80: 375}
    for target, config in CONFIGS.items():
        states = states_for_config(config)
        assert states is not None
        assert len(states) == expected_counts[target]
        assert states[0] == 0
        assert len(states) == len(set(states))
        assert all(packed._bit_count(state) <= config.cap for state in states)
    assert packed.transition_cost(states_for_config(CONFIGS[40]) or ()) == 115
    assert packed.transition_cost(states_for_config(CONFIGS[60]) or ()) == 203
    assert packed.transition_cost(states_for_config(CONFIGS[80]) or ()) == 591


def test_fixed_suite_audit_has_denominators_hashes_and_band_margin():
    audit_path = HERE / "submissions" / "packed_frontier_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["n_secrets"] == 256
    assert audit["repetitions"] == 8
    assert len(audit["suites"]) == 16
    assert len(SEARCH_SUITES) == 12
    assert len(GATE_SUITES) == 4
    assert [suite["key"] for suite in audit["suites"]] == list(
        SEARCH_SUITES + GATE_SUITES
    )
    for suite in audit["suites"]:
        inputs, masks, metadata = mp.mask_suite(
            suite_key=suite["key"],
            n_secrets=mp.FINAL_SECRETS,
            repetitions=mp.FINAL_REPS,
        )
        assert suite["instances"] == 2_048
        assert suite["sha256"] == suite_hash(inputs, masks, metadata)
        for target in EXPECTED:
            result = suite["targets"][str(target)]
            assert result["cost"] == EXPECTED[target][1]
            assert result["denominator"] == 2_048
            assert result["successes"] / result["denominator"] == result["recovery"]
    for target in EXPECTED:
        summary = audit["summary"][str(target)]
        draws = [
            suite["targets"][str(target)]
            for suite in audit["suites"]
        ]
        total_successes = sum(draw["successes"] for draw in draws)
        recoveries = [draw["recovery"] for draw in draws]
        assert summary["draws"] == 16
        assert summary["total_instances"] == 32_768
        assert summary["total_successes"] == total_successes
        assert summary["min_recovery"] == min(recoveries)
        assert summary["mean_recovery"] == total_successes / 32_768
        assert summary["max_recovery"] == max(recoveries)
        assert summary["min_recovery"] >= target / 100


def test_compact_weight_predicates_exhaustively():
    # The generated circuit applies these identities to three unsigned 6-bit
    # packed pivot cells.  Exhaust their complete 64^3 domain.
    for a in range(64):
        for b in range(64):
            for c in range(64):
                wanted = sum(packed._bit_count(cell) for cell in (a, b, c))
                parity = a ^ b ^ c
                majority = (a & b) | ((a ^ b) & c)

                union = a | b | c
                error1 = ((a + b + c) ^ union) | (union & (union - 1))
                got1 = bool(union if error1 == 0 else 0)

                clear1 = parity & (parity - 1)
                clear2 = clear1 & (clear1 - 1)
                case2a = clear1 if (clear2 | majority) == 0 else 0
                majority_clear1 = majority & (majority - 1)
                case2b = majority if (majority_clear1 | parity) == 0 else 0
                got2 = bool(case2a | case2b)

                clear3 = clear2 & (clear2 - 1)
                case3a = clear2 if (clear3 | majority) == 0 else 0
                both_nonzero = majority if parity else 0
                case3b = (
                    both_nonzero
                    if (clear1 | majority_clear1) == 0
                    else 0
                )
                got3 = bool(case3a | case3b)

                assert got1 == (wanted == 1)
                assert got2 == (wanted == 2)
                assert got3 == (wanted == 3)
