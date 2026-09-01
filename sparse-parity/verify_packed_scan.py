#!/usr/bin/env python3
"""Reproduce packed-scan artifacts, scores, and recorded validation suites."""
from __future__ import annotations

import argparse
from pathlib import Path

import mask_sparse_parity as mp
import packed_sparse_parity as packed


HERE = Path(__file__).resolve().parent
EXPECTED = {
    1: (10_732, 135_348, 0.2001953125),
    2: (13_752, 151_943, 0.5654296875),
    3: (23_325, 200_937, 0.8984375),
    5: (64_073, 409_001, 1.0),
}
RECORDED_KEYS = (
    "fresh-best5-a-20260901",
    "fresh-best5-b-20260901",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recorded",
        action="store_true",
        help="evaluate all caps on two fixed, independent 2,048-instance suites",
    )
    parser.add_argument(
        "--fresh",
        type=int,
        default=0,
        metavar="N",
        help="also evaluate cap 5 on N newly randomized final-sized suites",
    )
    args = parser.parse_args()
    if args.fresh < 0:
        parser.error("--fresh must be non-negative")

    states = packed.bounded_weight_gray_states(14, 5)
    lower = packed.bounded_weight_transition_lower_bound(14, 5)
    actual = packed.transition_cost(states)
    assert len(states) == 3_473
    assert actual == lower == 4_759
    print(f"cap-5 walk: {len(states):,} states, {actual:,} flips (tight)")
    print()
    print(" cap      lines       cost       dev recovery")
    print("----  ---------  ---------  -----------------")

    generated = {}
    for cap, (expected_lines, expected_cost, expected_recovery) in EXPECTED.items():
        ir = packed.generate_packed_scan(cap)
        generated[cap] = ir
        stored = (
            HERE / "submissions" / f"packedscan{cap}_mask32.ir"
        ).read_text().rstrip("\n")
        assert stored == ir, f"stored cap-{cap} IR differs from generator"

        result = mp.evaluate_mask(ir)
        lines = len(ir.splitlines())
        assert lines == expected_lines
        assert result.cost == expected_cost
        assert result.recovery == expected_recovery
        print(
            f"{cap:>4}  {lines:>9,}  {result.cost:>9,}  "
            f"{result.recovery:>17.10f}"
        )

    if args.recorded:
        print("\nrecorded final-sized suites")
        for key in RECORDED_KEYS:
            print(f"  {key}")
            for cap, ir in generated.items():
                result = mp.evaluate_mask(
                    ir,
                    suite_key=key,
                    n_secrets=mp.FINAL_SECRETS,
                    repetitions=mp.FINAL_REPS,
                )
                print(f"    cap {cap}: {result.recovery:.10f}")

    for index in range(args.fresh):
        result = mp.evaluate_mask(generated[5], suite_key=None)
        print(
            f"fresh cap-5 suite {index + 1}: {result.recovery:.10f} "
            f"({result.n_instances:,} instances)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
