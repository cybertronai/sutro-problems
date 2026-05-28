"""16x16 matmul submission: Claude-assisted annealed 66,300 IR.

This companion file loads the checked-in scored IR and verifies its score.  The
artifact was produced by simulated annealing over a leaderboard physical-address
IR, not as a raw semantic trace, so this reproducer intentionally does not invent
a raw-generation path.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXPECTED_SCORE = 66_300
IR_NAME = "best_66300.ir"


def generate_best_66300() -> str:
    """Return the checked-in scored IR for the 66,300 submission."""
    return Path(__file__).with_name(IR_NAME).read_text().strip()


if __name__ == "__main__":
    from matmul import score_16x16  # noqa: E402

    ir = generate_best_66300()
    cost = score_16x16(ir)
    print(f"{IR_NAME}  cost={cost:,}")
    assert cost == EXPECTED_SCORE, cost
