"""16x16 matmul submission: weighted-lifetime pressure + copy elimination.

This companion file follows the submission naming convention for
``weighted_lifetime_copyelim_66707.ir``. The final IR was produced by a
search pipeline rather than a compact closed-form schedule generator:

1. pressure-ranked suffix splitting over A/B/derived values,
2. DP chain coloring of value lifetimes,
3. generalized copy elimination over legal redirected-read windows.

The raw continuation trace is kept next to this file as
``weighted_lifetime_copyelim_66707.raw.ir``.
"""
from __future__ import annotations

from pathlib import Path


EXPECTED_SCORE = 66_707
IR_NAME = "weighted_lifetime_copyelim_66707.ir"


def generate_weighted_lifetime_copyelim_66707() -> str:
    """Return the checked-in final IR text."""
    return (Path(__file__).with_name(IR_NAME)).read_text().strip()


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matmul import score_16x16  # noqa: E402

    ir = generate_weighted_lifetime_copyelim_66707()
    out_path = Path(__file__).with_name(IR_NAME)
    out_path.write_text(ir + "\n")
    cost = score_16x16(ir)
    print(f"{IR_NAME}  cost={cost:,}")
    assert cost == EXPECTED_SCORE, cost
