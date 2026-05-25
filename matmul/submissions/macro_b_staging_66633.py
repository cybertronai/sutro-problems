"""16x16 matmul submission: Macro B-staging, score 66,633.

This companion file follows the submission naming convention for
``macro_b_staging_66633.ir``. The final IR was produced by a search pipeline
rather than a compact closed-form schedule generator. The winning candidate
was the ``b7_later_panel_prestage`` variant that copies B row 7 target
columns 10..15 from cheap address 1 after their first load, then redirects
the later panel reloads through those staged values.

The raw continuation trace is kept next to this file as
``macro_b_staging_66633.raw.ir``.
"""
from __future__ import annotations

from pathlib import Path


EXPECTED_SCORE = 66_633
IR_NAME = "macro_b_staging_66633.ir"


def generate_macro_b_staging_66633() -> str:
    """Return the checked-in final IR text."""
    return (Path(__file__).with_name(IR_NAME)).read_text().strip()


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matmul import score_16x16  # noqa: E402

    ir = generate_macro_b_staging_66633()
    out_path = Path(__file__).with_name(IR_NAME)
    out_path.write_text(ir + "\n")
    cost = score_16x16(ir)
    print(f"{IR_NAME}  cost={cost:,}")
    assert cost == EXPECTED_SCORE, cost
