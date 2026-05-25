"""16x16 matmul submission: cheap B capture, score 66,524.

This companion file follows the submission naming convention for
``cheap_capture_66524.ir``. The final IR was produced by a search pipeline
rather than a compact closed-form schedule generator.

The raw continuation trace is kept next to this file as
``cheap_capture_66524.raw.ir``.
"""
from __future__ import annotations

from pathlib import Path


EXPECTED_SCORE = 66_524
IR_NAME = "cheap_capture_66524.ir"


def generate_cheap_capture_66524() -> str:
    """Return the checked-in final IR text."""
    return (Path(__file__).with_name(IR_NAME)).read_text().strip()


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matmul import score_16x16  # noqa: E402

    ir = generate_cheap_capture_66524()
    out_path = Path(__file__).with_name(IR_NAME)
    out_path.write_text(ir + "\n")
    cost = score_16x16(ir)
    print(f"{IR_NAME}  cost={cost:,}")
    assert cost == EXPECTED_SCORE, cost
