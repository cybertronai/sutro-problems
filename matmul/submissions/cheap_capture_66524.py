"""16x16 matmul submission: cheap B capture, score 66,524.

This companion file starts from the 66,633 raw trace, applies the 26-value
late B-block cheap-capture edit, and then reruns value-lifetime coloring.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matmul.submissions import staged_trace_repro as repro


EXPECTED_SCORE = 66_524
IR_NAME = "cheap_capture_66524.ir"
RAW_NAME = "cheap_capture_66524.raw.ir"
BASE_RAW_NAME = "macro_b_staging_66633.raw.ir"
STAGED_CAPTURE_SPECS = (
    (5424, 731),
    (5433, 732),
    (5442, 733),
    (5451, 734),
    (5460, 735),
    (5469, 736),
    (5478, 737),
    (5487, 738),
    (5500, 739),
    (5509, 740),
    (5518, 741),
    (5527, 742),
    (5536, 743),
    (5545, 744),
    (5554, 745),
    (5563, 746),
    (5576, 747),
    (5585, 748),
    (5594, 749),
    (5603, 750),
    (5612, 751),
    (5621, 752),
    (5630, 753),
    (5639, 754),
    (5652, 755),
    (5661, 756),
)


def build_cheap_capture_66524_raw_ops() -> tuple[list[int], list[repro.Op], list[int]]:
    """Reproduce the raw semantic trace from the 66,633 raw trace."""
    inputs, ops, outputs = repro.load_raw(__file__, BASE_RAW_NAME)
    ops = repro.apply_staged_captures(
        inputs,
        ops,
        outputs,
        STAGED_CAPTURE_SPECS,
    )
    return inputs, ops, outputs


def generate_cheap_capture_66524_raw() -> str:
    """Reproduce ``cheap_capture_66524.raw.ir``."""
    return repro.emit_raw(*build_cheap_capture_66524_raw_ops())


def generate_cheap_capture_66524() -> str:
    """Reproduce the final colored IR from the generated raw trace."""
    return repro.color_ops(*build_cheap_capture_66524_raw_ops())


if __name__ == "__main__":
    from matmul import score_16x16  # noqa: E402

    ir = generate_cheap_capture_66524()
    raw = generate_cheap_capture_66524_raw()
    out_path = Path(__file__).with_name(IR_NAME)
    raw_path = Path(__file__).with_name(RAW_NAME)
    out_path.write_text(ir + "\n")
    raw_path.write_text(raw + "\n")
    cost = score_16x16(ir)
    print(f"{IR_NAME}  cost={cost:,}")
    assert cost == EXPECTED_SCORE, cost
