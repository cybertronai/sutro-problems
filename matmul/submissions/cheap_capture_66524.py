"""16x16 matmul submission: cheap B capture, score 66,524.

This companion file reproduces the final colored IR from the checked-in raw
semantic trace. The raw candidate captures 26 late B-block values from address
1, redirects later reloads through those staged values, and then relies on the
value-lifetime coloring pass to assign physical addresses.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matmul.submissions import search_value_coloring as coloring


EXPECTED_SCORE = 66_524
IR_NAME = "cheap_capture_66524.ir"
RAW_NAME = "cheap_capture_66524.raw.ir"


def _color_raw_trace(raw_name: str) -> str:
    """Run the value-lifetime coloring pass used for the submitted IR."""
    raw_path = Path(__file__).with_name(raw_name)
    inputs, ops, outputs = coloring.parse_ir(raw_path)
    values, op_values, input_values, output_values = coloring.to_values(
        inputs,
        ops,
        outputs,
    )
    assignment = coloring.allocate_dp_chains(values)
    return coloring.emit_ir(
        inputs,
        ops,
        values,
        op_values,
        input_values,
        output_values,
        assignment,
    ).strip()


def generate_cheap_capture_66524() -> str:
    """Reproduce the final IR from ``cheap_capture_66524.raw.ir``."""
    return _color_raw_trace(RAW_NAME)


if __name__ == "__main__":
    from matmul import score_16x16  # noqa: E402

    ir = generate_cheap_capture_66524()
    out_path = Path(__file__).with_name(IR_NAME)
    out_path.write_text(ir + "\n")
    cost = score_16x16(ir)
    print(f"{IR_NAME}  cost={cost:,}")
    assert cost == EXPECTED_SCORE, cost
