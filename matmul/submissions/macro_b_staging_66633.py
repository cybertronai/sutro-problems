"""16x16 matmul submission: Macro B-staging, score 66,633.

This companion file reproduces the final colored IR from the checked-in raw
semantic trace. The winning raw candidate was the ``b7_later_panel_prestage``
variant that copies B row 7 target columns 10..15 from cheap address 1 after
their first load, then redirects the later panel reloads through those staged
values.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matmul.submissions import search_value_coloring as coloring


EXPECTED_SCORE = 66_633
IR_NAME = "macro_b_staging_66633.ir"
RAW_NAME = "macro_b_staging_66633.raw.ir"


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


def generate_macro_b_staging_66633() -> str:
    """Reproduce the final IR from ``macro_b_staging_66633.raw.ir``."""
    return _color_raw_trace(RAW_NAME)


if __name__ == "__main__":
    from matmul import score_16x16  # noqa: E402

    ir = generate_macro_b_staging_66633()
    out_path = Path(__file__).with_name(IR_NAME)
    out_path.write_text(ir + "\n")
    cost = score_16x16(ir)
    print(f"{IR_NAME}  cost={cost:,}")
    assert cost == EXPECTED_SCORE, cost
