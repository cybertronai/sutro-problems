"""16x16 matmul submission: late copy-schedule motif bundle, score 66,400.

This companion file starts from the checked-in raw semantic trace and reruns
the value-lifetime coloring pass that produces the scored physical IR.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matmul.submissions import staged_trace_repro as repro


EXPECTED_SCORE = 66_400
IR_NAME = "motif_bundle_66400.ir"
RAW_NAME = "motif_bundle_66400.raw.ir"


def build_motif_bundle_66400_raw_ops() -> tuple[list[int], list[repro.Op], list[int]]:
    """Load the final raw semantic trace used for this submission."""
    return repro.load_raw(__file__, RAW_NAME)


def generate_motif_bundle_66400_raw() -> str:
    """Re-emit ``motif_bundle_66400.raw.ir`` in canonical raw form."""
    return repro.emit_raw(*build_motif_bundle_66400_raw_ops())


def generate_motif_bundle_66400() -> str:
    """Reproduce the final colored IR from the raw semantic trace."""
    return repro.color_ops(*build_motif_bundle_66400_raw_ops())


if __name__ == "__main__":
    from matmul import score_16x16  # noqa: E402

    ir = generate_motif_bundle_66400()
    raw = generate_motif_bundle_66400_raw()
    out_path = Path(__file__).with_name(IR_NAME)
    raw_path = Path(__file__).with_name(RAW_NAME)
    out_path.write_text(ir + "\n")
    raw_path.write_text(raw + "\n")
    cost = score_16x16(ir)
    print(f"{IR_NAME}  cost={cost:,}")
    assert cost == EXPECTED_SCORE, cost
