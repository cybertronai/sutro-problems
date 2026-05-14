"""16x16 matmul: five-direct output-repacked tail plus live-B evacuation.

This 68,039 submission is a small improvement over the 68,041 current-order
live-B evacuation record.  It keeps the same 39 live-B evacuation moves but
starts from a five-direct final tail base with cost 68,431, saving two more
weighted read-cost units before evacuation.

The sibling IR file is the canonical submission artifact.  This module returns
and validates that IR without depending on experiment-only helper modules.
"""
from __future__ import annotations

from pathlib import Path
import sys

N = 16
EXPECTED_COST = 68_039


def generate_output_repacked_tail_five_direct_live_b_16x16() -> str:
    return Path(__file__).with_suffix(".ir").read_text().strip()


def _parse_ir(ir: str) -> tuple[list[int], list[tuple[str, list[int]]], list[int]]:
    lines = [line.strip() for line in ir.splitlines() if line.strip()]
    inputs = [int(part) for part in lines[0].split(",")]
    outputs = [int(part) for part in lines[-1].split(",")]
    ops: list[tuple[str, list[int]]] = []
    for line in lines[1:-1]:
        op, _sep, rest = line.partition(" ")
        ops.append((op, [int(part) for part in rest.split(",")]))
    return inputs, ops, outputs


def _assert_submission_invariants(ir: str) -> None:
    inputs, ops, outputs = _parse_ir(ir)
    assert len(inputs) == 2 * N * N
    assert len(set(inputs)) == len(inputs)
    assert len(outputs) == N * N
    assert len(set(outputs)) == len(outputs)
    assert all(addr > 0 for addr in inputs + outputs)

    output_writes: set[int] = set()
    for op_index, (op, operands) in enumerate(ops, start=1):
        assert all(addr > 0 for addr in operands)
        if op == "copy":
            dest, _src = operands
        elif op in {"add", "sub", "mul"} and len(operands) in {2, 3}:
            dest = operands[0]
        else:
            raise AssertionError(f"bad op at {op_index}: {op} {operands}")
        output_writes.add(dest)

    missing_outputs = [addr for addr in outputs if addr not in output_writes]
    assert not missing_outputs, missing_outputs
    assert len(set(inputs) & set(outputs)) == 128


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from matmul import score_16x16  # noqa: E402

    ir = generate_output_repacked_tail_five_direct_live_b_16x16()
    _assert_submission_invariants(ir)
    cost = score_16x16(ir)
    assert cost == EXPECTED_COST, cost
    print(f"{Path(__file__).with_suffix('.ir').name}  cost={cost:,}")
