"""16x16 matmul: output-repacked tail plus live-B evacuation.

This is a 68,341 follow-up to ``output_repacked_tail_16x16``.  It preserves
the 68,390 schedule and adds a final lifetime rewrite:

* copy 22 still-live B inputs into cheap dead output homes;
* write those outputs into the B inputs' old homes;
* redirect the remaining B reads to the cheaper former output homes.

The scalar matmul arithmetic is unchanged.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from matmul.submissions.output_repacked_tail_16x16 import (
    N,
    generate_output_repacked_tail_16x16,
)

EXPECTED_COST = 68_341

# Each row is (output index, old output home, base op index, B source addr).
# The op index is zero-based in the pre-evacuation 68,390 IR.
LIVE_B_EVACUATIONS = (
    (192, 39, 5850, 290),
    (208, 40, 5852, 291),
    (224, 41, 5854, 292),
    (240, 42, 5856, 293),
    (193, 43, 5859, 294),
    (209, 44, 5861, 257),
    (225, 45, 5863, 258),
    (241, 46, 5865, 259),
    (194, 47, 5868, 260),
    (210, 48, 5870, 261),
    (226, 49, 5872, 262),
    (242, 50, 5874, 271),
    (195, 51, 5877, 272),
    (211, 52, 5879, 273),
    (227, 53, 5881, 274),
    (243, 54, 5883, 275),
    (196, 55, 5886, 276),
    (212, 56, 5888, 277),
    (228, 57, 5890, 278),
    (244, 58, 5892, 287),
    (197, 59, 5895, 288),
    (213, 60, 5897, 289),
)


def _parse_ir(ir: str) -> tuple[list[int], list[tuple[str, list[int]]], list[int]]:
    lines = [line.strip() for line in ir.splitlines() if line.strip()]
    inputs = [int(part) for part in lines[0].split(",")]
    outputs = [int(part) for part in lines[-1].split(",")]
    ops: list[tuple[str, list[int]]] = []
    for line in lines[1:-1]:
        op, _sep, rest = line.partition(" ")
        ops.append((op, [int(part) for part in rest.split(",")]))
    return inputs, ops, outputs


def _read_positions(op: str, operands: list[int]) -> list[int]:
    if op == "copy":
        return [1]
    if op in {"add", "sub", "mul"} and len(operands) == 3:
        return [1, 2]
    if op in {"add", "sub", "mul"} and len(operands) == 2:
        return [0, 1]
    raise AssertionError(f"bad op: {op} {operands}")


def _format_op(op: str, operands: list[int]) -> str:
    return f"{op} {','.join(map(str, operands))}"


def _apply_live_b_evacuations(ir: str) -> str:
    inputs, ops, outputs = _parse_ir(ir)
    by_write = {row[2]: row for row in LIVE_B_EVACUATIONS}
    assert len(by_write) == len(LIVE_B_EVACUATIONS)

    new_ops: list[str] = []
    for op_index, (op, operands) in enumerate(ops):
        rewritten = list(operands)
        for _output_index, old_home, write_index, source_addr in LIVE_B_EVACUATIONS:
            if op_index <= write_index:
                continue
            for read_index in _read_positions(op, rewritten):
                if rewritten[read_index] == source_addr:
                    rewritten[read_index] = old_home

        row = by_write.get(op_index)
        if row is not None:
            _output_index, old_home, _write_index, source_addr = row
            assert rewritten[0] == old_home
            new_ops.append(f"copy {old_home},{source_addr}")
            rewritten[0] = source_addr
        new_ops.append(_format_op(op, rewritten))

    new_outputs = list(outputs)
    for output_index, old_home, _write_index, source_addr in LIVE_B_EVACUATIONS:
        assert new_outputs[output_index] == old_home
        new_outputs[output_index] = source_addr

    return (
        ",".join(map(str, inputs))
        + "\n"
        + "\n".join(new_ops)
        + "\n"
        + ",".join(map(str, new_outputs))
    )


def generate_output_repacked_tail_live_b_16x16() -> str:
    return _apply_live_b_evacuations(generate_output_repacked_tail_16x16())


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
    assert len(set(inputs) & set(outputs)) == 160


if __name__ == "__main__":
    from matmul import score_16x16  # noqa: E402

    ir = generate_output_repacked_tail_live_b_16x16()
    _assert_submission_invariants(ir)
    cost = score_16x16(ir)
    assert cost == EXPECTED_COST, cost

    out_path = Path(__file__).with_suffix(".ir")
    out_path.write_text(ir + "\n")
    print(f"{out_path.name}  cost={cost:,}")
