"""Small raw-trace transforms used by staged matmul record reproducers."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from matmul import _cost
from matmul.submissions import search_value_coloring as coloring


Origin = tuple[str, int, int]
Op = tuple[str, list[int]]


def clone_ops(ops: Sequence[Op]) -> list[Op]:
    return [(op, list(operands)) for op, operands in ops]


def load_raw(sibling_file: str, raw_name: str) -> tuple[list[int], list[Op], list[int]]:
    return coloring.parse_ir(Path(sibling_file).with_name(raw_name))


def emit_raw(inputs: Sequence[int], ops: Sequence[Op], outputs: Sequence[int]) -> str:
    lines = [",".join(map(str, inputs))]
    lines.extend(f"{op} {','.join(map(str, operands))}" for op, operands in ops)
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def color_ops(inputs: list[int], ops: list[Op], outputs: list[int]) -> str:
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


def apply_staged_captures(
    inputs: Sequence[int],
    ops: Sequence[Op],
    outputs: Sequence[int],
    capture_specs: Sequence[tuple[int, int]],
    *,
    capture_addr: int = 1,
) -> list[Op]:
    """Insert semantic stage copies and redirect later expensive reloads.

    ``capture_specs`` contains ``(op_no, fresh_addr)`` pairs. At each capture
    op, the value currently defined into ``capture_addr`` is copied to
    ``fresh_addr``. Later reads of the same semantic value are redirected
    through that fresh address when they would otherwise read from a more
    expensive address.
    """
    capture_by_op = dict(capture_specs)
    current: dict[int, Origin] = {
        addr: ("input", index, addr)
        for index, addr in enumerate(inputs)
    }
    staged: dict[Origin, tuple[int, int]] = {}
    next_origin = 10_000_000
    out: list[Op] = []

    for op_no, (op, operands) in enumerate(clone_ops(ops), start=1):
        read_origins = {
            pos: current[operands[pos]]
            for pos in coloring.read_positions(op, operands)
        }
        new_operands = list(operands)
        for pos, origin in read_origins.items():
            if origin not in staged:
                continue
            fresh_addr, first_op = staged[origin]
            if (
                op_no > first_op
                and operands[pos] != capture_addr
                and _cost(operands[pos]) > _cost(capture_addr)
            ):
                new_operands[pos] = fresh_addr

        if op == "copy":
            dest_origin = read_origins[1]
        else:
            dest_origin = ("op", next_origin, op_no)
            next_origin += 1
        current[operands[0]] = dest_origin
        out.append((op, new_operands))

        if op_no in capture_by_op:
            if operands[0] != capture_addr:
                raise ValueError((op_no, op, operands, capture_addr))
            fresh_addr = capture_by_op[op_no]
            origin = current[capture_addr]
            staged[origin] = (fresh_addr, op_no)
            current[fresh_addr] = origin
            out.append(("copy", [fresh_addr, capture_addr]))

    return out
