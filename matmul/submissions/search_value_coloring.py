"""Scratch value-lifetime coloring experiments for 16x16 matmul IRs.

This parses a valid IR into SSA-like values, assigns compatible value
lifetimes to physical addresses, and emits a recolored IR.  It is a research
harness for schedule mutations; it does not replace the record generators.
"""
from __future__ import annotations

import argparse
import heapq
import math
import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Value:
    index: int
    define: int
    last: int
    reads: int


def addr_cost(addr: int) -> int:
    return math.isqrt(addr - 1) + 1


def parse_ir(path: Path) -> tuple[list[int], list[tuple[str, list[int]]], list[int]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    inputs = [int(part) for part in lines[0].split(",")]
    outputs = [int(part) for part in lines[-1].split(",")]
    ops: list[tuple[str, list[int]]] = []
    for line in lines[1:-1]:
        op, _sep, rest = line.partition(" ")
        ops.append((op, [int(part) for part in rest.split(",")]))
    return inputs, ops, outputs


def read_positions(op: str, operands: list[int]) -> list[int]:
    if op == "copy":
        return [1]
    if op in {"add", "sub", "mul"} and len(operands) == 3:
        return [1, 2]
    if op in {"add", "sub", "mul"} and len(operands) == 2:
        return [0, 1]
    raise ValueError((op, operands))


def to_values(
    inputs: list[int],
    ops: list[tuple[str, list[int]]],
    outputs: list[int],
) -> tuple[list[Value], list[list[int]], list[int], list[int]]:
    current = {addr: idx for idx, addr in enumerate(inputs)}
    mutable = [
        {"define": 0, "last": 0, "reads": 0}
        for _addr in inputs
    ]
    op_values: list[list[int]] = []

    for time, (op, operands) in enumerate(ops, start=1):
        read_value_ids: list[int] = []
        for pos in read_positions(op, operands):
            value_id = current[operands[pos]]
            mutable[value_id]["reads"] += 1
            mutable[value_id]["last"] = time
            read_value_ids.append(value_id)
        dest_value = len(mutable)
        mutable.append({"define": time, "last": time, "reads": 0})
        current[operands[0]] = dest_value
        op_values.append([dest_value, *read_value_ids])

    exit_time = len(ops) + 1
    output_values: list[int] = []
    for addr in outputs:
        value_id = current[addr]
        mutable[value_id]["reads"] += 1
        mutable[value_id]["last"] = exit_time
        output_values.append(value_id)

    values = [
        Value(
            index=index,
            define=item["define"],
            last=item["last"],
            reads=item["reads"],
        )
        for index, item in enumerate(mutable)
    ]
    return values, op_values, list(range(len(inputs))), output_values


def current_score(
    ops: list[tuple[str, list[int]]],
    outputs: list[int],
) -> int:
    total = 0
    for op, operands in ops:
        for pos in read_positions(op, operands):
            total += addr_cost(operands[pos])
    total += sum(addr_cost(addr) for addr in outputs)
    return total


def allocate_linear(
    values: list[Value],
    mode: str,
) -> dict[int, int]:
    """Greedy interval coloring.

    ``start_heavy`` processes values by start time and gives heavier intervals
    lower freed addresses.  ``weight_first`` processes by descending read
    weight and starts a maximal compatible chain for each cheap address.
    """
    live_values = [value for value in values if value.reads > 0]
    if mode == "start_heavy":
        available: list[int] = []
        active: list[tuple[int, int]] = []
        next_addr = 1
        assignment: dict[int, int] = {}
        for value in sorted(live_values, key=lambda v: (v.define, -v.reads, v.last)):
            while active and active[0][0] <= value.define:
                _last, addr = heapq.heappop(active)
                heapq.heappush(available, addr)
            if available:
                addr = heapq.heappop(available)
            else:
                addr = next_addr
                next_addr += 1
            assignment[value.index] = addr
            heapq.heappush(active, (value.last, addr))
        return assignment

    if mode == "weight_first":
        remaining = set(value.index for value in live_values)
        by_index = {value.index: value for value in live_values}
        assignment: dict[int, int] = {}
        addr = 1
        while remaining:
            last = -1
            while True:
                compatible = [
                    by_index[index]
                    for index in remaining
                    if by_index[index].define >= last
                ]
                if not compatible:
                    break
                value = max(
                    compatible,
                    key=lambda v: (v.reads, -(v.last - v.define), -v.define),
                )
                assignment[value.index] = addr
                remaining.remove(value.index)
                last = value.last
            addr += 1
        return assignment

    raise ValueError(mode)


def allocate_dp_chains(values: list[Value]) -> dict[int, int]:
    """Repeated maximum-weight compatible interval chains.

    This is the chain-decomposition heuristic that improved the upstream
    67,821 trace to 67,641.  Each physical address receives a maximum-read
    chain among the remaining non-overlapping value lifetimes.
    """
    remaining = [
        (value.define, value.last, value.reads, value.index)
        for value in values
        if value.reads > 0
    ]
    assignment: dict[int, int] = {}
    addr = 1

    def best_chain(
        items: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        ordered = sorted(
            items,
            key=lambda item: (item[1], item[1] - item[0], -item[2]),
        )
        ends = [item[1] for item in ordered]
        dp = [0] * (len(ordered) + 1)
        choose = [False] * len(ordered)
        prev = [0] * len(ordered)
        for i, (define, _last, reads, _index) in enumerate(ordered, start=1):
            j = bisect_right(ends, define, 0, i - 1)
            candidate = dp[j] + reads
            if candidate >= dp[i - 1]:
                dp[i] = candidate
                choose[i - 1] = True
                prev[i - 1] = j
            else:
                dp[i] = dp[i - 1]

        chain: list[tuple[int, int, int, int]] = []
        i = len(ordered)
        while i > 0:
            item = ordered[i - 1]
            if choose[i - 1] and dp[prev[i - 1]] + item[2] >= dp[i - 1]:
                chain.append(item)
                i = prev[i - 1]
            else:
                i -= 1
        return list(reversed(chain))

    while remaining:
        chain = best_chain(remaining)
        used = {item[3] for item in chain}
        for _define, _last, _reads, index in chain:
            assignment[index] = addr
        remaining = [item for item in remaining if item[3] not in used]
        addr += 1

    return assignment


def emit_ir(
    inputs: list[int],
    ops: list[tuple[str, list[int]]],
    values: list[Value],
    op_values: list[list[int]],
    input_values: list[int],
    output_values: list[int],
    assignment: dict[int, int],
) -> str:
    # Values with zero reads never need an address, except as dead writes.
    next_unused = max(assignment.values(), default=0) + 1
    full_assignment = dict(assignment)
    for value in values:
        if value.index not in full_assignment:
            full_assignment[value.index] = next_unused
            next_unused += 1

    input_addrs = [full_assignment[value_id] for value_id in input_values]
    output_addrs = [full_assignment[value_id] for value_id in output_values]
    lines = [",".join(map(str, input_addrs))]

    for (op, _operands), value_ids in zip(ops, op_values):
        dest = full_assignment[value_ids[0]]
        reads = [full_assignment[value_id] for value_id in value_ids[1:]]
        lines.append(f"{op} {','.join(map(str, [dest, *reads]))}")

    lines.append(",".join(map(str, output_addrs)))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ir", type=Path)
    parser.add_argument(
        "--mode",
        choices=("start_heavy", "weight_first", "dp_chains"),
        default="dp_chains",
    )
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()

    inputs, ops, outputs = parse_ir(args.ir)
    values, op_values, input_values, output_values = to_values(inputs, ops, outputs)
    if args.mode == "dp_chains":
        assignment = allocate_dp_chains(values)
    else:
        assignment = allocate_linear(values, args.mode)
    ir = emit_ir(inputs, ops, values, op_values, input_values, output_values, assignment)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from matmul import score_16x16

    checked = score_16x16(ir)
    print(f"mode={args.mode} score_16x16={checked:,} addrs={max(assignment.values())}")
    print(f"source_static={current_score(ops, outputs):,}")
    if args.write:
        args.write.write_text(ir + "\n")
        print(f"wrote={args.write}")


if __name__ == "__main__":
    main()
