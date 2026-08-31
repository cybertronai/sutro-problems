#!/usr/bin/env python3
"""Verify the 4x4 Sutro matmul score, symbolic correctness, and LP dual bound.

Usage:
    python3 verify_best_689.py best_689.ir best_689_certificate.json
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


def tier(addr: int) -> int:
    if addr <= 0:
        raise ValueError(f"addresses must be positive, got {addr}")
    return math.isqrt(addr - 1) + 1


Poly = Dict[Tuple[int, ...], int]


def poly_add(a: Poly, b: Poly, sign: int = 1) -> Poly:
    out = dict(a)
    for mon, coeff in b.items():
        out[mon] = out.get(mon, 0) + sign * coeff
        if out[mon] == 0:
            del out[mon]
    return out


def poly_mul(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            mon = tuple(sorted(ma + mb))
            if len(mon) > 2:
                raise ValueError("intermediate polynomial degree exceeds 2")
            out[mon] = out.get(mon, 0) + ca * cb
    return {m: c for m, c in out.items() if c}


@dataclass
class Value:
    start: int
    address: int
    reads: List[int] = field(default_factory=list)

    @property
    def end(self) -> int:
        if not self.reads:
            raise ValueError("certificate trace contains an unread value")
        return max(self.reads)

    @property
    def read_count(self) -> int:
        return len(self.reads)


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def verify(ir_path: Path, cert_path: Path) -> None:
    lines = [
        line.strip()
        for line in ir_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 3:
        raise ValueError("IR is too short")

    input_addresses = parse_csv_ints(lines[0])
    output_addresses = parse_csv_ints(lines[-1])
    if len(input_addresses) != 32:
        raise ValueError(f"expected 32 inputs, got {len(input_addresses)}")
    if len(set(input_addresses)) != 32:
        raise ValueError("input addresses must be distinct")
    if len(output_addresses) != 16:
        raise ValueError(f"expected 16 outputs, got {len(output_addresses)}")

    # Current physical memory, symbolic memory, and SSA trace.
    current_value: Dict[int, int] = {}
    memory: Dict[int, Poly] = {}
    values: List[Value] = []
    for variable, addr in enumerate(input_addresses):
        if addr <= 0:
            raise ValueError("all input addresses must be positive")
        current_value[addr] = len(values)
        values.append(Value(start=0, address=addr))
        memory[addr] = {(variable,): 1}

    score = 0
    operations = lines[1:-1]
    for op_index, line in enumerate(operations, start=1):
        pieces = line.split(None, 1)
        if len(pieces) != 2:
            raise ValueError(f"malformed instruction: {line!r}")
        opcode, raw_args = pieces
        args = parse_csv_ints(raw_args)
        read_time = 2 * op_index
        start_time = read_time + 1

        if opcode == "copy":
            if len(args) != 2:
                raise ValueError(f"copy needs 2 arguments: {line!r}")
            dest, src = args
            source_addresses = [src]
        elif opcode in {"add", "sub", "mul"}:
            if len(args) != 3:
                raise ValueError(f"{opcode} needs 3 arguments: {line!r}")
            dest, src1, src2 = args
            source_addresses = [src1, src2]
        else:
            raise ValueError(f"unsupported opcode in this certificate: {opcode}")

        for addr in [dest, *source_addresses]:
            if addr <= 0:
                raise ValueError(f"non-positive address in: {line!r}")
        for src in source_addresses:
            if src not in current_value:
                raise ValueError(f"read of undefined address {src} in: {line!r}")
            values[current_value[src]].reads.append(read_time)
            score += tier(src)

        # Reads occur before the destination write, so in-place operations work.
        if opcode == "copy":
            result = dict(memory[source_addresses[0]])
        elif opcode == "add":
            result = poly_add(memory[source_addresses[0]], memory[source_addresses[1]])
        elif opcode == "sub":
            result = poly_add(memory[source_addresses[0]], memory[source_addresses[1]], -1)
        else:
            result = poly_mul(memory[source_addresses[0]], memory[source_addresses[1]])

        current_value[dest] = len(values)
        values.append(Value(start=start_time, address=dest))
        memory[dest] = result

    output_time = 2 * (len(operations) + 1)
    actual: List[Poly] = []
    for addr in output_addresses:
        if addr not in current_value:
            raise ValueError(f"undefined output address: {addr}")
        values[current_value[addr]].reads.append(output_time)
        score += tier(addr)
        actual.append(memory[addr])

    expected: List[Poly] = []
    n = 4
    for i in range(n):
        for j in range(n):
            expected.append(
                {
                    tuple(sorted((i * n + k, n * n + k * n + j))): 1
                    for k in range(n)
                }
            )
    if actual != expected:
        raise AssertionError("symbolic outputs do not equal exact 4x4 matrix multiplication")

    if any(not value.reads for value in values):
        raise AssertionError("trace has unread values; certificate indexing would differ")

    # A feasible physical allocation cannot assign overlapping values to one cell.
    by_address: Dict[int, List[Tuple[int, int, int]]] = {}
    for index, value in enumerate(values):
        by_address.setdefault(value.address, []).append((value.start, value.end, index))
    for addr, intervals in by_address.items():
        intervals.sort()
        for left, right in zip(intervals, intervals[1:]):
            if left[1] >= right[0]:
                raise AssertionError(
                    f"overlapping SSA values share address {addr}: {left} and {right}"
                )

    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    claimed = int(cert["claimed_score"])
    max_tier = int(cert["max_tier"])
    times = [int(x) for x in cert["times"]]
    y = [int(x) for x in cert["dual_y"]]
    if len(y) != len(values):
        raise AssertionError(f"dual y has {len(y)} entries, trace has {len(values)} values")
    if int(cert["value_count"]) != len(values):
        raise AssertionError("certificate value count mismatch")
    if int(cert["operation_count"]) != len(operations):
        raise AssertionError("certificate operation count mismatch")

    z = {
        (int(t), int(tm)): int(value)
        for t, tm, value in cert["dual_z_nonzero"]
    }
    if any(value > 0 for value in z.values()):
        raise AssertionError("dual capacity multipliers must be non-positive")
    if times != sorted(set(times)):
        raise AssertionError("certificate times must be sorted and unique")

    # Dual feasibility:
    #   y_v + sum_{time: v live} z[tier,time] <= reads(v) * tier
    for value_index, value in enumerate(values):
        for t in range(1, max_tier + 1):
            lhs = y[value_index]
            lhs += sum(
                z.get((t, tm), 0)
                for tm in times
                if value.start <= tm <= value.end
            )
            rhs = value.read_count * t
            if lhs > rhs:
                raise AssertionError(
                    f"dual infeasible for value {value_index}, tier {t}: {lhs} > {rhs}"
                )

    dual_objective = sum(y)
    for t in range(1, max_tier + 1):
        capacity = 2 * t - 1
        dual_objective += capacity * sum(z.get((t, tm), 0) for tm in times)

    # There are fewer logical values than addresses through max_tier, so no
    # optimum needs a more expensive tier: give every value a globally unique
    # address in 1..value_count if necessary.
    if len(values) > max_tier * max_tier:
        raise AssertionError("max_tier is too small to dominate all possible allocations")

    if dual_objective != claimed:
        raise AssertionError(
            f"dual objective is {dual_objective}, certificate claims {claimed}"
        )
    if score != claimed:
        raise AssertionError(f"IR score is {score}, certificate claims {claimed}")

    print(f"PASS: symbolic 4x4 matrix multiplication is exact")
    print(f"PASS: static read score = {score}")
    print(f"PASS: integer LP dual lower bound = {dual_objective}")
    print(
        "CONCLUSION: no address assignment can improve this exact operation "
        f"schedule below {claimed}; a different schedule or arithmetic circuit is required."
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    verify(Path(sys.argv[1]), Path(sys.argv[2]))
