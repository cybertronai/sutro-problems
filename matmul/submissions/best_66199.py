"""Verify the 66,199 submission by score and exact polynomial semantics."""

from __future__ import annotations

import hashlib
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N = 16
EXPECTED_SCORE = 66_199
EXPECTED_SHA256 = "16ecbd1a27eb779b8334fecbd2735ed001d3255b51ed0616cac830b79296cd7b"
IR_PATH = Path(__file__).with_name("best_66199.ir")

Monomial = tuple[int, ...]
Polynomial = dict[Monomial, int]


def _add(left: Polynomial, right: Polynomial, sign: int = 1) -> Polynomial:
    result = dict(left)
    for monomial, coefficient in right.items():
        updated = result.get(monomial, 0) + sign * coefficient
        if updated:
            result[monomial] = updated
        else:
            result.pop(monomial, None)
    return result


def _multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            monomial = left_monomial + right_monomial
            result[monomial] = (
                result.get(monomial, 0) + left_coefficient * right_coefficient
            )
    return {
        monomial: coefficient for monomial, coefficient in result.items() if coefficient
    }


def _read_cost(address: int) -> int:
    if address < 1:
        raise ValueError(f"address must be positive: {address}")
    return math.isqrt(address - 1) + 1


def _prove(ir: str) -> tuple[Counter[str], Counter[str]]:
    lines = [
        line.strip() for line in ir.replace(";", "\n").splitlines() if line.strip()
    ]
    if len(lines) < 2:
        raise ValueError("IR needs an input line and an output line")

    input_addresses = [int(part) for part in lines[0].split(",")]
    output_addresses = [int(part) for part in lines[-1].split(",")]
    if len(input_addresses) != 2 * N * N:
        raise ValueError(f"expected 512 inputs, got {len(input_addresses)}")
    if len(set(input_addresses)) != len(input_addresses):
        raise ValueError("input addresses must be distinct")
    if len(output_addresses) != N * N:
        raise ValueError(f"expected 256 outputs, got {len(output_addresses)}")
    if any(address < 1 for address in input_addresses + output_addresses):
        raise ValueError("input and output addresses must be positive")

    memory: dict[int, Polynomial] = {
        address: {(variable,): 1} for variable, address in enumerate(input_addresses)
    }
    operation_counts: Counter[str] = Counter()
    read_costs: Counter[str] = Counter()

    for operation_number, line in enumerate(lines[1:-1], start=1):
        opcode, separator, operand_text = line.partition(" ")
        if not separator:
            raise ValueError(f"malformed operation {operation_number}: {line}")
        operands = [int(part) for part in operand_text.split(",")]
        if any(address < 1 for address in operands):
            raise ValueError(f"operation {operation_number} has a non-positive address")

        if opcode == "copy" and len(operands) == 2:
            destination, source = operands
            sources = (source,)
        elif opcode in {"add", "sub", "mul"} and len(operands) in {2, 3}:
            if len(operands) == 3:
                destination, source1, source2 = operands
            else:
                destination, source2 = operands
                source1 = destination
            sources = (source1, source2)
        else:
            raise ValueError(f"unsupported operation {operation_number}: {line}")

        for source in sources:
            if source not in memory:
                raise ValueError(
                    f"operation {operation_number} reads uninitialized address {source}"
                )
            read_costs[opcode] += _read_cost(source)

        if opcode == "copy":
            value = dict(memory[source])
        elif opcode == "add":
            value = _add(memory[source1], memory[source2])
        elif opcode == "sub":
            value = _add(memory[source1], memory[source2], -1)
        else:
            value = _multiply(memory[source1], memory[source2])
        memory[destination] = value
        operation_counts[opcode] += 1

    for index, address in enumerate(output_addresses):
        if address not in memory:
            raise ValueError(f"output {index} reads uninitialized address {address}")
        read_costs["output"] += _read_cost(address)
        row, column = divmod(index, N)
        expected = {
            (row * N + inner, N * N + inner * N + column): 1 for inner in range(N)
        }
        if memory[address] != expected:
            raise AssertionError(f"formal mismatch at output ({row}, {column})")

    return operation_counts, read_costs


if __name__ == "__main__":
    from matmul import score_16x16

    encoded = IR_PATH.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != EXPECTED_SHA256:
        raise AssertionError(f"SHA-256 mismatch: {digest}")

    ir = encoded.decode()
    official_score = score_16x16(ir)
    operation_counts, read_costs = _prove(ir)
    formal_score = sum(read_costs.values())
    if official_score != EXPECTED_SCORE or formal_score != EXPECTED_SCORE:
        raise AssertionError(
            f"score mismatch: official={official_score}, formal={formal_score}"
        )

    print(f"{IR_PATH.name}: score={official_score:,}, sha256={digest}")
    print(
        "formal proof: 256/256 outputs match over the free noncommutative integer polynomial ring"
    )
    print(f"operations: {dict(sorted(operation_counts.items()))}")
    print(f"read costs: {dict(sorted(read_costs.items()))}")
