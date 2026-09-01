"""Verify the 681-cost 4x4 matrix-multiplication submission.

Run from the repository root:

    python3 matmul/submissions/best_681.py
"""
from __future__ import annotations

import hashlib
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_SCORE = 681
EXPECTED_SHA256 = "05d1150da06eea9581e51c0617488c49bf52d02c41242920d40e2eb940e854a7"
IR_PATH = Path(__file__).with_name("best_681.ir")


def generate_best_681() -> str:
    """Return the exact submitted IR."""
    return IR_PATH.read_text(encoding="utf-8")


def _read_cost(address: int) -> int:
    if address < 1:
        raise ValueError(f"address must be positive: {address}")
    return math.isqrt(address - 1) + 1


def _breakdown(ir: str) -> tuple[Counter[str], Counter[str]]:
    """Return operation counts and read-cost contributions."""
    from matmul import _parse

    _inputs, operations, outputs = _parse(ir)
    operation_counts: Counter[str] = Counter()
    read_costs: Counter[str] = Counter()
    for opcode, operands in operations:
        if opcode == "copy" and len(operands) == 2:
            sources = operands[1:]
        elif opcode in {"add", "sub", "mul"} and len(operands) == 3:
            sources = operands[1:]
        elif opcode in {"add", "sub", "mul"} and len(operands) == 2:
            sources = operands
        else:
            raise ValueError(f"unsupported instruction: {opcode} {operands}")
        operation_counts[opcode] += 1
        read_costs[opcode] += sum(_read_cost(address) for address in sources)
    read_costs["output"] = sum(_read_cost(address) for address in outputs)
    return operation_counts, read_costs


def main() -> None:
    from matmul import score_4x4

    encoded = IR_PATH.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != EXPECTED_SHA256:
        raise AssertionError(f"SHA-256 mismatch: {digest}")

    ir = encoded.decode("utf-8")
    score = score_4x4(ir)
    if score != EXPECTED_SCORE:
        raise AssertionError(f"expected {EXPECTED_SCORE}, got {score}")

    operation_counts, read_costs = _breakdown(ir)
    expected_operations = Counter({"mul": 64, "add": 48, "copy": 13})
    expected_costs = Counter({"mul": 345, "add": 197, "copy": 72, "output": 67})
    if operation_counts != expected_operations:
        raise AssertionError(f"unexpected operations: {operation_counts}")
    if read_costs != expected_costs or sum(read_costs.values()) != EXPECTED_SCORE:
        raise AssertionError(f"unexpected read-cost breakdown: {read_costs}")

    print(f"{IR_PATH.name}: score={score}, sha256={digest}")
    print(f"operations: {dict(sorted(operation_counts.items()))}")
    print(f"read costs: {dict(sorted(read_costs.items()))}")


if __name__ == "__main__":
    main()
