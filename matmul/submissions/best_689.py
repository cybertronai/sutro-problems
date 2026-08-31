"""Reproduce and verify the 689-cost 4×4 matrix-multiplication submission.

Place this file and ``best_689.ir`` in ``matmul/submissions/`` and run from the
repository root:

    python3 matmul/submissions/best_689.py
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

EXPECTED_SCORE = 689
EXPECTED_SHA256 = "280f3a566c21858a37cc0642abff65857853f52545b656ec2bea72ef62d5122c"
IR_PATH = Path(__file__).with_name("best_689.ir")


def generate_best_689() -> str:
    """Return the exact leaderboard IR."""
    return IR_PATH.read_text(encoding="utf-8")


def _read_cost(address: int) -> int:
    if address < 1:
        raise ValueError(f"address must be positive: {address}")
    return math.isqrt(address - 1) + 1


def _breakdown(ir: str) -> tuple[Counter[str], Counter[str]]:
    lines = [line.strip() for line in ir.replace(";", "\n").splitlines() if line.strip()]
    operation_counts: Counter[str] = Counter()
    read_costs: Counter[str] = Counter()

    for line in lines[1:-1]:
        opcode, raw = line.split(None, 1)
        operands = [int(part) for part in raw.split(",")]
        if opcode == "copy":
            sources = operands[1:2]
        elif opcode in {"add", "sub", "mul"}:
            sources = operands[1:] if len(operands) == 3 else [operands[0], operands[1]]
        else:
            raise ValueError(f"unsupported opcode: {opcode}")
        operation_counts[opcode] += 1
        read_costs[opcode] += sum(_read_cost(address) for address in sources)

    output_addresses = [int(part) for part in lines[-1].split(",")]
    read_costs["output"] = sum(_read_cost(address) for address in output_addresses)
    return operation_counts, read_costs


if __name__ == "__main__":
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
    if sum(read_costs.values()) != EXPECTED_SCORE:
        raise AssertionError(f"breakdown does not sum to {EXPECTED_SCORE}: {read_costs}")

    print(f"{IR_PATH.name}: score={score}, sha256={digest}")
    print(f"operations: {dict(sorted(operation_counts.items()))}")
    print(f"read costs: {dict(sorted(read_costs.items()))}")
