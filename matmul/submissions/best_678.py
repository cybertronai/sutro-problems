"""Verify the 678-cost 4x4 matrix-multiplication submission."""
from __future__ import annotations

import hashlib
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_SCORE = 678
EXPECTED_SHA256 = "1e024f1aefdafc044fd825c2174bc3c3b7293c4845fad46dfa0fb81ef6ad7805"
EXPECTED_OPERATIONS = Counter({"mul": 64, "add": 48, "copy": 14})
EXPECTED_COSTS = Counter({"mul": 353, "add": 185, "copy": 73, "output": 67})
IR_PATH = Path(__file__).with_name("best_678.ir")


def generate_best_678() -> str:
    return IR_PATH.read_text(encoding="utf-8")


def _read_cost(address: int) -> int:
    return math.isqrt(address - 1) + 1


def main() -> None:
    from matmul import _parse, score_4x4

    encoded = IR_PATH.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != EXPECTED_SHA256:
        raise AssertionError(f"SHA-256 mismatch: {digest}")

    ir = encoded.decode("utf-8")
    score = score_4x4(ir)
    if score != EXPECTED_SCORE:
        raise AssertionError(f"expected {EXPECTED_SCORE}, got {score}")

    inputs, operations, outputs = _parse(ir)
    counts = Counter(opcode for opcode, _ in operations)
    costs: Counter[str] = Counter()
    for opcode, operands in operations:
        sources = operands[1:] if opcode == "copy" or len(operands) == 3 else operands
        costs[opcode] += sum(_read_cost(address) for address in sources)
    costs["output"] = sum(_read_cost(address) for address in outputs)

    if len(inputs) != 32 or len(set(inputs)) != 32:
        raise AssertionError("expected 32 distinct inputs")
    if len(outputs) != 16 or len(set(outputs)) != 16:
        raise AssertionError("expected 16 distinct outputs")
    if counts != EXPECTED_OPERATIONS:
        raise AssertionError(f"unexpected operations: {counts}")
    if costs != EXPECTED_COSTS or sum(costs.values()) != score:
        raise AssertionError(f"unexpected read costs: {costs}")

    print(f"{IR_PATH.name}: score={score}, sha256={digest}")
    print(f"operations: {dict(sorted(counts.items()))}")
    print(f"read costs: {dict(sorted(costs.items()))}")


if __name__ == "__main__":
    main()
