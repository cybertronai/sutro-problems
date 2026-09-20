"""Verify the frozen 667-cost 4x4 matrix-multiplication submission."""
from __future__ import annotations

import hashlib
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_SCORE = 667
EXPECTED_SHA256 = "7bf91dafc04584b6ce8f4d4d5e4de917ea426e47d84b413b8a068f78e143d7a3"
EXPECTED_OPERATIONS = Counter({"mul": 64, "add": 48, "copy": 20})
EXPECTED_COSTS = Counter({"mul": 319, "add": 178, "copy": 102, "output": 68})
IR_PATH = Path(__file__).with_suffix(".ir")


def generate_best_667() -> str:
    """Load the hash-pinned IR; this does not rerun the search."""
    encoded = IR_PATH.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != EXPECTED_SHA256:
        raise AssertionError(f"SHA-256 mismatch: {digest}")
    return encoded.decode("utf-8")


def _read_cost(address: int) -> int:
    return math.isqrt(address - 1) + 1


def main() -> None:
    from matmul import _parse, score_4x4

    ir = generate_best_667()
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

    print(f"{IR_PATH.name}: score={score}, sha256={EXPECTED_SHA256}")
    print(f"operations: {dict(sorted(counts.items()))}")
    print(f"read costs: {dict(sorted(costs.items()))}")


if __name__ == "__main__":
    main()
