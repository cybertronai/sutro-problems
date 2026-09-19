"""Reproduce and verify the 674-cost 4x4 matrix-multiplication submission.

Replays the winning 20-instruction replacement against the hash-pinned 675
record. The local search that discovered this replacement is not rerun.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IR_PATH = Path(__file__).with_suffix(".ir")
BASE_PATH = IR_PATH.with_name("best_675.ir")
BASE_SHA256 = "761d0401876c4c1fe8f51c0ff7ecf3676b7abca1f56b4e36715d70426d35427e"
EXPECTED_SCORE = 674
EXPECTED_SHA256 = "9370c290d54e5ef2edb2ae69986f65c8932e68ca9f409574dd2ee96b7137094b"
EXPECTED_OPERATIONS = Counter({"mul": 64, "add": 48, "copy": 16})
EXPECTED_COSTS = Counter({"mul": 341, "add": 185, "copy": 80, "output": 68})

# Instructions 17-36 (one-based, excluding input/output placement lines).
REPLACEMENT = """mul 2,3,21
mul 1,3,13
mul 3,4,9
add 3,3,1
mul 1,4,11
add 2,1,2
copy 1,5
mul 4,1,15
mul 1,1,23
add 2,2,1
mul 1,6,7
add 1,1,3
copy 3,17
add 37,1,4
copy 1,20
mul 4,3,1
mul 1,6,1
add 17,1,2
copy 1,26
mul 2,1,11"""


def generate_best_674() -> str:
    """Reconstruct the submitted IR without a compiler or search dependencies."""
    source = BASE_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != BASE_SHA256:
        raise AssertionError("675 baseline SHA-256 mismatch")
    lines = source.decode("utf-8").splitlines()
    operations = lines[1:-1]
    operations[16:36] = REPLACEMENT.splitlines()
    ir = "\n".join([lines[0], *operations, lines[-1]]) + "\n"
    if hashlib.sha256(ir.encode("utf-8")).hexdigest() != EXPECTED_SHA256:
        raise AssertionError("reconstructed IR SHA-256 mismatch")
    return ir


def verify() -> int:
    from matmul import _parse, score_4x4

    data = IR_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise AssertionError("artifact SHA-256 mismatch")
    ir = data.decode("utf-8")
    if ir != generate_best_674():
        raise AssertionError("reconstruction differs from artifact")
    score = score_4x4(ir)
    if score != EXPECTED_SCORE:
        raise AssertionError(f"expected {EXPECTED_SCORE}, got {score}")

    inputs, operations, outputs = _parse(ir)
    counts = Counter(opcode for opcode, _ in operations)
    costs: Counter[str] = Counter()
    reads = len(outputs)
    for opcode, operands in operations:
        sources = operands[1:] if opcode == "copy" or len(operands) == 3 else operands
        costs[opcode] += sum(math.isqrt(a - 1) + 1 for a in sources)
        reads += len(sources)
    costs["output"] = sum(math.isqrt(a - 1) + 1 for a in outputs)

    if len(inputs) != 32 or len(set(inputs)) != 32:
        raise AssertionError("expected 32 distinct inputs")
    if len(outputs) != 16 or len(set(outputs)) != 16:
        raise AssertionError("expected 16 distinct outputs")
    if counts != EXPECTED_OPERATIONS:
        raise AssertionError(f"unexpected operations: {counts}")
    if costs != EXPECTED_COSTS or sum(costs.values()) != score:
        raise AssertionError(f"unexpected read costs: {costs}")
    maximum = max(inputs + outputs + [a for _, aa in operations for a in aa])
    if reads != 256 or maximum != 37:
        raise AssertionError(f"unexpected paid reads or maximum address: {reads}, {maximum}")
    return score


if __name__ == "__main__":
    print(f"{IR_PATH.name}: score={verify()}, sha256={EXPECTED_SHA256}")
    print("All 16 outputs match exact integer polynomials; reconstruction matches the artifact.")
    print(f"Operations: {dict(EXPECTED_OPERATIONS)}")
    print(f"Read costs: {dict(EXPECTED_COSTS)}")
