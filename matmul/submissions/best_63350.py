"""Verify the frozen 63,350-cost 16x16 matrix multiplication program.

Verification uses only the standard library and existing repository helpers.
The optimization that produced this artifact used external LP/MILP tooling;
loading the artifact does not rerun that search.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IR_PATH = Path(__file__).with_suffix(".ir")
EXPECTED_SCORE = 63350
EXPECTED_SHA256 = "3cf3b8dea456fa0a7dba2e5fab3a31af0dc971fd11d7b04332bcfd967d3259af"
EXPECTED_OPERATIONS = {"copy": 2167, "mul": 4096, "add": 3840}
EXPECTED_READ_COSTS = {"copy": 20666, "mul": 18344, "add": 20082, "output": 4258}
EXPECTED_MAX_ADDRESS = 620
DEPENDENCIES = {
    "matmul/matmul.py": "cb701af7e34aa330492a76e43483c7953bb55a883c95398524bcc551a15fd4d9",
    "matmul/__init__.py": "b81bd729b9013020e40278c36956069bc54eda91a152595233b9355eabe60853",
    "matmul/submissions/best_66178.py": "56045f1436cfce083c570069917e9242bfa11f2df91b27bd1a64cbca3c086284",
}


def generate_best_63350() -> str:
    """Load the hash-pinned artifact; this is not an optimizer replay."""
    data = IR_PATH.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise AssertionError(f"artifact SHA-256 mismatch: {digest}")
    return data.decode("utf-8")


def verify() -> int:
    for name, expected in DEPENDENCIES.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"verification dependency changed: {name}")

    from matmul import _parse, score_16x16
    from matmul.submissions.best_66178 import _prove

    ir = generate_best_63350()
    official = score_16x16(ir)
    counts, costs = _prove(ir)
    if official != EXPECTED_SCORE or sum(costs.values()) != EXPECTED_SCORE:
        raise AssertionError(
            f"score mismatch: official={official}, independent={sum(costs.values())}"
        )
    if dict(counts) != EXPECTED_OPERATIONS or dict(costs) != EXPECTED_READ_COSTS:
        raise AssertionError(f"operation/read accounting changed: {counts}, {costs}")
    inputs, operations, outputs = _parse(ir)
    maximum = max(
        inputs + outputs + [address for _, operands in operations for address in operands]
    )
    if maximum != EXPECTED_MAX_ADDRESS:
        raise AssertionError(f"maximum address changed: {maximum}")
    if any(op == "copy" and operands[0] == operands[1] for op, operands in operations):
        raise AssertionError("artifact contains a redundant self-copy")
    return official


if __name__ == "__main__":
    print(f"Verified score={verify():,}; all 256 outputs match exact integer polynomials.")
    print(f"SHA-256: {EXPECTED_SHA256}")
    print(f"Operations: {EXPECTED_OPERATIONS}")
    print(f"Read costs: {EXPECTED_READ_COSTS}")
