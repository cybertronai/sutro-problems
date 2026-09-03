"""Load and verify the 65,084-cost 16x16 matmul submission."""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matmul import score_16x16  # noqa: E402
from matmul.submissions.best_66178 import _prove  # noqa: E402

EXPECTED_SCORE = 65_084
EXPECTED_SHA256 = "600502a7a2b10f762f145a682b21ecb9062ee81eb3723b250d9a2dbccab03c3e"
EXPECTED_OPERATIONS = Counter({"mul": 4_096, "add": 3_840, "copy": 1_649})
EXPECTED_READ_COSTS = Counter(
    {"add": 24_897, "copy": 21_539, "mul": 14_300, "output": 4_348}
)
IR_PATH = Path(__file__).with_name("best_65084.ir")


def generate_best_65084() -> str:
    return IR_PATH.read_text(encoding="utf-8")


def main() -> None:
    ir = generate_best_65084()
    digest = hashlib.sha256(ir.encode()).hexdigest()
    operations, read_costs = _prove(ir)
    official_score = score_16x16(ir)

    if digest != EXPECTED_SHA256:
        raise AssertionError(f"SHA-256 mismatch: {digest}")
    if (
        official_score != EXPECTED_SCORE
        or sum(read_costs.values()) != EXPECTED_SCORE
    ):
        raise AssertionError(f"score mismatch: {official_score}, {read_costs}")
    if operations != EXPECTED_OPERATIONS or read_costs != EXPECTED_READ_COSTS:
        raise AssertionError(f"breakdown mismatch: {operations}, {read_costs}")

    print(f"{IR_PATH.name}: score={official_score:,}, sha256={digest}")
    print(f"operations: {dict(sorted(operations.items()))}")
    print(f"read costs: {dict(sorted(read_costs.items()))}")


if __name__ == "__main__":
    main()
