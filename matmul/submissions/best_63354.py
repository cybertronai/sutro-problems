"""Verify all 256 exact matrix outputs and the 63,354 weighted-read score."""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXPECTED_SCORE = 63354
EXPECTED_SHA256 = "615bbea8519ead81f3074dacb0e9789e4bdb093759be92c3f497c1506cfda779"
EXPECTED_OPERATIONS = {"copy": 2168, "mul": 4096, "add": 3840}
EXPECTED_READ_COSTS = {"copy": 20696, "mul": 18348, "add": 20081, "output": 4229}
DEPENDENCIES = {
    "matmul/matmul.py": "cb701af7e34aa330492a76e43483c7953bb55a883c95398524bcc551a15fd4d9",
    "matmul/__init__.py": "b81bd729b9013020e40278c36956069bc54eda91a152595233b9355eabe60853",
    "matmul/submissions/best_66178.py": "56045f1436cfce083c570069917e9242bfa11f2df91b27bd1a64cbca3c086284",
}


def verify():
    for name, expected in DEPENDENCIES.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    from matmul import score_16x16
    from matmul.submissions.best_66178 import _prove

    data = Path(__file__).with_suffix(".ir").read_bytes()
    assert hashlib.sha256(data).hexdigest() == EXPECTED_SHA256
    score = score_16x16(data.decode())
    operations, reads = _prove(data.decode())
    assert score == EXPECTED_SCORE == sum(reads.values())
    assert (
        dict(operations) == EXPECTED_OPERATIONS and dict(reads) == EXPECTED_READ_COSTS
    )
    return score


if __name__ == "__main__":
    print(
        f"Verified score={verify():,}; all 256 outputs match exact integer polynomials."
    )
    print(f"SHA-256: {EXPECTED_SHA256}")
    print(f"Operations: {EXPECTED_OPERATIONS}")
    print(f"Read costs: {EXPECTED_READ_COSTS}")
