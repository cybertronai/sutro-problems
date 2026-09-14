"""Verify the 63,639 record and its exact fixed-trace allocation bound."""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matmul import _parse, score_16x16
from matmul.submissions.best_66178 import _prove

EXPECTED_SCORE = 63_639
EXPECTED_SHA256 = "4f1ce0c343cc0dd3d78b56be6907fd8a9bb59e93af0e9b3293028788fa5f61d0"
CERTIFICATE_SHA256 = "93949653b0d86e1bb9be16aa366a8a1b1518677e088b784f9edc4cda29e07370"
ORIGINAL_SHA256 = "e8adc50783a64088e6864ecfccb7e18a74814e24cfd6e1f9e3045e61fb37f2df"
EXPECTED_OPERATIONS = {"copy": 1947, "mul": 4096, "add": 3840}
EXPECTED_READ_COSTS = {"copy": 20558, "mul": 18471, "add": 20271, "output": 4339}
IR_PATH = Path(__file__).with_suffix(".ir")
CERTIFICATE_PATH = Path(__file__).with_suffix(".certificate.json")
ORIGINAL_PATH = IR_PATH.with_name("best_63819.ir")


def _read_pinned(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise AssertionError(f"{path.name}: SHA-256 mismatch")
    return data.decode("utf-8")


def _trace(ir):
    """Derive SSA intervals and a copy-erased, ordered arithmetic DAG."""
    inputs, operations, outputs = _parse(ir)
    current = {address: value for value, address in enumerate(inputs)}
    if len(current) != len(inputs):
        raise ValueError("input addresses must be distinct")
    births = [-1] * len(inputs)
    ends = [0] * len(inputs)
    reads = [0] * len(inputs)
    origins = [("input", value) for value in range(len(inputs))]
    arithmetic = []
    for index, (opcode, operands) in enumerate(operations):
        if opcode == "copy" and len(operands) == 2:
            addresses = operands[1:]
        elif opcode in {"add", "sub", "mul"} and len(operands) in {2, 3}:
            addresses = operands[1:] if len(operands) == 3 else operands
        else:
            raise ValueError(f"unsupported instruction: {opcode} {operands}")
        sources = tuple(current[address] for address in addresses)
        for source in sources:
            reads[source] += 1
            ends[source] = index
        if opcode == "copy":
            origin = origins[sources[0]]
        else:
            arithmetic.append((opcode, tuple(origins[source] for source in sources)))
            origin = ("arithmetic", len(arithmetic) - 1)
        current[operands[0]] = len(births)
        births.append(index)
        # Even an unread destination needs a cell after its write.
        ends.append(index + 1)
        reads.append(0)
        origins.append(origin)
    for address in outputs:
        value = current[address]
        reads[value] += 1
        ends[value] = len(operations)
    intervals = list(zip(births, ends, reads))
    canonical = (len(inputs), tuple(arithmetic), tuple(origins[current[a]] for a in outputs))
    return intervals, canonical, len(operations) + 1


def _dual_bound(intervals, gaps, certificate):
    """Reprice every SSA value exactly over all positive integer tiers."""
    tiers = certificate["modeled_tiers"]
    if not isinstance(tiers, int) or tiers < 1:
        raise ValueError("modeled tiers must be a positive integer")
    prices = {}
    for row in certificate["capacity_prices"]:
        if len(row) != 4 or any(type(x) is not int for x in row):
            raise ValueError("capacity prices require four integers")
        tier, gap, numerator, denominator = row
        if not (1 <= tier <= tiers and 0 <= gap < gaps):
            raise ValueError("capacity price index outside trace")
        if numerator < 0 or denominator <= 0:
            raise ValueError("capacity prices must be nonnegative rationals")
        if (tier, gap) in prices:
            raise ValueError("duplicate capacity price")
        prices[tier, gap] = Fraction(numerator, denominator)
    prefix = []
    for tier in range(1, tiers + 1):
        cumulative = [Fraction(0)]
        for gap in range(gaps):
            cumulative.append(cumulative[-1] + prices.get((tier, gap), 0))
        prefix.append(cumulative)
    bound = Fraction(0)
    for birth, end, reads in intervals:
        # Occupancy is birth < gap <= end. A dying source can share the
        # destination's address because reads happen before the write.
        choices = [reads * tier + rent[end + 1] - rent[birth + 1]
                   for tier, rent in enumerate(prefix, 1)]
        # All higher tiers have zero rent. Their cheapest read cost occurs
        # at tiers+1, so this single choice covers the entire infinite tail.
        choices.append(reads * (tiers + 1))
        bound += min(choices)
    return bound - sum((2 * tier - 1) * price
                       for (tier, _gap), price in prices.items())


def verify():
    ir = _read_pinned(IR_PATH, EXPECTED_SHA256)
    score = score_16x16(ir)
    operations, costs = _prove(ir)
    if score != EXPECTED_SCORE or sum(costs.values()) != EXPECTED_SCORE:
        raise AssertionError("official or independent score mismatch")
    if operations != EXPECTED_OPERATIONS:
        raise AssertionError("operation counts mismatch")
    if costs != EXPECTED_READ_COSTS:
        raise AssertionError("read-cost breakdown mismatch")
    intervals, arithmetic, gaps = _trace(ir)
    original = _read_pinned(ORIGINAL_PATH, ORIGINAL_SHA256)
    if arithmetic != _trace(original)[1]:
        raise AssertionError("original arithmetic DAG or order changed")
    certificate = json.loads(_read_pinned(CERTIFICATE_PATH, CERTIFICATE_SHA256))
    if certificate["ir_sha256"] != EXPECTED_SHA256:
        raise AssertionError("certificate names a different IR")
    bound = _dual_bound(intervals, gaps, certificate)
    if bound != score:
        raise AssertionError(f"exact allocation bound {bound} does not equal {score}")
    return {"score": score, "lower_bound": str(bound),
            "operations": dict(operations), "read_costs": dict(costs)}


if __name__ == "__main__":
    result = verify()
    print(f"{IR_PATH.name}: score={result['score']:,}, sha256={EXPECTED_SHA256}")
    print("Exact symbolic proof: 256/256 outputs; arithmetic DAG and order preserved.")
    print("Fixed-trace optimum: feasible score = exact rational dual = 63,639.")
    print(f"operations: {result['operations']}")
    print(f"read costs: {result['read_costs']}")
