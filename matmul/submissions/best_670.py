"""Verify the frozen 670-cost witness and its fixed-trace allocation certificate.

The certificate fixes arithmetic, instruction order, copies, source bindings,
and distinct overlapping residence versions. It does not bound all programs.
Only the Python standard library is needed, including for exact dual checking.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matmul import score_4x4

EXPECTED_SCORE = 670
EXPECTED_SHA256 = "cd48466797857b35a063de73eb5e222b98b4d3f1056691bbddd5b3c37b73e32a"
EXPECTED_CERTIFICATE_SHA256 = "6bc4a36557a41217c64c44b1419b2cd5a3a929c88fa9039eb0a856ae91a688dd"
EXPECTED_OPERATIONS = {"mul": 64, "add": 48, "copy": 19}
EXPECTED_READ_COSTS = {"mul": 321, "add": 185, "copy": 95, "output": 69}
IR_PATH = Path(__file__).with_suffix(".ir")
CERTIFICATE_PATH = Path(__file__).with_suffix(".certificate.json")
CERTIFICATE_SCOPE = (
    "Fixed arithmetic, instruction order, copies, source bindings, and distinct "
    "overlapping residence versions; not a lower bound for all programs."
)

Polynomial = dict[tuple[int, ...], int]


@dataclass
class Version:
    """A value residence with a half-open [birth, last-read) lifetime."""

    birth: int
    end: int
    reads: int = 0


def generate_best_670() -> str:
    """Load the frozen witness; this is not a search or reconstruction."""
    encoded = IR_PATH.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != EXPECTED_SHA256:
        raise AssertionError("frozen witness SHA-256 mismatch")
    return encoded.decode("utf-8")


def _read_cost(address: int) -> int:
    if address < 1:
        raise ValueError("addresses must be positive")
    return math.isqrt(address - 1) + 1


def _polynomial(opcode: str, values: list[Polynomial]) -> Polynomial:
    if opcode == "copy":
        return dict(values[0])
    left, right = values
    result: Polynomial = {}
    if opcode == "mul":
        if left and right and max(map(len, left)) + max(map(len, right)) > 2:
            raise ValueError("polynomial degree exceeds two")
        for first, a in left.items():
            for second, b in right.items():
                monomial = tuple(sorted(first + second))
                result[monomial] = result.get(monomial, 0) + a * b
    else:
        result.update(left)
        sign = -1 if opcode == "sub" else 1
        for monomial, coefficient in right.items():
            result[monomial] = result.get(monomial, 0) + sign * coefficient
    return {monomial: coefficient for monomial, coefficient in result.items() if coefficient}


def _prove(ir: str, n: int = 4) -> tuple[dict, list[Version]]:
    """Independently parse and interpret the IR over exact integer polynomials."""
    lines = [line.strip() for line in ir.replace(";", "\n").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("IR needs input and output lines")
    inputs = [int(part) for part in lines[0].split(",")]
    outputs = [int(part) for part in lines[-1].split(",")]
    if len(inputs) != 2 * n * n or len(set(inputs)) != len(inputs):
        raise ValueError("expected distinct matrix inputs")
    if len(outputs) != n * n:
        raise ValueError("unexpected matrix output count")
    addresses = inputs + outputs
    if min(addresses) < 1:
        raise ValueError("addresses must be positive")

    memory = {address: {(i,): 1} for i, address in enumerate(inputs)}
    resident = {address: i for i, address in enumerate(inputs)}
    versions = [Version(0, 0) for _ in inputs]
    operations: Counter[str] = Counter()
    costs: Counter[str] = Counter()

    def read(address: int, step: int, opcode: str) -> Polynomial:
        if address not in memory:
            raise ValueError(f"{opcode} reads uninitialized address {address}")
        version = versions[resident[address]]
        version.reads += 1
        version.end = step
        costs[opcode] += _read_cost(address)
        return memory[address]

    for step, line in enumerate(lines[1:-1], 1):
        opcode, separator, rest = line.partition(" ")
        if not separator:
            raise ValueError(f"malformed instruction: {line}")
        operands = [int(part) for part in rest.split(",")]
        if not operands or min(operands) < 1:
            raise ValueError("addresses must be positive")
        addresses.extend(operands)
        if opcode == "copy" and len(operands) == 2:
            destination, *sources = operands
        elif opcode in {"add", "sub", "mul"} and len(operands) in {2, 3}:
            destination = operands[0]
            sources = operands[1:] if len(operands) == 3 else operands
        else:
            raise ValueError(f"unsupported instruction: {line}")
        # Both reads refer to the old residence, even when the destination aliases.
        values = [read(source, step, opcode) for source in sources]
        memory[destination] = _polynomial(opcode, values)
        resident[destination] = len(versions)
        versions.append(Version(step, step))
        operations[opcode] += 1

    exit_step = len(lines) - 1
    for index, address in enumerate(outputs):
        row, column = divmod(index, n)
        expected = {
            (row * n + k, n * n + k * n + column): 1 for k in range(n)
        }
        if read(address, exit_step, "output") != expected:
            raise AssertionError(f"symbolic output mismatch at ({row}, {column})")

    # Last-read-then-write reuse is legal: a residence ending at t does not
    # overlap one born at t. Unread versions have empty lifetimes.
    peak_liveness = max(
        sum(version.birth <= step < version.end for version in versions)
        for step in range(exit_step)
    )
    return {
        "score": sum(costs.values()),
        "operations": dict(sorted(operations.items())),
        "costs": dict(sorted(costs.items())),
        "reads": sum(version.reads for version in versions),
        "versions": len(versions),
        "max_address": max(addresses),
        "peak_liveness": peak_liveness,
    }, versions


def check_certificate(versions: list[Version], certificate: dict, score: int) -> dict:
    """Check a tier-allocation LP dual using rational arithmetic only.

    Tier t has 2*t-1 cells. Each simultaneous clique can use at most that
    many residences in the tier. With beta <= 0, the checked inequalities
    alpha[v] + sum(beta[clique,t]) <= reads[v]*t give a lower bound by
    weak duality. Higher tiers use beta=0 and are checked in one step.
    """
    count = len(versions)
    if not isinstance(certificate, dict):
        raise ValueError("certificate must be a JSON object")
    tiers = certificate.get("tiers")
    if type(tiers) is not int or tiers != _read_cost(count):
        raise ValueError("unexpected certificate tier count")
    cliques = certificate.get("cliques")
    if not isinstance(cliques, list):
        raise ValueError("certificate cliques must be a list")

    memberships: list[list[int]] = [[] for _ in versions]
    for index, clique in enumerate(cliques):
        if not isinstance(clique, list) or any(
            type(v) is not int or not 0 <= v < count for v in clique
        ):
            raise ValueError("invalid clique version index")
        if len(clique) != len(set(clique)):
            raise ValueError("duplicate version in clique")
        if clique and max(versions[v].birth for v in clique) >= min(
            versions[v].end for v in clique
        ):
            raise AssertionError("clique versions are not simultaneously live")
        for v in clique:
            memberships[v].append(index)

    def rationals(name: str, length: int) -> list[Fraction]:
        values = certificate.get(name)
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"unexpected {name} dimensions")
        if any(not isinstance(value, str) for value in values):
            raise ValueError(f"{name} must contain exact rational strings")
        return [Fraction(value) for value in values]

    alpha = rationals("alpha", count)
    beta = rationals("beta", len(cliques) * tiers)
    if any(value > 0 for value in beta):
        raise AssertionError("dual beta must be nonpositive")
    for v, version in enumerate(versions):
        for j in range(tiers):
            lhs = alpha[v] + sum(beta[q * tiers + j] for q in memberships[v])
            if lhs > version.reads * (j + 1):
                raise AssertionError(f"dual inequality failed for version {v}, tier {j + 1}")
        # Reads are nonnegative, so this proves the constraint for every
        # omitted tier t >= tiers+1 without an address-compression assumption.
        if alpha[v] > version.reads * (tiers + 1):
            raise AssertionError(f"higher-tier dual inequality failed for version {v}")
    lower = sum(alpha) + sum(
        value * (2 * (index % tiers) + 1) for index, value in enumerate(beta)
    )
    bound = certificate.get("bound")
    if not isinstance(bound, str):
        raise ValueError("bound must be an exact rational string")
    if lower != Fraction(bound) or lower != score:
        raise AssertionError("certificate lower bound does not match witness score")
    return {
        "fixed_trace_lower_bound": str(lower),
        "certificate_tiers": tiers,
        "cliques": len(cliques),
        "dual_inequalities": count * tiers,
        "higher_tiers_verified": True,
        "scope": CERTIFICATE_SCOPE,
    }


def verify() -> dict:
    ir = generate_best_670()
    certificate_bytes = CERTIFICATE_PATH.read_bytes()
    certificate_digest = hashlib.sha256(certificate_bytes).hexdigest()
    if certificate_digest != EXPECTED_CERTIFICATE_SHA256:
        raise AssertionError("frozen certificate SHA-256 mismatch")
    official_score = score_4x4(ir)
    report, versions = _prove(ir)
    if official_score != EXPECTED_SCORE or report["score"] != EXPECTED_SCORE:
        raise AssertionError("official or independent score mismatch")
    if report["operations"] != EXPECTED_OPERATIONS:
        raise AssertionError("operation counts mismatch")
    if report["costs"] != EXPECTED_READ_COSTS:
        raise AssertionError("read-cost breakdown mismatch")
    for name, expected in (("versions", 163), ("max_address", 37), ("peak_liveness", 36)):
        if report[name] != expected:
            raise AssertionError(f"{name} mismatch")
    report.update(check_certificate(versions, json.loads(certificate_bytes), official_score))
    report.update(sha256=EXPECTED_SHA256, certificate_sha256=certificate_digest)
    return report


if __name__ == "__main__":
    report = verify()
    print(f"{IR_PATH.name}: score={report['score']}, sha256={report['sha256']}")
    print(json.dumps(report, indent=2))
