"""Dally v3 IR scorer for the symmetry (palindrome-detection) problem.

This implements the target classification task from Rumelhart, Hinton &
Williams (1986): given n input bits, output 1 if the pattern is a palindrome
(x[i] == x[n-1-i] for all i < n//2), 0 otherwise.

The canonical 6-bit variant (SIX) is the direct match to the RHW1986
symmetry task, where a 6-2-1 MLP is trained to detect 6-bit palindromes.
The 8-bit variant (EIGHT) doubles the number of pairs as a harder target.

The function has a closed polynomial form over ±1-encoded inputs
    f(x) = prod_{i=0}^{n//2 - 1} (1 + x[i]*x[n-1-i]) / 2^(n//2)
which requires only mul and add — no sigmoid needed — making it a
natural Dally v3 IR target.

Scorer inputs  : n integers (0 or 1) placed at addresses 1..n.
Scorer output  : 1 integer — 1 if the pattern is a palindrome, else 0.
Correctness    : the IR must classify every possible input pattern correctly
                 (all 2^n patterns are tested exhaustively).
Cost           : static IR cost — sum of ceil(sqrt(addr-1))+1 over every
                 address read, charged once for each read, plus output reads.
"""
from __future__ import annotations

import math
import operator
from typing import Callable, List, Tuple


# ---------------------------------------------------------------------------
# Problem configurations
# ---------------------------------------------------------------------------

N_BITS_SIX   = 6   # canonical RHW1986 task (3 pairs, 8/64 palindromes)
N_BITS_EIGHT = 8   # harder variant (4 pairs, 16/256 palindromes)


def is_palindrome(pattern: List[int]) -> bool:
    """True iff pattern[i] == pattern[n-1-i] for all mirror pairs."""
    n = len(pattern)
    return all(pattern[i] == pattern[n - 1 - i] for i in range(n // 2))


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def _cost(addr: int) -> int:
    if not isinstance(addr, int) or addr < 1:
        raise ValueError(f"addresses must be positive integers; got {addr!r}")
    return math.isqrt(addr - 1) + 1


def _to_signed_8bit(val: int) -> int:
    val &= 0xFF
    return val - 0x100 if val >= 0x80 else val


# ---------------------------------------------------------------------------
# Static compiler + fast array simulator (Dally v3 IR)
# ---------------------------------------------------------------------------

def _safe_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("integer division or modulo by zero")
    return a // b


_CMP_OPS = {
    "eq": operator.eq, "ne": operator.ne, "lt": operator.lt,
    "le": operator.le, "gt": operator.gt, "ge": operator.ge,
}

_BINARY_OPS = {
    "add": operator.add, "sub": operator.sub, "mul": operator.mul,
    "div": _safe_div, "and": operator.and_, "or": operator.or_,
    "xor": operator.xor,
}

_UNARY_OPS = {
    "copy": lambda x: x, "not": operator.invert, "abs": abs,
}


def _compile_ir(ir: str) -> Tuple[Callable[[List[int]], List[int]], int, int]:
    text = ir.replace(";", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if len(lines) > 100_000:
        raise ValueError("IR exceeds maximum allowed length (100,000 instructions)")
    if len(lines) < 2:
        raise ValueError("IR needs at least an input line and an output line")

    def parse_addrs(line: str) -> List[int]:
        addrs = [int(x) for x in line.split(",") if x.strip()]
        for a in addrs:
            if a < 1 or a.bit_length() > 64:
                raise ValueError(f"invalid address {a}")
        return addrs

    try:
        input_addrs  = parse_addrs(lines[0])
        output_addrs = parse_addrs(lines[-1])
    except ValueError as e:
        raise ValueError(f"malformed input/output line: {e}")

    if len(set(input_addrs)) != len(input_addrs):
        raise ValueError("input addresses must be distinct")

    init = set(input_addrs)
    cost = 0
    ops: list = []

    for ln in lines[1:-1]:
        head, _, rest = ln.partition(" ")
        raw = [x.strip() for x in rest.split(",") if x.strip()] if rest else []

        try:
            if head == "set":
                if len(raw) != 2:
                    raise ValueError("needs 2 operands")
                dest, literal = int(raw[0]), int(raw[1])
                if not (-128 <= literal <= 255):
                    raise ValueError("literal out of bounds")
                reads = []
                ops.append((0, dest, _to_signed_8bit(literal), 0, None))

            elif head == "cmp":
                if len(raw) != 4:
                    raise ValueError("needs 4 operands")
                dest, a, b = int(raw[0]), int(raw[1]), int(raw[2])
                pred = raw[3]
                if pred not in _CMP_OPS:
                    raise ValueError("invalid predicate")
                reads = [a, b]
                ops.append((1, dest, a, b, _CMP_OPS[pred]))

            elif head == "select":
                if len(raw) != 4:
                    raise ValueError("needs 4 operands")
                dest, c, t, f = int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])
                reads = [c, t, f]
                ops.append((2, dest, c, t, f))

            elif head in _UNARY_OPS:
                if len(raw) != 2:
                    raise ValueError("needs 2 operands")
                dest, a = int(raw[0]), int(raw[1])
                reads = [a]
                ops.append((3, dest, a, 0, _UNARY_OPS[head]))

            elif head in _BINARY_OPS:
                if len(raw) not in (2, 3):
                    raise ValueError("needs 2 or 3 operands")
                dest = int(raw[0])
                s1, s2 = (
                    (int(raw[1]), int(raw[2])) if len(raw) == 3
                    else (dest, int(raw[1]))
                )
                reads = [s1, s2]
                ops.append((4, dest, s1, s2, _BINARY_OPS[head]))

            else:
                raise ValueError(f"unknown op: {head!r}")

            if dest < 1 or dest.bit_length() > 64:
                raise ValueError(f"invalid dest address {dest}")

            for src in reads:
                if src not in init:
                    raise ValueError(f"uninitialized read: {src}")
                cost += _cost(src)

            init.add(dest)

        except ValueError as e:
            raise ValueError(f"malformed instruction '{ln}': {e}")

    for a in output_addrs:
        if a not in init:
            raise ValueError(f"output addr {a} never written")
        cost += _cost(a)

    sorted_addrs  = sorted(list(init))
    addr_to_idx   = {a: i for i, a in enumerate(sorted_addrs)}
    in_idx        = [addr_to_idx[a] for a in input_addrs]
    out_idx       = [addr_to_idx[a] for a in output_addrs]
    n_mem         = len(sorted_addrs)

    fast_ops = []
    for kind, dest, arg1, arg2, aux in ops:
        di = addr_to_idx[dest]
        if kind == 0:
            fast_ops.append((0, di, arg1, 0, None))
        elif kind == 1:
            fast_ops.append((1, di, addr_to_idx[arg1], addr_to_idx[arg2], aux))
        elif kind == 2:
            fast_ops.append((2, di, addr_to_idx[arg1], addr_to_idx[arg2],
                             addr_to_idx[aux]))
        elif kind == 3:
            fast_ops.append((3, di, addr_to_idx[arg1], 0, aux))
        elif kind == 4:
            fast_ops.append((4, di, addr_to_idx[arg1], addr_to_idx[arg2], aux))

    def simulate_fn(inputs: List[int]) -> List[int]:
        mem = [0] * n_mem
        for i, val in zip(in_idx, inputs):
            mem[i] = val
        for kind, dest, arg1, arg2, aux in fast_ops:
            if kind == 0:
                mem[dest] = arg1
            elif kind == 1:
                mem[dest] = 1 if aux(mem[arg1], mem[arg2]) else 0
            elif kind == 2:
                mem[dest] = mem[arg2] if mem[arg1] else mem[aux]
            elif kind == 3:
                mem[dest] = _to_signed_8bit(aux(mem[arg1]))
            elif kind == 4:
                mem[dest] = _to_signed_8bit(aux(mem[arg1], mem[arg2]))
        return [mem[i] for i in out_idx]

    return simulate_fn, cost, len(input_addrs)


def _simulate(ir: str, inputs: List[int]) -> Tuple[List[int], int]:
    simulate_fn, cost, expected = _compile_ir(ir)
    if len(inputs) != expected:
        raise ValueError(f"IR declares {expected} inputs; {len(inputs)} provided")
    return simulate_fn(inputs), cost


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def score(ir: str, n_bits: int = N_BITS_SIX) -> int:
    """Score an IR that classifies n_bits-wide palindromes.

    The IR must accept exactly n_bits inputs (0 or 1) and produce exactly one
    output (0 or 1).  It is tested against all 2**n_bits possible inputs.
    Returns the static IR cost (same for every input).
    Raises ValueError on any correctness failure.
    """
    simulate_fn, static_cost, n_inputs = _compile_ir(ir)
    if n_inputs != n_bits:
        raise ValueError(
            f"IR declares {n_inputs} inputs; expected {n_bits} for {n_bits}-bit palindrome"
        )

    for code in range(1 << n_bits):
        pattern = [(code >> i) & 1 for i in range(n_bits)]
        expected = [1 if is_palindrome(pattern) else 0]
        actual = simulate_fn(pattern)
        if len(actual) != 1:
            raise ValueError(
                f"IR must produce exactly 1 output; got {len(actual)} outputs"
            )
        if actual != expected:
            raise ValueError(
                f"incorrect output for pattern {pattern}: "
                f"expected {expected[0]}, got {actual[0]}"
            )

    return static_cost


def score_six(ir: str) -> int:
    """Score against all 64 6-bit patterns (canonical RHW1986 task)."""
    return score(ir, n_bits=N_BITS_SIX)


def score_eight(ir: str) -> int:
    """Score against all 256 8-bit patterns (harder variant)."""
    return score(ir, n_bits=N_BITS_EIGHT)


# ---------------------------------------------------------------------------
# Baseline generator
# ---------------------------------------------------------------------------

def generate_baseline(n_bits: int = N_BITS_SIX) -> str:
    """Generate the CMP+AND palindrome classifier for n_bits inputs.

    Layout: inputs x[0]..x[n-1] at addresses 1..n.  Each palindrome pair
    (x[i], x[n-1-i]) is tested with ``cmp dest, a, b, eq``, writing the
    result back to addr i+1 (overwriting the input).  A chain of ``and``
    instructions then accumulates the result at address 1.

    Static cost for n=6: 20.  For n=8: 29.
    """
    n_pairs = n_bits // 2
    lines = [",".join(str(i) for i in range(1, n_bits + 1))]
    for i in range(n_pairs):
        j    = n_bits - 1 - i  # mirror position (0-based)
        dest = i + 1            # destination address
        a    = i + 1            # address of x[i]  (= dest, cmp reads before write)
        b    = j + 1            # address of x[j]
        lines.append(f"cmp {dest}, {a}, {b}, eq")
    for k in range(2, n_pairs + 1):
        lines.append(f"and 1, {k}")
    lines.append("1")
    return "\n".join(lines)


def generate_baseline_six() -> str:
    return generate_baseline(N_BITS_SIX)


def generate_baseline_eight() -> str:
    return generate_baseline(N_BITS_EIGHT)


__all__ = [
    "N_BITS_SIX", "N_BITS_EIGHT",
    "is_palindrome",
    "score", "score_six", "score_eight",
    "generate_baseline", "generate_baseline_six", "generate_baseline_eight",
]


if __name__ == "__main__":
    import os
    here    = os.path.dirname(os.path.abspath(__file__))
    ir_dir  = os.path.join(here, "submissions")
    os.makedirs(ir_dir, exist_ok=True)

    artifacts = [
        ("baseline_six.ir",   generate_baseline_six(),   score_six),
        ("baseline_eight.ir", generate_baseline_eight(), score_eight),
    ]

    for name, ir, scorer in artifacts:
        cost = scorer(ir)
        path = os.path.join(ir_dir, name)
        with open(path, "w") as f:
            f.write(ir + "\n")
        n_ops = len([ln for ln in ir.splitlines() if ln.strip()]) - 2
        print(f"  {name:<20}  cost={cost:>6,}  ops={n_ops:>3}  -> {path}")
