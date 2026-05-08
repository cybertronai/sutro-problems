"""Energy-efficient sparse-parity scorer + baselines.

Scores IR programs that recover the secret 2-subset of bits from m=5
training rows over ``{0, 1}^3`` and predict 5 test labels under the
[simplified Dally model](https://github.com/cybertronai/simplified-dally-model)
using the
[v2 instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets/v2)
(``add``, ``sub``, ``mul``, ``copy``, ``and``, ``or``, ``xor``, ``not``,
``set``).

**Cost model (v2).** Processor at the origin, memory laid out as a
2D upper half-plane indexed by **positive integers**; the cell at
linear index ``addr`` sits at Manhattan distance ``⌈√addr⌉`` from the
core. Each operand read pays that distance; writes and arithmetic are
free; inputs are placed for free at caller-specified addresses; every
output address pays one standard read at exit. ``set`` writes an
integer immediate without reading anything, so it is free.

Three-address-code IR (one instruction per line; ``;`` is also a
line separator so that single-line strings work):

    1,2                   ← input placement: a@1, b@2
    xor 3,1,2             ← mem[3] = mem[1] ^ mem[2]; reads ⌈√1⌉ + ⌈√2⌉
    3                     ← exit: read mem[3]; cost ⌈√3⌉

Supported ops (all from v2):

* ``add  dest, src1, src2``  — ``mem[dest] = mem[src1] + mem[src2]``
* ``sub  dest, src1, src2``  — ``mem[dest] = mem[src1] - mem[src2]``
* ``mul  dest, src1, src2``  — ``mem[dest] = mem[src1] * mem[src2]``
* ``and  dest, src1, src2``  — ``mem[dest] = mem[src1] & mem[src2]``
* ``or   dest, src1, src2``  — ``mem[dest] = mem[src1] | mem[src2]``
* ``xor  dest, src1, src2``  — ``mem[dest] = mem[src1] ^ mem[src2]``
* ``copy dest, src``         — ``mem[dest] = mem[src]``  (1 read)
* ``not  dest, src``         — ``mem[dest] = ~mem[src]`` (1 read)
* ``set  dest, K``           — ``mem[dest] = K`` (free; K is a literal)

Two-operand short form for the binary ops: ``xor dest, src`` is wire
sugar for ``xor dest, dest, src`` (in-place). Addresses must be
positive integers; ``addr ≤ 0`` raises. ``set``'s second operand is
an integer literal, not an address.
"""
from __future__ import annotations

import math
from itertools import combinations
from random import Random
from typing import Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------
# Problem spec
# --------------------------------------------------------------------------

N_BITS = 3
K_SECRET = 2
M_TRAIN = 5
M_TEST = 5


def _label(row: Sequence[int], subset: Sequence[int]) -> int:
    """XOR (parity) of the bits at positions in ``subset``."""
    p = 0
    for j in subset:
        p ^= row[j]
    return p


def _identifiable(X: Sequence[Sequence[int]], y: Sequence[int], k: int) -> bool:
    """True iff exactly one k-subset of columns explains every label in y."""
    n = len(X[0])
    matches = 0
    for subset in combinations(range(n), k):
        if all(_label(row, subset) == y_i for row, y_i in zip(X, y)):
            matches += 1
            if matches > 1:
                return False
    return matches == 1


def generate(seed: int = 0) -> Tuple[
    List[List[int]], List[int], List[List[int]], List[int], List[int]
]:
    """Return ``(X_train, y_train, X_test, y_test, secret)``.

    The training rows are resampled until the secret is the unique
    weight-k subset matching y_train. The expected number of resamples
    is ~1 (E[false subsets] = 2 · 2^-5 = 0.0625 for n=3, k=2, m=5).
    """
    rng = Random(seed)
    secret = sorted(rng.sample(range(N_BITS), K_SECRET))
    while True:
        X_train = [
            [rng.choice((0, 1)) for _ in range(N_BITS)]
            for _ in range(M_TRAIN)
        ]
        y_train = [_label(row, secret) for row in X_train]
        if _identifiable(X_train, y_train, K_SECRET):
            break
    X_test = [
        [rng.choice((0, 1)) for _ in range(N_BITS)]
        for _ in range(M_TEST)
    ]
    y_test = [_label(row, secret) for row in X_test]
    return X_train, y_train, X_test, y_test, secret


def solve_bruteforce(
    X: Sequence[Sequence[int]],
    y: Sequence[int],
    k: int = K_SECRET,
) -> List[int]:
    """Return the unique k-subset of columns matching every label in y."""
    n = len(X[0])
    for subset in combinations(range(n), k):
        if all(_label(row, subset) == y_i for row, y_i in zip(X, y)):
            return list(subset)
    raise RuntimeError("no k-subset matches the training labels")


def predict(X: Sequence[Sequence[int]], subset: Sequence[int]) -> List[int]:
    return [_label(row, subset) for row in X]


def accuracy(y_pred: Sequence[int], y_true: Sequence[int]) -> float:
    return sum(a == b for a, b in zip(y_pred, y_true)) / len(y_true)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def _cost(addr: int) -> int:
    """``⌈√addr⌉`` for a positive integer ``addr``; raises otherwise."""
    if not isinstance(addr, int) or addr < 1:
        raise ValueError(
            f"addresses must be positive integers; got {addr!r}")
    return math.isqrt(addr - 1) + 1


def _check_addrs(addrs, where):
    for a in addrs:
        if not isinstance(a, int) or a < 1:
            raise ValueError(
                f"{where}: addresses must be positive integers; got {a!r}")


# ---------------------------------------------------------------------------
# Parser + simulator
# ---------------------------------------------------------------------------

_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "and": lambda a, b: a & b,
    "or":  lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
}
_UNARY = {
    "copy": lambda a: a,
    "not":  lambda a: ~a,
}


def _parse(ir: str):
    text = ir.replace(";", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("IR needs at least an input line and an output line")
    input_addrs = [int(x) for x in lines[0].split(",")]
    output_addrs = [int(x) for x in lines[-1].split(",")]
    _check_addrs(input_addrs,  "input line")
    _check_addrs(output_addrs, "output line")
    ops = []
    for ln in lines[1:-1]:
        head, _, rest = ln.partition(" ")
        if not rest:
            raise ValueError(f"malformed instruction: {ln!r}")
        operands = [int(x) for x in rest.split(",")]
        if head == "set":
            if len(operands) != 2:
                raise ValueError(f"set needs 2 operands (dest, K); got {operands}")
            _check_addrs(operands[:1], "`set` dest")
            # operands[1] is a literal — any integer is allowed.
        else:
            _check_addrs(operands, f"`{head}` operands")
        ops.append((head, operands))
    return input_addrs, ops, output_addrs


def _simulate(ir: str, inputs: List[int]) -> Tuple[List[int], int]:
    input_addrs, ops, output_addrs = _parse(ir)
    if len(input_addrs) != len(inputs):
        raise ValueError(
            f"IR declares {len(input_addrs)} inputs; {len(inputs)} provided")
    if len(set(input_addrs)) != len(input_addrs):
        raise ValueError("input addresses must be distinct")
    mem: Dict[int, int] = {a: v for a, v in zip(input_addrs, inputs)}
    cost = 0
    for op, oprs in ops:
        if op == "set":
            dest, literal = oprs
            mem[dest] = literal  # no read, no cost
            continue
        if op in _UNARY:
            if len(oprs) != 2:
                raise ValueError(f"{op} needs 2 operands; got {oprs}")
            dest, src = oprs
            if src not in mem:
                raise ValueError(
                    f"{op} {dest},{src} reads uninitialized addr {src}")
            cost += _cost(src)
            mem[dest] = _UNARY[op](mem[src])
            continue
        if op not in _BINARY:
            raise ValueError(
                f"unknown op: {op!r}  (v2 supports "
                f"add/sub/mul/copy/and/or/xor/not/set)")
        if len(oprs) == 3:
            dest, s1, s2 = oprs
        elif len(oprs) == 2:
            dest, s2 = oprs
            s1 = dest
        else:
            raise ValueError(f"{op} needs 2 or 3 operands; got {oprs}")
        for src in (s1, s2):
            if src not in mem:
                raise ValueError(
                    f"{op} {','.join(map(str,oprs))} reads "
                    f"uninitialized addr {src}")
        cost += _cost(s1) + _cost(s2)
        mem[dest] = _BINARY[op](mem[s1], mem[s2])
    outputs = []
    for a in output_addrs:
        if a not in mem:
            raise ValueError(f"output addr {a} never written")
        cost += _cost(a)
        outputs.append(mem[a])
    return outputs, cost


# ---------------------------------------------------------------------------
# Test instance + scorer
# ---------------------------------------------------------------------------

# Inputs are passed in this order (matches the IR address declaration
# order in ``generate_baseline``):
#   X_train (row-major, 15 values)
#   y_train               (5 values)
#   X_test  (row-major, 15 values)
_N_INPUTS = N_BITS * M_TRAIN + M_TRAIN + N_BITS * M_TEST  # = 35


def _sparse_parity_test() -> Tuple[List[int], List[int]]:
    """Deterministic seed=0 instance.

    Returns ``(inputs, expected)`` where ``inputs`` is the 35-value flat
    list ``[X_train..., y_train..., X_test...]`` and ``expected`` is the
    5-element ``y_test`` the IR's outputs must match.
    """
    X_tr, y_tr, X_te, y_te, _ = generate(seed=0)
    inputs = (
        [v for row in X_tr for v in row]
        + list(y_tr)
        + [v for row in X_te for v in row]
    )
    expected = list(y_te)
    return inputs, expected


def score_sparse_parity(ir: str) -> int:
    """Verify the IR predicts y_test on the seed=0 instance and return
    its total Dally read-cost."""
    inputs, expected = _sparse_parity_test()
    actual, cost = _simulate(ir, inputs)
    if actual != expected:
        raise ValueError(
            f"correctness failed:\n  got      {actual}\n"
            f"  expected {expected}")
    return cost


# ---------------------------------------------------------------------------
# Baseline generator — one xor per test row
# ---------------------------------------------------------------------------

def generate_baseline() -> str:
    """Naive predictor: discover the secret subset in Python (free,
    just like matmul's algorithm choice is free) and emit one ``xor``
    per test row reading the two secret columns of ``X_test``.

    Layout (worst case — bulk arrays placed contiguously, output after):
      X_train at addrs 1..15   (row-major)
      y_train at addrs 16..20
      X_test  at addrs 21..35  (row-major)
      pred    at addrs 36..40  (output)
    """
    X_tr, y_tr, _, _, _ = generate(seed=0)
    s0, s1 = solve_bruteforce(X_tr, y_tr)  # k=2

    X_test_at = lambda i, j: 1 + N_BITS * M_TRAIN + M_TRAIN + i * N_BITS + j
    pred_at   = lambda i: 1 + _N_INPUTS + i

    inputs  = list(range(1, _N_INPUTS + 1))
    outputs = [pred_at(i) for i in range(M_TEST)]

    lines = [",".join(map(str, inputs))]
    for i in range(M_TEST):
        lines.append(
            f"xor {pred_at(i)},{X_test_at(i, s0)},{X_test_at(i, s1)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


__all__ = [
    "N_BITS", "K_SECRET", "M_TRAIN", "M_TEST",
    "generate", "solve_bruteforce", "predict", "accuracy",
    "score_sparse_parity", "generate_baseline",
]


# ---------------------------------------------------------------------------
# Reproducer for the record-history IR file (``python sparse_parity.py``).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    ir_dir = os.path.join(here, "submissions")
    os.makedirs(ir_dir, exist_ok=True)
    artifacts = [
        ("baseline.ir", generate_baseline(), score_sparse_parity),
    ]
    for name, ir, scorer in artifacts:
        cost = scorer(ir)
        path = os.path.join(ir_dir, name)
        with open(path, "w") as f:
            f.write(ir)
            f.write("\n")
        n_ops = len(ir.splitlines()) - 2
        print(f"  {name:<14} cost={cost:>5,}  ops={n_ops:>3,}  -> {path}")
