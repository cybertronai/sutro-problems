"""Sparse parity: recover k=2 secret bits among n=3 from m=5 training rows.

Each row of X is in {0, 1}^3 (2 secret bits + 1 noise bit). The label is
the XOR (parity) of the k secret bits: ``y[i] = XOR(X[i, j] for j in S)``.
The training set is chosen (by rejection sampling) so that the secret
subset is the *unique* 2-subset of columns consistent with the training
labels — therefore the brute-force solver always recovers it, and the
5-row test set is classified at 100%.

This module also exposes a v1-IR scorer (``score_sparse_parity``) and a
naive baseline IR generator (``generate_baseline``), mirroring the
[simplified Dally model](https://github.com/cybertronai/simplified-dally-model)
+ [v1 instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets/v1)
(``add``, ``sub``, ``mul``, ``copy`` from v0, plus ``and``, ``or``, ``xor``,
``not``). v1 is the natural fit: parity is a single ``xor`` per test row.
Read-cost = ⌈√addr⌉ per operand; writes and arithmetic are free; inputs
are placed for free at caller-specified addresses; every output address
pays one standard read at exit.
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


# --------------------------------------------------------------------------
# v1 IR cost model — same simplified Dally model as ../matmul, with the
# v1 instruction set: v0's ``add``, ``sub``, ``mul``, ``copy`` plus
# bitwise ``and``, ``or``, ``xor`` (binary) and ``not`` (unary).
# --------------------------------------------------------------------------

def _cost(addr: int) -> int:
    if not isinstance(addr, int) or addr < 1:
        raise ValueError(
            f"addresses must be positive integers; got {addr!r}")
    return math.isqrt(addr - 1) + 1


def _check_addrs(addrs, where):
    for a in addrs:
        if not isinstance(a, int) or a < 1:
            raise ValueError(
                f"{where}: addresses must be positive integers; got {a!r}")


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
    _check_addrs(input_addrs, "input line")
    _check_addrs(output_addrs, "output line")
    ops = []
    for ln in lines[1:-1]:
        head, _, rest = ln.partition(" ")
        if not rest:
            raise ValueError(f"malformed instruction: {ln!r}")
        operands = [int(x) for x in rest.split(",")]
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
                f"unknown op: {op!r}  (v1 supports "
                f"add/sub/mul/copy/and/or/xor/not)")
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


# --------------------------------------------------------------------------
# Scoring + baseline IR
# --------------------------------------------------------------------------

# Inputs are passed in this order (matches the IR address declaration
# order in ``generate_baseline``):
#   X_train (row-major, 15 values)
#   y_train               (5 values)
#   X_test  (row-major, 15 values)
_N_INPUTS = N_BITS * M_TRAIN + M_TRAIN + N_BITS * M_TEST  # = 35


def score_sparse_parity(ir: str) -> int:
    """Run *ir* on the canonical (seed=0) instance, verify it predicts
    y_test exactly, and return the total Dally read-cost."""
    X_tr, y_tr, X_te, y_te, _ = generate(seed=0)
    inputs = (
        [v for row in X_tr for v in row]
        + list(y_tr)
        + [v for row in X_te for v in row]
    )
    actual, cost = _simulate(ir, inputs)
    if actual != list(y_te):
        raise ValueError(
            f"prediction mismatch:\n  got      {actual}\n"
            f"  expected {list(y_te)}")
    return cost


def generate_baseline() -> str:
    """Naive predictor IR for the seed=0 instance.

    The Python ``solve_bruteforce`` discovers ``S`` (free, just like
    matmul's algorithm choice is free); the IR then emits one ``xor``
    per test row reading the two secret columns of ``X_test``.

    Layout:
      X_train at addrs 1..15   (row-major)
      y_train at addrs 16..20
      X_test  at addrs 21..35  (row-major)
      pred    at addrs 36..40  (output)
    """
    X_tr, y_tr, _, _, _ = generate(seed=0)
    secret = solve_bruteforce(X_tr, y_tr)
    s0, s1 = secret  # k=2

    X_test_at = lambda i, j: 1 + N_BITS * M_TRAIN + M_TRAIN + i * N_BITS + j
    pred_at = lambda i: 1 + _N_INPUTS + i

    inputs = list(range(1, _N_INPUTS + 1))
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


# --------------------------------------------------------------------------
# Reproducer for the record-history IR file (``python sparse_parity.py``).
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    X_tr, y_tr, X_te, y_te, secret = generate(seed=0)
    found = solve_bruteforce(X_tr, y_tr)
    acc = accuracy(predict(X_te, found), y_te)
    print(f"secret    = {secret}")
    print(f"recovered = {found}")
    print(f"test acc  = {acc:.0%}")

    ir = generate_baseline()
    cost = score_sparse_parity(ir)
    here = os.path.dirname(os.path.abspath(__file__))
    sub_dir = os.path.join(here, "submissions")
    os.makedirs(sub_dir, exist_ok=True)
    path = os.path.join(sub_dir, "baseline.ir")
    with open(path, "w") as f:
        f.write(ir)
        f.write("\n")
    n_ops = len(ir.splitlines()) - 2
    print(f"baseline  cost={cost:>4}  ops={n_ops}  -> {path}")
