"""Energy-efficient sparse-parity scorer + baselines.

Scores IR programs that recover the secret 2-subset of bits from m=4
training rows over ``{0, 1}^3`` and predict 32 test labels under the
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
M_TRAIN = 4
M_TEST = 32  # 4 × 8


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
    is ~1 (E[false subsets] = 2 · 2^-4 = 0.125 for n=3, k=2, m=4).
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
# Test instances + scorer
# ---------------------------------------------------------------------------

# Inputs are passed in this order (matches the IR address declaration
# order in ``generate_baseline``):
#   X_train (row-major, 15 values)
#   y_train               (5 values)
#   X_test  (row-major, 15 values)
_N_INPUTS = N_BITS * M_TRAIN + M_TRAIN + N_BITS * M_TEST  # = 35


def _instance(seed: int) -> Tuple[List[int], List[int]]:
    """Return ``(inputs, expected)`` for one seed.

    ``inputs`` is the 35-value flat list ``[X_train..., y_train..., X_test...]``
    that the IR receives; ``expected`` is the 5-element ``y_test`` the IR's
    outputs must match.
    """
    X_tr, y_tr, X_te, y_te, _ = generate(seed=seed)
    inputs = (
        [v for row in X_tr for v in row]
        + list(y_tr)
        + [v for row in X_te for v in row]
    )
    return inputs, list(y_te)


def _seeds_covering_all_secrets() -> Tuple[int, ...]:
    """Smallest set of seeds (scanning seed=0,1,2,...) that exercises
    every possible secret 2-subset of {0, 1, 2}. There are ``C(3,2)=3``
    such subsets, so this returns three seeds. Computed once at import."""
    seen: Dict[Tuple[int, ...], int] = {}
    for seed in range(200):
        _, _, _, _, secret = generate(seed=seed)
        key = tuple(secret)
        if key not in seen:
            seen[key] = seed
            if len(seen) == 3:
                break
    if len(seen) != 3:
        raise RuntimeError("could not cover all 3 secrets within 200 seeds")
    return tuple(sorted(seen.values()))


_CANONICAL_SEEDS: Tuple[int, ...] = _seeds_covering_all_secrets()


def score_sparse_parity(ir: str) -> int:
    """Verify the IR predicts y_test correctly on instances covering
    every possible secret subset, and return its Dally read-cost.

    The scorer walks ``_CANONICAL_SEEDS`` (one seed per distinct secret).
    Cost is determined by the IR alone — the same set of operand reads
    is performed regardless of input *values* — so the scorer also
    asserts the cost is identical across seeds (a cheap robustness
    check that the IR's control flow is data-independent).
    """
    seen_cost: int | None = None
    for seed in _CANONICAL_SEEDS:
        inputs, expected = _instance(seed)
        actual, cost = _simulate(ir, inputs)
        if actual != expected:
            raise ValueError(
                f"correctness failed (seed={seed}):\n  got      {actual}\n"
                f"  expected {expected}")
        if seen_cost is None:
            seen_cost = cost
        elif cost != seen_cost:
            raise ValueError(
                f"non-deterministic cost across seeds: "
                f"{seen_cost} (earlier) vs {cost} (seed={seed})")
    assert seen_cost is not None
    return seen_cost


# ---------------------------------------------------------------------------
# Baseline generator — try every candidate subset, AND-reduce match,
# OR-combine predictions
# ---------------------------------------------------------------------------

def generate_baseline() -> str:
    """General predictor IR — works for any seed.

    Mirrors the brute-force solver in pure v2 IR. For each of the
    ``C(3, 2) = 3`` candidate 2-subsets ``T = (t0, t1)``:

      * Compute ``matched_T_i = 1 XOR (y_train[i] XOR X_train[i,t0]
        XOR X_train[i,t1])`` — that's 1 iff T matches row i.
      * ``ind_T = AND_i matched_T_i`` — 1 iff T matches every row.

    By identifiability, exactly one ``ind_T`` is 1 (the true secret).
    Each test row is then predicted as
    ``OR_T (ind_T AND (X_test[j,t0] XOR X_test[j,t1]))`` — the OR
    selects the lone non-zero term.

    Memory layout (computed from M_TRAIN, M_TEST, N_BITS, K_SECRET):
      pred       starts at 1                       (output, M_TEST cells)
      X_train    next                              (M_TRAIN × N_BITS, row-major)
      y_train    next                              (M_TRAIN cells)
      X_test     next                              (M_TEST × N_BITS, row-major)
      ONE        next                              (constant 1 via ``set``, free)
      tmp        next                              (scratch, reused)
      parity     next                              (scratch, reused)
      matched_T  next                              (C(n,k) × M_TRAIN cells)
      ind_T      next                              (C(n,k) cells)
      predT      next                              (scratch, reused per test row)
      term_T     next                              (C(n,k) cells, reused per row)
    """
    candidates = list(combinations(range(N_BITS), K_SECRET))
    n_cands = len(candidates)

    pred_base   = 1
    X_tr_base   = pred_base + M_TEST
    y_tr_base   = X_tr_base + N_BITS * M_TRAIN
    X_te_base   = y_tr_base + M_TRAIN
    ONE         = X_te_base + N_BITS * M_TEST
    TMP         = ONE + 1
    PARITY      = TMP + 1
    matched_base = PARITY + 1
    ind_T_base  = matched_base + n_cands * M_TRAIN
    PREDT       = ind_T_base + n_cands
    term_base   = PREDT + 1

    pred_at    = lambda j: pred_base + j
    X_tr_at    = lambda i, c: X_tr_base + i * N_BITS + c
    y_tr_at    = lambda i: y_tr_base + i
    X_te_at    = lambda j, c: X_te_base + j * N_BITS + c
    matched_at = lambda T_idx, i: matched_base + T_idx * M_TRAIN + i
    ind_T_at   = lambda T_idx: ind_T_base + T_idx
    term_at    = lambda T_idx: term_base + T_idx

    inputs = (
        [X_tr_at(i, c) for i in range(M_TRAIN) for c in range(N_BITS)]
        + [y_tr_at(i) for i in range(M_TRAIN)]
        + [X_te_at(j, c) for j in range(M_TEST) for c in range(N_BITS)]
    )
    outputs = [pred_at(j) for j in range(M_TEST)]

    lines = [",".join(map(str, inputs))]
    lines.append(f"set {ONE},1")

    # --- decoding: ind_T per candidate ----------------------------------
    for T_idx, (t0, t1) in enumerate(candidates):
        for i in range(M_TRAIN):
            lines.append(f"xor {TMP},{y_tr_at(i)},{X_tr_at(i, t0)}")
            lines.append(f"xor {PARITY},{TMP},{X_tr_at(i, t1)}")
            lines.append(f"xor {matched_at(T_idx, i)},{PARITY},{ONE}")
        # ind_T = AND of all matched_T_i
        lines.append(
            f"and {ind_T_at(T_idx)},{matched_at(T_idx, 0)},{matched_at(T_idx, 1)}")
        for i in range(2, M_TRAIN):
            lines.append(f"and {ind_T_at(T_idx)},{matched_at(T_idx, i)}")

    # --- predictions: pred[j] = OR_T (ind_T AND predT) ------------------
    for j in range(M_TEST):
        for T_idx, (t0, t1) in enumerate(candidates):
            lines.append(f"xor {PREDT},{X_te_at(j, t0)},{X_te_at(j, t1)}")
            lines.append(f"and {term_at(T_idx)},{ind_T_at(T_idx)},{PREDT}")
        lines.append(f"or {pred_at(j)},{term_at(0)},{term_at(1)}")
        lines.append(f"or {pred_at(j)},{term_at(2)}")

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
