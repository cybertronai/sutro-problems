"""Energy-efficient sparse-parity scorer + baselines.

Scores IR programs that recover the unknown secret bit-subset from a
small number of labeled training rows and predict the labels of a
held-out test set, under the simplified Dally model using the v3
instruction set with all data values constrained to a signed 8-bit
ALU (``[-128, 127]``). The same scorer covers both the SMALL
(n=3 bits, k=2 secret, 4 train / 32 test) and MEDIUM (n=8 bits, k=3
secret, 8 train / 64 test) configurations defined below.
"""
from __future__ import annotations

import math
import operator
import secrets
from collections import namedtuple
from itertools import combinations
from random import Random
from typing import Callable, Iterable, List, Sequence, Tuple

# --------------------------------------------------------------------------
# Problem spec
# --------------------------------------------------------------------------

Spec = namedtuple("Spec", "n_bits k_secret m_train m_test")

SMALL = Spec(n_bits=3, k_secret=2, m_train=4, m_test=32)
MEDIUM = Spec(n_bits=8, k_secret=3, m_train=8, m_test=64)

N_BITS = SMALL.n_bits
K_SECRET = SMALL.k_secret
M_TRAIN = SMALL.m_train
M_TEST = SMALL.m_test


def _label(row: Sequence[int], subset: Sequence[int]) -> int:
    """XOR (parity) of the bits at positions in ``subset``."""
    p = 0
    for j in subset:
        p ^= row[j]
    return p


def _matches_all(X: Sequence[Sequence[int]], y: Sequence[int], subset: Sequence[int]) -> bool:
    """Fast iterative short-circuit for row parities."""
    for row, y_i in zip(X, y):
        if _label(row, subset) != y_i:
            return False
    return True


def _count_matches(X: Sequence[Sequence[int]], y: Sequence[int], combs: Iterable[Sequence[int]]) -> int:
    """Counts matching subsets, efficiently stopping early if > 1 match occurs."""
    matches = 0
    for subset in combs:
        if _matches_all(X, y, subset):
            matches += 1
            if matches > 1:
                break
    return matches


def _identifiable(X: Sequence[Sequence[int]], y: Sequence[int], k: int) -> bool:
    """True iff exactly one k-subset of columns explains every label in y."""
    return _count_matches(X, y, combinations(range(len(X[0])), k)) == 1


def generate(seed: int | None = None, *, spec: Spec = SMALL) -> Tuple[
    List[List[int]], List[int], List[List[int]], List[int], List[int]
]:
    # HARDENING: Fallback to SystemRandom (secrets) to prevent test-case predictability
    if seed is None:
        seed = secrets.randbits(256)

    rng = Random(seed)
    secret = sorted(rng.sample(range(spec.n_bits), spec.k_secret))
    all_combs = list(combinations(range(spec.n_bits), spec.k_secret))

    while True:
        X_train = [
            [rng.choice((0, 1)) for _ in range(spec.n_bits)]
            for _ in range(spec.m_train)
        ]
        y_train = [_label(row, secret) for row in X_train]

        # Validates that exactly one k-subset of columns explains every label
        if _count_matches(X_train, y_train, all_combs) == 1:
            break

    X_test = [
        [rng.choice((0, 1)) for _ in range(spec.n_bits)]
        for _ in range(spec.m_test)
    ]
    y_test = [_label(row, secret) for row in X_test]
    return X_train, y_train, X_test, y_test, secret


def solve_bruteforce(X: Sequence[Sequence[int]], y: Sequence[int], k: int = SMALL.k_secret) -> List[int]:
    for subset in combinations(range(len(X[0])), k):
        if _matches_all(X, y, subset):
            return list(subset)
    raise RuntimeError("no k-subset matches the training labels")


def predict(X: Sequence[Sequence[int]], subset: Sequence[int]) -> List[int]:
    return [_label(row, subset) for row in X]


def accuracy(y_pred: Sequence[int], y_true: Sequence[int]) -> float:
    return sum(a == b for a, b in zip(y_pred, y_true)) / len(y_true)


# ---------------------------------------------------------------------------
# Cost model & Security Helpers
# ---------------------------------------------------------------------------

def _cost(addr: int) -> int:
    if not isinstance(addr, int) or addr < 1:
        raise ValueError(f"addresses must be positive integers; got {addr!r}")
    return math.isqrt(addr - 1) + 1


def _to_signed_8bit(val: int) -> int:
    """Normalize integer to canonical signed 8-bit form [-128, 127]."""
    val &= 0xFF
    return val - 0x100 if val >= 0x80 else val


# ---------------------------------------------------------------------------
# Static Compiler + Fast Array Simulator
# ---------------------------------------------------------------------------

def _safe_div(a: int, b: int) -> int:
    if b == 0: raise ZeroDivisionError("integer division or modulo by zero")
    return a // b

_CMP_OPS = {
    "eq": operator.eq, "ne": operator.ne, "lt": operator.lt,
    "le": operator.le, "gt": operator.gt, "ge": operator.ge
}

_BINARY_OPS = {
    "add": operator.add, "sub": operator.sub, "mul": operator.mul,
    "div": _safe_div,    "and": operator.and_, "or": operator.or_, "xor": operator.xor
}

_UNARY_OPS = {
    "copy": lambda x: x, "not": operator.invert, "abs": abs
}


def _compile_ir(ir: str) -> Tuple[Callable[[List[int]], List[int]], int, int]:
    text = ir.replace(";", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Protect grader against DoS (Denial of Service) via bloated IR files
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
        input_addrs = parse_addrs(lines[0])
        output_addrs = parse_addrs(lines[-1])
    except ValueError as e:
        raise ValueError(f"malformed input/output line: {e}")

    if len(set(input_addrs)) != len(input_addrs):
        raise ValueError("input addresses must be distinct")

    init = set(input_addrs)
    cost = 0
    ops = []

    # 1. Validation, Translation, and Static Cost Determination
    for ln in lines[1:-1]:
        head, _, rest = ln.partition(" ")
        raw = [x.strip() for x in rest.split(",") if x.strip()] if rest else []

        try:
            if head == "set":
                if len(raw) != 2: raise ValueError("needs 2 operands")
                dest, literal = int(raw[0]), int(raw[1])
                if not (-128 <= literal <= 255): raise ValueError("literal out of bounds")
                reads = []
                ops.append((0, dest, _to_signed_8bit(literal), 0, None))

            elif head == "cmp":
                if len(raw) != 4: raise ValueError("needs 4 operands")
                dest, a, b = int(raw[0]), int(raw[1]), int(raw[2])
                pred = raw[3]
                if pred not in _CMP_OPS: raise ValueError("invalid predicate")
                reads = [a, b]
                ops.append((1, dest, a, b, _CMP_OPS[pred]))

            elif head == "select":
                if len(raw) != 4: raise ValueError("needs 4 operands")
                dest, c, t, f = int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])
                reads = [c, t, f]
                ops.append((2, dest, c, t, f))

            elif head in _UNARY_OPS:
                if len(raw) != 2: raise ValueError("needs 2 operands")
                dest, a = int(raw[0]), int(raw[1])
                reads = [a]
                ops.append((3, dest, a, 0, _UNARY_OPS[head]))

            elif head in _BINARY_OPS:
                if len(raw) not in (2, 3): raise ValueError("needs 2 or 3 operands")
                dest = int(raw[0])
                s1, s2 = (int(raw[1]), int(raw[2])) if len(raw) == 3 else (dest, int(raw[1]))
                reads = [s1, s2]
                ops.append((4, dest, s1, s2, _BINARY_OPS[head]))

            else:
                raise ValueError(f"unknown op: {head!r}")

            # General validation
            if dest < 1 or dest.bit_length() > 64:
                raise ValueError(f"invalid dest address {dest}")

            for src in reads:
                if src not in init: raise ValueError(f"uninitialized read: {src}")
                cost += _cost(src)

            init.add(dest)

        except ValueError as e:
            raise ValueError(f"malformed instruction '{ln}': {e}")

    for a in output_addrs:
        if a not in init:
            raise ValueError(f"output addr {a} never written")
        cost += _cost(a)

    # 2. Build dense execution map and dynamically packed instruction list
    sorted_addrs = sorted(list(init))
    addr_to_idx = {a: i for i, a in enumerate(sorted_addrs)}

    in_idx = [addr_to_idx[a] for a in input_addrs]
    out_idx = [addr_to_idx[a] for a in output_addrs]
    n_mem = len(sorted_addrs)

    fast_ops = []
    for kind, dest, arg1, arg2, aux in ops:
        if kind == 0:    fast_ops.append((0, addr_to_idx[dest], arg1, 0, None))
        elif kind == 1:  fast_ops.append((1, addr_to_idx[dest], addr_to_idx[arg1], addr_to_idx[arg2], aux))
        elif kind == 2:  fast_ops.append((2, addr_to_idx[dest], addr_to_idx[arg1], addr_to_idx[arg2], addr_to_idx[aux]))
        elif kind == 3:  fast_ops.append((3, addr_to_idx[dest], addr_to_idx[arg1], 0, aux))
        elif kind == 4:  fast_ops.append((4, addr_to_idx[dest], addr_to_idx[arg1], addr_to_idx[arg2], aux))

    def simulate_fn(inputs: List[int]) -> List[int]:
        mem = [0] * n_mem
        for i, val in zip(in_idx, inputs):
            mem[i] = val

        for kind, dest, arg1, arg2, aux in fast_ops:
            if kind == 0:     # set
                mem[dest] = arg1
            elif kind == 1:   # cmp
                mem[dest] = 1 if aux(mem[arg1], mem[arg2]) else 0
            elif kind == 2:   # select
                mem[dest] = mem[arg2] if mem[arg1] else mem[aux]
            elif kind == 3:   # unary
                mem[dest] = _to_signed_8bit(aux(mem[arg1]))
            elif kind == 4:   # binary
                mem[dest] = _to_signed_8bit(aux(mem[arg1], mem[arg2]))

        return [mem[i] for i in out_idx]

    return simulate_fn, cost, len(input_addrs)


def _simulate(ir: str, inputs: List[int]) -> Tuple[List[int], int]:
    simulate_fn, cost, expected_inputs = _compile_ir(ir)
    if len(inputs) != expected_inputs:
        raise ValueError(f"IR declares {expected_inputs} inputs; {len(inputs)} provided")
    return simulate_fn(inputs), cost


# ---------------------------------------------------------------------------
# Test instances + scorer
# ---------------------------------------------------------------------------

def _n_inputs(spec: Spec) -> int:
    return spec.n_bits * spec.m_train + spec.m_train + spec.n_bits * spec.m_test


def _instance(seed: int, spec: Spec) -> Tuple[List[int], List[int]]:
    X_tr, y_tr, X_te, y_te, _ = generate(seed=seed, spec=spec)
    inputs = (
        [v for row in X_tr for v in row]
        + list(y_tr)
        + [v for row in X_te for v in row]
    )
    return inputs, list(y_te)


def _canonical_seeds(spec: Spec, max_seeds: int, rng: Random | None = None) -> Tuple[int, ...]:
    if rng is None:
        # HARDENING: Enforce SystemRandom to prevent CI global seed predictability
        rng = secrets.SystemRandom()

    n_secrets = math.comb(spec.n_bits, spec.k_secret)
    target_secrets = min(max_seeds, n_secrets)

    seeds = []
    seen_secrets = set()
    used_seeds = set()

    for _ in range(500 * max(max_seeds, n_secrets)):
        if len(seeds) >= max_seeds and len(seen_secrets) >= target_secrets:
            break

        # HARDENING: Expand PRNG range to 256 bits.
        seed = rng.randrange(1 << 256)
        if seed in used_seeds:
            continue

        rng_peek = Random(seed)
        secret = tuple(sorted(rng_peek.sample(range(spec.n_bits), spec.k_secret)))

        # Ensure we don't accidentally fill up our max_seeds before discovering unique secrets
        if max_seeds - len(seeds) <= target_secrets - len(seen_secrets) and secret in seen_secrets:
            continue

        seeds.append(seed)
        used_seeds.add(seed)
        seen_secrets.add(secret)
    else:
        raise RuntimeError(f"could not draw {max_seeds} instances covering {target_secrets} distinct secrets")

    return tuple(seeds)


def _score(ir: str, spec: Spec, max_seeds: int, threshold: float = 1.0) -> int:
    simulate_fn, static_cost, expected_inputs = _compile_ir(ir)
    if expected_inputs != _n_inputs(spec):
        raise ValueError(f"IR declares {expected_inputs} inputs; {_n_inputs(spec)} provided")

    seeds = _canonical_seeds(spec, max_seeds=max_seeds)
    for i, seed in enumerate(seeds):
        inputs, expected = _instance(seed, spec)

        try:
            actual = simulate_fn(inputs)
        except Exception:
            # HARDENING: Mask exceptions so attackers can't intentionally fail logic
            raise ValueError("execution failed on a hidden test instance.") from None

        if len(actual) != len(expected):
            raise ValueError(
                f"output count mismatch on hidden test instance {i+1}/{max_seeds}."
            )

        if threshold >= 1.0:
            if actual != expected:
                # HARDENING: Stop the Oracle Leak. Do not echo arrays to the console.
                raise ValueError(
                    f"correctness failed on hidden test instance {i+1}/{max_seeds}. "
                    "(Test arrays are hidden securely to prevent hardcoding)"
                )
        else:
            n_correct = sum(1 for a, e in zip(actual, expected) if a == e)
            frac = n_correct / len(expected)
            if frac < threshold:
                raise ValueError(
                    f"correctness below {threshold:.0%} on hidden test instance "
                    f"{i+1}/{max_seeds} (got {frac:.0%}). "
                    "(Test arrays are hidden securely to prevent hardcoding)"
                )

    return static_cost


def score_small(ir: str) -> int:
    return _score(ir, SMALL, max_seeds=64)


def score_medium(ir: str) -> int:
    return _score(ir, MEDIUM, max_seeds=8)


def score_medium_approx50(ir: str) -> int:
    """Medium-instance scorer that requires only ≥ 50 % per-seed accuracy."""
    return _score(ir, MEDIUM, max_seeds=8, threshold=0.5)


# ---------------------------------------------------------------------------
# Baseline generator
# ---------------------------------------------------------------------------

def _generate_baseline(spec: Spec) -> str:
    candidates = list(combinations(range(spec.n_bits), spec.k_secret))
    n_cands = len(candidates)

    pred_base    = 1
    X_tr_base    = pred_base + spec.m_test
    y_tr_base    = X_tr_base + spec.n_bits * spec.m_train
    X_te_base    = y_tr_base + spec.m_train
    ONE          = X_te_base + spec.n_bits * spec.m_test
    TMP          = ONE + 1
    PARITY       = TMP + 1
    matched_base = PARITY + 1
    ind_T_base   = matched_base + n_cands * spec.m_train
    PREDT        = ind_T_base + n_cands
    term_base    = PREDT + 1

    def pred_at(j): return pred_base + j
    def X_tr_at(i, c): return X_tr_base + i * spec.n_bits + c
    def y_tr_at(i): return y_tr_base + i
    def X_te_at(j, c): return X_te_base + j * spec.n_bits + c
    def matched_at(T_idx, i): return matched_base + T_idx * spec.m_train + i
    def ind_T_at(T_idx): return ind_T_base + T_idx
    def term_at(T_idx): return term_base + T_idx

    inputs = (
        [X_tr_at(i, c) for i in range(spec.m_train) for c in range(spec.n_bits)]
        + [y_tr_at(i) for i in range(spec.m_train)]
        + [X_te_at(j, c) for j in range(spec.m_test) for c in range(spec.n_bits)]
    )
    outputs = [pred_at(j) for j in range(spec.m_test)]

    lines = [",".join(map(str, inputs))]

    def emit(op: str, *args: int) -> None:
        lines.append(f"{op} " + ",".join(map(str, args)))

    emit("set", ONE, 1)

    # --- decoding: ind_T per candidate ----------------------------------
    for T_idx, T in enumerate(candidates):
        for i in range(spec.m_train):
            emit("xor", TMP, y_tr_at(i), X_tr_at(i, T[0]))
            for k in range(1, spec.k_secret - 1):
                emit("xor", TMP, X_tr_at(i, T[k]))
            emit("xor", PARITY, TMP, X_tr_at(i, T[-1]))
            emit("xor", matched_at(T_idx, i), PARITY, ONE)

        emit("and", ind_T_at(T_idx), matched_at(T_idx, 0), matched_at(T_idx, 1))
        for i in range(2, spec.m_train):
            emit("and", ind_T_at(T_idx), matched_at(T_idx, i))

    # --- predictions: pred[j] = OR_T (ind_T AND predT) ------------------
    for j in range(spec.m_test):
        for T_idx, T in enumerate(candidates):
            emit("xor", PREDT, X_te_at(j, T[0]), X_te_at(j, T[1]))
            for k in range(2, spec.k_secret):
                emit("xor", PREDT, X_te_at(j, T[k]))
            emit("and", term_at(T_idx), ind_T_at(T_idx), PREDT)

        emit("or", pred_at(j), term_at(0), term_at(1))
        for T_idx in range(2, n_cands):
            emit("or", pred_at(j), term_at(T_idx))

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def generate_baseline_small() -> str:
    return _generate_baseline(SMALL)


def generate_baseline_medium() -> str:
    return _generate_baseline(MEDIUM)


__all__ = [
    "Spec", "SMALL", "MEDIUM",
    "N_BITS", "K_SECRET", "M_TRAIN", "M_TEST",
    "generate", "solve_bruteforce", "predict", "accuracy",
    "score_small", "generate_baseline_small",
    "score_medium", "generate_baseline_medium",
    "score_medium_approx50",
]


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    ir_dir = os.path.join(here, "submissions")
    os.makedirs(ir_dir, exist_ok=True)

    artifacts = [
        ("baseline_small.ir",  generate_baseline_small(),  score_small),
        ("baseline_medium.ir", generate_baseline_medium(), score_medium),
    ]

    for name, ir, scorer in artifacts:
        cost = scorer(ir)
        path = os.path.join(ir_dir, name)
        with open(path, "w") as f:
            f.write(ir + "\n")
        n_ops = len(ir.splitlines()) - 2
        print(f"  {name:<20} cost={cost:>9,}  ops={n_ops:>6,}  -> {path}")
