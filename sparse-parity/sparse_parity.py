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


def _check_addrs(addrs: Sequence[int], where: str) -> None:
    for a in addrs:
        if not isinstance(a, int) or a < 1:
            raise ValueError(f"{where}: addresses must be positive integers; got {a!r}")
        # HARDENING: Prevent massive layout offsets attempting OS memory exploits
        if a.bit_length() > 64:
            raise ValueError(f"{where}: address exceeds 64-bit bounds")


def _to_signed_8bit(val: int) -> int:
    """Normalize integer to canonical signed 8-bit form [-128, 127]."""
    val &= 0xFF
    return val - 0x100 if val >= 0x80 else val


# ---------------------------------------------------------------------------
# Static Compiler + Fast Array Simulator
# ---------------------------------------------------------------------------

_BINARY = {"add", "sub", "mul", "div", "and", "or", "xor"}
_UNARY = {"copy", "not", "abs"}
_CMP_PRED = {"eq", "ne", "lt", "le", "gt", "ge"}


def _parse(ir: str) -> Tuple[List[int], List[Tuple[str, List[int]]], List[int]]:
    text = ir.replace(";", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # HARDENING: Protect grader against DoS (Denial of Service) via bloated IR files
    if len(lines) > 100_000:
        raise ValueError("IR exceeds maximum allowed length (100,000 instructions)")
    if len(lines) < 2:
        raise ValueError("IR needs at least an input line and an output line")

    try:
        input_addrs = [int(x) for x in lines[0].split(",") if x.strip()]
        output_addrs = [int(x) for x in lines[-1].split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"malformed input/output line: {e}")

    _check_addrs(input_addrs, "input line")
    _check_addrs(output_addrs, "output line")

    if len(set(input_addrs)) != len(input_addrs):
        raise ValueError("input addresses must be distinct")

    ops = []
    for ln in lines[1:-1]:
        head, _, rest = ln.partition(" ")
        if not rest:
            raise ValueError(f"malformed instruction: {ln!r}")
        raw = [tok.strip() for tok in rest.split(",")]

        if head == "set":
            if len(raw) != 2: raise ValueError(f"set needs 2 operands (dest, K); got {raw}")
            literal = int(raw[1])
            # HARDENING: Limit immediate values to 8-bit byte range.
            if not (-128 <= literal <= 255):
                raise ValueError("set literal must fit in 8 bits (-128..255)")
            operands = [int(raw[0]), literal]
            _check_addrs(operands[:1], "`set` dest")

        elif head == "cmp":
            if len(raw) != 4: raise ValueError(f"cmp needs 4 operands (dest, a, b, pred); got {raw}")
            pred = raw[3]
            if pred not in _CMP_PRED: raise ValueError(f"cmp predicate must be one of {sorted(_CMP_PRED)}")
            operands = [int(raw[0]), int(raw[1]), int(raw[2]), pred]
            _check_addrs(operands[:3], "`cmp` operands")

        else:
            try: operands = [int(x) for x in raw]
            except ValueError: raise ValueError(f"malformed integer in `{head}` operands: {raw}")
            _check_addrs(operands, f"`{head}` operands")

            if head in _UNARY and len(operands) != 2:
                raise ValueError(f"{head} needs 2 operands; got {operands}")
            elif head in _BINARY and len(operands) not in (2, 3):
                raise ValueError(f"{head} needs 2 or 3 operands; got {operands}")
            elif head == "select" and len(operands) != 4:
                raise ValueError(f"select needs 4 operands (dest, c, t, f); got {operands}")
            elif head not in _UNARY | _BINARY | {"select"}:
                raise ValueError(f"unknown op: {head!r}")

        ops.append((head, operands))

    return input_addrs, ops, output_addrs


def _compile_ir(ir: str) -> Tuple[Callable[[List[int]], List[int]], int, int]:
    input_addrs, ops, output_addrs = _parse(ir)

    # 1. Validation and Static Cost Determination
    cost = 0
    init = set(input_addrs)

    for op, oprs in ops:
        dest = oprs[0]

        if op == "set":
            read_addrs = []
        elif op == "cmp":
            read_addrs = oprs[1:3]
        elif op == "select":
            read_addrs = oprs[1:4]
        elif op in _UNARY:
            read_addrs = [oprs[1]]
        elif op in _BINARY:
            read_addrs = oprs[1:3] if len(oprs) == 3 else [dest, oprs[1]]
        else:
            read_addrs = []

        for src in read_addrs:
            if src not in init:
                raise ValueError(f"{op} reads uninitialized addr {src}")
            cost += _cost(src)

        init.add(dest)

    for a in output_addrs:
        if a not in init:
            raise ValueError(f"output addr {a} never written")
        cost += _cost(a)

    # 2. Build dense execution map and packed instruction list
    sorted_addrs = sorted(list(init))
    addr_to_idx = {a: i for i, a in enumerate(sorted_addrs)}

    in_idx = [addr_to_idx[a] for a in input_addrs]
    out_idx = [addr_to_idx[a] for a in output_addrs]
    n_mem = len(sorted_addrs)

    fast_ops = []
    for op, oprs in ops:
        dest_idx = addr_to_idx[oprs[0]]
        if op == "set":
            # HARDENING: Immediate values securely clamped at compile-time
            fast_ops.append((0, dest_idx, _to_signed_8bit(oprs[1]), 0, 0))
        elif op == "cmp":
            fast_ops.append((1, dest_idx, addr_to_idx[oprs[1]], addr_to_idx[oprs[2]], oprs[3]))
        elif op == "select":
            fast_ops.append((2, dest_idx, addr_to_idx[oprs[1]], addr_to_idx[oprs[2]], addr_to_idx[oprs[3]]))
        elif op in _UNARY:
            fast_ops.append((3, dest_idx, addr_to_idx[oprs[1]], 0, op))
        elif op in _BINARY:
            s1_idx = addr_to_idx[oprs[1] if len(oprs) == 3 else oprs[0]]
            s2_idx = addr_to_idx[oprs[2] if len(oprs) == 3 else oprs[1]]
            fast_ops.append((4, dest_idx, s1_idx, s2_idx, op))

    def simulate_fn(inputs: List[int]) -> List[int]:
        mem = [0] * n_mem
        for i, val in zip(in_idx, inputs):
            mem[i] = val

        for kind, dest, arg1, arg2, aux in fast_ops:
            if kind == 0:
                mem[dest] = arg1
            elif kind == 1:
                a, b = mem[arg1], mem[arg2]
                if aux == "eq": res = a == b
                elif aux == "ne": res = a != b
                elif aux == "lt": res = a < b
                elif aux == "le": res = a <= b
                elif aux == "gt": res = a > b
                else: res = a >= b
                mem[dest] = 1 if res else 0
            elif kind == 2:
                mem[dest] = mem[arg2] if mem[arg1] else mem[aux]
            elif kind == 3:
                src = mem[arg1]
                if aux == "copy": res = src
                elif aux == "not": res = ~src
                else: res = abs(src)
                mem[dest] = _to_signed_8bit(res)
            elif kind == 4:
                s1, s2 = mem[arg1], mem[arg2]
                if aux == "add": res = s1 + s2
                elif aux == "sub": res = s1 - s2
                elif aux == "mul": res = s1 * s2
                elif aux == "div":
                    if s2 == 0: raise ZeroDivisionError("integer division or modulo by zero")
                    res = s1 // s2
                elif aux == "and": res = s1 & s2
                elif aux == "or": res = s1 | s2
                else: res = s1 ^ s2
                mem[dest] = _to_signed_8bit(res)

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

    max_draws = 500 * max(max_seeds, n_secrets)
    for _ in range(max_draws):
        # HARDENING: Decouple secret coverage check from total array size.
        if len(seeds) >= max_seeds and len(seen_secrets) >= target_secrets:
            break

        # HARDENING: Expand PRNG range to 256 bits.
        seed = rng.randrange(1 << 256)
        if seed in used_seeds:
            continue

        rng_peek = Random(seed)
        secret = tuple(sorted(rng_peek.sample(range(spec.n_bits), spec.k_secret)))

        # Ensure we don't accidentally fill up our max_seeds before discovering unique secrets
        slots_left = max_seeds - len(seeds)
        secrets_needed = target_secrets - len(seen_secrets)
        if slots_left <= secrets_needed and secret in seen_secrets:
            continue

        seeds.append(seed)
        used_seeds.add(seed)
        seen_secrets.add(secret)

    if len(seeds) < max_seeds or len(seen_secrets) < target_secrets:
        raise RuntimeError(
            f"could not draw {max_seeds} instances covering {target_secrets} distinct secrets"
        )
    return tuple(seeds)


def _score(ir: str, spec: Spec, max_seeds: int) -> int:
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
            # to scrape evaluation datasets off the tracebacks (Oracle exploit).
            raise ValueError("execution failed on a hidden test instance.") from None

        if actual != expected:
            # HARDENING: Stop the Oracle Leak. Do not echo arrays to the console.
            raise ValueError(
                f"correctness failed on hidden test instance {i+1}/{max_seeds}. "
                "(Test arrays are hidden securely to prevent hardcoding)"
            )

    return static_cost


def score_small(ir: str) -> int:
    return _score(ir, SMALL, max_seeds=64)


def score_medium(ir: str) -> int:
    return _score(ir, MEDIUM, max_seeds=8)


def _score_approx(ir: str, spec: Spec, max_seeds: int, threshold: float) -> int:
    """Approximate-correctness scorer — passes if every canonical seed
    achieves at least *threshold* fraction of correct outputs."""
    simulate_fn, static_cost, expected_inputs = _compile_ir(ir)
    if expected_inputs != _n_inputs(spec):
        raise ValueError(f"IR declares {expected_inputs} inputs; {_n_inputs(spec)} provided")

    seeds = _canonical_seeds(spec, max_seeds=max_seeds)
    for i, seed in enumerate(seeds):
        inputs, expected = _instance(seed, spec)
        try:
            actual = simulate_fn(inputs)
        except Exception:
            raise ValueError("execution failed on a hidden test instance.") from None
        if len(actual) != len(expected):
            raise ValueError(
                f"output count mismatch on hidden test instance {i+1}/{max_seeds}.")
        n_correct = sum(1 for a, e in zip(actual, expected) if a == e)
        frac = n_correct / len(expected)
        if frac < threshold:
            raise ValueError(
                f"correctness below {threshold:.0%} on hidden test instance "
                f"{i+1}/{max_seeds} (got {frac:.0%}). "
                "(Test arrays are hidden securely to prevent hardcoding)"
            )
    return static_cost


def score_medium_approx50(ir: str) -> int:
    """Medium-instance scorer that requires only ≥ 50 % per-seed accuracy."""
    return _score_approx(ir, MEDIUM, max_seeds=8, threshold=0.5)


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
    lines.append(f"set {ONE},1")

    # --- decoding: ind_T per candidate ----------------------------------
    for T_idx, T in enumerate(candidates):
        for i in range(spec.m_train):
            lines.append(f"xor {TMP},{y_tr_at(i)},{X_tr_at(i, T[0])}")
            for k in range(1, spec.k_secret - 1):
                lines.append(f"xor {TMP},{X_tr_at(i, T[k])}")
            lines.append(f"xor {PARITY},{TMP},{X_tr_at(i, T[spec.k_secret - 1])}")
            lines.append(f"xor {matched_at(T_idx, i)},{PARITY},{ONE}")

        lines.append(f"and {ind_T_at(T_idx)},{matched_at(T_idx, 0)},{matched_at(T_idx, 1)}")
        for i in range(2, spec.m_train):
            lines.append(f"and {ind_T_at(T_idx)},{matched_at(T_idx, i)}")

    # --- predictions: pred[j] = OR_T (ind_T AND predT) ------------------
    for j in range(spec.m_test):
        for T_idx, T in enumerate(candidates):
            lines.append(f"xor {PREDT},{X_te_at(j, T[0])},{X_te_at(j, T[1])}")
            for k in range(2, spec.k_secret):
                lines.append(f"xor {PREDT},{X_te_at(j, T[k])}")
            lines.append(f"and {term_at(T_idx)},{ind_T_at(T_idx)},{PREDT}")

        lines.append(f"or {pred_at(j)},{term_at(0)},{term_at(1)}")
        for T_idx in range(2, n_cands):
            lines.append(f"or {pred_at(j)},{term_at(T_idx)}")

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
            f.write(ir)
            f.write("\n")
        n_ops = len(ir.splitlines()) - 2
        print(f"  {name:<20} cost={cost:>9,}  ops={n_ops:>6,}  -> {path}")
