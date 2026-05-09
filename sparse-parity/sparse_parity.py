"""Energy-efficient sparse-parity scorer + baselines.

Scores IR programs that recover the secret 2-subset of bits from m=4
training rows over ``{0, 1}^3`` and predict 32 test labels under the
simplified Dally model using the v3 instruction set, with all data
values constrained to a signed 8-bit ALU (``[-128, 127]``).
"""
from __future__ import annotations

import math
import secrets
from collections import namedtuple
from itertools import combinations
from random import Random
from typing import Callable, Dict, List, Sequence, Tuple

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


def _identifiable(X: Sequence[Sequence[int]], y: Sequence[int], k: int) -> bool:
    """True iff exactly one k-subset of columns explains every label in y."""
    n = len(X[0])
    matches = 0
    # Iterative short-circuit bypasses heavy frame creations of Python `all()`
    for subset in combinations(range(n), k):
        match = True
        for row, y_i in zip(X, y):
            p = 0
            for j in subset:
                p ^= row[j]
            if p != y_i:
                match = False
                break
        if match:
            matches += 1
            if matches > 1:
                return False
    return matches == 1


def generate(seed: int | None = None, *, spec: Spec = SMALL) -> Tuple[
    List[List[int]], List[int], List[List[int]], List[int], List[int]
]:
    # HARDENING: Fallback to SystemRandom (secrets) to prevent test-case predictability
    # if evaluated inside globally seeded test platforms (like CI environments).
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

        matches = 0
        for subset in all_combs:
            match = True
            for row, y_i in zip(X_train, y_train):
                p = 0
                for j in subset:
                    p ^= row[j]
                if p != y_i:
                    match = False
                    break
            if match:
                matches += 1
                if matches > 1:
                    break
        if matches == 1:
            break

    X_test = [
        [rng.choice((0, 1)) for _ in range(spec.n_bits)]
        for _ in range(spec.m_test)
    ]
    y_test = [_label(row, secret) for row in X_test]
    return X_train, y_train, X_test, y_test, secret


def solve_bruteforce(
    X: Sequence[Sequence[int]],
    y: Sequence[int],
    k: int = SMALL.k_secret,
) -> List[int]:
    n = len(X[0])
    for subset in combinations(range(n), k):
        match = True
        for row, y_i in zip(X, y):
            p = 0
            for j in subset:
                p ^= row[j]
            if p != y_i:
                match = False
                break
        if match:
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
    if not isinstance(addr, int) or addr < 1:
        raise ValueError(f"addresses must be positive integers; got {addr!r}")
    return math.isqrt(addr - 1) + 1


def _check_addrs(addrs, where):
    for a in addrs:
        if not isinstance(a, int) or a < 1:
            raise ValueError(f"{where}: addresses must be positive integers; got {a!r}")
        # HARDENING: Prevent massive layout offsets attempting OS memory exploits
        if a.bit_length() > 64:
            raise ValueError(f"{where}: address exceeds 64-bit bounds")


# ---------------------------------------------------------------------------
# Static Compiler + Fast Array Simulator
# ---------------------------------------------------------------------------

_BINARY = {"add", "sub", "mul", "div", "and", "or", "xor"}
_UNARY = {"copy", "not", "abs"}
_CMP_PRED = {"eq", "ne", "lt", "le", "gt", "ge"}

def _parse(ir: str) -> Tuple[List[int], List, List[int]]:
    text = ir.replace(";", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # HARDENING: Protect grader against DoS (Denial of Service) via bloated IR files
    if len(lines) > 100_000:
        raise ValueError("IR exceeds maximum allowed length (100,000 instructions)")
    if len(lines) < 2:
        raise ValueError("IR needs at least an input line and an output line")

    try:
        input_addrs = [int(x) for x in lines[0].split(",")]
        output_addrs = [int(x) for x in lines[-1].split(",")]
    except ValueError as e:
        raise ValueError(f"malformed input/output line: {e}")

    _check_addrs(input_addrs,  "input line")
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
            # HARDENING: Limit immediate values to 8-bit byte range. Accept either
            # signed (-128..127) or unsigned (0..255) interpretation; store time
            # below normalizes to canonical signed [-128, 127] so arithmetic stays
            # consistent with the byte-ALU semantics used for add/sub/mul/etc.
            if not (-128 <= literal <= 255):
                raise ValueError(
                    "set literal must fit in 8 bits (-128..255)")
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
            elif head not in _UNARY and head not in _BINARY and head != "select":
                raise ValueError(f"unknown op: {head!r}")

        ops.append((head, operands))

    return input_addrs, ops, output_addrs


def _compile_ir(ir: str) -> Tuple[Callable[[List[int]], List[int]], int, int]:
    input_addrs, ops, output_addrs = _parse(ir)

    # 1. Validation and Static Cost Determination Run Exactly Once
    cost = 0
    init = set(input_addrs)
    for op, oprs in ops:
        if op == "set":
            dest, literal = oprs
            init.add(dest)
        elif op == "cmp":
            dest, a, b, pred = oprs
            for src in (a, b):
                if src not in init: raise ValueError(f"cmp reads uninitialized addr {src}")
            cost += _cost(a) + _cost(b)
            init.add(dest)
        elif op == "select":
            dest, c, t, f = oprs
            for src in (c, t, f):
                if src not in init: raise ValueError(f"select reads uninitialized addr {src}")
            cost += _cost(c) + _cost(t) + _cost(f)
            init.add(dest)
        elif op in _UNARY:
            dest, src = oprs
            if src not in init: raise ValueError(f"{op} {dest},{src} reads uninitialized addr {src}")
            cost += _cost(src)
            init.add(dest)
        elif op in _BINARY:
            if len(oprs) == 3: dest, s1, s2 = oprs
            else: dest, s2 = oprs; s1 = dest
            for src in (s1, s2):
                if src not in init: raise ValueError(f"{op} {','.join(map(str,oprs))} reads uninitialized addr {src}")
            cost += _cost(s1) + _cost(s2)
            init.add(dest)

    for a in output_addrs:
        if a not in init: raise ValueError(f"output addr {a} never written")
        cost += _cost(a)

    # 2. Build dense execution map and packed instruction list
    all_addrs = set(input_addrs) | set(output_addrs)
    for op, oprs in ops:
        all_addrs.add(oprs[0])
        if op == "set": pass
        elif op == "cmp": all_addrs.update(oprs[1:3])
        else: all_addrs.update(oprs[1:])

    sorted_addrs = sorted(list(all_addrs))
    addr_to_idx = {a: i for i, a in enumerate(sorted_addrs)}

    in_idx = [addr_to_idx[a] for a in input_addrs]
    out_idx = [addr_to_idx[a] for a in output_addrs]
    n_mem = len(sorted_addrs)

    fast_ops = []
    for op, oprs in ops:
        if op == "set":
            fast_ops.append((0, addr_to_idx[oprs[0]], oprs[1], 0, 0))
        elif op == "cmp":
            fast_ops.append((1, addr_to_idx[oprs[0]], addr_to_idx[oprs[1]], addr_to_idx[oprs[2]], oprs[3]))
        elif op == "select":
            fast_ops.append((2, addr_to_idx[oprs[0]], addr_to_idx[oprs[1]], addr_to_idx[oprs[2]], addr_to_idx[oprs[3]]))
        elif op in _UNARY:
            fast_ops.append((3, addr_to_idx[oprs[0]], addr_to_idx[oprs[1]], 0, op))
        elif op in _BINARY:
            if len(oprs) == 3: dest, s1, s2 = oprs
            else: dest, s2 = oprs; s1 = dest
            fast_ops.append((4, addr_to_idx[dest], addr_to_idx[s1], addr_to_idx[s2], op))

    def simulate_fn(inputs: List[int]) -> List[int]:
        mem = [0] * n_mem
        for i, val in zip(in_idx, inputs):
            mem[i] = val

        for instr in fast_ops:
            kind = instr[0]
            if kind == 0:
                # Normalize the literal to canonical signed-byte form so
                # `set X, 128` and `set X, -128` produce the same in-memory value.
                v = instr[2] & 0xFF
                if v >= 0x80:
                    v -= 0x100
                mem[instr[1]] = v
            elif kind == 1:
                pred = instr[4]
                a = mem[instr[2]]
                b = mem[instr[3]]
                if pred == "eq": res = a == b
                elif pred == "ne": res = a != b
                elif pred == "lt": res = a < b
                elif pred == "le": res = a <= b
                elif pred == "gt": res = a > b
                else: res = a >= b
                mem[instr[1]] = 1 if res else 0
            elif kind == 2:
                mem[instr[1]] = mem[instr[3]] if mem[instr[2]] else mem[instr[4]]
            elif kind == 3:
                head_u = instr[4]
                src = mem[instr[2]]
                if head_u == "copy": res = src
                elif head_u == "not": res = ~src
                else: res = abs(src)

                # HARDENING: Force strict 8-bit (signed byte) bounds so Python big-int
                # limits SIMD exploits. Values live in [-128, 127] post-mask.
                res &= 0xFF
                if res >= 0x80:
                    res -= 0x100
                mem[instr[1]] = res

            elif kind == 4:
                head_b = instr[4]
                s1 = mem[instr[2]]
                s2 = mem[instr[3]]
                if head_b == "add": res = s1 + s2
                elif head_b == "sub": res = s1 - s2
                elif head_b == "mul": res = s1 * s2
                elif head_b == "div":
                    if s2 == 0: raise ZeroDivisionError("integer division or modulo by zero")
                    res = s1 // s2
                elif head_b == "and": res = s1 & s2
                elif head_b == "or": res = s1 | s2
                else: res = s1 ^ s2

                # HARDENING: Force strict 64-bit bounds
                res &= 0xFFFFFFFFFFFFFFFF
                if res >= 0x8000000000000000:
                    res -= 0x10000000000000000
                mem[instr[1]] = res

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


def _canonical_seeds(
    spec: Spec, max_seeds: int, rng: "Random | None" = None,
) -> Tuple[int, ...]:
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
        # HARDENING: Decouple secret coverage check from total array size to ensure
        # max_seeds are ALWAYS evaluated, instead of quitting after exactly 3 loops for SMALL
        if len(seeds) >= max_seeds and len(seen_secrets) >= target_secrets:
            break

        # HARDENING: Expand PRNG range to 256 bits, thwarting lookup tables mappings
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

    pred_at    = lambda j: pred_base + j
    X_tr_at    = lambda i, c: X_tr_base + i * spec.n_bits + c
    y_tr_at    = lambda i: y_tr_base + i
    X_te_at    = lambda j, c: X_te_base + j * spec.n_bits + c
    matched_at = lambda T_idx, i: matched_base + T_idx * spec.m_train + i
    ind_T_at   = lambda T_idx: ind_T_base + T_idx
    term_at    = lambda T_idx: term_base + T_idx

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
            lines.append(
                f"xor {PARITY},{TMP},{X_tr_at(i, T[spec.k_secret - 1])}")
            lines.append(f"xor {matched_at(T_idx, i)},{PARITY},{ONE}")

        lines.append(
            f"and {ind_T_at(T_idx)},"
            f"{matched_at(T_idx, 0)},{matched_at(T_idx, 1)}")
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
