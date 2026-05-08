"""Energy-efficient sparse-parity scorer + baselines.

Scores IR programs that recover the secret 2-subset of bits from m=4
training rows over ``{0, 1}^3`` and predict 32 test labels under the
[simplified Dally model](https://github.com/cybertronai/simplified-dally-model)
using the
[v3 instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets/v3)
(v2 ops plus ``div``, ``abs``, ``cmp``, ``select`` — sufficient for
straight-line GF(2) Gaussian elimination with branchless partial
pivoting).

**Cost model (v3).** Processor at the origin, memory laid out as a
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

Supported ops (all from v3):

* ``add    dest, src1, src2``       — ``mem[dest] = mem[src1] + mem[src2]``
* ``sub    dest, src1, src2``       — ``mem[dest] = mem[src1] - mem[src2]``
* ``mul    dest, src1, src2``       — ``mem[dest] = mem[src1] * mem[src2]``
* ``div    dest, src1, src2``       — ``mem[dest] = mem[src1] // mem[src2]``
* ``and    dest, src1, src2``       — ``mem[dest] = mem[src1] & mem[src2]``
* ``or     dest, src1, src2``       — ``mem[dest] = mem[src1] | mem[src2]``
* ``xor    dest, src1, src2``       — ``mem[dest] = mem[src1] ^ mem[src2]``
* ``copy   dest, src``              — ``mem[dest] = mem[src]``  (1 read)
* ``not    dest, src``              — ``mem[dest] = ~mem[src]`` (1 read)
* ``abs    dest, src``              — ``mem[dest] = |mem[src]|`` (1 read)
* ``set    dest, K``                — ``mem[dest] = K`` (free; K is a literal)
* ``cmp    dest, a, b, pred``       — ``mem[dest] = 1 if mem[a] <pred> mem[b]
                                       else 0``; ``pred`` ∈
                                       {``eq, ne, lt, le, gt, ge``}; reads a + b
* ``select dest, c, t, f``          — ``mem[dest] = mem[t] if mem[c] else mem[f]``;
                                       reads c + t + f

Two-operand short form for the binary ops: ``xor dest, src`` is wire
sugar for ``xor dest, dest, src`` (in-place). Addresses must be
positive integers; ``addr ≤ 0`` raises. ``set``'s second operand is
an integer literal, not an address; ``cmp``'s last operand is a
predicate keyword, not an address.
"""
from __future__ import annotations

import math
from collections import namedtuple
from itertools import combinations
from random import Random
from typing import Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------
# Problem spec
# --------------------------------------------------------------------------

Spec = namedtuple("Spec", "n_bits k_secret m_train m_test")

#: Small instance — 3 bits, 2 secret, 4 train / 32 test. Parity recovery
#: is identifiable with 4 random training rows (E[false subsets] ≈ 0.125).
SMALL = Spec(n_bits=3, k_secret=2, m_train=4, m_test=32)

#: Medium instance — 8 bits, 3 secret, 8 train / 64 test. Identifiability
#: also high-probability per draw (E[false subsets] = (C(8,3)-1) · 2⁻⁸ ≈ 0.215).
MEDIUM = Spec(n_bits=8, k_secret=3, m_train=8, m_test=64)

# Backward-compat aliases for the small instance.
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
    for subset in combinations(range(n), k):
        if all(_label(row, subset) == y_i for row, y_i in zip(X, y)):
            matches += 1
            if matches > 1:
                return False
    return matches == 1


def generate(seed: int = 0, *, spec: Spec = SMALL) -> Tuple[
    List[List[int]], List[int], List[List[int]], List[int], List[int]
]:
    """Return ``(X_train, y_train, X_test, y_test, secret)``.

    The training rows are resampled until the secret is the unique
    weight-k subset matching y_train.
    """
    rng = Random(seed)
    secret = sorted(rng.sample(range(spec.n_bits), spec.k_secret))
    while True:
        X_train = [
            [rng.choice((0, 1)) for _ in range(spec.n_bits)]
            for _ in range(spec.m_train)
        ]
        y_train = [_label(row, secret) for row in X_train]
        if _identifiable(X_train, y_train, spec.k_secret):
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
    "div": lambda a, b: a // b,  # integer division
    "and": lambda a, b: a & b,
    "or":  lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
}
_UNARY = {
    "copy": lambda a: a,
    "not":  lambda a: ~a,
    "abs":  lambda a: abs(a),
}
_CMP_PRED = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a <  b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a >  b,
    "ge": lambda a, b: a >= b,
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
        raw = [tok.strip() for tok in rest.split(",")]
        if head == "set":
            if len(raw) != 2:
                raise ValueError(f"set needs 2 operands (dest, K); got {raw}")
            operands = [int(raw[0]), int(raw[1])]
            _check_addrs(operands[:1], "`set` dest")
            # operands[1] is a literal — any integer is allowed.
        elif head == "cmp":
            if len(raw) != 4:
                raise ValueError(
                    f"cmp needs 4 operands (dest, a, b, pred); got {raw}")
            pred = raw[3]
            if pred not in _CMP_PRED:
                raise ValueError(
                    f"cmp predicate must be one of {sorted(_CMP_PRED)}; "
                    f"got {pred!r}")
            operands = [int(raw[0]), int(raw[1]), int(raw[2]), pred]
            _check_addrs(operands[:3], "`cmp` operands")
        else:
            operands = [int(x) for x in raw]
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
        if op == "cmp":
            dest, a, b, pred = oprs
            for src in (a, b):
                if src not in mem:
                    raise ValueError(
                        f"cmp reads uninitialized addr {src}")
            cost += _cost(a) + _cost(b)
            mem[dest] = 1 if _CMP_PRED[pred](mem[a], mem[b]) else 0
            continue
        if op == "select":
            if len(oprs) != 4:
                raise ValueError(
                    f"select needs 4 operands (dest, c, t, f); got {oprs}")
            dest, c, t, f = oprs
            for src in (c, t, f):
                if src not in mem:
                    raise ValueError(
                        f"select reads uninitialized addr {src}")
            cost += _cost(c) + _cost(t) + _cost(f)
            mem[dest] = mem[t] if mem[c] else mem[f]
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
                f"unknown op: {op!r}  (v3 supports add/sub/mul/div/copy/"
                f"and/or/xor/not/abs/set/cmp/select)")
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

def _n_inputs(spec: Spec) -> int:
    """Number of input values: X_train + y_train + X_test, flat."""
    return spec.n_bits * spec.m_train + spec.m_train + spec.n_bits * spec.m_test


def _instance(seed: int, spec: Spec) -> Tuple[List[int], List[int]]:
    """Return ``(inputs, expected)`` for one seed under *spec*.

    ``inputs`` is the flat list ``[X_train..., y_train..., X_test...]``
    that the IR receives; ``expected`` is the ``y_test`` the IR's
    outputs must match.
    """
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
    """Random seeds covering distinct secrets, drawn from *rng* (defaults
    to a fresh nondeterministic ``Random()``).

    Returns up to ``max_seeds`` seeds, each producing a different secret
    subset. Drawing seeds randomly (instead of scanning 0, 1, 2, …)
    means the scorer cannot be gamed by an IR that hard-codes its
    predictions to a specific known seed list.
    """
    if rng is None:
        rng = Random()
    n_secrets = math.comb(spec.n_bits, spec.k_secret)
    target = min(max_seeds, n_secrets)
    seen: Dict[Tuple[int, ...], int] = {}
    # Coupon-collector bound: E[draws to collect ``target`` of N secrets]
    # is bounded by N · H_N for full coverage; allow a generous multiple.
    max_draws = 50 * (n_secrets + 1)
    for _ in range(max_draws):
        if len(seen) >= target:
            break
        seed = rng.randrange(1 << 31)
        _, _, _, _, secret = generate(seed=seed, spec=spec)
        key = tuple(secret)
        if key not in seen:
            seen[key] = seed
    if len(seen) < target:
        raise RuntimeError(
            f"could not draw {target} distinct secrets in {max_draws} attempts")
    return tuple(sorted(seen.values()))


def _score(ir: str, spec: Spec, max_seeds: int) -> int:
    """Run *ir* on a fresh random batch of canonical seeds for *spec*,
    verify outputs, and return the (data-independent) Dally read-cost.

    Seeds are drawn at random each call (see ``_canonical_seeds``), so
    the scorer cannot be gamed by an IR that hard-codes predictions to
    a known seed list.
    """
    seeds = _canonical_seeds(spec, max_seeds=max_seeds)
    seen_cost: int | None = None
    for seed in seeds:
        inputs, expected = _instance(seed, spec)
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


def score_small(ir: str) -> int:
    """Score *ir* on the small instance (3 random seeds — one per secret).

    Cost is data-independent for any given IR, so the returned value is
    deterministic; only the *which-instance* sampling is random.
    """
    return _score(ir, SMALL, max_seeds=64)


def score_medium(ir: str) -> int:
    """Score *ir* on the medium instance (8 random distinct secrets out
    of C(8,3)=56). See ``score_small`` for the data-independence note."""
    return _score(ir, MEDIUM, max_seeds=8)


# ---------------------------------------------------------------------------
# Baseline generator — try every candidate subset, AND-reduce match,
# OR-combine predictions
# ---------------------------------------------------------------------------

def _generate_baseline(spec: Spec) -> str:
    """General predictor IR — works for any seed of *spec*.

    Mirrors the brute-force solver in pure v2 IR. For each of the
    ``C(n, k)`` candidate k-subsets ``T = (t0, ..., t_{k-1})``:

      * Compute ``parity_T_i = y_train[i] XOR X_train[i,t0] XOR ...
        XOR X_train[i,t_{k-1}]`` — 0 iff T matches row i.
      * ``matched_T_i = parity_T_i XOR 1`` — 1 iff T matches row i.
      * ``ind_T = AND_i matched_T_i`` — 1 iff T matches every row.

    By identifiability, exactly one ``ind_T`` is 1 (the true secret).
    Each test row is then predicted as
    ``OR_T (ind_T AND (X_test[j,t0] XOR ... XOR X_test[j,t_{k-1}]))`` —
    the OR selects the lone non-zero term.

    Memory layout (computed from spec):
      pred       starts at 1                       (output, m_test cells)
      X_train    next                              (m_train × n_bits, row-major)
      y_train    next                              (m_train cells)
      X_test     next                              (m_test × n_bits, row-major)
      ONE        next                              (constant 1 via ``set``, free)
      tmp        next                              (scratch, reused)
      parity     next                              (scratch, reused)
      matched_T  next                              (C(n,k) × m_train cells)
      ind_T      next                              (C(n,k) cells)
      predT      next                              (scratch, reused per test row)
      term_T     next                              (C(n,k) cells, reused per row)
    """
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
            # parity_T_i = y_train[i] XOR X_train[i,t0] XOR ... XOR X_train[i,t_{k-1}]
            lines.append(f"xor {TMP},{y_tr_at(i)},{X_tr_at(i, T[0])}")
            for k in range(1, spec.k_secret - 1):
                lines.append(f"xor {TMP},{X_tr_at(i, T[k])}")
            lines.append(
                f"xor {PARITY},{TMP},{X_tr_at(i, T[spec.k_secret - 1])}")
            # matched_T_i = parity XOR 1
            lines.append(f"xor {matched_at(T_idx, i)},{PARITY},{ONE}")
        # ind_T = AND of all matched_T_i
        lines.append(
            f"and {ind_T_at(T_idx)},"
            f"{matched_at(T_idx, 0)},{matched_at(T_idx, 1)}")
        for i in range(2, spec.m_train):
            lines.append(f"and {ind_T_at(T_idx)},{matched_at(T_idx, i)}")

    # --- predictions: pred[j] = OR_T (ind_T AND predT) ------------------
    for j in range(spec.m_test):
        for T_idx, T in enumerate(candidates):
            # predT = X_test[j,t0] XOR X_test[j,t1] XOR ... XOR X_test[j,t_{k-1}]
            lines.append(f"xor {PREDT},{X_te_at(j, T[0])},{X_te_at(j, T[1])}")
            for k in range(2, spec.k_secret):
                lines.append(f"xor {PREDT},{X_te_at(j, T[k])}")
            lines.append(f"and {term_at(T_idx)},{ind_T_at(T_idx)},{PREDT}")
        # pred[j] = OR_T term_T
        lines.append(f"or {pred_at(j)},{term_at(0)},{term_at(1)}")
        for T_idx in range(2, n_cands):
            lines.append(f"or {pred_at(j)},{term_at(T_idx)}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def generate_baseline_small() -> str:
    """Baseline predictor IR for the *small* instance (n=3, k=2, 4/32)."""
    return _generate_baseline(SMALL)


def generate_baseline_medium() -> str:
    """Baseline predictor IR for the *medium* instance (n=8, k=3, 8/64)."""
    return _generate_baseline(MEDIUM)


__all__ = [
    "Spec", "SMALL", "MEDIUM",
    "N_BITS", "K_SECRET", "M_TRAIN", "M_TEST",
    "generate", "solve_bruteforce", "predict", "accuracy",
    "score_small", "generate_baseline_small",
    "score_medium", "generate_baseline_medium",
]


# ---------------------------------------------------------------------------
# Reproducer for the record-history IR files (``python sparse_parity.py``).
# ---------------------------------------------------------------------------

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
