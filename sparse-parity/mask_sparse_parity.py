"""Mask-recovery sparse parity: the test-set-free scaled tier.

The submission's job is to output the secret itself: given X_train
(m_train x n_bits) and y_train, emit the n_bits-cell 0/1 mask of the
hidden k-subset.  An instance scores 1 on an exact match and 0
otherwise -- because training sets are conditioned on unique
identifiability, "the weight-k solution consistent with the training
rows" is well-defined and unique, so exact-match scoring is complete
and ungameable.  The accuracy axis of the curve is the aggregate
**secret recovery rate**; the chance floor is 1/C(n,k) ~ 5e-6, i.e.
effectively zero (no complement pairing or advantage normalization is
needed, and even k would be legal in this mode).

Sparse parity makes the dropped test set provably redundant: a circuit
that has not identified the secret predicts test rows at exactly 50%
(any strict-subset parity is uncorrelated with the true one), so joint
test accuracy was always (1 + recovery)/2.  A fixed standard evaluator
can turn a recovered mask into labels for a constant ~4.4k reads/row --
a constant that cannot change rankings, so it is not scored here.

    MASK32:  n_bits=32, k_secret=5, m_train=18, cap = 2,000,000 ops

The cap is raised from the joint tier's 250k so that a known family
reaches 100% recovery (see ``generate_scan``); margins per brute-force
family: packed candidate enumeration needs ~4.2M ops (~2.1x the cap),
naive ~26M (~13x).  Under the cap, plain enumeration tops out near an
8% recovery ceiling at enormous energy.

Why are the instruction counts so large?  The ISA is straight-line:
there are no loops (every iteration of every loop is emitted as its own
instructions -- the 2^14-step scan is literally unrolled 16,383 times),
no branches (every data-dependent choice becomes a cmp/select chain
over all possibilities: picking the pivot row among 18 rows costs ~7
instructions per row, not one branch), and no indirect addressing
(reading M[pivot_row, c] where pivot_row is data takes an 18-way select
chain).  A ten-line looped Gaussian elimination therefore compiles to
~10^5 instructions, and instruction count ~= algorithmic work x
branchless overhead (~2-7x).

Reference family, ``generate_scan(n_steps)``: full-width branchless
GF(2) Gaussian elimination, then extraction of the null-space basis
(dimension n - m_train = 14 w.h.p.), then a Gray-code walk through the
solution space w = s0 + sum(a_j * basis_j), capturing any weight-k
visitor -- which, by identifiability, is the secret.  ``n_steps``
sweeps 0 (min-support GE alone: ~4% expected recovery, 6% on the dev
suite -- per-suite noise is ~+-2pp since success is secret-dependent)
to 2^14 - 1 (the whole solution space, ~100% recovery), tracing a
concave energy-vs-recovery curve (see ``generate_scan``).

This module is self-contained: it carries the IR compiler and energy
model, the vectorized batch simulator, and the GF(2) building blocks
it needs, so ``import mask_sparse_parity`` is the whole dependency.
"""
from __future__ import annotations

import hashlib
import math
import operator
import secrets as _secrets
from collections import namedtuple
from functools import lru_cache
from itertools import combinations
from random import Random
from typing import Callable, List, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Tier spec
# --------------------------------------------------------------------------

Spec = namedtuple("Spec", "n_bits k_secret m_train m_test")

MASK32 = Spec(n_bits=32, k_secret=5, m_train=18, m_test=0)
# Cap counts every line of the IR text (input/output lines included),
# matching _compile_ir.
OP_CAP = 2_000_000

SUITE_VERSION = "mask-sparse-parity-v1"
DEV_SUITE_KEY = "mask-dev"
DEV_SECRETS, DEV_REPS = 128, 8        # 1,024 instances
FINAL_SECRETS, FINAL_REPS = 256, 8    # 2,048 instances

MaskResult = namedtuple("MaskResult", "cost recovery n_instances")


# --------------------------------------------------------------------------
# IR text -> executable program: the v3 instruction set, its energy
# model, and a scalar simulator.  Shared by every generator below.
# --------------------------------------------------------------------------

def _cost(addr: int) -> int:
    if not isinstance(addr, int) or addr < 1:
        raise ValueError(f"addresses must be positive integers; got {addr!r}")
    return math.isqrt(addr - 1) + 1

def _to_signed_8bit(val: int) -> int:
    """Normalize integer to canonical signed 8-bit form [-128, 127]."""
    val &= 0xFF
    return val - 0x100 if val >= 0x80 else val

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

def _compile_ir(
    ir: str, max_instructions: int = 100_000
) -> Tuple[Callable[[List[int]], List[int]], int, int]:
    text = ir.replace(";", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Protect grader against DoS (Denial of Service) via bloated IR files.
    # The cap is per-benchmark: 100k for the classic instances; scaled tiers
    # pass their own (it is part of each tier's contract).
    if len(lines) > max_instructions:
        raise ValueError(
            f"IR exceeds maximum allowed length ({max_instructions:,} instructions)"
        )
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

# --------------------------------------------------------------------------
# Vectorized simulator: runs one IR across a whole batch of instances
# at once (numpy per instruction), plus the deterministic suite RNG.
# --------------------------------------------------------------------------

def _digest_rng(*parts) -> Random:
    """Random stream derived from SHA-256 of the joined parts."""
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode()).digest()
    return Random(int.from_bytes(digest, "big"))

_MAX_BATCH_CELLS = 1 << 26  # int16 cells of IR memory held at once (~128 MB)

_V_CMP = {
    "eq": np.equal, "ne": np.not_equal, "lt": np.less,
    "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal,
}

_V_UNARY = {"copy": lambda x: x, "not": np.bitwise_not, "abs": np.abs}

def _v_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if np.any(b == 0):
        raise ZeroDivisionError("integer division or modulo by zero")
    return np.floor_divide(a, b)

_V_BINARY = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": _v_div,
    "and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor,
}

def _wrap8(v: np.ndarray) -> np.ndarray:
    """Vectorized ``_to_signed_8bit``."""
    return ((v & 0xFF) ^ 0x80) - 0x80

def _compile_vector(
    ir: str, max_instructions: int = 100_000
) -> Tuple[Callable[[np.ndarray], np.ndarray], int, int]:
    """Compile IR into a batched executor: (n_instances, n_inputs) int16 ->
    (n_instances, n_outputs) int16.  Validation and the static read cost are
    delegated to ``_compile_ir`` so both engines agree."""
    _, static_cost, n_inputs = _compile_ir(ir, max_instructions)

    text = ir.replace(";", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    input_addrs = [int(x) for x in lines[0].split(",") if x.strip()]
    output_addrs = [int(x) for x in lines[-1].split(",") if x.strip()]

    init = set(input_addrs)
    ops = []
    for ln in lines[1:-1]:
        head, _, rest = ln.partition(" ")
        raw = [x.strip() for x in rest.split(",") if x.strip()] if rest else []
        if head == "set":
            dest = int(raw[0])
            ops.append(("set", dest, _to_signed_8bit(int(raw[1])), None, None))
        elif head == "cmp":
            dest = int(raw[0])
            ops.append(("cmp", dest, int(raw[1]), int(raw[2]), _V_CMP[raw[3]]))
        elif head == "select":
            dest = int(raw[0])
            ops.append(("select", dest, int(raw[1]), int(raw[2]), int(raw[3])))
        elif head in _V_UNARY:
            dest = int(raw[0])
            ops.append(("unary", dest, int(raw[1]), None, _V_UNARY[head]))
        else:  # binary (validated by _compile_ir)
            dest = int(raw[0])
            s1, s2 = (int(raw[1]), int(raw[2])) if len(raw) == 3 else (dest, int(raw[1]))
            ops.append(("binary", dest, s1, s2, _V_BINARY[head]))
        init.add(dest)

    addr_to_idx = {a: i for i, a in enumerate(sorted(init))}
    n_mem = len(addr_to_idx)
    in_idx = [addr_to_idx[a] for a in input_addrs]
    out_idx = [addr_to_idx[a] for a in output_addrs]

    fast_ops = []
    for kind, dest, a, b, aux in ops:
        d = addr_to_idx[dest]
        if kind == "set":
            fast_ops.append(("set", d, a, None, None))
        elif kind == "cmp":
            fast_ops.append(("cmp", d, addr_to_idx[a], addr_to_idx[b], aux))
        elif kind == "select":
            fast_ops.append(("select", d, addr_to_idx[a], addr_to_idx[b], addr_to_idx[aux]))
        elif kind == "unary":
            fast_ops.append(("unary", d, addr_to_idx[a], None, aux))
        else:
            fast_ops.append(("binary", d, addr_to_idx[a], addr_to_idx[b], aux))

    def _run_chunk(inputs: np.ndarray) -> np.ndarray:
        mem = np.zeros((n_mem, inputs.shape[0]), dtype=np.int16)
        mem[in_idx] = inputs.T
        for kind, d, a, b, aux in fast_ops:
            if kind == "set":
                mem[d] = a
            elif kind == "cmp":
                mem[d] = aux(mem[a], mem[b]).astype(np.int16)
            elif kind == "select":
                mem[d] = np.where(mem[a] != 0, mem[b], mem[aux])
            elif kind == "unary":
                mem[d] = _wrap8(aux(mem[a]))
            else:
                mem[d] = _wrap8(aux(mem[a], mem[b]))
        return mem[out_idx].T

    def run(inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim != 2 or inputs.shape[1] != len(in_idx):
            raise ValueError(
                f"IR declares {len(in_idx)} inputs; got shape {inputs.shape}"
            )
        # Bound working memory: an adversarial IR can declare ~100k distinct
        # addresses, so cap n_mem * batch at _MAX_BATCH_CELLS int16 cells
        # (~128 MB) and process the instance axis in chunks.
        rows = max(1, _MAX_BATCH_CELLS // n_mem)
        if inputs.shape[0] <= rows:
            return _run_chunk(inputs)
        return np.vstack([
            _run_chunk(inputs[i:i + rows])
            for i in range(0, inputs.shape[0], rows)
        ])

    return run, static_cost, n_inputs

# --------------------------------------------------------------------------
# GF(2) building blocks: candidate columns, unique-identifiability
# checks, secret sampling, and the branchless information-set-decoding
# generator that ``generate_isd_mask`` wraps.
# --------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _candidate_columns(n_bits: int, k_secret: int) -> np.ndarray:
    return np.array(list(combinations(range(n_bits), k_secret)), dtype=np.int64)

def _unique_ksparse(X: np.ndarray, y: np.ndarray, cand: np.ndarray) -> bool:
    """Exactly one k-subset of columns explains every label.  Each
    column's m_train bits are packed into one integer signature, so the
    check over all C(n,k) candidates is a handful of vectorized XORs."""
    weights = (1 << np.arange(X.shape[0], dtype=np.int64))
    col_sig = weights @ X.astype(np.int64)          # (n,)
    y_sig = int(weights @ y.astype(np.int64))
    sig = col_sig[cand[:, 0]]
    for j in range(1, cand.shape[1]):
        sig = sig ^ col_sig[cand[:, j]]
    return int((sig == y_sig).sum()) == 1

def _sample_secrets(spec: Spec, n_secrets: int, rng: Random) -> List[Tuple[int, ...]]:
    n_possible = math.comb(spec.n_bits, spec.k_secret)
    if n_secrets > n_possible:
        raise ValueError(
            f"cannot sample {n_secrets} distinct secrets; C(n,k)={n_possible}"
        )
    seen, out = set(), []
    while len(out) < n_secrets:
        s = tuple(sorted(rng.sample(range(spec.n_bits), spec.k_secret)))
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _isd_subsets(
    spec: Spec, n_restarts: int, *, seed: int | None = None
) -> List[List[int]]:
    """Information sets for the T restarts.

    Default (``seed=None``) keeps the original deterministic rotation:
    restart t uses columns [(stride*t + j) % n for j < m_train].  The
    rotation has period exactly n = 32 (gcd(stride, n) = 1), so past
    T = 32 every restart re-deals a subset an earlier restart already
    used and recovery plateaus.

    ``seed`` is not None switches to independent random information
    sets: each restart draws a fresh uniform m_train-subset of the
    n_bits columns once, at generation time, from a seeded RNG (the
    emitted circuit stays deterministic).  Independent draws stack as
    1 - (1 - p)^T instead of saturating."""
    n, m = spec.n_bits, spec.m_train
    if seed is not None:
        rng = Random(seed)
        return [sorted(rng.sample(range(n), m)) for _ in range(n_restarts)]
    stride = 7 if math.gcd(7, n) == 1 else 5
    return [[(stride * t + j) % n for j in range(m)] for t in range(n_restarts)]

def generate_isd(
    n_restarts: int = 1,
    n_outputs: int | None = None,
    *,
    spec: Spec = MASK32,
    mask_output: bool = False,
    op_cap: int = OP_CAP,
    subset_seed: int | None = None,
) -> str:
    """T-restart information-set-decoding circuit.

    Each restart copies the training columns of one m_train-sized column
    subset into an augmented matrix, runs branchless GF(2) row reduction
    (select-based pivoting, following ge_small.py), reads out the
    zero-free-variable solution, and accepts it only if it has weight
    k_secret AND reproduces every training label -- by unique
    identifiability an accepted solution is exactly the secret.  Accepted
    solutions OR into a full-width secret mask; the first ``n_outputs``
    test rows are labeled by the O(n) mask predictor, the rest default
    to 0 (exactly 50% under complement pairing).

    With ``mask_output=True`` the circuit targets the test-set-free mask
    task instead: inputs are X_train and y_train only, the outputs are
    the n_bits mask cells themselves, and no prediction phase is
    emitted.
    """
    n, m, k, m_test = spec.n_bits, spec.m_train, spec.k_secret, spec.m_test
    f = m_test if n_outputs is None else n_outputs
    if not 0 <= f <= m_test:
        raise ValueError(f"n_outputs must be in [0, {m_test}]")
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")
    n_sub = m          # information-set size = m_train (square system)
    n_aug = n_sub + 1
    subsets = _isd_subsets(spec, n_restarts, seed=subset_seed)

    # ---- layout: hottest scratch at the lowest addresses ----------------
    a = 1
    def alloc(sz):
        nonlocal a
        base = a; a += sz; return base

    M_base   = alloc(m * n_aug)          # augmented matrix (reused per restart)
    PR_base  = alloc(n_aug)              # pivot-row buffer
    ZERO     = alloc(1); ONE = alloc(1); M_VAL = alloc(1); K_VAL = alloc(1)
    ROW_base = alloc(m)                  # row-index constants
    used_base = alloc(m)
    pivot_base = alloc(n_sub)            # pivot row per subset column
    s_sub_base = alloc(n_sub)            # per-restart solution (subset coords)
    pivot_idx = alloc(1); found = alloc(1); bit = alloc(1)
    not_used = alloc(1); eligible = alloc(1); is_first = alloc(1)
    is_match = alloc(1); is_other = alloc(1); do_xor = alloc(1)
    a_tmp = alloc(1); b_tmp = alloc(1)
    weight = alloc(1); ok = alloc(1); err = alloc(1); acc = alloc(1)
    s_final_base = alloc(n)              # full-width secret mask
    pred_base = alloc(m_test)
    X_tr_base = alloc(n * m)
    y_tr_base = alloc(m)
    X_te_base = alloc(n * m_test)

    M_at    = lambda i, j: M_base + i * n_aug + j
    PR_at   = lambda j: PR_base + j
    ROW_at  = lambda r: ROW_base + r
    used_at = lambda r: used_base + r
    pivot_at = lambda c: pivot_base + c
    s_sub_at = lambda c: s_sub_base + c
    s_at    = lambda c: s_final_base + c
    pred_at = lambda j: pred_base + j
    X_tr_at = lambda i, c: X_tr_base + i * n + c
    y_tr_at = lambda i: y_tr_base + i
    X_te_at = lambda j, c: X_te_base + j * n + c

    inputs = [X_tr_at(i, c) for i in range(m) for c in range(n)] + [
        y_tr_at(i) for i in range(m)
    ]
    if mask_output:
        outputs = [s_at(c) for c in range(n)]
    else:
        inputs += [X_te_at(j, c) for j in range(m_test) for c in range(n)]
        outputs = [pred_at(j) for j in range(m_test)]
    lines = [",".join(map(str, inputs))]
    emit = lambda s: lines.append(s)

    # ---- constants and mask init ---------------------------------------
    emit(f"set {ZERO},0"); emit(f"set {ONE},1")
    emit(f"set {M_VAL},{m}"); emit(f"set {K_VAL},{k}")
    for r in range(m):
        emit(f"set {ROW_at(r)},{r}")
    for c in range(n):
        emit(f"set {s_at(c)},0")

    for t, cols in enumerate(subsets):
        # -- load [X_train[:, cols] | y] and reset pivot bookkeeping ------
        for i in range(m):
            for j, c in enumerate(cols):
                emit(f"copy {M_at(i, j)},{X_tr_at(i, c)}")
            emit(f"copy {M_at(i, n_sub)},{y_tr_at(i)}")
        for r in range(m):
            emit(f"set {used_at(r)},0")

        # -- branchless GF(2) RREF over the subset columns ----------------
        for col in range(n_sub):
            emit(f"copy {pivot_idx},{M_VAL}")
            emit(f"copy {found},{ZERO}")
            for r in range(m):
                emit(f"copy {bit},{M_at(r, col)}")
                emit(f"select {not_used},{used_at(r)},{ZERO},{ONE}")
                emit(f"and {eligible},{bit},{not_used}")
                emit(f"select {is_first},{found},{ZERO},{eligible}")
                emit(f"select {pivot_idx},{is_first},{ROW_at(r)},{pivot_idx}")
                emit(f"or {used_at(r)},{is_first}")
                emit(f"or {found},{eligible}")
            emit(f"copy {pivot_at(col)},{pivot_idx}")
            for j in range(n_aug):
                emit(f"copy {PR_at(j)},{ZERO}")
                for r in range(m):
                    emit(f"cmp {is_match},{pivot_idx},{ROW_at(r)},eq")
                    emit(f"select {PR_at(j)},{is_match},{M_at(r, j)},{PR_at(j)}")
            for r in range(m):
                emit(f"cmp {is_match},{pivot_idx},{ROW_at(r)},eq")
                emit(f"select {is_other},{is_match},{ZERO},{ONE}")
                emit(f"copy {bit},{M_at(r, col)}")
                emit(f"and {do_xor},{is_other},{bit}")
                for j in range(n_aug):
                    emit(f"copy {a_tmp},{M_at(r, j)}")
                    emit(f"xor {b_tmp},{M_at(r, j)},{PR_at(j)}")
                    emit(f"select {M_at(r, j)},{do_xor},{b_tmp},{a_tmp}")

        # -- zero-free-variable readout -----------------------------------
        for c in range(n_sub):
            emit(f"copy {s_sub_at(c)},{ZERO}")
            for r in range(m):
                emit(f"cmp {is_match},{pivot_at(c)},{ROW_at(r)},eq")
                emit(f"select {s_sub_at(c)},{is_match},{M_at(r, n_sub)},{s_sub_at(c)}")

        # -- accept iff weight == k AND every training row verifies -------
        emit(f"add {weight},{s_sub_at(0)},{s_sub_at(1)}")
        for c in range(2, n_sub):
            emit(f"add {weight},{s_sub_at(c)}")
        emit(f"cmp {ok},{weight},{K_VAL},eq")
        emit(f"copy {err},{ZERO}")
        for i in range(m):
            emit(f"and {acc},{s_sub_at(0)},{X_tr_at(i, cols[0])}")
            for j in range(1, n_sub):
                emit(f"and {a_tmp},{s_sub_at(j)},{X_tr_at(i, cols[j])}")
                emit(f"xor {acc},{a_tmp}")
            emit(f"xor {acc},{y_tr_at(i)}")
            emit(f"or {err},{acc}")
        emit(f"select {ok},{err},{ZERO},{ok}")

        # -- accumulate the (unique) verified secret into s_final ---------
        for j, c in enumerate(cols):
            emit(f"and {a_tmp},{ok},{s_sub_at(j)}")
            emit(f"or {s_at(c)},{a_tmp}")

    # ---- mask predictor on the first f rows (joint task only) -----------
    if not mask_output:
        for j in range(m_test):
            if j >= f:
                emit(f"set {pred_at(j)},0")
                continue
            emit(f"and {acc},{s_at(0)},{X_te_at(j, 0)}")
            for c in range(1, n):
                emit(f"and {a_tmp},{s_at(c)},{X_te_at(j, c)}")
                emit(f"xor {acc},{a_tmp}")
            emit(f"copy {pred_at(j)},{acc}")

    lines.append(",".join(map(str, outputs)))
    ir = "\n".join(lines)
    if len(lines) > op_cap:
        raise ValueError(
            f"generated ISD IR has {len(lines) - 2:,} ops, over the {op_cap:,} cap"
        )
    return ir

# --------------------------------------------------------------------------
# Mask tier: deterministic suite, scoring, and the reference
# generator families.
# --------------------------------------------------------------------------

def _n_inputs(spec: Spec) -> int:
    return spec.n_bits * spec.m_train + spec.m_train


# --------------------------------------------------------------------------
# Suite: training instances only
# --------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _mask_suite_cached(
    spec: Spec, n_secrets: int, repetitions: int, suite_key: str
) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    cand = _candidate_columns(spec.n_bits, spec.k_secret)
    key_rng = _digest_rng(SUITE_VERSION, suite_key, "secrets", n_secrets)
    sampled = _sample_secrets(spec, n_secrets, key_rng)

    inputs_rows, mask_rows, meta = [], [], []
    for secret in sampled:
        secret_cols = list(secret)
        mask = np.zeros(spec.n_bits, dtype=np.int16)
        mask[secret_cols] = 1
        for rep in range(repetitions):
            rng = _digest_rng(SUITE_VERSION, suite_key, "train", secret, rep)
            while True:
                X = np.array(
                    [[rng.getrandbits(1) for _ in range(spec.n_bits)]
                     for _ in range(spec.m_train)],
                    dtype=np.int16,
                )
                y = np.bitwise_xor.reduce(X[:, secret_cols], axis=1)
                if _unique_ksparse(X, y, cand):
                    break
            inputs_rows.append(np.concatenate([X.ravel(), y]))
            mask_rows.append(mask)
            meta.append((secret, rep))

    inputs = np.stack(inputs_rows).astype(np.int16)
    masks = np.stack(mask_rows).astype(np.int16)
    inputs.flags.writeable = False
    masks.flags.writeable = False
    return inputs, masks, tuple(meta)


def mask_suite(
    *,
    spec: Spec = MASK32,
    n_secrets: int | None = None,
    repetitions: int | None = None,
    suite_key: str | None = DEV_SUITE_KEY,
):
    """Training-only suite for the mask task.  ``suite_key=None`` draws a
    fresh hidden SystemRandom key (adjudication: FINAL_SECRETS sample,
    built outside the cache); a string key gives the cached deterministic
    dev suite at DEV_SECRETS."""
    final = suite_key is None
    if n_secrets is None:
        n_secrets = FINAL_SECRETS if final else DEV_SECRETS
    if repetitions is None:
        repetitions = FINAL_REPS if final else DEV_REPS
    if final:
        key = "fresh-" + _secrets.token_hex(16)
        return _mask_suite_cached.__wrapped__(spec, n_secrets, repetitions, key)
    return _mask_suite_cached(spec, n_secrets, repetitions, suite_key)


def evaluate_mask(
    ir: str,
    *,
    spec: Spec = MASK32,
    n_secrets: int | None = None,
    repetitions: int | None = None,
    suite_key: str | None = DEV_SUITE_KEY,
    engine: str = "vector",
) -> MaskResult:
    """Score an IR on the mask task: fraction of instances whose n_bits
    output cells exactly equal the secret mask, plus static read cost."""
    inputs, masks, _ = mask_suite(
        spec=spec, n_secrets=n_secrets, repetitions=repetitions,
        suite_key=suite_key,
    )
    expected_inputs = _n_inputs(spec)

    if engine == "vector":
        run, cost, n_in = _compile_vector(ir, OP_CAP)
        if n_in != expected_inputs:
            raise ValueError(f"IR declares {n_in} inputs; {expected_inputs} required")
        outputs = run(inputs)
    elif engine == "reference":
        simulate_fn, cost, n_in = _compile_ir(ir, OP_CAP)
        if n_in != expected_inputs:
            raise ValueError(f"IR declares {n_in} inputs; {expected_inputs} required")
        outputs = np.array(
            [simulate_fn(list(map(int, row))) for row in inputs], dtype=np.int16
        )
    else:
        raise ValueError(f"unknown engine {engine!r}")

    if outputs.shape[1] != spec.n_bits:
        raise ValueError(
            f"IR produces {outputs.shape[1]} outputs; {spec.n_bits} required"
        )
    exact = (outputs == masks).all(axis=1)
    return MaskResult(
        cost=cost,
        recovery=float(exact.mean()),
        n_instances=int(masks.shape[0]),
    )


# --------------------------------------------------------------------------
# Reference circuits
# --------------------------------------------------------------------------

def generate_isd_mask(
    n_restarts: int = 1, *, spec: Spec = MASK32, subset_seed: int | None = None
) -> str:
    """ISD restart family on the mask task (no prediction phase).

    ``subset_seed`` is not None draws independent random information
    sets instead of the default period-32 rotation (see
    ``_isd_subsets``)."""
    joint_spec = Spec(spec.n_bits, spec.k_secret, spec.m_train, 0)
    return generate_isd(
        n_restarts, spec=joint_spec, mask_output=True, op_cap=OP_CAP,
        subset_seed=subset_seed,
    )


def generate_enum_mask(
    n_candidates: int, *, spec: Spec = MASK32
) -> str:
    """Capped candidate enumeration on the mask task: check the first q
    of C(n,k) candidate subsets against the training labels, OR the
    (at most one) match into the output mask."""
    from itertools import combinations

    n, m, k = spec.n_bits, spec.m_train, spec.k_secret
    C = math.comb(n, k)
    if not 1 <= n_candidates <= C:
        raise ValueError(f"n_candidates must be in [1, {C}]")
    candidates = list(combinations(range(n), k))[:n_candidates]

    a = 1
    def alloc(sz):
        nonlocal a
        base = a; a += sz; return base

    ONE = alloc(1); TMP = alloc(1); PARITY = alloc(1); IND = alloc(1)
    mask_base = alloc(n)
    X_tr_base = alloc(n * m)
    y_tr_base = alloc(m)

    mask_at = lambda c: mask_base + c
    X_at = lambda i, c: X_tr_base + i * n + c
    y_at = lambda i: y_tr_base + i

    inputs = [X_at(i, c) for i in range(m) for c in range(n)] + [
        y_at(i) for i in range(m)
    ]
    lines = [",".join(map(str, inputs))]
    emit = lines.append

    emit(f"set {ONE},1")
    for c in range(n):
        emit(f"set {mask_at(c)},0")

    for T in candidates:
        for i in range(m):
            emit(f"xor {TMP},{y_at(i)},{X_at(i, T[0])}")
            for j in range(1, k - 1):
                emit(f"xor {TMP},{X_at(i, T[j])}")
            emit(f"xor {PARITY},{TMP},{X_at(i, T[-1])}")
            emit(f"xor {PARITY},{ONE}")
            if i == 0:
                emit(f"copy {IND},{PARITY}")
            else:
                emit(f"and {IND},{PARITY}")
        for c in T:
            emit(f"or {mask_at(c)},{IND}")

    lines.append(",".join(str(mask_at(c)) for c in range(n)))
    ir = "\n".join(lines)
    if len(lines) > OP_CAP:
        raise ValueError(
            f"enumeration IR has {len(lines) - 2:,} ops, over the {OP_CAP:,} cap"
        )
    return ir


def _weight_order_flips(G: int, cap: int) -> List[List[int]]:
    """Flip schedule for a coefficient-weight-ordered walk.

    Visits the empty coefficient vector, then every index subset of the
    G free variables in increasing Hamming weight, up to ``cap``.  For
    each transition returns the list of Gray slots whose basis vector
    must be XORed into the running solution.  With ~2.2 average ones in
    the secret's coefficient vector (measured on the dev suite), the
    secret is reached in tens-to-hundreds of visits instead of the
    thousands a reflected-Gray walk needs."""
    sets: List[tuple] = [()]
    for w in range(1, cap + 1):
        sets.extend(combinations(range(G), w))
    flips, prev = [], frozenset()
    for cur in sets:
        cur_s = frozenset(cur)
        flips.append(sorted(cur_s.symmetric_difference(prev)))
        prev = cur_s
    return flips


def generate_scan(
    n_steps: int | None = None,
    *,
    spec: Spec = MASK32,
    joint: bool = False,
    op_cap: int = OP_CAP,
    walk: str = "gray",
    weight_cap: int | None = None,
) -> str:
    """GE + null-space Gray scan: the family that reaches 100%.

    Phase 1: branchless full-width GF(2) RREF of [X_train | y] with
    select-based pivoting (pivot for column c = first unused row with a
    1 there; free columns get a sentinel).
    Phase 2: read the zero-free-variable solution s0, extract the
    null-space basis (basis vector of free column f has a 1 at f and
    the RREF entries M[pivot_row(c), f] at pivot columns c), and gather
    the basis vectors into Gray-variable slots by each free column's
    rank among free columns.
    Phase 3: walk ``n_steps`` of the reflected Gray code over the
    G = n - m_train abstract free variables; each step XORs one basis
    vector into the current solution w, recomputes weight(w), and
    captures w into the output whenever weight == k_secret -- every
    visited w solves the training system, so a weight-k visitor IS the
    secret (unique identifiability).  s0 itself is checked before the
    walk, so ``n_steps=0`` is exactly min-support GE.

    Recovery grows concavely in n_steps -- much faster than the naive
    (n_steps + 1) / 2^G line -- because the secret's Gray coefficient
    vector is low-weight (its restriction to the ~14 free columns
    averages ~2.2 ones), so the walk from the all-zero assignment tends
    to visit it early: measured on the dev suite, s=0 -> 6%, s=1,023 ->
    49%, s=4,095 -> 74% (median capture step ~1,100 of 16,383).
    n_steps = 2^G - 1 visits the entire solution space -> ~100%
    (rank-deficient training draws, ~2^-14 of instances, may still miss).

    With ``joint=True`` the circuit targets a joint train+test tier
    instead (``spec.m_test`` > 0): it additionally reads X_test and
    outputs all m_test labels via the O(n) mask predictor over the
    captured secret -- no output truncation.
    """
    n, m, k = spec.n_bits, spec.m_train, spec.k_secret
    G = n - m                       # null-space dimension w.h.p.
    max_steps = (1 << G) - 1
    s = max_steps if n_steps is None else n_steps
    if not 0 <= s <= max_steps:
        raise ValueError(f"n_steps must be in [0, {max_steps}]")
    if walk not in ("gray", "weight"):
        raise ValueError(f"unknown walk {walk!r}")
    n_aug = n + 1

    a = 1
    def alloc(sz):
        nonlocal a
        base = a; a += sz; return base

    # scan-phase state first: it is read ~200x per Gray step
    w_base   = alloc(n)
    out_base = alloc(n)
    WSUM = alloc(1); OK = alloc(1)
    FB_base = alloc(G * n)          # basis vectors in Gray-variable order
    # RREF working set
    M_base  = alloc(m * n_aug)
    PR_base = alloc(n_aug)
    ZERO = alloc(1); ONE = alloc(1); M_VAL = alloc(1); K_VAL = alloc(1)
    ROW_base  = alloc(m)
    JC_base   = alloc(G)            # constants 0..G-1 for rank matching
    used_base = alloc(m)
    pivot_base = alloc(n)
    free_base  = alloc(n)
    rank_base  = alloc(n)
    B_base     = alloc(n * n)       # raw basis entries B[f][c]
    pivot_idx = alloc(1); found = alloc(1); bit = alloc(1)
    not_used = alloc(1); eligible = alloc(1); is_first = alloc(1)
    is_match = alloc(1); is_other = alloc(1); do_xor = alloc(1)
    a_tmp = alloc(1); b_tmp = alloc(1)
    X_tr_base = alloc(n * m)
    y_tr_base = alloc(m)
    if joint:
        if spec.m_test <= 0:
            raise ValueError("joint=True requires spec.m_test > 0")
        pred_base = alloc(spec.m_test)
        X_te_base = alloc(n * spec.m_test)
        pred_at = lambda j: pred_base + j
        X_te_at = lambda j, c: X_te_base + j * n + c

    w_at    = lambda c: w_base + c
    out_at  = lambda c: out_base + c
    FB_at   = lambda j, c: FB_base + j * n + c
    M_at    = lambda i, j: M_base + i * n_aug + j
    PR_at   = lambda j: PR_base + j
    ROW_at  = lambda r: ROW_base + r
    JC_at   = lambda j: JC_base + j
    used_at = lambda r: used_base + r
    pivot_at = lambda c: pivot_base + c
    free_at  = lambda c: free_base + c
    rank_at  = lambda c: rank_base + c
    B_at     = lambda f, c: B_base + f * n + c
    X_at     = lambda i, c: X_tr_base + i * n + c
    y_at     = lambda i: y_tr_base + i

    inputs = [X_at(i, c) for i in range(m) for c in range(n)] + [
        y_at(i) for i in range(m)
    ]
    if joint:
        inputs += [
            X_te_at(j, c) for j in range(spec.m_test) for c in range(n)
        ]
    lines = [",".join(map(str, inputs))]
    emit = lines.append

    # ---- constants ------------------------------------------------------
    emit(f"set {ZERO},0"); emit(f"set {ONE},1")
    emit(f"set {M_VAL},{m}"); emit(f"set {K_VAL},{k}")
    for r in range(m):
        emit(f"set {ROW_at(r)},{r}")
        emit(f"set {used_at(r)},0")
    for j in range(G):
        emit(f"set {JC_at(j)},{j}")
    for c in range(n):
        emit(f"set {out_at(c)},0")

    # ---- load augmented matrix ------------------------------------------
    for i in range(m):
        for c in range(n):
            emit(f"copy {M_at(i, c)},{X_at(i, c)}")
        emit(f"copy {M_at(i, n)},{y_at(i)}")

    # ---- phase 1: RREF over all n columns -------------------------------
    for col in range(n):
        emit(f"copy {pivot_idx},{M_VAL}")
        emit(f"copy {found},{ZERO}")
        for r in range(m):
            emit(f"copy {bit},{M_at(r, col)}")
            emit(f"select {not_used},{used_at(r)},{ZERO},{ONE}")
            emit(f"and {eligible},{bit},{not_used}")
            emit(f"select {is_first},{found},{ZERO},{eligible}")
            emit(f"select {pivot_idx},{is_first},{ROW_at(r)},{pivot_idx}")
            emit(f"or {used_at(r)},{is_first}")
            emit(f"or {found},{eligible}")
        emit(f"copy {pivot_at(col)},{pivot_idx}")
        emit(f"select {free_at(col)},{found},{ZERO},{ONE}")
        for j in range(n_aug):
            emit(f"copy {PR_at(j)},{ZERO}")
            for r in range(m):
                emit(f"cmp {is_match},{pivot_idx},{ROW_at(r)},eq")
                emit(f"select {PR_at(j)},{is_match},{M_at(r, j)},{PR_at(j)}")
        for r in range(m):
            emit(f"cmp {is_match},{pivot_idx},{ROW_at(r)},eq")
            emit(f"select {is_other},{is_match},{ZERO},{ONE}")
            emit(f"copy {bit},{M_at(r, col)}")
            emit(f"and {do_xor},{is_other},{bit}")
            for j in range(n_aug):
                emit(f"copy {a_tmp},{M_at(r, j)}")
                emit(f"xor {b_tmp},{M_at(r, j)},{PR_at(j)}")
                emit(f"select {M_at(r, j)},{do_xor},{b_tmp},{a_tmp}")

    # ---- phase 2a: w = s0 (zero-free-variable solution) ------------------
    for c in range(n):
        emit(f"copy {w_at(c)},{ZERO}")
        for r in range(m):
            emit(f"cmp {is_match},{pivot_at(c)},{ROW_at(r)},eq")
            emit(f"select {w_at(c)},{is_match},{M_at(r, n)},{w_at(c)}")

    # ---- phase 2b: raw basis entries B[f][c] -----------------------------
    # basis vector of free column f: 1 at f, M[pivot_row(c), f] at pivot
    # columns c, 0 at other free columns (the dynamic read yields 0 there).
    for f in range(n):
        for c in range(n):
            if f == c:
                emit(f"copy {B_at(f, c)},{free_at(f)}")
                continue
            emit(f"copy {a_tmp},{ZERO}")
            for r in range(m):
                emit(f"cmp {is_match},{pivot_at(c)},{ROW_at(r)},eq")
                emit(f"select {a_tmp},{is_match},{M_at(r, f)},{a_tmp}")
            emit(f"and {B_at(f, c)},{a_tmp},{free_at(f)}")

    # ---- phase 2c: rank of each column among free columns ----------------
    emit(f"copy {rank_at(0)},{ZERO}")
    for c in range(1, n):
        emit(f"add {rank_at(c)},{rank_at(c - 1)},{free_at(c - 1)}")

    # ---- phase 2d: gather basis vectors into Gray slots ------------------
    for j in range(G):
        for c in range(n):
            emit(f"copy {FB_at(j, c)},{ZERO}")
            for f in range(n):
                emit(f"cmp {is_match},{rank_at(f)},{JC_at(j)},eq")
                emit(f"and {is_match},{free_at(f)}")
                emit(f"and {a_tmp},{is_match},{B_at(f, c)}")
                emit(f"or {FB_at(j, c)},{a_tmp}")

    # ---- phase 3: Gray-code scan with weight-k capture -------------------
    def capture():
        emit(f"add {WSUM},{w_at(0)},{w_at(1)}")
        for c in range(2, n):
            emit(f"add {WSUM},{w_at(c)}")
        emit(f"cmp {OK},{WSUM},{K_VAL},eq")
        for c in range(n):
            emit(f"select {out_at(c)},{OK},{w_at(c)},{out_at(c)}")

    capture()                       # n_steps=0 == min-support GE
    if walk == "gray":
        for i in range(1, s + 1):
            j = (i & -i).bit_length() - 1   # reflected-Gray flip schedule
            for c in range(n):
                emit(f"xor {w_at(c)},{FB_at(j, c)}")
            capture()
    elif walk == "weight":
        if weight_cap is None:
            raise ValueError("walk='weight' requires weight_cap")
        if not 0 <= weight_cap <= G:
            raise ValueError(f"weight_cap must be in [0, {G}]")
        for flips in _weight_order_flips(G, weight_cap):
            for j in flips:
                for c in range(n):
                    emit(f"xor {w_at(c)},{FB_at(j, c)}")
            capture()
    else:
        raise ValueError(f"unknown walk {walk!r}")

    if joint:
        # ---- joint mode: label every test row from the captured mask ----
        for j in range(spec.m_test):
            emit(f"and {WSUM},{out_at(0)},{X_te_at(j, 0)}")
            for c in range(1, n):
                emit(f"and {a_tmp},{out_at(c)},{X_te_at(j, c)}")
                emit(f"xor {WSUM},{a_tmp}")
            emit(f"copy {pred_at(j)},{WSUM}")
        lines.append(",".join(str(pred_at(j)) for j in range(spec.m_test)))
    else:
        lines.append(",".join(str(out_at(c)) for c in range(n)))
    ir = "\n".join(lines)
    if len(lines) > op_cap:
        raise ValueError(
            f"scan IR has {len(lines) - 2:,} ops, over the {op_cap:,} cap"
        )
    return ir


__all__ = [
    "MASK32", "OP_CAP", "SUITE_VERSION", "DEV_SUITE_KEY",
    "DEV_SECRETS", "DEV_REPS", "FINAL_SECRETS", "FINAL_REPS",
    "MaskResult", "mask_suite", "evaluate_mask",
    "generate_scan", "generate_isd_mask", "generate_enum_mask",
    "_weight_order_flips",
]


if __name__ == "__main__":
    import os, time
    here = os.path.dirname(os.path.abspath(__file__))

    for label, ir_gen in [
        ("scan s=0 (min-support GE)", lambda: generate_scan(0)),
        ("scan s=4095", lambda: generate_scan(4095)),
        ("scan full (s=16383)", lambda: generate_scan()),
        ("ISD T=6", lambda: generate_isd_mask(6)),
        ("enum q=15000", lambda: generate_enum_mask(15000)),
    ]:
        t0 = time.time()
        ir = ir_gen()
        n_ops = len(ir.splitlines()) - 2
        res = evaluate_mask(ir)
        print(
            f"{label:>26}: ops={n_ops:>9,}  cost={res.cost:>11,}  "
            f"recovery={res.recovery:.3f}  [{time.time() - t0:.1f}s]"
        )

    with open(os.path.join(here, "submissions", "scan_full_mask32.ir"), "w") as fh:
        fh.write(generate_scan() + "\n")
    print("wrote submissions/scan_full_mask32.ir")
