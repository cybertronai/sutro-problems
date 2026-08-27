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
"""
from __future__ import annotations

import math
import secrets as _secrets
from functools import lru_cache
from typing import List, Tuple

import numpy as np

import sparse_parity
from sparse_parity import Spec
import approx_sparse_parity as ap
import scaled_sparse_parity as sp

# --------------------------------------------------------------------------
# Tier spec
# --------------------------------------------------------------------------

MASK32 = Spec(n_bits=32, k_secret=5, m_train=18, m_test=0)
# Cap counts every line of the IR text (input/output lines included),
# matching sparse_parity._compile_ir.
OP_CAP = 2_000_000

SUITE_VERSION = "mask-sparse-parity-v1"
DEV_SUITE_KEY = "mask-dev"
DEV_SECRETS, DEV_REPS = 128, 8        # 1,024 instances
FINAL_SECRETS, FINAL_REPS = 256, 8    # 2,048 instances

from collections import namedtuple

MaskResult = namedtuple("MaskResult", "cost recovery n_instances")


def _n_inputs(spec: Spec) -> int:
    return spec.n_bits * spec.m_train + spec.m_train


# --------------------------------------------------------------------------
# Suite: training instances only
# --------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _mask_suite_cached(
    spec: Spec, n_secrets: int, repetitions: int, suite_key: str
) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    cand = sp._candidate_columns(spec.n_bits, spec.k_secret)
    key_rng = ap._digest_rng(SUITE_VERSION, suite_key, "secrets", n_secrets)
    sampled = sp._sample_secrets(spec, n_secrets, key_rng)

    inputs_rows, mask_rows, meta = [], [], []
    for secret in sampled:
        secret_cols = list(secret)
        mask = np.zeros(spec.n_bits, dtype=np.int16)
        mask[secret_cols] = 1
        for rep in range(repetitions):
            rng = ap._digest_rng(SUITE_VERSION, suite_key, "train", secret, rep)
            while True:
                X = np.array(
                    [[rng.getrandbits(1) for _ in range(spec.n_bits)]
                     for _ in range(spec.m_train)],
                    dtype=np.int16,
                )
                y = np.bitwise_xor.reduce(X[:, secret_cols], axis=1)
                if sp._unique_ksparse(X, y, cand):
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
        run, cost, n_in = ap._compile_vector(ir, OP_CAP)
        if n_in != expected_inputs:
            raise ValueError(f"IR declares {n_in} inputs; {expected_inputs} required")
        outputs = run(inputs)
    elif engine == "reference":
        simulate_fn, cost, n_in = sparse_parity._compile_ir(ir, OP_CAP)
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

def generate_isd_mask(n_restarts: int = 1, *, spec: Spec = MASK32) -> str:
    """ISD restart family on the mask task (no prediction phase)."""
    joint_spec = Spec(spec.n_bits, spec.k_secret, spec.m_train, 0)
    return sp.generate_isd(
        n_restarts, spec=joint_spec, mask_output=True, op_cap=OP_CAP
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


def generate_scan(
    n_steps: int | None = None,
    *,
    spec: Spec = MASK32,
    joint: bool = False,
    op_cap: int = OP_CAP,
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
    for i in range(1, s + 1):
        j = (i & -i).bit_length() - 1   # reflected-Gray flip schedule
        for c in range(n):
            emit(f"xor {w_at(c)},{FB_at(j, c)}")
        capture()

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
