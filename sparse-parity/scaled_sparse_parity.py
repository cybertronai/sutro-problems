"""Scaled sparse-parity tier: n=32, where enumeration is priced out.

Stage-1 implementation of the scaled accuracy-vs-energy benchmark:

    SCALED32:  n_bits=32, k_secret=5, m_train=18, m_test=256, cap=250k ops

Design targets (see docs/ report for the full derivation):

* C(32,5) = 201,376 candidate secrets and a null-space gap of
  n - m_train = 14, so under the 250,000-instruction cap BOTH brute-force
  families die: try-each-candidate needs ~26M instructions (a capped
  circuit can check <1% of candidates) and full solution-space
  (null-space) enumeration needs ~2^14 Gray-code steps at ~65 measured
  instructions each (~1.1M total, ~4x the cap).  Polynomial algorithms
  -- Gaussian elimination and
  its randomized information-set-decoding (ISD) restarts -- become the
  intended solution family.
* m_train=18 is just above the identifiability threshold
  log2(C) ~= 17.6: rejection sampling accepts ~46% of training draws.
* m_test=256 was chosen from the measured decode/predict energy split of
  the reference ISD circuit so that neither side of the joint
  train-plus-test program dominates the energy account.

Scoring is joint (one IR reads X_train, y_train, X_test and outputs all
test labels; energy is the whole program's static read cost) and
aggregate, as in ``approx_sparse_parity``.  Because C is large the suite
samples secrets instead of enumerating them:

* dev mode -- a fixed public suite key, cached: deterministic, fast
  iteration, but fully precomputable (never adjudicate on it).
* final mode (``suite_key=None``) -- a fresh hidden key from
  ``secrets.SystemRandom`` per scoring run: nothing about the secrets or
  instances is minable, at the price of ~+-1-2pp run-to-run noise in
  measured advantage.

Reference circuit: ``generate_isd(T, f)`` -- T Prange/ISD restarts.  Each
restart runs branchless GF(2) Gaussian elimination on a distinct
18-column subset (an "information set"), reads the zero-free-variable
solution, and verifies it (weight k AND consistent with all training
rows).  Unique identifiability makes any verified solution THE secret,
so restarts combine by OR.  Each restart independently succeeds with
probability prod_{i<k} (m-i)/(n-i) ~= 4.25%.  Prediction is the O(n)
mask predictor on the first f test rows.
"""
from __future__ import annotations

import math
import secrets as _secrets
from functools import lru_cache
from itertools import combinations
from random import Random
from typing import List, Sequence, Tuple

import numpy as np

import sparse_parity
from sparse_parity import Spec
import approx_sparse_parity as ap
from approx_sparse_parity import EvalResult

# --------------------------------------------------------------------------
# Tier spec
# --------------------------------------------------------------------------

SCALED32 = Spec(n_bits=32, k_secret=5, m_train=18, m_test=256)
# Per-tier instruction cap (part of the contract).  Matching
# sparse_parity._compile_ir, the cap counts every line of the IR text --
# the input-address and output-address lines included.
OP_CAP = 250_000

DEV_SUITE_KEY = "scaled-dev"
DEV_SECRETS, DEV_REPS = 128, 8        # 1,024 instances
FINAL_SECRETS, FINAL_REPS = 256, 8    # 2,048 instances

SUITE_VERSION = "scaled-sparse-parity-v1"


def isd_success_probability(spec: Spec = SCALED32) -> float:
    """Per-restart Prange success rate: P(secret inside a random
    information set) = prod_{i<k} (m_train - i) / (n_bits - i)."""
    p = 1.0
    for i in range(spec.k_secret):
        p *= (spec.m_train - i) / (spec.n_bits - i)
    return p


# --------------------------------------------------------------------------
# Suite: sampled secrets, packed-signature identifiability
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
    seen, out = set(), []
    while len(out) < n_secrets:
        s = tuple(sorted(rng.sample(range(spec.n_bits), spec.k_secret)))
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


@lru_cache(maxsize=4)
def _scaled_suite_cached(
    spec: Spec, n_secrets: int, repetitions: int, suite_key: str
) -> Tuple[np.ndarray, np.ndarray, Tuple[Tuple[Tuple[int, ...], int], ...]]:
    if repetitions < 2 or repetitions % 2:
        raise ValueError("repetitions must be even (complement-paired tests)")
    if spec.k_secret % 2 == 0:
        raise ValueError("complement pairing requires odd k_secret")

    cand = _candidate_columns(spec.n_bits, spec.k_secret)
    mask = (1 << spec.n_bits) - 1
    key_rng = ap._digest_rng(SUITE_VERSION, suite_key, "secrets", n_secrets)
    sampled = _sample_secrets(spec, n_secrets, key_rng)

    inputs_rows, y_rows, meta = [], [], []
    for secret in sampled:
        secret_cols = list(secret)
        for t in range(repetitions // 2):
            rng = ap._digest_rng(SUITE_VERSION, suite_key, "test", secret, t)
            base = [rng.getrandbits(spec.n_bits) for _ in range(spec.m_test)]
            test_blocks = [base, [r ^ mask for r in base]]
            for half, rows in enumerate(test_blocks):
                rep = 2 * t + half
                trng = ap._digest_rng(SUITE_VERSION, suite_key, "train", secret, rep)
                while True:
                    X = np.array(
                        [[trng.getrandbits(1) for _ in range(spec.n_bits)]
                         for _ in range(spec.m_train)],
                        dtype=np.int16,
                    )
                    y = np.bitwise_xor.reduce(X[:, secret_cols], axis=1)
                    if _unique_ksparse(X, y, cand):
                        break
                X_te = ap._rows_to_bits(rows, spec.n_bits)
                y_te = np.bitwise_xor.reduce(X_te[:, secret_cols], axis=1)
                inputs_rows.append(np.concatenate([X.ravel(), y, X_te.ravel()]))
                y_rows.append(y_te)
                meta.append((secret, rep))

    inputs = np.stack(inputs_rows).astype(np.int16)
    y_test = np.stack(y_rows).astype(np.int16)
    inputs.flags.writeable = False
    y_test.flags.writeable = False
    return inputs, y_test, tuple(meta)


def scaled_suite(
    *,
    spec: Spec = SCALED32,
    n_secrets: int | None = None,
    repetitions: int | None = None,
    suite_key: str | None = DEV_SUITE_KEY,
):
    """Sampled-secret evaluation suite.  ``suite_key=None`` draws a fresh
    hidden key from SystemRandom (final/adjudication mode: unminable,
    non-reproducible) and defaults to the larger FINAL_SECRETS sample;
    a string key gives the cached deterministic dev suite at DEV_SECRETS.
    Explicit ``n_secrets``/``repetitions`` override either default."""
    final = suite_key is None
    if n_secrets is None:
        n_secrets = FINAL_SECRETS if final else DEV_SECRETS
    if repetitions is None:
        repetitions = FINAL_REPS if final else DEV_REPS
    if final:
        # fresh hidden key: build outside the cache (the key is discarded,
        # so a cached entry would be unreachable and only evict dev suites)
        key = "fresh-" + _secrets.token_hex(16)
        return _scaled_suite_cached.__wrapped__(spec, n_secrets, repetitions, key)
    return _scaled_suite_cached(spec, n_secrets, repetitions, suite_key)


def evaluate_scaled(
    ir: str,
    *,
    spec: Spec = SCALED32,
    n_secrets: int | None = None,
    repetitions: int | None = None,
    suite_key: str | None = DEV_SUITE_KEY,
    engine: str = "vector",
) -> EvalResult:
    """Joint train+test scoring under the tier's instruction cap.

    Aggregates accuracy over all sampled secrets, repetitions and test
    rows; returns the IR's static read cost alongside.  Development
    defaults to the deterministic dev suite (DEV_SECRETS x DEV_REPS);
    ``suite_key=None`` runs adjudication on fresh hidden randomness at
    the larger FINAL_SECRETS x FINAL_REPS sample.
    """
    inputs, y_test, _ = scaled_suite(
        spec=spec, n_secrets=n_secrets, repetitions=repetitions,
        suite_key=suite_key,
    )
    expected_inputs = sparse_parity._n_inputs(spec)

    if engine == "vector":
        run, cost, n_inputs = ap._compile_vector(ir, OP_CAP)
        if n_inputs != expected_inputs:
            raise ValueError(f"IR declares {n_inputs} inputs; {expected_inputs} required")
        outputs = run(inputs)
    elif engine == "reference":
        simulate_fn, cost, n_inputs = sparse_parity._compile_ir(ir, OP_CAP)
        if n_inputs != expected_inputs:
            raise ValueError(f"IR declares {n_inputs} inputs; {expected_inputs} required")
        outputs = np.array(
            [simulate_fn(list(map(int, row))) for row in inputs], dtype=np.int16
        )
    else:
        raise ValueError(f"unknown engine {engine!r}")

    if outputs.shape[1] != spec.m_test:
        raise ValueError(
            f"IR produces {outputs.shape[1]} outputs; {spec.m_test} required"
        )
    n_correct = int((outputs == y_test).sum())
    n_labels = int(y_test.size)
    raw = n_correct / n_labels
    return EvalResult(
        cost=cost, raw_accuracy=raw, advantage=2.0 * raw - 1.0,
        n_instances=int(y_test.shape[0]), n_labels=n_labels,
    )


# --------------------------------------------------------------------------
# Reference circuit: T-restart ISD (Prange) + mask predictor
# --------------------------------------------------------------------------

def _isd_subsets(spec: Spec, n_restarts: int) -> List[List[int]]:
    """Deterministic rotating information sets: restart t uses columns
    [(stride*t + j) mod n for j < m_train].  stride is coprime to n so
    consecutive subsets overlap as little as possible."""
    n, m = spec.n_bits, spec.m_train
    stride = 7 if math.gcd(7, n) == 1 else 5
    return [[(stride * t + j) % n for j in range(m)] for t in range(n_restarts)]


def generate_isd(
    n_restarts: int = 1,
    n_outputs: int | None = None,
    *,
    spec: Spec = SCALED32,
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
    """
    n, m, k, m_test = spec.n_bits, spec.m_train, spec.k_secret, spec.m_test
    f = m_test if n_outputs is None else n_outputs
    if not 0 <= f <= m_test:
        raise ValueError(f"n_outputs must be in [0, {m_test}]")
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")
    n_sub = m          # information-set size = m_train (square system)
    n_aug = n_sub + 1
    subsets = _isd_subsets(spec, n_restarts)

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

    inputs = (
        [X_tr_at(i, c) for i in range(m) for c in range(n)]
        + [y_tr_at(i) for i in range(m)]
        + [X_te_at(j, c) for j in range(m_test) for c in range(n)]
    )
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

    # ---- mask predictor on the first f rows -----------------------------
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
    if len(lines) > OP_CAP:
        raise ValueError(
            f"generated ISD IR has {len(lines) - 2:,} ops, over the {OP_CAP:,} cap"
        )
    return ir


__all__ = [
    "SCALED32", "OP_CAP", "SUITE_VERSION",
    "DEV_SUITE_KEY", "DEV_SECRETS", "DEV_REPS",
    "FINAL_SECRETS", "FINAL_REPS",
    "isd_success_probability", "scaled_suite", "evaluate_scaled",
    "generate_isd",
]


if __name__ == "__main__":
    import os, time
    here = os.path.dirname(os.path.abspath(__file__))

    for T in (1, 3, 6):
        ir = generate_isd(T)
        n_ops = len(ir.splitlines()) - 2
        t0 = time.time()
        res = evaluate_scaled(ir)
        dt = time.time() - t0
        pred = 1 - (1 - isd_success_probability()) ** T
        print(
            f"ISD T={T}: ops={n_ops:>7,}  cost={res.cost:>9,}  "
            f"advantage={res.advantage:+.4f} (analytic ~{pred:.4f})  "
            f"[{dt:.1f}s]"
        )

    ir = generate_isd(6)
    path = os.path.join(here, "submissions", "isd6_scaled32.ir")
    with open(path, "w") as fh:
        fh.write(ir + "\n")
    print(f"wrote {path}")
