"""Approximate sparse-parity: a deterministic accuracy-vs-energy benchmark.

Extends the exact-recovery benchmark in ``sparse_parity.py`` with a graded
target: a submitted IR program no longer has to label every test row
correctly -- it earns a point on an accuracy-vs-energy curve instead.

Problem size (see README for the derivation):

    n_bits=12, k_secret=3, m_train=8, m_test=32

* C(12,3) = 220 candidate secrets, and log2(220) ~= 7.8, so the 8 training
  labels sit almost exactly at the information-theoretic threshold.
* 220 < 256, so a candidate's identity fits an 8-bit training signature --
  the largest 3-sparse problem for which that is true.
* m_train=8 < n_bits=12 means GF(2) linear algebra alone cannot solve the
  task: a random interpolant scores ~53%, minimum-support Gaussian
  elimination ~64% -- the higher targets require sparse search.
* 32 test rows keep decoder energy and prediction energy comparable for an
  optimized solver, so partial decoding and partial prediction both matter.

Scoring is deterministic and stratified: every one of the 220 secrets is
evaluated with ``repetitions`` independent training instances derived from
SHA-256 of (suite version, suite key, secret, repetition).  Each training
set is rejection-sampled until the secret is uniquely identifiable, so the
maximum achievable accuracy is always 100% and the benchmark measures
approximation quality, not dataset ambiguity.  Test rows come in complement
pairs split across consecutive repetitions: because k is odd, a row and its
complement always carry opposite labels, so any constant guess scores
exactly 50% and accuracy aggregates are balanced by construction.

Accuracy is aggregated over all secrets, repetitions, and test rows (never
thresholded per instance) and reported both raw and as the normalized
advantage over random guessing::

    advantage = 2 * raw_accuracy - 1

Energy is the same static read cost (simplified Dally model, v3
instruction set) returned by ``sparse_parity._compile_ir``.

Requires numpy (the batched evaluator executes the IR over every suite
instance at once; a scalar ``engine="reference"`` path reuses the pure-
Python simulator for cross-checking).
"""
from __future__ import annotations

import hashlib
import math
from collections import namedtuple
from functools import lru_cache
from itertools import combinations
from random import Random
from typing import Callable, List, Sequence, Tuple

import numpy as np

import sparse_parity
from sparse_parity import Spec

# --------------------------------------------------------------------------
# Problem spec + suite parameters
# --------------------------------------------------------------------------

APPROX = Spec(n_bits=12, k_secret=3, m_train=8, m_test=32)

SUITE_VERSION = "approx-sparse-parity-v1"
PUBLIC_SUITE_KEY = "public"

DEV_REPETITIONS = 8      # fast iteration:      220 * 8   =  1,760 instances
FINAL_REPETITIONS = 32   # leaderboard scoring: 220 * 32  =  7,040 instances
FULL_AUDIT_REPETITIONS = 128  # 128 * 32 = 4096 = every 12-bit test row

EvalResult = namedtuple(
    "EvalResult", "cost raw_accuracy advantage n_instances n_labels"
)


def _secrets(spec: Spec) -> List[Tuple[int, ...]]:
    return list(combinations(range(spec.n_bits), spec.k_secret))


def n_secrets(spec: Spec = APPROX) -> int:
    return math.comb(spec.n_bits, spec.k_secret)


# --------------------------------------------------------------------------
# Deterministic suite construction
# --------------------------------------------------------------------------

def _digest_rng(*parts) -> Random:
    """Random stream derived from SHA-256 of the joined parts."""
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode()).digest()
    return Random(int.from_bytes(digest, "big"))


def _rows_to_bits(rows: Sequence[int], n_bits: int) -> np.ndarray:
    """(len(rows), n_bits) 0/1 matrix; bit j of the row integer is column j."""
    r = np.asarray(rows, dtype=np.int64)[:, None]
    return ((r >> np.arange(n_bits)) & 1).astype(np.int16)


def _train_for_secret(
    spec: Spec, secret: Tuple[int, ...], rng: Random, cand: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Rejection-sample a training set on which ``secret`` is the unique
    matching k-subset.  Returns (X_train (m,n), y_train (m,))."""
    secret_cols = list(secret)
    while True:
        X = np.array(
            [[rng.getrandbits(1) for _ in range(spec.n_bits)]
             for _ in range(spec.m_train)],
            dtype=np.int16,
        )
        y = np.bitwise_xor.reduce(X[:, secret_cols], axis=1)
        cols = X.T  # (n, m)
        par = cols[cand[:, 0]]
        for j in range(1, spec.k_secret):
            par = par ^ cols[cand[:, j]]
        if int(((par == y).all(axis=1)).sum()) == 1:
            return X, y


def _test_blocks_for_secret(
    spec: Spec,
    secret: Tuple[int, ...],
    repetitions: int,
    suite_key: str,
    full_cube: bool,
) -> List[List[int]]:
    """Test rows (as integers) for each repetition of one secret.

    Sampled mode: repetition pairs (2t, 2t+1) share a draw -- 2t gets
    ``m_test`` distinct random rows, 2t+1 gets their bitwise complements in
    the same order, so labels are exactly balanced across each pair.
    Full-cube mode: a deterministic permutation of all 2**n rows split into
    blocks of ``m_test`` (every possible test row appears exactly once).
    """
    n_rows = 1 << spec.n_bits
    if full_cube:
        perm = list(range(n_rows))
        _digest_rng(SUITE_VERSION, suite_key, "cube", secret).shuffle(perm)
        return [
            perm[rep * spec.m_test:(rep + 1) * spec.m_test]
            for rep in range(repetitions)
        ]
    blocks: List[List[int]] = []
    mask = n_rows - 1
    for t in range(repetitions // 2):
        rng = _digest_rng(SUITE_VERSION, suite_key, "test", secret, t)
        base = rng.sample(range(n_rows), spec.m_test)
        blocks.append(base)
        blocks.append([r ^ mask for r in base])
    return blocks


@lru_cache(maxsize=8)
def _suite_cached(
    spec: Spec, repetitions: int, suite_key: str, full_cube: bool
) -> Tuple[np.ndarray, np.ndarray, Tuple[Tuple[Tuple[int, ...], int], ...]]:
    if full_cube:
        if (1 << spec.n_bits) % spec.m_test:
            raise ValueError("m_test must divide 2**n_bits for full-cube audit")
        repetitions = (1 << spec.n_bits) // spec.m_test
    else:
        if repetitions < 2 or repetitions % 2:
            raise ValueError("repetitions must be even (complement-paired tests)")
        if spec.k_secret % 2 == 0:
            raise ValueError(
                "complement pairing balances labels only for odd k_secret; "
                "use full_cube=True for even k"
            )

    cand = np.array(_secrets(spec), dtype=np.int64)
    inputs_rows, y_rows, meta = [], [], []
    for secret in _secrets(spec):
        blocks = _test_blocks_for_secret(
            spec, secret, repetitions, suite_key, full_cube
        )
        for rep in range(repetitions):
            rng = _digest_rng(SUITE_VERSION, suite_key, "train", secret, rep)
            X_tr, y_tr = _train_for_secret(spec, secret, rng, cand)
            X_te = _rows_to_bits(blocks[rep], spec.n_bits)
            y_te = np.bitwise_xor.reduce(X_te[:, list(secret)], axis=1)
            inputs_rows.append(
                np.concatenate([X_tr.ravel(), y_tr, X_te.ravel()])
            )
            y_rows.append(y_te)
            meta.append((secret, rep))

    inputs = np.stack(inputs_rows).astype(np.int16)
    y_test = np.stack(y_rows).astype(np.int16)
    inputs.flags.writeable = False
    y_test.flags.writeable = False
    return inputs, y_test, tuple(meta)


def suite(
    *,
    spec: Spec = APPROX,
    repetitions: int = DEV_REPETITIONS,
    suite_key: str = PUBLIC_SUITE_KEY,
    full_cube: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Tuple[Tuple[Tuple[int, ...], int], ...]]:
    """Deterministic evaluation suite.

    Returns ``(inputs, y_test, meta)`` where ``inputs`` is a
    ``(n_instances, n_bits*m_train + m_train + n_bits*m_test)`` matrix of IR
    inputs (one row per instance), ``y_test`` the matching
    ``(n_instances, m_test)`` labels, and ``meta`` the ``(secret,
    repetition)`` behind each row.  ``full_cube=True`` always evaluates the
    whole test cube (``repetitions`` is ignored -- normalized here so the
    cache holds one copy per key).
    """
    if full_cube:
        if (1 << spec.n_bits) % spec.m_test:
            raise ValueError("m_test must divide 2**n_bits for full-cube audit")
        repetitions = (1 << spec.n_bits) // spec.m_test
    return _suite_cached(spec, repetitions, suite_key, full_cube)


# --------------------------------------------------------------------------
# Batched IR evaluation
# --------------------------------------------------------------------------

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
    """Vectorized ``sparse_parity._to_signed_8bit``."""
    return ((v & 0xFF) ^ 0x80) - 0x80


def _compile_vector(ir: str) -> Tuple[Callable[[np.ndarray], np.ndarray], int, int]:
    """Compile IR into a batched executor: (n_instances, n_inputs) int16 ->
    (n_instances, n_outputs) int16.  Validation and the static read cost are
    delegated to ``sparse_parity._compile_ir`` so both engines agree."""
    _, static_cost, n_inputs = sparse_parity._compile_ir(ir)

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
            ops.append(("set", dest, sparse_parity._to_signed_8bit(int(raw[1])), None, None))
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
# Scorer
# --------------------------------------------------------------------------

def evaluate(
    ir: str,
    *,
    spec: Spec = APPROX,
    repetitions: int = DEV_REPETITIONS,
    suite_key: str = PUBLIC_SUITE_KEY,
    full_cube: bool = False,
    engine: str = "vector",
) -> EvalResult:
    """Run ``ir`` over the deterministic suite; return cost and accuracy.

    Aggregates correctness over every secret, repetition, and test row --
    there is no per-instance threshold.  ``engine="reference"`` uses the
    scalar simulator from ``sparse_parity`` (slow; for cross-checking).
    """
    inputs, y_test, _ = suite(
        spec=spec, repetitions=repetitions, suite_key=suite_key,
        full_cube=full_cube,
    )
    expected_inputs = sparse_parity._n_inputs(spec)

    if engine == "vector":
        run, cost, n_inputs = _compile_vector(ir)
        if n_inputs != expected_inputs:
            raise ValueError(f"IR declares {n_inputs} inputs; {expected_inputs} required")
        outputs = run(inputs)
    elif engine == "reference":
        simulate_fn, cost, n_inputs = sparse_parity._compile_ir(ir)
        if n_inputs != expected_inputs:
            raise ValueError(f"IR declares {n_inputs} inputs; {expected_inputs} required")
        outputs = np.array([simulate_fn(list(map(int, row))) for row in inputs],
                           dtype=np.int16)
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
        cost=cost,
        raw_accuracy=raw,
        advantage=2.0 * raw - 1.0,
        n_instances=int(y_test.shape[0]),
        n_labels=n_labels,
    )


def score_approx(
    ir: str,
    min_advantage: float,
    *,
    repetitions: int = FINAL_REPETITIONS,
    suite_key: str = PUBLIC_SUITE_KEY,
    spec: Spec = APPROX,
) -> int:
    """Return the static read cost if the IR's aggregate normalized
    advantage meets ``min_advantage``; raise ``ValueError`` otherwise.

    The default public ``suite_key`` is for development: it is fully
    precomputable (labels included), so a submission built against it can
    mine a few points of spurious measured advantage.  Official rankings
    near a threshold should be confirmed by re-scoring with a held-out
    ``suite_key`` (same code, private key) and/or the full-cube audit.
    """
    res = evaluate(ir, spec=spec, repetitions=repetitions, suite_key=suite_key)
    if res.advantage + 1e-12 < min_advantage:
        raise ValueError(
            f"advantage {res.advantage:.4f} (raw accuracy "
            f"{res.raw_accuracy:.4f}) below target {min_advantage:.4f}"
        )
    return res.cost


def score_approx_t25(ir: str) -> int:
    """Leaderboard target: advantage >= 0.25 (raw accuracy >= 62.5%)."""
    return score_approx(ir, 0.25)


def score_approx_t50(ir: str) -> int:
    """Leaderboard target: advantage >= 0.50 (raw accuracy >= 75%)."""
    return score_approx(ir, 0.50)


def score_approx_t90(ir: str) -> int:
    """Leaderboard target: advantage >= 0.90 (raw accuracy >= 95%)."""
    return score_approx(ir, 0.90)


# --------------------------------------------------------------------------
# Parameterized approximate baseline
# --------------------------------------------------------------------------

def generate_approx_baseline(
    n_candidates: int | None = None,
    n_outputs: int | None = None,
    *,
    spec: Spec = APPROX,
) -> str:
    """Try-each-candidate baseline that searches only the first
    ``n_candidates`` of the C(n,k) secrets and computes only the first
    ``n_outputs`` test predictions (remaining outputs default to 0).

    Because every suite instance is uniquely identifiable, decoding
    succeeds iff the secret is among the searched candidates, so with
    q candidates and f computed outputs the aggregate advantage is exactly
    (q / C(n,k)) * (f / m_test) while energy shrinks with both knobs.
    """
    C = math.comb(spec.n_bits, spec.k_secret)
    q = C if n_candidates is None else n_candidates
    f = spec.m_test if n_outputs is None else n_outputs
    if not 0 <= q <= C:
        raise ValueError(f"n_candidates must be in [0, {C}]")
    if not 0 <= f <= spec.m_test:
        raise ValueError(f"n_outputs must be in [0, {spec.m_test}]")
    candidates = _secrets(spec)[:q]

    pred_base    = 1
    X_tr_base    = pred_base + spec.m_test
    y_tr_base    = X_tr_base + spec.n_bits * spec.m_train
    X_te_base    = y_tr_base + spec.m_train
    ONE          = X_te_base + spec.n_bits * spec.m_test
    TMP          = ONE + 1
    PARITY       = TMP + 1
    matched_base = PARITY + 1
    ind_base     = matched_base + q * spec.m_train
    PREDT        = ind_base + q
    term_base    = PREDT + 1

    def pred_at(j): return pred_base + j
    def X_tr_at(i, c): return X_tr_base + i * spec.n_bits + c
    def y_tr_at(i): return y_tr_base + i
    def X_te_at(j, c): return X_te_base + j * spec.n_bits + c
    def matched_at(t, i): return matched_base + t * spec.m_train + i
    def ind_at(t): return ind_base + t
    def term_at(t): return term_base + t

    inputs = (
        [X_tr_at(i, c) for i in range(spec.m_train) for c in range(spec.n_bits)]
        + [y_tr_at(i) for i in range(spec.m_train)]
        + [X_te_at(j, c) for j in range(spec.m_test) for c in range(spec.n_bits)]
    )
    outputs = [pred_at(j) for j in range(spec.m_test)]

    lines = [",".join(map(str, inputs))]

    def emit(op: str, *args: int) -> None:
        lines.append(f"{op} " + ",".join(map(str, args)))

    if q:
        emit("set", ONE, 1)

    # --- decoding: ind[t] = 1 iff candidate t explains every train label --
    for t, T in enumerate(candidates):
        for i in range(spec.m_train):
            emit("xor", TMP, y_tr_at(i), X_tr_at(i, T[0]))
            for k in range(1, spec.k_secret - 1):
                emit("xor", TMP, X_tr_at(i, T[k]))
            emit("xor", PARITY, TMP, X_tr_at(i, T[-1]))
            emit("xor", matched_at(t, i), PARITY, ONE)
        emit("and", ind_at(t), matched_at(t, 0), matched_at(t, 1))
        for i in range(2, spec.m_train):
            emit("and", ind_at(t), matched_at(t, i))

    # --- predictions: pred[j] = OR_t (ind[t] AND parity_t(x_j)) ----------
    for j in range(spec.m_test):
        if j >= f or not q:
            emit("set", pred_at(j), 0)
            continue
        for t, T in enumerate(candidates):
            emit("xor", PREDT, X_te_at(j, T[0]), X_te_at(j, T[1]))
            for k in range(2, spec.k_secret):
                emit("xor", PREDT, X_te_at(j, T[k]))
            emit("and", term_at(t), ind_at(t), PREDT)
        if q == 1:
            emit("copy", pred_at(j), term_at(0))
        else:
            emit("or", pred_at(j), term_at(0), term_at(1))
            for t in range(2, q):
                emit("or", pred_at(j), term_at(t))

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def generate_mask_baseline(
    n_candidates: int | None = None,
    n_outputs: int | None = None,
    *,
    spec: Spec = APPROX,
) -> str:
    """Two-phase approximate baseline: decode once into an n-bit secret
    mask, then predict each output in O(n) instead of O(candidates).

    Phase 1 matches the first ``n_candidates`` secrets against the training
    labels (as in ``generate_approx_baseline``) and ORs the resulting
    one-hot indicators into per-column mask bits.  Phase 2 computes each of
    the first ``n_outputs`` predictions as XOR_c(mask_c AND x_c) -- its cost
    no longer grows with the number of candidates searched, which shifts
    the whole accuracy-energy curve left of the try-each-candidate family.
    The aggregate advantage is the same exact (q/C) * (f/m_test).
    """
    C = math.comb(spec.n_bits, spec.k_secret)
    q = C if n_candidates is None else n_candidates
    f = spec.m_test if n_outputs is None else n_outputs
    if not 0 <= q <= C:
        raise ValueError(f"n_candidates must be in [0, {C}]")
    if not 0 <= f <= spec.m_test:
        raise ValueError(f"n_outputs must be in [0, {spec.m_test}]")
    candidates = _secrets(spec)[:q]

    pred_base    = 1
    X_tr_base    = pred_base + spec.m_test
    y_tr_base    = X_tr_base + spec.n_bits * spec.m_train
    X_te_base    = y_tr_base + spec.m_train
    ONE          = X_te_base + spec.n_bits * spec.m_test
    TMP          = ONE + 1
    PARITY       = TMP + 1
    ACC          = PARITY + 1
    mask_base    = ACC + 1
    matched_base = mask_base + spec.n_bits
    ind_base     = matched_base + q * spec.m_train

    def pred_at(j): return pred_base + j
    def X_tr_at(i, c): return X_tr_base + i * spec.n_bits + c
    def y_tr_at(i): return y_tr_base + i
    def X_te_at(j, c): return X_te_base + j * spec.n_bits + c
    def mask_at(c): return mask_base + c
    def matched_at(t, i): return matched_base + t * spec.m_train + i
    def ind_at(t): return ind_base + t

    inputs = (
        [X_tr_at(i, c) for i in range(spec.m_train) for c in range(spec.n_bits)]
        + [y_tr_at(i) for i in range(spec.m_train)]
        + [X_te_at(j, c) for j in range(spec.m_test) for c in range(spec.n_bits)]
    )
    outputs = [pred_at(j) for j in range(spec.m_test)]

    lines = [",".join(map(str, inputs))]

    def emit(op: str, *args: int) -> None:
        lines.append(f"{op} " + ",".join(map(str, args)))

    if q:
        emit("set", ONE, 1)

    # --- phase 1a: ind[t] = 1 iff candidate t explains the training set --
    for t, T in enumerate(candidates):
        for i in range(spec.m_train):
            emit("xor", TMP, y_tr_at(i), X_tr_at(i, T[0]))
            for k in range(1, spec.k_secret - 1):
                emit("xor", TMP, X_tr_at(i, T[k]))
            emit("xor", PARITY, TMP, X_tr_at(i, T[-1]))
            emit("xor", matched_at(t, i), PARITY, ONE)
        emit("and", ind_at(t), matched_at(t, 0), matched_at(t, 1))
        for i in range(2, spec.m_train):
            emit("and", ind_at(t), matched_at(t, i))

    # --- phase 1b: mask[c] = OR of ind[t] over candidates containing c ---
    if q or f:
        for c in range(spec.n_bits):
            ts = [t for t, T in enumerate(candidates) if c in T]
            if not ts:
                emit("set", mask_at(c), 0)
            elif len(ts) == 1:
                emit("copy", mask_at(c), ind_at(ts[0]))
            else:
                emit("or", mask_at(c), ind_at(ts[0]), ind_at(ts[1]))
                for t in ts[2:]:
                    emit("or", mask_at(c), ind_at(t))

    # --- phase 2: pred[j] = XOR_c (mask[c] AND x_j[c]) -------------------
    for j in range(spec.m_test):
        if j >= f:
            emit("set", pred_at(j), 0)
            continue
        emit("and", ACC, mask_at(0), X_te_at(j, 0))
        for c in range(1, spec.n_bits):
            emit("and", TMP, mask_at(c), X_te_at(j, c))
            emit("xor", ACC, TMP)
        emit("copy", pred_at(j), ACC)

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


__all__ = [
    "APPROX", "EvalResult",
    "SUITE_VERSION", "PUBLIC_SUITE_KEY",
    "DEV_REPETITIONS", "FINAL_REPETITIONS", "FULL_AUDIT_REPETITIONS",
    "n_secrets", "suite", "evaluate",
    "score_approx", "score_approx_t25", "score_approx_t50", "score_approx_t90",
    "generate_approx_baseline", "generate_mask_baseline",
]


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    ir_dir = os.path.join(here, "submissions")
    os.makedirs(ir_dir, exist_ok=True)

    artifacts = [
        ("approx_baseline_full.ir", generate_approx_baseline()),
        ("mask_baseline_full.ir", generate_mask_baseline()),
    ]
    for name, ir in artifacts:
        res = evaluate(ir)
        path = os.path.join(ir_dir, name)
        with open(path, "w") as fh:
            fh.write(ir + "\n")
        n_ops = len(ir.splitlines()) - 2
        print(
            f"  {name:<24} cost={res.cost:>9,}  ops={n_ops:>6,}  "
            f"advantage={res.advantage:.3f}  -> {path}"
        )
