"""GF(2) Gaussian-elimination submission for the small sparse-parity
instance (n=3, k=2, m_train=4, m_test=32).

Builds the augmented matrix ``[A | b]`` directly from the (X_train,
y_train) inputs in scratchpad memory, then performs branchless GF(2)
row-reduction with full pivot tracking using v3 ``cmp`` + ``select``.
The pivot for column *c* is the first **unused** row containing a 1 in
column *c* (rather than the first row with index ≥ *c*) — this is the
correct GF(2) pivoting rule and handles the case where a free
column comes before a pivot column.

When ``rank(A) = n`` (about 70 % of identifiable seeds for this size),
the augmented column at the pivot rows is exactly the secret indicator.
When ``rank = n − 1`` (about 30 %), one column is free; the affine
solution space contains two vectors and the weight-``k`` one is the
true secret. The branchless weight-fix flips the free variable — *and*
every pivot row whose RREF entry in the free column is 1 — when
``weight(default) ≠ k``.

For the rare ``rank = n − 2 = 1`` case (about 0.6 % of seeds), two
columns are free and the single-flip fix isn't sufficient. The IR
verifies ``A·s_GE = b`` over the training rows and, on mismatch, falls
back to brute-force enumeration of the ``C(3, 2) = 3`` weight-``k``
candidates (the unique consistent one is selected via OR-combination
of per-candidate match indicators).

Each test row is then labeled as ``XOR_c (s[c] AND X_test[j][c])``.

Run ``python ge_small.py`` to regenerate ``ge_small.ir`` and print
its cost.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import sparse_parity  # noqa: E402
from sparse_parity import SMALL, score_small  # noqa: E402


def _generate_ge(spec) -> str:
    """Branchless GF(2) GE + brute-force fallback IR for any *spec*."""
    n = spec.n_bits
    m = spec.m_train
    n_aug = n + 1
    m_test = spec.m_test

    # ----- memory layout ---------------------------------------------------
    pred_base = 1
    X_tr_base = pred_base + m_test                 # 33
    y_tr_base = X_tr_base + n * m                  # 45
    X_te_base = y_tr_base + m                      # 49
    M_base    = X_te_base + n * m_test             # 145  (m × n_aug)
    PR_base   = M_base + m * n_aug                 # 161  pivot-row buffer (n_aug)
    scratch   = PR_base + n_aug                    # 165

    ZERO          = scratch + 0
    ONE           = scratch + 1
    M_VAL         = scratch + 2                    # sentinel = m
    K_VAL         = scratch + 3                    # = k_secret
    ROW_BASE      = scratch + 4                    # ROW_r at base + r
    used_base     = ROW_BASE + m                   # used[r]   (m cells)
    pivot_for_col = used_base + m                  # pivot_row_for_col[c] (n cells)
    is_free_base  = pivot_for_col + n              # is_free[c] (n cells)
    s_base        = is_free_base + n               # s[c]       (n cells)

    # work registers (reused)
    s_brute_base  = s_base + n                     # brute-force candidate per col
    s_final_base  = s_brute_base + n               # selected s used for predictions
    pivot_idx     = s_final_base + n
    found         = pivot_idx + 1
    bit           = found + 1
    not_used      = bit + 1
    eligible      = not_used + 1
    is_first      = eligible + 1
    is_match      = is_first + 1
    is_other      = is_match + 1
    do_xor        = is_other + 1
    a_tmp         = do_xor + 1
    b_tmp         = a_tmp + 1
    pred_tmp      = b_tmp + 1
    weight_addr   = pred_tmp + 1
    need_flip     = weight_addr + 1
    flip_bit      = need_flip + 1
    err_addr      = flip_bit + 1
    parity_t      = err_addr + 1
    matched_t     = parity_t + 1
    ind_t_base    = matched_t + 1                  # ind_T per candidate

    pred_at  = lambda j: pred_base + j
    X_tr_at  = lambda i, c: X_tr_base + i * n + c
    y_tr_at  = lambda i: y_tr_base + i
    X_te_at  = lambda j, c: X_te_base + j * n + c
    M_at     = lambda i, j: M_base + i * n_aug + j
    PR_at    = lambda j: PR_base + j
    ROW_AT   = lambda r: ROW_BASE + r
    used_at  = lambda r: used_base + r
    pivot_at = lambda c: pivot_for_col + c
    is_free_at = lambda c: is_free_base + c
    s_at     = lambda c: s_base + c
    s_brute_at = lambda c: s_brute_base + c
    s_final_at = lambda c: s_final_base + c
    ind_t_at = lambda t_idx: ind_t_base + t_idx

    inputs = (
        [X_tr_at(i, c) for i in range(m) for c in range(n)]
        + [y_tr_at(i) for i in range(m)]
        + [X_te_at(j, c) for j in range(m_test) for c in range(n)]
    )
    outputs = [pred_at(j) for j in range(m_test)]

    lines = [",".join(map(str, inputs))]

    # ----- constants -------------------------------------------------------
    lines.append(f"set {ZERO},0")
    lines.append(f"set {ONE},1")
    lines.append(f"set {M_VAL},{m}")
    lines.append(f"set {K_VAL},{spec.k_secret}")
    for r in range(m):
        lines.append(f"set {ROW_AT(r)},{r}")
        lines.append(f"set {used_at(r)},0")

    # ----- copy [X_train | y_train] into augmented matrix M ----------------
    for i in range(m):
        for c in range(n):
            lines.append(f"copy {M_at(i, c)},{X_tr_at(i, c)}")
        lines.append(f"copy {M_at(i, n)},{y_tr_at(i)}")

    # ----- GF(2) RREF with full pivot tracking -----------------------------
    for col in range(n):
        # 1. find first unused row r with M[r][col] = 1
        lines.append(f"copy {pivot_idx},{M_VAL}")
        lines.append(f"copy {found},{ZERO}")
        for r in range(m):
            lines.append(f"copy {bit},{M_at(r, col)}")
            # not_used = 1 - used[r] = select(used[r], 0, 1)
            lines.append(f"select {not_used},{used_at(r)},{ZERO},{ONE}")
            lines.append(f"and {eligible},{bit},{not_used}")
            lines.append(f"select {is_first},{found},{ZERO},{eligible}")
            lines.append(
                f"select {pivot_idx},{is_first},{ROW_AT(r)},{pivot_idx}")
            # used[r] |= is_first (only the first eligible row is marked used)
            lines.append(f"or {used_at(r)},{is_first}")
            lines.append(f"or {found},{eligible}")
        # save pivot row index for this col (sentinel m if free)
        lines.append(f"copy {pivot_at(col)},{pivot_idx}")
        # is_free[col] = 1 - found
        lines.append(f"select {is_free_at(col)},{found},{ZERO},{ONE}")

        # 2. build pivot-row buffer PR[j] = M[pivot_idx][j] (branchless via
        #    a select chain over all candidate rows).
        for j in range(n_aug):
            lines.append(f"copy {PR_at(j)},{ZERO}")
            for r in range(m):
                lines.append(f"cmp {is_match},{pivot_idx},{ROW_AT(r)},eq")
                lines.append(
                    f"select {PR_at(j)},{is_match},{M_at(r, j)},{PR_at(j)}")

        # 3. eliminate col from every row r != pivot_idx that has M[r][col]=1
        for r in range(m):
            lines.append(f"cmp {is_match},{pivot_idx},{ROW_AT(r)},eq")
            lines.append(f"select {is_other},{is_match},{ZERO},{ONE}")
            lines.append(f"copy {bit},{M_at(r, col)}")
            lines.append(f"and {do_xor},{is_other},{bit}")
            for j in range(n_aug):
                lines.append(f"copy {a_tmp},{M_at(r, j)}")
                lines.append(f"xor {b_tmp},{M_at(r, j)},{PR_at(j)}")
                lines.append(
                    f"select {M_at(r, j)},{do_xor},{b_tmp},{a_tmp}")

    # ----- read out default solution s_0 from RREF -------------------------
    # For each col c, s_0[c] = M[pivot_for_col[c]][n] if c is a pivot col
    # else 0. Read M[pivot_for_col[c]][n] dynamically via select chain.
    for c in range(n):
        lines.append(f"copy {s_at(c)},{ZERO}")
        for r in range(m):
            lines.append(f"cmp {is_match},{pivot_at(c)},{ROW_AT(r)},eq")
            lines.append(
                f"select {s_at(c)},{is_match},{M_at(r, n)},{s_at(c)}")

    # ----- weight-fix for rank-deficient case ------------------------------
    # weight(s_0); if != k_secret, flip the free column. When the free
    # column f is flipped from 0 to 1, every pivot row whose RREF entry
    # M[pivot_for_col[c]][f] is 1 must also have its s[c] flipped.
    lines.append(f"add {weight_addr},{s_at(0)},{s_at(1)}")
    for c in range(2, n):
        lines.append(f"add {weight_addr},{s_at(c)}")
    lines.append(f"cmp {need_flip},{weight_addr},{K_VAL},ne")

    for c in range(n):
        # Off-diagonal contribution: for each free col f != c, if
        # M[pivot_for_col[c]][f] = 1, flipping s[f] flips s[c] too.
        for f in range(n):
            if f == c:
                continue
            # read M[pivot_for_col[c]][f] dynamically
            lines.append(f"copy {a_tmp},{ZERO}")
            for r in range(m):
                lines.append(f"cmp {is_match},{pivot_at(c)},{ROW_AT(r)},eq")
                lines.append(
                    f"select {a_tmp},{is_match},{M_at(r, f)},{a_tmp}")
            lines.append(f"and {flip_bit},{need_flip},{is_free_at(f)}")
            lines.append(f"and {flip_bit},{a_tmp}")
            lines.append(f"xor {s_at(c)},{flip_bit}")
        # Diagonal contribution: if c is the free col, flip its 0 to 1.
        lines.append(f"and {flip_bit},{need_flip},{is_free_at(c)}")
        lines.append(f"xor {s_at(c)},{flip_bit}")

    # ----- verify A·s_GE = b on training rows AND weight(s_GE) = k --------
    # In rank ≤ n−2 cases the affine solution space has more than one
    # weight-k vector candidate, so an "A·s = b" check alone cannot
    # distinguish; the weight check rules out solution-space vectors
    # whose weight ≠ k.
    lines.append(f"copy {err_addr},{ZERO}")
    for i in range(m):
        lines.append(f"and {pred_tmp},{s_at(0)},{X_tr_at(i, 0)}")
        for c in range(1, n):
            lines.append(f"and {a_tmp},{s_at(c)},{X_tr_at(i, c)}")
            lines.append(f"xor {pred_tmp},{a_tmp}")
        lines.append(f"xor {pred_tmp},{y_tr_at(i)}")
        lines.append(f"or {err_addr},{pred_tmp}")
    # weight(s_GE) check
    lines.append(f"add {weight_addr},{s_at(0)},{s_at(1)}")
    for c in range(2, n):
        lines.append(f"add {weight_addr},{s_at(c)}")
    lines.append(f"cmp {a_tmp},{weight_addr},{K_VAL},ne")
    lines.append(f"or {err_addr},{a_tmp}")

    # ----- brute-force fallback (rare rank ≤ n−2 case) ---------------------
    # For each weight-k candidate T, compute ind_T = AND_i ((y_train[i] XOR
    # XOR_T(X_train[i])) XOR 1), then s_brute[c] = OR over T containing c
    # of ind_T. By identifiability, exactly one ind_T is 1.
    candidates = list(__import__("itertools").combinations(range(n), spec.k_secret))
    for t_idx, t in enumerate(candidates):
        # ind_T = AND over rows of matched_T_i
        for i in range(m):
            # parity_T_i = y XOR X[i,t0] XOR X[i,t1] XOR ...
            lines.append(f"xor {parity_t},{y_tr_at(i)},{X_tr_at(i, t[0])}")
            for k_idx in range(1, len(t)):
                lines.append(f"xor {parity_t},{X_tr_at(i, t[k_idx])}")
            # matched_i = parity XOR 1 (NOT in 0/1)
            if i == 0:
                lines.append(f"xor {ind_t_at(t_idx)},{parity_t},{ONE}")
            else:
                lines.append(f"xor {matched_t},{parity_t},{ONE}")
                lines.append(f"and {ind_t_at(t_idx)},{matched_t}")
    # s_brute[c] = OR over candidates containing c of ind_T
    for c in range(n):
        contributing = [
            t_idx for t_idx, t in enumerate(candidates) if c in t
        ]
        first = True
        for t_idx in contributing:
            if first:
                lines.append(
                    f"copy {s_brute_at(c)},{ind_t_at(t_idx)}")
                first = False
            else:
                lines.append(f"or {s_brute_at(c)},{ind_t_at(t_idx)}")

    # ----- pick s_final = err ? s_brute : s_GE -----------------------------
    for c in range(n):
        lines.append(
            f"select {s_final_at(c)},{err_addr},{s_brute_at(c)},{s_at(c)}")

    # ----- predictions: pred[j] = XOR_c (s_final[c] AND X_test[j][c]) ------
    for j in range(m_test):
        lines.append(f"and {pred_tmp},{s_final_at(0)},{X_te_at(j, 0)}")
        for c in range(1, n):
            lines.append(f"and {a_tmp},{s_final_at(c)},{X_te_at(j, c)}")
            lines.append(f"xor {pred_tmp},{a_tmp}")
        lines.append(f"copy {pred_at(j)},{pred_tmp}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def generate_ge_small() -> str:
    """GE-based predictor IR for the small instance."""
    return _generate_ge(SMALL)


if __name__ == "__main__":
    ir = generate_ge_small()
    cost = score_small(ir)
    out = os.path.join(_HERE, "ge_small.ir")
    with open(out, "w") as f:
        f.write(ir)
        f.write("\n")
    n_ops = len(ir.splitlines()) - 2
    print(f"  ge_small.ir       cost={cost:>5,}  ops={n_ops:>4,}  -> {out}")
