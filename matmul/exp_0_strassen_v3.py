"""Strassen v3: stream-accumulate Mi into 4 C-block accumulators.

Vs v2 we no longer materialize separate 7 Mi result tiles; instead, the
inner kernel's sC-flush directly accumulates into the relevant C-block
accumulators.  This reduces the number of "hot" cells in the cost-3..6
region from 7×64=448 down to ~4×64=256 (the four C accumulators).
"""
from __future__ import annotations

import os
import sys
import time
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from matmul import score_16x16  # noqa: E402
from addr_renamer import rename_optimal  # noqa: E402

N = 16
H = 8
TI, TJ = 8, 4

A_BASE = 100_000
B_BASE = 110_000
C_BASE = 120_000

A_OP = 60_000   # 64 cells: current Mi's A operand
B_OP = 60_064   # 64 cells: current Mi's B operand

# Four 8x8 C accumulators (also 64 cells each)
ACC_BASE = 50_000


# Inner kernel scratch
SA_SCRATCH = 1_000
TMP        = 1_001
SB_BASE    = 1_002
SC_BASE    = 1_010   # 32 cells


def A_at(i, j): return A_BASE + i * N + j
def B_at(i, j): return B_BASE + i * N + j
def C_at(i, j): return C_BASE + i * N + j


def A_blk(bi, bj, ii, jj): return A_at(bi * H + ii, bj * H + jj)
def B_blk(bi, bj, ii, jj): return B_at(bi * H + ii, bj * H + jj)


def Aop_at(ii, jj): return A_OP + ii * H + jj
def Bop_at(ii, jj): return B_OP + ii * H + jj


def acc_at(blk_idx, ii, jj):
    """blk_idx in {0=C11, 1=C12, 2=C21, 3=C22}."""
    return ACC_BASE + blk_idx * 64 + ii * H + jj


def emit_combo(lines, dest_fn, lhs_fn, rhs_fn, op):
    for ii in range(H):
        for jj in range(H):
            lines.append(
                f"{op} {dest_fn(ii, jj)},{lhs_fn(ii, jj)},{rhs_fn(ii, jj)}"
            )


def emit_copy_block(lines, dest_fn, src_fn):
    for ii in range(H):
        for jj in range(H):
            lines.append(f"copy {dest_fn(ii, jj)},{src_fn(ii, jj)}")


def emit_inner_8x8_streaming(lines, A_fn, B_fn, contributions, init_flags):
    """Compute Mi = A @ B (8x8 sa-cache) and stream-accumulate sC into the
    listed C accumulators.

    contributions: list of (blk_idx, sign) where sign in {'+','-'}.  After
       each TJ-strip's sC is finalized, for each (blk_idx, sign):
         if init_flags[blk_idx] is False: copy acc[blk_idx] = sC (then set True)
         elif sign == '+': add acc[blk_idx] += sC
         else:             sub acc[blk_idx] -= sC
       init_flags is mutated.

    Returns: nothing; init_flags updated in place.
    """
    nbj = H // TJ  # = 2
    SA = SA_SCRATCH
    sB = lambda jj: SB_BASE + jj
    sC = lambda ii, jj: SC_BASE + ii * TJ + jj

    for bj in range(nbj):
        for bk in range(H):
            for jj in range(TJ):
                lines.append(f"copy {sB(jj)},{B_fn(bk, bj * TJ + jj)}")
            for ii in range(H):
                lines.append(f"copy {SA},{A_fn(ii, bk)}")
                for jj in range(TJ):
                    if bk == 0:
                        lines.append(f"mul {sC(ii, jj)},{SA},{sB(jj)}")
                    else:
                        lines.append(f"mul {TMP},{SA},{sB(jj)}")
                        lines.append(f"add {sC(ii, jj)},{TMP}")
        # Stream sC into all contributing accumulators.
        for ii in range(H):
            for jj in range(TJ):
                col = bj * TJ + jj
                src = sC(ii, jj)
                for blk_idx, sign in contributions:
                    dst = acc_at(blk_idx, ii, col)
                    if not init_flags[blk_idx]:
                        # Initialize.  If sign is '-', need to negate.
                        # For '-': we'd need a 0 cell.  But since the FIRST
                        # contribution is always positive in a valid Strassen
                        # ordering (we'll ensure so), init is always copy.
                        if sign == '-':
                            # Fall back: create acc[ii,jj] = 0 - sC[ii,jj]
                            # We need a constant-zero cell.  Use a different
                            # approach: this 'init with negation' should not
                            # happen if we order Mi contributions properly.
                            raise RuntimeError(
                                "negation-init not yet supported; "
                                "reorder Mi contributions to put a + first")
                        lines.append(f"copy {dst},{src}")
                    elif sign == '+':
                        lines.append(f"add {dst},{src}")
                    else:
                        lines.append(f"sub {dst},{src}")
        # After this bj-strip, init_flags persist (the next strip touches
        # different columns, but the same blk).  For correctness we want
        # init_flags[blk] = True for ALL columns once the first bj-strip
        # has touched it.  Since each strip only touches columns
        # bj*TJ..bj*TJ+3, but acc_at is indexed by full (ii, col), the
        # init flag is per-blk_idx, and each bj-strip writes 32 distinct
        # cells.  After bj=0 we've init'd 32 of 64 cells; bj=1 init's the
        # other 32.  So we need to track init per (blk, column_strip).
        # For simplicity, treat init_flags as per-blk: set True after
        # bj=0 has written all its cells, at which point bj=1 cells are
        # ALSO uninitialized (different cells!).  Bug.

        # FIX: track init per-cell or per-strip.  Simplest: track per
        # (blk_idx, bj_strip).  Since we go bj=0 then bj=1, and each strip
        # writes distinct cells, the init-flag for that blk should apply
        # only AFTER both strips have written.  But within Mi, we want
        # init for the next Mi's contribution.  Each Mi's strip writes the
        # SAME 32 cells as Mi-prev's strip (same column range, same blk).
        # So init for a given (blk, col-range) is set after FIRST Mi
        # touches that strip.

    # Track init at per-blk level since we always do bj=0 then bj=1 in
    # this kernel — the first Mi to touch blk init's BOTH strips by
    # going through bj=0 then bj=1.  Good enough.
    for blk_idx, _ in contributions:
        init_flags[blk_idx] = True


def build_strassen_ir() -> str:
    inputs = ([A_at(i, j) for i in range(N) for j in range(N)] +
              [B_at(i, j) for i in range(N) for j in range(N)])
    outputs = [C_at(i, j) for i in range(N) for j in range(N)]

    lines = [",".join(map(str, inputs))]

    A11 = lambda ii, jj: A_blk(0, 0, ii, jj)
    A12 = lambda ii, jj: A_blk(0, 1, ii, jj)
    A21 = lambda ii, jj: A_blk(1, 0, ii, jj)
    A22 = lambda ii, jj: A_blk(1, 1, ii, jj)
    B11 = lambda ii, jj: B_blk(0, 0, ii, jj)
    B12 = lambda ii, jj: B_blk(0, 1, ii, jj)
    B21 = lambda ii, jj: B_blk(1, 0, ii, jj)
    B22 = lambda ii, jj: B_blk(1, 1, ii, jj)

    # Strassen plan: each row = (lhs_op, lhs_a, lhs_b, rhs_op, rhs_a, rhs_b,
    # contributions).
    # Contributions are pairs (blk_idx, sign).
    # blk_idx: 0=C11, 1=C12, 2=C21, 3=C22
    # C11 = M1 + M4 - M5 + M7  (so M1+, M4+, M5-, M7+)
    # C12 = M3 + M5            (M3+, M5+)
    # C21 = M2 + M4            (M2+, M4+)
    # C22 = M1 - M2 + M3 + M6  (M1+, M2-, M3+, M6+)
    plans = [
        # M1 = (A11+A22)(B11+B22): C11+, C22+
        ("add", A11, A22, "add", B11, B22, [(0, '+'), (3, '+')]),
        # M2 = (A21+A22) B11    : C21+, C22-
        ("add", A21, A22, "copy", B11, None, [(2, '+'), (3, '-')]),
        # M3 = A11 (B12-B22)    : C12+, C22+
        ("copy", A11, None, "sub", B12, B22, [(1, '+'), (3, '+')]),
        # M4 = A22 (B21-B11)    : C11+, C21+
        ("copy", A22, None, "sub", B21, B11, [(0, '+'), (2, '+')]),
        # M5 = (A11+A12) B22    : C11-, C12+
        ("add", A11, A12, "copy", B22, None, [(0, '-'), (1, '+')]),
        # M6 = (A21-A11)(B11+B12): C22+
        ("sub", A21, A11, "add", B11, B12, [(3, '+')]),
        # M7 = (A12-A22)(B21+B22): C11+
        ("sub", A12, A22, "add", B21, B22, [(0, '+')]),
    ]

    # Reorder so the FIRST contribution to each blk_idx has sign '+'.  Look
    # at plans in order:
    # M1: C11+, C22+ (init both, both +) ✓
    # M2: C21+, C22- (C21 init+, C22 already init by M1, -, OK)
    # M3: C12+, C22+ (C12 init+, OK)
    # M4: C11+, C21+ (already init)
    # M5: C11-, C12+ (already init)
    # M6: C22+
    # M7: C11+
    # So all init's are positive.

    init_flags = [False, False, False, False]

    for (la, lhs_a, lhs_b, ra, rhs_a, rhs_b, contribs) in plans:
        # Build A_OP
        if la == "copy":
            emit_copy_block(lines, Aop_at, lhs_a)
        else:
            emit_combo(lines, Aop_at, lhs_a, lhs_b, la)
        # Build B_OP
        if ra == "copy":
            emit_copy_block(lines, Bop_at, rhs_a)
        else:
            emit_combo(lines, Bop_at, rhs_a, rhs_b, ra)
        # Run inner kernel and stream-accumulate
        emit_inner_8x8_streaming(lines, Aop_at, Bop_at, contribs, init_flags)

    # Final: copy 4 accumulators to C output region.
    for ii in range(H):
        for jj in range(H):
            lines.append(f"copy {C_at(ii, jj)},{acc_at(0, ii, jj)}")
            lines.append(f"copy {C_at(ii, H + jj)},{acc_at(1, ii, jj)}")
            lines.append(f"copy {C_at(H + ii, jj)},{acc_at(2, ii, jj)}")
            lines.append(f"copy {C_at(H + ii, H + jj)},{acc_at(3, ii, jj)}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


if __name__ == "__main__":
    stop_path = os.path.join(HERE, "STOP_SIGNAL_0")
    if os.path.exists(stop_path):
        os.remove(stop_path)
        print("STOP_SIGNAL — halting.")
        sys.exit(0)

    with open(os.path.join(HERE, "events.jsonl"), "a") as f:
        f.write(json.dumps({
            "ts": int(time.time()),
            "type": "exp_start",
            "lane": 0,
            "exp": os.path.basename(__file__),
        }) + "\n")

    ir = build_strassen_ir()
    print(f"raw IR ops: {len(ir.splitlines()) - 2}")
    raw_cost = score_16x16(ir)
    print(f"raw cost  (no rename): {raw_cost:,}")
    new_ir, info = rename_optimal(ir)
    new_cost = score_16x16(new_ir)
    print(f"renamed cost         : {new_cost:,}")

    PREV_RECORD = 73602
    if new_cost < PREV_RECORD:
        out_path = os.path.join(HERE, "records", f"record_{new_cost}_lane0.ir")
        with open(out_path, "w") as f:
            f.write(new_ir + "\n")
        print(f"NEW RECORD: {new_cost} (saved to {out_path})")
        with open(os.path.join(HERE, "events.jsonl"), "a") as f:
            f.write(json.dumps({
                "ts": int(time.time()),
                "type": "new_record",
                "cost": new_cost,
                "prev": PREV_RECORD,
                "lane": 0,
                "file": os.path.basename(__file__),
            }) + "\n")
    else:
        print(f"no new record (current best {PREV_RECORD})")
