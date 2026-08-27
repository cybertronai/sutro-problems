"""Strassen 2x2 over 8x8 sub-blocks for 16x16 matmul.

View 16x16 = [[A11, A12], [A21, A22]] * [[B11, B12], [B21, B22]] where each
sub-block is 8x8. Use Strassen's 7 multiplications instead of 8:

  M1=(A11+A22)(B11+B22)
  M2=(A21+A22)*B11
  M3=A11*(B12-B22)
  M4=A22*(B21-B11)
  M5=(A11+A12)*B22
  M6=(A21-A11)*(B11+B12)
  M7=(A12-A22)*(B21+B22)

Then:
  C11 = M1 + M4 - M5 + M7
  C12 = M3 + M5
  C21 = M2 + M4
  C22 = M1 - M2 + M3 + M6

Each 8x8 inner kernel is implemented sa-cache style (Ti=8, Tj=4 mirrored).
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
H = 8           # half size
TI, TJ = 8, 4   # inner kernel tile shape


# -----------------------------------------------------------------------------
# Address layout
# -----------------------------------------------------------------------------
# We allocate addresses in big bulk regions, then rename optimally at the end.
# So absolute values don't matter much; only that they don't collide.

# Inputs A, B at fixed bulk addresses
A_BASE = 100_000  # A bulk: 256 cells
B_BASE = 110_000  # B bulk: 256 cells
C_BASE = 120_000  # C bulk: 256 cells (output)

# 14 derived 8x8 operands (each 64 cells) at offsets we pick:
# Order: A11+A22, B11+B22, A21+A22, B12-B22, B21-B11, A11+A12, A21-A11, B11+B12,
#        A12-A22, B21+B22.  We need these 10 *summed* operands.
# Actually 14 is overcounting since some are 'just' a sub-block.  Let's enumerate
# what each Mi needs:
#   M1: (A11+A22) and (B11+B22)              -> 2 derived
#   M2: A21+A22 and B11 (sub-block)          -> 1 derived + 1 sub
#   M3: A11 and B12-B22                      -> 1 sub + 1 derived
#   M4: A22 and B21-B11                      -> 1 sub + 1 derived
#   M5: A11+A12 and B22                      -> 1 derived + 1 sub
#   M6: A21-A11 and B11+B12                  -> 2 derived
#   M7: A12-A22 and B21+B22                  -> 2 derived
# Distinct derived combos: A11+A22, A21+A22, A11+A12, A21-A11, A12-A22  (5 from A)
#                          B11+B22, B12-B22, B21-B11, B11+B12, B21+B22  (5 from B)
# So 10 derived 8x8 operands of 64 cells each = 640 cells.  Plus inner kernel
# scratchpads, plus 7 result tiles M1..M7 (8x8 = 64 each, 448 cells), plus
# C-block bulk.

DER_BASE = 200_000  # 10 derived operands, 64 cells each
M_BASE   = 210_000  # 7 Mi result tiles, 64 cells each

# Inner kernel scratchpads (re-used across the 7 Mi calls): keep them low so
# rename_optimal will move them to the cheapest addresses.
SA_SCRATCH = 1_000   # cached A element
TMP        = 1_001
SB_BASE    = 1_002   # 4 cells (Tj=4)
SC_BASE    = 1_010   # 32 cells (Ti*Tj = 8*4 = 32)


def A_at(i, j):
    """A[i,j] for 0<=i,j<16."""
    return A_BASE + i * N + j


def B_at(i, j):
    return B_BASE + i * N + j


def C_at(i, j):
    return C_BASE + i * N + j


# Sub-block accessors: blk in {(0,0),(0,1),(1,0),(1,1)} -> 8x8 region of A or B.
def A_blk(bi, bj, ii, jj):
    return A_at(bi * H + ii, bj * H + jj)


def B_blk(bi, bj, ii, jj):
    return B_at(bi * H + ii, bj * H + jj)


# Derived-operand addresses: idx in 0..9, 64-cell block
DERIVED_NAMES = [
    "A11pA22",  # 0
    "A21pA22",  # 1
    "A11pA12",  # 2
    "A21mA11",  # 3
    "A12mA22",  # 4
    "B11pB22",  # 5
    "B12mB22",  # 6
    "B21mB11",  # 7
    "B11pB12",  # 8
    "B21pB22",  # 9
]


def D_at(idx, ii, jj):
    """Cell (ii,jj) of derived operand idx."""
    return DER_BASE + idx * 64 + ii * H + jj


def M_at(midx, ii, jj):
    """Cell (ii,jj) of Mi result, midx in 0..6."""
    return M_BASE + midx * 64 + ii * H + jj


# -----------------------------------------------------------------------------
# Linear-combo helpers
# -----------------------------------------------------------------------------

def emit_combo(lines, dest_addr_fn, lhs_addr_fn, rhs_addr_fn, op):
    """Emit 64 instructions: dest[ii,jj] = lhs[ii,jj] op rhs[ii,jj].

    op is 'add' or 'sub'.
    """
    for ii in range(H):
        for jj in range(H):
            lines.append(
                f"{op} {dest_addr_fn(ii, jj)},{lhs_addr_fn(ii, jj)},{rhs_addr_fn(ii, jj)}"
            )


# -----------------------------------------------------------------------------
# Inner 8x8 kernel: sa-cache style, Ti=8, Tj=4
# -----------------------------------------------------------------------------

def emit_inner_8x8(lines, A_fn, B_fn, M_fn):
    """Compute M[ii,jj] = sum_kk A[ii,kk] * B[kk,jj] for 8x8.

    A_fn, B_fn, M_fn are callables (ii,jj) -> address.
    Uses SA_SCRATCH, TMP, SB_BASE, SC_BASE scratchpads.

    Following the 16x16 sa-cache pattern, with N->H=8, Ti=8, Tj=4.
    Loop order: bj (over Tj-strips), bk (full reduction), ii, jj.
    Using bi only one iteration since Ti=8 covers the whole row.
    """
    nbj = H // TJ  # = 2
    SA = SA_SCRATCH
    sB = lambda jj: SB_BASE + jj
    sC = lambda ii, jj: SC_BASE + ii * TJ + jj

    for bj in range(nbj):
        for bk in range(H):
            # Load 4 B cells into sB
            for jj in range(TJ):
                lines.append(f"copy {sB(jj)},{B_fn(bk, bj * TJ + jj)}")
            # Inner: for ii in 0..7, cache A[ii,bk] then mul-accumulate sC
            for ii in range(H):
                lines.append(f"copy {SA},{A_fn(ii, bk)}")
                for jj in range(TJ):
                    if bk == 0:
                        lines.append(f"mul {sC(ii, jj)},{SA},{sB(jj)}")
                    else:
                        lines.append(f"mul {TMP},{SA},{sB(jj)}")
                        lines.append(f"add {sC(ii, jj)},{TMP}")
        # Flush sC to M-tile for this Tj-strip
        for ii in range(H):
            for jj in range(TJ):
                lines.append(f"copy {M_fn(ii, bj * TJ + jj)},{sC(ii, jj)}")


# -----------------------------------------------------------------------------
# Build full Strassen IR
# -----------------------------------------------------------------------------

def build_strassen_ir() -> str:
    inputs = ([A_at(i, j) for i in range(N) for j in range(N)] +
              [B_at(i, j) for i in range(N) for j in range(N)])
    outputs = [C_at(i, j) for i in range(N) for j in range(N)]

    lines = [",".join(map(str, inputs))]

    # ---------- Phase 1: derive 10 8x8 combos ----------
    # A operands: bi indexes the row-block, bj the col-block of A
    # A11 = A[0:8, 0:8]      ((0,0))
    # A12 = A[0:8, 8:16]     ((0,1))
    # A21 = A[8:16, 0:8]     ((1,0))
    # A22 = A[8:16, 8:16]    ((1,1))
    A11 = lambda ii, jj: A_blk(0, 0, ii, jj)
    A12 = lambda ii, jj: A_blk(0, 1, ii, jj)
    A21 = lambda ii, jj: A_blk(1, 0, ii, jj)
    A22 = lambda ii, jj: A_blk(1, 1, ii, jj)
    B11 = lambda ii, jj: B_blk(0, 0, ii, jj)
    B12 = lambda ii, jj: B_blk(0, 1, ii, jj)
    B21 = lambda ii, jj: B_blk(1, 0, ii, jj)
    B22 = lambda ii, jj: B_blk(1, 1, ii, jj)

    D = lambda idx: (lambda ii, jj: D_at(idx, ii, jj))

    # 0: A11+A22
    emit_combo(lines, D(0), A11, A22, "add")
    # 1: A21+A22
    emit_combo(lines, D(1), A21, A22, "add")
    # 2: A11+A12
    emit_combo(lines, D(2), A11, A12, "add")
    # 3: A21-A11
    emit_combo(lines, D(3), A21, A11, "sub")
    # 4: A12-A22
    emit_combo(lines, D(4), A12, A22, "sub")
    # 5: B11+B22
    emit_combo(lines, D(5), B11, B22, "add")
    # 6: B12-B22
    emit_combo(lines, D(6), B12, B22, "sub")
    # 7: B21-B11
    emit_combo(lines, D(7), B21, B11, "sub")
    # 8: B11+B12
    emit_combo(lines, D(8), B11, B12, "add")
    # 9: B21+B22
    emit_combo(lines, D(9), B21, B22, "add")

    # ---------- Phase 2: 7 inner 8x8 matmuls ----------
    M = lambda midx: (lambda ii, jj: M_at(midx, ii, jj))

    # M1 = (A11+A22)(B11+B22)
    emit_inner_8x8(lines, D(0), D(5), M(0))
    # M2 = (A21+A22) B11
    emit_inner_8x8(lines, D(1), B11, M(1))
    # M3 = A11 (B12-B22)
    emit_inner_8x8(lines, A11, D(6), M(2))
    # M4 = A22 (B21-B11)
    emit_inner_8x8(lines, A22, D(7), M(3))
    # M5 = (A11+A12) B22
    emit_inner_8x8(lines, D(2), B22, M(4))
    # M6 = (A21-A11)(B11+B12)
    emit_inner_8x8(lines, D(3), D(8), M(5))
    # M7 = (A12-A22)(B21+B22)
    emit_inner_8x8(lines, D(4), D(9), M(6))

    # ---------- Phase 3: combine M's into C ----------
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    M_(0)  # noqa  # just to silence unused warning if linting

    M0 = lambda ii, jj: M_at(0, ii, jj)
    M1 = lambda ii, jj: M_at(1, ii, jj)
    M2 = lambda ii, jj: M_at(2, ii, jj)
    M3 = lambda ii, jj: M_at(3, ii, jj)
    M4 = lambda ii, jj: M_at(4, ii, jj)
    M5 = lambda ii, jj: M_at(5, ii, jj)
    M6 = lambda ii, jj: M_at(6, ii, jj)

    # C11[ii,jj] = M0+M3-M4+M6  at C[ii, jj]
    # C12[ii,jj] = M2+M4        at C[ii, 8+jj]
    # C21[ii,jj] = M1+M3        at C[8+ii, jj]
    # C22[ii,jj] = M0-M1+M2+M5  at C[8+ii, 8+jj]

    # We can fuse: write C-cell using a free tmp.  For each output cell, do
    # 3-add chains with intermediate writes to C itself.
    #
    # Note: writes are free, so we can build C in place by doing a chain of
    # add/sub instructions whose dest is C.  Each subsequent add reads C
    # (which we just wrote) plus an Mi cell.

    for ii in range(H):
        for jj in range(H):
            # C11 = M0 + M3 - M4 + M6
            c = C_at(ii, jj)
            lines.append(f"add {c},{M0(ii, jj)},{M3(ii, jj)}")
            lines.append(f"sub {c},{c},{M4(ii, jj)}")
            lines.append(f"add {c},{c},{M6(ii, jj)}")
            # C12 = M2 + M4
            c = C_at(ii, H + jj)
            lines.append(f"add {c},{M2(ii, jj)},{M4(ii, jj)}")
            # C21 = M1 + M3
            c = C_at(H + ii, jj)
            lines.append(f"add {c},{M1(ii, jj)},{M3(ii, jj)}")
            # C22 = M0 - M1 + M2 + M5
            c = C_at(H + ii, H + jj)
            lines.append(f"sub {c},{M0(ii, jj)},{M1(ii, jj)}")
            lines.append(f"add {c},{c},{M2(ii, jj)}")
            lines.append(f"add {c},{c},{M5(ii, jj)}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def M_(_):
    """No-op (keeps the lint-shaped reference)."""
    return None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # STOP_SIGNAL check
    stop_path = os.path.join(HERE, "STOP_SIGNAL_0")
    if os.path.exists(stop_path):
        os.remove(stop_path)
        print("STOP_SIGNAL — halting.")
        sys.exit(0)

    # Log experiment start
    with open(os.path.join(HERE, "events.jsonl"), "a") as f:
        f.write(json.dumps({
            "ts": int(time.time()),
            "type": "exp_start",
            "lane": 0,
            "exp": os.path.basename(__file__),
        }) + "\n")

    ir = build_strassen_ir()
    print(f"raw IR ops: {len(ir.splitlines()) - 2}")

    # Verify correctness directly first (raw layout)
    raw_cost = score_16x16(ir)
    print(f"raw cost  (no rename): {raw_cost:,}")

    # Rename and re-score
    new_ir, info = rename_optimal(ir)
    new_cost = score_16x16(new_ir)
    print(f"renamed cost         : {new_cost:,}")
    print(f"renamer info         : orig_est={info['orig_cost_estimate']:,} "
          f"remap_est={info['remapped_cost_estimate']:,} gap={info['gap']:,}")

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
