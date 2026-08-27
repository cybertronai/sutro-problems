"""Strassen v2: pack derived A and B 8x8 operands into fixed scratchpads.

Key change vs v1: instead of materializing 5 different A-derivatives at 5
distinct 64-cell regions, we have ONE shared "current A operand" scratchpad
(64 cells) that we recompute before each Mi inner kernel.  Same for B.
This packs the read traffic onto fewer cells, letting them migrate to
cheaper addresses after rename_optimal.
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


# Inputs A, B at fixed bulk addresses (renamer will move them)
A_BASE = 100_000
B_BASE = 110_000
C_BASE = 120_000

# Shared 8x8 scratchpads for current Mi's A-operand and B-operand
A_OP = 60_000   # 64 cells
B_OP = 60_064   # 64 cells

# 7 Mi result tiles (each 64 cells)
M_BASE = 70_000

# Inner kernel scratch
SA_SCRATCH = 1_000
TMP        = 1_001
SB_BASE    = 1_002   # 4 cells
SC_BASE    = 1_010   # 32 cells (Ti*Tj = 8*4 = 32)


def A_at(i, j): return A_BASE + i * N + j
def B_at(i, j): return B_BASE + i * N + j
def C_at(i, j): return C_BASE + i * N + j


def A_blk(bi, bj, ii, jj):
    return A_at(bi * H + ii, bj * H + jj)


def B_blk(bi, bj, ii, jj):
    return B_at(bi * H + ii, bj * H + jj)


def Aop_at(ii, jj):
    """Current Mi's A-operand cell."""
    return A_OP + ii * H + jj


def Bop_at(ii, jj):
    return B_OP + ii * H + jj


def M_at(midx, ii, jj):
    return M_BASE + midx * 64 + ii * H + jj


def emit_combo(lines, dest_fn, lhs_fn, rhs_fn, op):
    """dest[ii,jj] = lhs op rhs for an 8x8 region.  op in {'add','sub'}."""
    for ii in range(H):
        for jj in range(H):
            lines.append(
                f"{op} {dest_fn(ii, jj)},{lhs_fn(ii, jj)},{rhs_fn(ii, jj)}"
            )


def emit_copy_block(lines, dest_fn, src_fn):
    """dest[ii,jj] = src[ii,jj] for 8x8 (used to move sub-block into A_OP/B_OP)."""
    for ii in range(H):
        for jj in range(H):
            lines.append(f"copy {dest_fn(ii, jj)},{src_fn(ii, jj)}")


def emit_inner_8x8(lines, A_fn, B_fn, M_fn):
    """Compute M[ii,jj] = sum_kk A[ii,kk] * B[kk,jj] for 8x8 (sa-cache)."""
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
        for ii in range(H):
            for jj in range(TJ):
                lines.append(f"copy {M_fn(ii, bj * TJ + jj)},{sC(ii, jj)}")


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

    M = lambda midx: (lambda ii, jj: M_at(midx, ii, jj))

    # Strassen schedule: for each Mi, build (A_OP, B_OP) then run kernel.
    #   M1=(A11+A22)(B11+B22)
    #   M2=(A21+A22)*B11
    #   M3=A11*(B12-B22)
    #   M4=A22*(B21-B11)
    #   M5=(A11+A12)*B22
    #   M6=(A21-A11)*(B11+B12)
    #   M7=(A12-A22)*(B21+B22)

    plans = [
        # (lhs_op, lhs_a, lhs_b,  rhs_op, rhs_a, rhs_b)
        # 'op' is 'add'/'sub'/'copy' (for sub-block direct).
        ("add", A11, A22, "add", B11, B22),  # M1
        ("add", A21, A22, "copy", B11, None),  # M2
        ("copy", A11, None, "sub", B12, B22),  # M3
        ("copy", A22, None, "sub", B21, B11),  # M4
        ("add", A11, A12, "copy", B22, None),  # M5
        ("sub", A21, A11, "add", B11, B12),  # M6
        ("sub", A12, A22, "add", B21, B22),  # M7
    ]

    for midx, (la, lhs_a, lhs_b, ra, rhs_a, rhs_b) in enumerate(plans):
        # Build A_OP from lhs_a (op) lhs_b
        if la == "copy":
            emit_copy_block(lines, Aop_at, lhs_a)
        else:
            emit_combo(lines, Aop_at, lhs_a, lhs_b, la)
        # Build B_OP
        if ra == "copy":
            emit_copy_block(lines, Bop_at, rhs_a)
        else:
            emit_combo(lines, Bop_at, rhs_a, rhs_b, ra)
        # Run inner 8x8 kernel
        emit_inner_8x8(lines, Aop_at, Bop_at, M(midx))

    # Combine into C
    M0 = lambda ii, jj: M_at(0, ii, jj)
    M1 = lambda ii, jj: M_at(1, ii, jj)
    M2 = lambda ii, jj: M_at(2, ii, jj)
    M3 = lambda ii, jj: M_at(3, ii, jj)
    M4 = lambda ii, jj: M_at(4, ii, jj)
    M5 = lambda ii, jj: M_at(5, ii, jj)
    M6 = lambda ii, jj: M_at(6, ii, jj)

    for ii in range(H):
        for jj in range(H):
            c = C_at(ii, jj)
            lines.append(f"add {c},{M0(ii, jj)},{M3(ii, jj)}")
            lines.append(f"sub {c},{c},{M4(ii, jj)}")
            lines.append(f"add {c},{c},{M6(ii, jj)}")
            c = C_at(ii, H + jj)
            lines.append(f"add {c},{M2(ii, jj)},{M4(ii, jj)}")
            c = C_at(H + ii, jj)
            lines.append(f"add {c},{M1(ii, jj)},{M3(ii, jj)}")
            c = C_at(H + ii, H + jj)
            lines.append(f"sub {c},{M0(ii, jj)},{M1(ii, jj)}")
            lines.append(f"add {c},{c},{M2(ii, jj)}")
            lines.append(f"add {c},{c},{M5(ii, jj)}")

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
