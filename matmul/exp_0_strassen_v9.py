"""Strassen v9: fuse A_OP into multi-cell SA scratchpad (one SA per ii).

Loop structure inside each Mi inner kernel:

    for bk in range(8):
        # Compute SA[ii] = A_lhs[ii,bk] op A_rhs[ii,bk] for all ii at this bk.
        # (8 add/sub/copy instructions, each reading 1-2 A_in cells.)
        for ii in range(8):
            add SA[ii], A_lhs[ii,bk], A_rhs[ii,bk]   # 2 A_in reads
        for bj in range(nbj):
            for jj in range(TJ):
                copy sB[jj], B_OP[bk, bj*TJ+jj]
            for ii in range(8):
                for jj in range(TJ):
                    if bk == 0:
                        mul sC[ii, bj*TJ+jj], SA[ii], sB[jj]
                    else:
                        mul tmp, SA[ii], sB[jj]
                        add sC[ii, bj*TJ+jj], tmp

This eliminates the per-Mi A_OP scratchpad (64 cells × 4 reads/cell/Mi
= 1792 reads at cost ~9 each, saving ~16k).  In exchange, SA grows from
1 cell to 8 cells (~4-5k extra cost).  Net savings ~11k.

Output cells of C still receive Mi contributions via the streaming flush
of sC at the end of each Mi kernel (same as v8).
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

A_BASE = 100_000
B_BASE = 110_000
C_BASE = 120_000

# B_OP is still a packed scratchpad (B operand re-used across nbj strips so
# packing is necessary).
B_OP = 60_064

# 8 SA cells (one per ii)
SA_BASE   = 1_000   # 8 cells
TMP       = 1_010
SB_BASE   = 1_020   # up to 8 cells
SC_BASE   = 1_030   # up to 64 cells (TJ=8 case)


def A_at(i, j): return A_BASE + i * N + j
def B_at(i, j): return B_BASE + i * N + j
def C_at(i, j): return C_BASE + i * N + j
def A_blk(bi, bj, ii, jj): return A_at(bi * H + ii, bj * H + jj)
def B_blk(bi, bj, ii, jj): return B_at(bi * H + ii, bj * H + jj)
def Bop_at(ii, jj): return B_OP + ii * H + jj


def C_block_at(blk_idx, ii, jj):
    if blk_idx == 0: return C_at(ii, jj)
    if blk_idx == 1: return C_at(ii, H + jj)
    if blk_idx == 2: return C_at(H + ii, jj)
    if blk_idx == 3: return C_at(H + ii, H + jj)
    raise ValueError(blk_idx)


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


def emit_inner_8x8_fused(lines, A_op, A_lhs_fn, A_rhs_fn,
                         B_fn, contributions, init_flags, TJ):
    """Inner 8x8 kernel with fused SA computation.

    A_op: 'add'/'sub'/'copy' specifying how SA[ii] is built from A_lhs/A_rhs.
    A_rhs_fn may be None for 'copy'.

    SA is an 8-cell array (one per ii).  Loop order: bk outer, bj inner,
    so SA[ii] computed once per (bk) iteration is reused across all bj
    strips.

    To make this correct, sC is indexed by (ii, full_col) — so it's a
    full 64-cell array (sC[ii, 0..H-1]).
    """
    nbj = H // TJ
    SA = lambda ii: SA_BASE + ii
    sB = lambda jj: SB_BASE + jj
    sC = lambda ii, full_col: SC_BASE + ii * H + full_col

    for bk in range(H):
        # Compute SA[ii] for this bk (8 instructions)
        for ii in range(H):
            if A_op == "copy":
                lines.append(f"copy {SA(ii)},{A_lhs_fn(ii, bk)}")
            else:
                lines.append(
                    f"{A_op} {SA(ii)},{A_lhs_fn(ii, bk)},{A_rhs_fn(ii, bk)}"
                )
        for bj in range(nbj):
            for jj in range(TJ):
                lines.append(f"copy {sB(jj)},{B_fn(bk, bj * TJ + jj)}")
            for ii in range(H):
                for jj in range(TJ):
                    full_col = bj * TJ + jj
                    if bk == 0:
                        lines.append(f"mul {sC(ii, full_col)},{SA(ii)},{sB(jj)}")
                    else:
                        lines.append(f"mul {TMP},{SA(ii)},{sB(jj)}")
                        lines.append(f"add {sC(ii, full_col)},{TMP}")
    # Flush: after all bk's, sC is fully accumulated.
    for ii in range(H):
        for col in range(H):
            src = sC(ii, col)
            for blk_idx, sign in contributions:
                dst = C_block_at(blk_idx, ii, col)
                if not init_flags[blk_idx]:
                    if sign == '-':
                        raise RuntimeError("negation-init")
                    lines.append(f"copy {dst},{src}")
                elif sign == '+':
                    lines.append(f"add {dst},{src}")
                else:
                    lines.append(f"sub {dst},{src}")
    for blk_idx, _ in contributions:
        init_flags[blk_idx] = True


def build_strassen_ir(TJ=4) -> str:
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

    plans = [
        # (A_op, A_lhs, A_rhs, B_op, B_lhs, B_rhs, contributions)
        ("add", A11, A22, "add", B11, B22, [(0, '+'), (3, '+')]),  # M1
        ("add", A21, A22, "copy", B11, None, [(2, '+'), (3, '-')]),  # M2
        ("copy", A11, None, "sub", B12, B22, [(1, '+'), (3, '+')]),  # M3
        ("copy", A22, None, "sub", B21, B11, [(0, '+'), (2, '+')]),  # M4
        ("add", A11, A12, "copy", B22, None, [(0, '-'), (1, '+')]),  # M5
        ("sub", A21, A11, "add", B11, B12, [(3, '+')]),  # M6
        ("sub", A12, A22, "add", B21, B22, [(0, '+')]),  # M7
    ]

    init_flags = [False, False, False, False]
    for (la, lhs_a, lhs_b, ra, rhs_a, rhs_b, contribs) in plans:
        # Build B_OP (still packed)
        if ra == "copy":
            emit_copy_block(lines, Bop_at, rhs_a)
        else:
            emit_combo(lines, Bop_at, rhs_a, rhs_b, ra)
        # Run fused inner kernel; SA[ii] is computed inline from A inputs.
        emit_inner_8x8_fused(lines, la, lhs_a, lhs_b, Bop_at,
                              contribs, init_flags, TJ)

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

    PREV_RECORD = 73602
    best_cost = None
    best_TJ = None
    best_ir = None
    for TJ in [1, 2, 4, 8]:
        ir = build_strassen_ir(TJ)
        try:
            new_ir, info = rename_optimal(ir)
            new_cost = score_16x16(new_ir)
            print(f"  TJ={TJ}: renamed cost = {new_cost:,}")
            if best_cost is None or new_cost < best_cost:
                best_cost = new_cost
                best_TJ = TJ
                best_ir = new_ir
        except Exception as e:
            print(f"  TJ={TJ}: error: {e}")

    print(f"\nBest TJ={best_TJ}: {best_cost:,}")

    if best_cost is not None and best_cost < PREV_RECORD:
        out_path = os.path.join(HERE, "records", f"record_{best_cost}_lane0.ir")
        with open(out_path, "w") as f:
            f.write(best_ir + "\n")
        print(f"NEW RECORD: {best_cost} (saved to {out_path})")
        with open(os.path.join(HERE, "events.jsonl"), "a") as f:
            f.write(json.dumps({
                "ts": int(time.time()),
                "type": "new_record",
                "cost": best_cost,
                "prev": PREV_RECORD,
                "lane": 0,
                "file": os.path.basename(__file__),
            }) + "\n")
    else:
        print(f"no new record (current best {PREV_RECORD})")
