"""Strassen v10: outer-product inner kernel + direct C accumulation.

Inner kernel for each Mi is an outer-product 8x8 with sB caching the full
B-row (NB=8).

Loop structure per Mi:
    for k in 0..7:
        for jj in 0..7: copy sB[jj], B_OP[k, jj]
        for i in 0..7:
            copy SA, A_OP[i, k]    (or compute SA inline from helpers — let's
                                    keep the simple pre-computed A_OP for now)
            for jj in 0..7:
                if k==0: mul tmp_M[i, jj], SA, sB[jj]
                else:    mul tmp, SA, sB[jj]; add tmp_M[i, jj], tmp

After Mi inner kernel: stream tmp_M into C-blocks (same as v8).

Vs sa-cache inner: outer-product fits sC into a per-Mi 64-cell tile (vs
sa-cache's 32-cell strip-shared tile).
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

A_OP = 60_000  # 64 cells (shared across all 7 Mi)
B_OP = 60_064  # 64 cells

SA       = 1_000
TMP      = 1_001
SB_BASE  = 1_002   # 8 cells (NB=8)
SC_BASE  = 1_010   # 64 cells (full Mi tile per Mi, shared across Mi's)


def A_at(i, j): return A_BASE + i * N + j
def B_at(i, j): return B_BASE + i * N + j
def C_at(i, j): return C_BASE + i * N + j
def A_blk(bi, bj, ii, jj): return A_at(bi * H + ii, bj * H + jj)
def B_blk(bi, bj, ii, jj): return B_at(bi * H + ii, bj * H + jj)
def Aop_at(ii, jj): return A_OP + ii * H + jj
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


def emit_inner_outer_product(lines, A_fn, B_fn, contributions, init_flags):
    """Outer-product inner kernel.  sC is a 64-cell scratch (shared across Mi)
    that is fully populated by k=0..7 accumulation.  After kernel finishes,
    sC is streamed into C-block accumulators.
    """
    sB = lambda jj: SB_BASE + jj
    sC = lambda ii, jj: SC_BASE + ii * H + jj

    for k in range(H):
        for jj in range(H):
            lines.append(f"copy {sB(jj)},{B_fn(k, jj)}")
        for i in range(H):
            lines.append(f"copy {SA},{A_fn(i, k)}")
            for jj in range(H):
                if k == 0:
                    lines.append(f"mul {sC(i, jj)},{SA},{sB(jj)}")
                else:
                    lines.append(f"mul {TMP},{SA},{sB(jj)}")
                    lines.append(f"add {sC(i, jj)},{TMP}")
    # Flush: stream sC into C-block accumulators
    for ii in range(H):
        for jj in range(H):
            src = sC(ii, jj)
            for blk_idx, sign in contributions:
                dst = C_block_at(blk_idx, ii, jj)
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

    plans = [
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
        if la == "copy":
            emit_copy_block(lines, Aop_at, lhs_a)
        else:
            emit_combo(lines, Aop_at, lhs_a, lhs_b, la)
        if ra == "copy":
            emit_copy_block(lines, Bop_at, rhs_a)
        else:
            emit_combo(lines, Bop_at, rhs_a, rhs_b, ra)
        emit_inner_outer_product(lines, Aop_at, Bop_at, contribs, init_flags)

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
    ir = build_strassen_ir()
    raw_cost = score_16x16(ir)
    print(f"raw cost: {raw_cost:,}")
    new_ir, info = rename_optimal(ir)
    new_cost = score_16x16(new_ir)
    print(f"renamed: {new_cost:,}")

    if new_cost < PREV_RECORD:
        out_path = os.path.join(HERE, "records", f"record_{new_cost}_lane0.ir")
        with open(out_path, "w") as f:
            f.write(new_ir + "\n")
        print(f"NEW RECORD: {new_cost}")
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
