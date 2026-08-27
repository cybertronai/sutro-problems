"""Strassen v6: combine stream-derive (1 A_in read per cell) with packing
(single A_OP / B_OP slot used for all 7 Mi).

Key idea: between Mi kernels, recompute the next Mi's A operand IN A_OP
using whatever scratch state we still hold.  We need to be careful to
sequence the Mi kernels such that the A operand can be derived efficiently
from currently-held scratch.

Concrete schedule for A side:
  Per (ii, jj) loop:
    copy t11 = A11; copy t22 = A22 ; copy t12 = A12; copy t21 = A21
    -- This costs 4 A_in reads per cell, 256 total.
    -- Then for each Mi sequentially compute A_OP[ii,jj] from t's:
    M1: A_OP = t11+t22 (2 t reads)
    M2: A_OP = t21+t22 (2 t reads)
    M3: A_OP = t11      (1 t read)
    M4: A_OP = t22      (1 t read)
    M5: A_OP = t11+t12  (2 t reads)
    M6: A_OP = t21-t11  (2 t reads)
    M7: A_OP = t12-t22  (2 t reads)
  -- BUT: A_OP is needed for the Mi inner kernel AT THAT POINT.  So we
  can't compute all 7 at once at this (ii,jj).  We need to interleave:
  compute A_OP cell-by-cell INSIDE the inner kernel loop.

Restructure: for each Mi, do its full inner kernel reading A_OP cells.
A_OP gets *re-derived per Mi*.  In the derive step, we read t-scratches
(cheap, cost-1) instead of A_in (cost ~17).

Plan:
  Phase pre-A: copy each A[i,j] into a scratch slot AT[i,j].  That's
  256 cells of "A scratch" (one per A_in cell).  Then for derived combos,
  we read AT[i,j] at low cost.

  But 256 scratch cells is a LOT — they each get just a few reads, and
  push other hotter cells out of the cheap region.

This idea may not work as a one-shot.  The right approach is probably to
keep packed A_OP and B_OP, and add a one-time copy of each A_in cell into
a "long" scratch region — but that copy reads A_in once, then derived
combos read scratch (cheap).  The scratch region is 256 cells, sitting at
addrs ~150-400 maybe.  Each scratch cell is read 2-4 times.

Hmm.  Actually the BEST move is simpler: just combine v3 with cascaded
derivation — D2 = D1 + D4, etc.  D1 stays in A_OP for M1.  After M1,
compute D2 = D1 + (precomputed-D4-stored-elsewhere).  We need somewhere
to store the "cascade helpers" D4, D5.  That's 128 cells of "warm"
storage, but if used 1-2 times each, they cost a moderate amount.
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

# Single shared A_OP and B_OP scratch slots (64 cells each).
A_OP = 60_000
B_OP = 60_064

# Cascade helpers for A-derivatives:
#   We compute D1 (A11+A22) into A_OP, run M1, then need to compute D2
#   (A21+A22) and D3 (A11+A12), D4 (A21-A11), D5 (A12-A22).
#   To save A_in reads, we precompute helpers:
#     H_A21 = A21 (copy 1 read of A21)   → used in D2, D4
#     H_A12 = A12 (copy 1 read of A12)   → used in D3, D5
#   After helpers exist, derived combos can be computed from A11/A22 +
#   helpers, OR from prior D outputs.
#
# Helpers (64 cells each):
H_A21 = 61_000
H_A12 = 61_064
# We don't need H_A11/H_A22 because A11 is used in M3 directly and A22
# in M4 directly.  We can read A11 once when computing D1, use it,
# overwrite A_OP later.  But for D3 = A11+A12 we again need A11 — we'd
# need H_A11.  Hmm.
H_A11 = 61_128
H_A22 = 61_192
# So we use 4 helper regions, each 64 cells.  Each helper is a 1:1 copy
# of an A sub-block, read multiple times in the various derived combos.
# A_in is then read ONCE per cell (to populate the helper).

# B-side same structure.
H_B11 = 62_000
H_B12 = 62_064
H_B21 = 62_128
H_B22 = 62_192

ACC_BASE = 50_000

# Scratch
SA_SCRATCH = 1_000
TMP        = 1_001
SB_BASE    = 1_002
SC_BASE    = 1_010


def A_at(i, j): return A_BASE + i * N + j
def B_at(i, j): return B_BASE + i * N + j
def C_at(i, j): return C_BASE + i * N + j
def A_blk(bi, bj, ii, jj): return A_at(bi * H + ii, bj * H + jj)
def B_blk(bi, bj, ii, jj): return B_at(bi * H + ii, bj * H + jj)
def Aop_at(ii, jj): return A_OP + ii * H + jj
def Bop_at(ii, jj): return B_OP + ii * H + jj
def acc_at(blk_idx, ii, jj): return ACC_BASE + blk_idx * 64 + ii * H + jj


def emit_copy_block(lines, dest_fn, src_fn):
    for ii in range(H):
        for jj in range(H):
            lines.append(f"copy {dest_fn(ii, jj)},{src_fn(ii, jj)}")


def emit_combo_block(lines, dest_fn, lhs_fn, rhs_fn, op):
    for ii in range(H):
        for jj in range(H):
            lines.append(
                f"{op} {dest_fn(ii, jj)},{lhs_fn(ii, jj)},{rhs_fn(ii, jj)}"
            )


def emit_inner_8x8_streaming(lines, A_fn, B_fn, contributions, init_flags, TJ):
    nbj = H // TJ
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
                col = bj * TJ + jj
                src = sC(ii, jj)
                for blk_idx, sign in contributions:
                    dst = acc_at(blk_idx, ii, col)
                    if not init_flags[blk_idx]:
                        if sign == '-':
                            raise RuntimeError("negation-init not supported")
                        lines.append(f"copy {dst},{src}")
                    elif sign == '+':
                        lines.append(f"add {dst},{src}")
                    else:
                        lines.append(f"sub {dst},{src}")
    for blk_idx, _ in contributions:
        init_flags[blk_idx] = True


def H_A11_at(ii, jj): return H_A11 + ii * H + jj
def H_A12_at(ii, jj): return H_A12 + ii * H + jj
def H_A21_at(ii, jj): return H_A21 + ii * H + jj
def H_A22_at(ii, jj): return H_A22 + ii * H + jj
def H_B11_at(ii, jj): return H_B11 + ii * H + jj
def H_B12_at(ii, jj): return H_B12 + ii * H + jj
def H_B21_at(ii, jj): return H_B21 + ii * H + jj
def H_B22_at(ii, jj): return H_B22 + ii * H + jj


def build_strassen_ir(TJ=4) -> str:
    inputs = ([A_at(i, j) for i in range(N) for j in range(N)] +
              [B_at(i, j) for i in range(N) for j in range(N)])
    outputs = [C_at(i, j) for i in range(N) for j in range(N)]

    lines = [",".join(map(str, inputs))]

    # Phase 0: copy each A sub-block and B sub-block into helper regions.
    # Each A_in cell read once (256 total reads on A_in).
    A11 = lambda ii, jj: A_blk(0, 0, ii, jj)
    A12 = lambda ii, jj: A_blk(0, 1, ii, jj)
    A21 = lambda ii, jj: A_blk(1, 0, ii, jj)
    A22 = lambda ii, jj: A_blk(1, 1, ii, jj)
    B11 = lambda ii, jj: B_blk(0, 0, ii, jj)
    B12 = lambda ii, jj: B_blk(0, 1, ii, jj)
    B21 = lambda ii, jj: B_blk(1, 0, ii, jj)
    B22 = lambda ii, jj: B_blk(1, 1, ii, jj)

    emit_copy_block(lines, H_A11_at, A11)
    emit_copy_block(lines, H_A12_at, A12)
    emit_copy_block(lines, H_A21_at, A21)
    emit_copy_block(lines, H_A22_at, A22)
    emit_copy_block(lines, H_B11_at, B11)
    emit_copy_block(lines, H_B12_at, B12)
    emit_copy_block(lines, H_B21_at, B21)
    emit_copy_block(lines, H_B22_at, B22)

    # Phase 1: For each Mi, compute A_OP and B_OP from helpers, run kernel.
    plans = [
        # (build_a_op fn, build_b_op fn, contributions)
        # A_OP = h_A11 + h_A22 ; B_OP = h_B11 + h_B22
        (
            lambda L: emit_combo_block(L, Aop_at, H_A11_at, H_A22_at, "add"),
            lambda L: emit_combo_block(L, Bop_at, H_B11_at, H_B22_at, "add"),
            [(0, '+'), (3, '+')],
        ),
        # M2: A_OP = h_A21 + h_A22 ; B_OP = h_B11
        (
            lambda L: emit_combo_block(L, Aop_at, H_A21_at, H_A22_at, "add"),
            lambda L: emit_copy_block(L, Bop_at, H_B11_at),
            [(2, '+'), (3, '-')],
        ),
        # M3: A_OP = h_A11 ; B_OP = h_B12 - h_B22
        (
            lambda L: emit_copy_block(L, Aop_at, H_A11_at),
            lambda L: emit_combo_block(L, Bop_at, H_B12_at, H_B22_at, "sub"),
            [(1, '+'), (3, '+')],
        ),
        # M4: A_OP = h_A22 ; B_OP = h_B21 - h_B11
        (
            lambda L: emit_copy_block(L, Aop_at, H_A22_at),
            lambda L: emit_combo_block(L, Bop_at, H_B21_at, H_B11_at, "sub"),
            [(0, '+'), (2, '+')],
        ),
        # M5: A_OP = h_A11 + h_A12 ; B_OP = h_B22
        (
            lambda L: emit_combo_block(L, Aop_at, H_A11_at, H_A12_at, "add"),
            lambda L: emit_copy_block(L, Bop_at, H_B22_at),
            [(0, '-'), (1, '+')],
        ),
        # M6: A_OP = h_A21 - h_A11 ; B_OP = h_B11 + h_B12
        (
            lambda L: emit_combo_block(L, Aop_at, H_A21_at, H_A11_at, "sub"),
            lambda L: emit_combo_block(L, Bop_at, H_B11_at, H_B12_at, "add"),
            [(3, '+')],
        ),
        # M7: A_OP = h_A12 - h_A22 ; B_OP = h_B21 + h_B22
        (
            lambda L: emit_combo_block(L, Aop_at, H_A12_at, H_A22_at, "sub"),
            lambda L: emit_combo_block(L, Bop_at, H_B21_at, H_B22_at, "add"),
            [(0, '+')],
        ),
    ]

    init_flags = [False, False, False, False]
    for build_a, build_b, contribs in plans:
        build_a(lines)
        build_b(lines)
        emit_inner_8x8_streaming(lines, Aop_at, Bop_at, contribs, init_flags, TJ)

    # Phase 3: copy 4 accumulators to C output.
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
