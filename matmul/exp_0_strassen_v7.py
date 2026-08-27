"""Strassen v7: cascade derived combos via D2 = D1 + D4 trick.

Order Mi to maximize derived-combo cascading:
  1. M3: A_OP = A11      (copy A11)         B_OP = ?
  2. M4: A_OP = A22      (copy A22)         B_OP = ?
  3. M1: A_OP = A11+A22 = (M3-A_OP) + (M4-A_OP)?  But A_OP got overwritten.

Strategy: store cascade-helper values in a small set of fixed locations,
then derive all 5 combos from those.  E.g.,:
  Step 1: D_A11 = h_A11_op (64-cell scratchpad)
  Step 2: D_A22 = h_A22_op
  Step 3: D_A12 = h_A12_op   (used in D5 = A12-A22)
  Step 4: D_A21 = h_A21_op   (used in D4 = A21-A11)

Then derived combos read these helpers (cheap?) instead of A_in.

But this is exactly v6.  The issue was 4 helper regions × 64 cells = 256
cells × 4 reads each = 1024 read-cost in mid-cost addrs.

Alternative: only stream-cache cells that are read 3+ times.  Looking at
A_in original reads (in v3):
  A11: 4 reads (D1, D3, D4, M3)
  A22: 4 reads (D1, D2, D5, M4)
  A12: 2 reads (D3, D5)
  A21: 2 reads (D2, D4)

Caching A11 saves 3 reads (4 reads on cold A11 → 1 read on cold A11 + 4
reads on cheap cache).  At cost diff ~17-7 = 10 per read, savings 30/cell
× 64 = 1920.  Per A11 cache cell: 4 reads at ~cost 10 = 40 cost.  Net
gain per cell: 30 - 40/3 ≈ 20.  Marginally positive.

Similarly for A22.  Caching A12, A21 (read only 2 times each) is a wash.

So selectively cache A11, A22, B11, B22 — the 4 sub-blocks read 4 times
each.  That's 4×64 = 256 helper cells.

This is roughly what v6 did but only for 4 sub-blocks.  Let me just try
it.
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

A_OP = 60_000
B_OP = 60_064

# Cache only the 4-times-used sub-blocks.
H_A11 = 61_000
H_A22 = 61_064
H_B11 = 61_128
H_B22 = 61_192

ACC_BASE = 50_000

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
def H_A11_at(ii, jj): return H_A11 + ii * H + jj
def H_A22_at(ii, jj): return H_A22 + ii * H + jj
def H_B11_at(ii, jj): return H_B11 + ii * H + jj
def H_B22_at(ii, jj): return H_B22 + ii * H + jj


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

    # Phase 0: cache the 4-time-used sub-blocks (A11, A22, B11, B22).
    # Each cell of A11/A22 read once here.  A12, A21, B12, B21 are not
    # cached (read directly 2 times in derived combos).
    emit_copy_block(lines, H_A11_at, A11)
    emit_copy_block(lines, H_A22_at, A22)
    emit_copy_block(lines, H_B11_at, B11)
    emit_copy_block(lines, H_B22_at, B22)

    # Phase 1: 7 Mi kernels.  Use h_A11/h_A22/h_B11/h_B22 in derived combos.
    plans = [
        # M1: A_OP = h_A11 + h_A22 ; B_OP = h_B11 + h_B22
        (
            lambda L: emit_combo_block(L, Aop_at, H_A11_at, H_A22_at, "add"),
            lambda L: emit_combo_block(L, Bop_at, H_B11_at, H_B22_at, "add"),
            [(0, '+'), (3, '+')],
        ),
        # M2: A_OP = A21 + h_A22 ; B_OP = h_B11
        (
            lambda L: emit_combo_block(L, Aop_at, A21, H_A22_at, "add"),
            lambda L: emit_copy_block(L, Bop_at, H_B11_at),
            [(2, '+'), (3, '-')],
        ),
        # M3: A_OP = h_A11 ; B_OP = B12 - h_B22
        (
            lambda L: emit_copy_block(L, Aop_at, H_A11_at),
            lambda L: emit_combo_block(L, Bop_at, B12, H_B22_at, "sub"),
            [(1, '+'), (3, '+')],
        ),
        # M4: A_OP = h_A22 ; B_OP = B21 - h_B11
        (
            lambda L: emit_copy_block(L, Aop_at, H_A22_at),
            lambda L: emit_combo_block(L, Bop_at, B21, H_B11_at, "sub"),
            [(0, '+'), (2, '+')],
        ),
        # M5: A_OP = h_A11 + A12 ; B_OP = h_B22
        (
            lambda L: emit_combo_block(L, Aop_at, H_A11_at, A12, "add"),
            lambda L: emit_copy_block(L, Bop_at, H_B22_at),
            [(0, '-'), (1, '+')],
        ),
        # M6: A_OP = A21 - h_A11 ; B_OP = h_B11 + B12
        (
            lambda L: emit_combo_block(L, Aop_at, A21, H_A11_at, "sub"),
            lambda L: emit_combo_block(L, Bop_at, H_B11_at, B12, "add"),
            [(3, '+')],
        ),
        # M7: A_OP = A12 - h_A22 ; B_OP = B21 + h_B22
        (
            lambda L: emit_combo_block(L, Aop_at, A12, H_A22_at, "sub"),
            lambda L: emit_combo_block(L, Bop_at, B21, H_B22_at, "add"),
            [(0, '+')],
        ),
    ]

    init_flags = [False, False, False, False]
    for build_a, build_b, contribs in plans:
        build_a(lines)
        build_b(lines)
        emit_inner_8x8_streaming(lines, Aop_at, Bop_at, contribs, init_flags, TJ)

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
