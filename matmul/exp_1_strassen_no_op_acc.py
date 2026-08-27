"""Lane 1: Strassen no-mat + 4 ACC blocks + alias C with A.

Key idea: accumulate Mi contributions into 4 separate 8x8 ACC blocks
(256 cells) instead of writing to C blocks directly during inner kernel.
After all 7 Mi finish, copy ACC → C addresses, where C aliases A.

This separates the "A reads" from the "C writes" cleanly:
  Phase 1: 7 Mi inner kernels (read A bulk + B bulk, write to ACC)
  Phase 2: 256 copy ACC -> C-aliased-A addresses

By the time Phase 2 starts, A is fully consumed.  Phase 2 writes to
addresses that USED to hold A — that's safe.

Cost:
  Cells: 256 A + 256 B + 256 ACC + 0 C (aliased) + 64 sC + scratch = 832
  Plus 256 ACC→C copy reads (1 per cell)
  Plus 256 exit reads on C-aliased-A addresses

Compare to no_mat-direct: 842 cells, 16,512 reads.
This version: ~832 cells, ~16,512 + 256 copy reads = 16,768 reads.
So slightly more reads but possibly better cell layout?
Renamer would put ACC cells (read 4-5 times each) where the C bulk used
to live... actually with alias, A-addrs get 4 reads (A use) + 1 (exit) = 5 reads.
Without alias, A-addrs got 4 reads, C-addrs got 1 read.
So alias merges these into one region with 5 reads each = renamer prefers it.

EXACT count of A reads in no_mat: each A cell is read by Mi that use it.
A11 in M1 (1) + M3 (1) + M5 (1) + M6 (1) = 4
A12 in M5 (1) + M7 (1) = 2
A21 in M2 (1) + M6 (1) = 2
A22 in M1 (1) + M2 (1) + M4 (1) + M7 (1) = 4

So 64 A11 cells x 4 reads, 64 A12 x 2, 64 A21 x 2, 64 A22 x 4 = 768 reads on A.
With alias, +256 exit reads = 1024 reads on the 256 alias cells.
A11 cells get 4+1=5 reads, A22 cells get 5 reads, A12/A21 get 3 reads.

Total: 5*128 + 3*128 = 1024 reads on aliased cells.
Without alias: 4*128 + 2*128 = 768 reads on A + 256 exit = 1024.
SAME read count.  But cell count drops from 256+256 to 256.

Hmm, then renamer with alias has 256 fewer cells -> bulk shifts down.
That's the savings.
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
C_BASE_NOALIAS = 120_000
ACC_BASE = 50_000

SA_SCRATCH = 1_000
TMP1       = 1_001
SB_BASE    = 1_010
SC_BASE    = 1_020


def A_at(i, j): return A_BASE + i * N + j
def B_at(i, j): return B_BASE + i * N + j
def A_blk(bi, bj, ii, jj): return A_at(bi * H + ii, bj * H + jj)
def B_blk(bi, bj, ii, jj): return B_at(bi * H + ii, bj * H + jj)


def acc_at(blk_idx, ii, jj):
    return ACC_BASE + blk_idx * 64 + ii * H + jj


def emit_inner_no_mat_to_acc(lines, lhs_a_fn, lhs_b_fn, lhs_op,
                              rhs_a_fn, rhs_b_fn, rhs_op,
                              contributions, init_flags, TJ):
    """Write Mi contribs to ACC blocks (not direct C)."""
    nbj = H // TJ
    SA = SA_SCRATCH
    sB = lambda jj: SB_BASE + jj
    sC = lambda ii, jj: SC_BASE + ii * TJ + jj

    for bj in range(nbj):
        for bk in range(H):
            for jj in range(TJ):
                col = bj * TJ + jj
                if rhs_op == "copy":
                    lines.append(f"copy {sB(jj)},{rhs_a_fn(bk, col)}")
                else:
                    lines.append(
                        f"{rhs_op} {sB(jj)},{rhs_a_fn(bk, col)},{rhs_b_fn(bk, col)}"
                    )
            for ii in range(H):
                if lhs_op == "copy":
                    lines.append(f"copy {SA},{lhs_a_fn(ii, bk)}")
                else:
                    lines.append(
                        f"{lhs_op} {SA},{lhs_a_fn(ii, bk)},{lhs_b_fn(ii, bk)}"
                    )
                for jj in range(TJ):
                    if bk == 0:
                        lines.append(f"mul {sC(ii, jj)},{SA},{sB(jj)}")
                    else:
                        lines.append(f"mul {TMP1},{SA},{sB(jj)}")
                        lines.append(f"add {sC(ii, jj)},{TMP1}")
        # Stream sC into ACC blocks
        for ii in range(H):
            for jj in range(TJ):
                col = bj * TJ + jj
                src = sC(ii, jj)
                for blk_idx, sign in contributions:
                    dst = acc_at(blk_idx, ii, col)
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


def build_strassen_no_mat_acc(TJ=8, alias_C_with_A=True):
    inputs = ([A_at(i, j) for i in range(N) for j in range(N)] +
              [B_at(i, j) for i in range(N) for j in range(N)])

    if alias_C_with_A:
        def C_at(i, j): return A_at(i, j)
    else:
        def C_at(i, j): return C_BASE_NOALIAS + i * N + j
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
        emit_inner_no_mat_to_acc(lines, lhs_a, lhs_b, la, rhs_a, rhs_b, ra,
                                 contribs, init_flags, TJ)

    # Phase 2: copy ACC -> C addresses (which alias A in alias mode).
    # The last A read happened in M7's inner kernel.  All ACC writes
    # complete before Phase 2.  Now we can safely overwrite A addrs.
    for ii in range(H):
        for jj in range(H):
            lines.append(f"copy {C_at(ii, jj)},{acc_at(0, ii, jj)}")
            lines.append(f"copy {C_at(ii, H + jj)},{acc_at(1, ii, jj)}")
            lines.append(f"copy {C_at(H + ii, jj)},{acc_at(2, ii, jj)}")
            lines.append(f"copy {C_at(H + ii, H + jj)},{acc_at(3, ii, jj)}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def log_event(d):
    with open(os.path.join(HERE, "events.jsonl"), "a") as f:
        f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    stop_path = os.path.join(HERE, "STOP_SIGNAL_1")
    if os.path.exists(stop_path):
        os.remove(stop_path)
        print("STOP_SIGNAL_1 — halting.")
        sys.exit(0)

    log_event({
        "ts": int(time.time()),
        "type": "exp_start",
        "lane": 1,
        "exp": os.path.basename(__file__),
    })

    PREV_RECORD = 73602
    results = []

    print("V_no_mat_acc (alias=True):")
    for TJ in [1, 2, 4, 8]:
        try:
            ir = build_strassen_no_mat_acc(TJ=TJ, alias_C_with_A=True)
            new_ir, info = rename_optimal(ir)
            cost = score_16x16(new_ir)
            results.append((f"alias", TJ, cost, new_ir))
            print(f"  TJ={TJ}: {cost:,}")
        except Exception as e:
            print(f"  TJ={TJ}: ERROR {e}")

    print("\nV_no_mat_acc (alias=False, control):")
    for TJ in [1, 2, 4, 8]:
        try:
            ir = build_strassen_no_mat_acc(TJ=TJ, alias_C_with_A=False)
            new_ir, info = rename_optimal(ir)
            cost = score_16x16(new_ir)
            results.append((f"noAlias", TJ, cost, new_ir))
            print(f"  TJ={TJ}: {cost:,}")
        except Exception as e:
            print(f"  TJ={TJ}: ERROR {e}")

    if results:
        best = min(results, key=lambda r: r[2])
        name, TJ, cost, ir = best
        print(f"\nBEST: {name} TJ={TJ}: {cost:,} vs record {PREV_RECORD}")
        if cost < PREV_RECORD:
            out_path = os.path.join(HERE, "records", f"record_{cost}_lane1.ir")
            with open(out_path, "w") as f:
                f.write(ir + "\n")
            print(f"NEW RECORD: {out_path}")
            log_event({"ts": int(time.time()), "type": "new_record",
                       "cost": cost, "prev": PREV_RECORD, "lane": 1,
                       "file": os.path.basename(__file__)})
        else:
            print("no new record")
            log_event({"ts": int(time.time()), "type": "exp_done",
                       "lane": 1, "best_cost": cost, "best_variant": name,
                       "best_TJ": TJ, "exp": os.path.basename(__file__)})
