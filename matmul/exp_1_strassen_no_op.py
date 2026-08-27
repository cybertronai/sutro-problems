"""Lane 1: Strassen with NO derived-operand materialization.

Read A and B bulk directly in inner kernel.  For sum/sub Mi (e.g. A11+A22),
the inner kernel does (mul tmp1, A11[i,k], B[...]; mul tmp2, A22[i,k], B[...];
add tmp_sum, tmp1, tmp2; ... ).

This eliminates 64 A_OP + 64 B_OP cells (saving 128 cells of mid-cost addrs)
at the cost of more bulk reads.

Cost analysis per Mi:
  Materialize-version: 64 mat-reads + ~2240 inner-reads = 2304 reads on 64 op-cells (HOT)
  No-mat: ~3000 reads on bulk cells (less hot but bulk is cheap-mid range)

Worth a shot — at minimum, fewer cells means bulk gets cheaper addresses.
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

SA_SCRATCH = 1_000
TMP1       = 1_001
TMP2       = 1_002
TMP_SUM    = 1_003
SB_BASE    = 1_010
SC_BASE    = 1_020


def A_at(i, j): return A_BASE + i * N + j
def B_at(i, j): return B_BASE + i * N + j
def C_at(i, j): return C_BASE + i * N + j
def A_blk(bi, bj, ii, jj): return A_at(bi * H + ii, bj * H + jj)
def B_blk(bi, bj, ii, jj): return B_at(bi * H + ii, bj * H + jj)


def C_block_at(blk_idx, ii, jj):
    if blk_idx == 0: return C_at(ii, jj)
    if blk_idx == 1: return C_at(ii, H + jj)
    if blk_idx == 2: return C_at(H + ii, jj)
    if blk_idx == 3: return C_at(H + ii, H + jj)
    raise ValueError(blk_idx)


def emit_inner_no_mat(lines, lhs_a_fn, lhs_b_fn, lhs_op,
                      rhs_a_fn, rhs_b_fn, rhs_op,
                      contributions, init_flags, TJ):
    """Inner 8x8 kernel reading bulk A and B directly.

    For each (i, j, k):
      build SA = lhs_op(A[i,k], A2[i,k])  if combo, else SA = A[i,k]
      build SB[j] = rhs_op(B[k,j], B2[k,j])  if combo, else SB[j] = B[k,j]
      mul tmp, SA, SB[j] -> sC[i,j]

    SA cached as 1 cell.  SB cached as TJ cells (per (k, bj)).
    """
    nbj = H // TJ
    SA = SA_SCRATCH
    sB = lambda jj: SB_BASE + jj
    sC = lambda ii, jj: SC_BASE + ii * TJ + jj

    for bj in range(nbj):
        for bk in range(H):
            # Build sB
            for jj in range(TJ):
                col = bj * TJ + jj
                if rhs_op == "copy":
                    lines.append(f"copy {sB(jj)},{rhs_a_fn(bk, col)}")
                else:
                    lines.append(
                        f"{rhs_op} {sB(jj)},{rhs_a_fn(bk, col)},{rhs_b_fn(bk, col)}"
                    )
            for ii in range(H):
                # Build SA
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
        # Stream sC into C blocks
        for ii in range(H):
            for jj in range(TJ):
                col = bj * TJ + jj
                src = sC(ii, jj)
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


def build_strassen_no_mat(TJ=4) -> str:
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
        emit_inner_no_mat(lines, lhs_a, lhs_b, la, rhs_a, rhs_b, ra,
                          contribs, init_flags, TJ)

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

    print("V_no_mat: Strassen with no derived-operand materialization")
    for TJ in [1, 2, 4, 8]:
        try:
            ir = build_strassen_no_mat(TJ=TJ)
            new_ir, info = rename_optimal(ir)
            cost = score_16x16(new_ir)
            results.append((f"no_mat", TJ, cost, new_ir))
            print(f"  TJ={TJ}: {cost:,}")
        except Exception as e:
            print(f"  TJ={TJ}: ERROR {e}")

    if results:
        best = min(results, key=lambda r: r[2])
        name, TJ, cost, ir = best
        print(f"\nBEST: {name} TJ={TJ}: {cost:,}")
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
