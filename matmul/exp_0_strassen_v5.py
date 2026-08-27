"""Strassen v5: stream A_in (and B_in) through scratch slots to compute all
derived combos with ONE read per input cell.

Key optimization vs v3: each input A[i,j] cell is read exactly once (into a
scratch slot t11/t12/t21/t22), then all 5 derived A combos for that
(ii,jj) position are computed from those scratch slots.  Same for B.

This saves ~512 reads on each of A_in and B_in, which were the dominant
'cold reads' in v3 (2-4 reads/cell at cost 14-22 = ~17 each, so ~17,000
saved on A and another ~10,000 on B).  We also still pack the *output* of
the derived combos into 5 distinct 64-cell regions.
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

# Five 8x8 derived A operands and five derived B operands.
DA_BASE = 200_000  # 5 derived A's × 64 = 320 cells
DB_BASE = 201_000

# Per-cell scratch slots used during derive phase (also reused as SA/TMP later;
# we allocate distinct addresses here, but rename_optimal can collapse them
# since they share addresses naturally if we pick the same).
T11 = 1_000   # current A11 scratch
T12 = 1_001
T21 = 1_002
T22 = 1_003
U11 = 1_010   # current B11 scratch
U12 = 1_011
U21 = 1_012
U22 = 1_013

# Inner-kernel scratch
SA_SCRATCH = T11   # reuse: T11 and SA can share since no temporal overlap
TMP        = T12   # reuse: TMP and T12 can share

SB_BASE    = 1_100   # up to 8 cells
SC_BASE    = 1_120   # 32 cells (TJ=4)

# Four C-block accumulators
ACC_BASE = 50_000


def A_at(i, j): return A_BASE + i * N + j
def B_at(i, j): return B_BASE + i * N + j
def C_at(i, j): return C_BASE + i * N + j


def A_blk(bi, bj, ii, jj): return A_at(bi * H + ii, bj * H + jj)
def B_blk(bi, bj, ii, jj): return B_at(bi * H + ii, bj * H + jj)


def DA_at(idx, ii, jj):
    """Derived A operand idx in 0..4."""
    return DA_BASE + idx * 64 + ii * H + jj


def DB_at(idx, ii, jj):
    return DB_BASE + idx * 64 + ii * H + jj


def acc_at(blk_idx, ii, jj):
    return ACC_BASE + blk_idx * 64 + ii * H + jj


def emit_derive_A(lines):
    """One pass: read each A cell once, compute D1..D5 and also copy A11/A22
    sub-blocks into "DA[5]"=A11_OP and "DA[6]"=A22_OP slots for M3/M4 use.

    Indexing:
      D0 = A11+A22 (M1)
      D1 = A21+A22 (M2)
      D2 = A11+A12 (M5)
      D3 = A21-A11 (M6)
      D4 = A12-A22 (M7)
      D5 = A11      (M3) — copy of A11
      D6 = A22      (M4) — copy of A22
    """
    for ii in range(H):
        for jj in range(H):
            a11 = A_blk(0, 0, ii, jj)
            a12 = A_blk(0, 1, ii, jj)
            a21 = A_blk(1, 0, ii, jj)
            a22 = A_blk(1, 1, ii, jj)
            # Stream A inputs into 4 scratch slots (1 read each).
            lines.append(f"copy {T11},{a11}")
            lines.append(f"copy {T12},{a12}")
            lines.append(f"copy {T21},{a21}")
            lines.append(f"copy {T22},{a22}")
            # Compute derivatives using cheap scratch reads.
            lines.append(f"add {DA_at(0, ii, jj)},{T11},{T22}")  # D0 = A11+A22
            lines.append(f"add {DA_at(1, ii, jj)},{T21},{T22}")  # D1 = A21+A22
            lines.append(f"add {DA_at(2, ii, jj)},{T11},{T12}")  # D2 = A11+A12
            lines.append(f"sub {DA_at(3, ii, jj)},{T21},{T11}")  # D3 = A21-A11
            lines.append(f"sub {DA_at(4, ii, jj)},{T12},{T22}")  # D4 = A12-A22
            lines.append(f"copy {DA_at(5, ii, jj)},{T11}")        # D5 = A11
            lines.append(f"copy {DA_at(6, ii, jj)},{T22}")        # D6 = A22


def emit_derive_B(lines):
    """Same for B, with:
      DB0 = B11+B22 (M1)
      DB1 = B11      (M2) — copy
      DB2 = B12-B22 (M3)
      DB3 = B21-B11 (M4)
      DB4 = B22     (M5)
      DB5 = B11+B12 (M6)
      DB6 = B21+B22 (M7)
    """
    for ii in range(H):
        for jj in range(H):
            b11 = B_blk(0, 0, ii, jj)
            b12 = B_blk(0, 1, ii, jj)
            b21 = B_blk(1, 0, ii, jj)
            b22 = B_blk(1, 1, ii, jj)
            lines.append(f"copy {U11},{b11}")
            lines.append(f"copy {U12},{b12}")
            lines.append(f"copy {U21},{b21}")
            lines.append(f"copy {U22},{b22}")
            lines.append(f"add {DB_at(0, ii, jj)},{U11},{U22}")  # B11+B22
            lines.append(f"copy {DB_at(1, ii, jj)},{U11}")        # B11
            lines.append(f"sub {DB_at(2, ii, jj)},{U12},{U22}")  # B12-B22
            lines.append(f"sub {DB_at(3, ii, jj)},{U21},{U11}")  # B21-B11
            lines.append(f"copy {DB_at(4, ii, jj)},{U22}")        # B22
            lines.append(f"add {DB_at(5, ii, jj)},{U11},{U12}")  # B11+B12
            lines.append(f"add {DB_at(6, ii, jj)},{U21},{U22}")  # B21+B22


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

    # Phase 1: derive A and B operands using one-pass streaming.
    emit_derive_A(lines)
    emit_derive_B(lines)

    # Phase 2: 7 inner kernels.  A operand is DA[i], B operand is DB[i].
    DA = lambda idx: (lambda ii, jj: DA_at(idx, ii, jj))
    DB = lambda idx: (lambda ii, jj: DB_at(idx, ii, jj))

    plans = [
        # (DA_idx, DB_idx, contributions)
        (0, 0, [(0, '+'), (3, '+')]),  # M1: D0*DB0 -> C11+, C22+
        (1, 1, [(2, '+'), (3, '-')]),  # M2: D1*DB1 -> C21+, C22-
        (5, 2, [(1, '+'), (3, '+')]),  # M3: D5(A11)*DB2 -> C12+, C22+
        (6, 3, [(0, '+'), (2, '+')]),  # M4: D6(A22)*DB3 -> C11+, C21+
        (2, 4, [(0, '-'), (1, '+')]),  # M5: D2*DB4 -> C11-, C12+
        (3, 5, [(3, '+')]),             # M6: D3*DB5 -> C22+
        (4, 6, [(0, '+')]),             # M7: D4*DB6 -> C11+
    ]

    init_flags = [False, False, False, False]
    for da, db, contribs in plans:
        emit_inner_8x8_streaming(lines, DA(da), DB(db), contribs, init_flags, TJ)

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
