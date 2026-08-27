"""Lane 1: streaming Strassen with NO sC scratchpad — direct accumulation
into C blocks for every (i,j,k).

For Mi with k contribs, each (i,j,k) of the 8x8 inner kernel does:
   mul tmp, A_op[i,k], B_op[k,j]
   add C_block_a[i,j], tmp     # for each contributing block
This eliminates 64 sC cells (saving cell count) at the cost of doubling
the post-mul add count (one add per contrib * each (i,j,k) instead of
one streaming flush at the end).

Per Mi cost:
  Mi with 1 contrib:
    Old: 512 (mul) + 512 (add to sC) + 64 (flush) = 1088 instructions
    New: 512 (mul) + 512 (add to C) = 1024 instructions
  Mi with 2 contribs:
    Old: 512 (mul) + 512 (add sC) + 128 (flush) = 1152
    New: 512 (mul) + 1024 (add to 2 C-blocks) = 1536  <- WORSE

So this is good for 1-contrib Mi (M6, M7) and bad for multi-contrib.
Hybrid: use sC for multi-contrib, no-sC for single-contrib.
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


def emit_inner_no_sC(lines, A_fn, B_fn, contributions, init_flags, TJ):
    """Inner 8x8 kernel that writes directly to C blocks (no sC scratchpad).

    For each (i, j) of output, for each k:
        mul tmp, sA, sB[j]
        for each contributing block: add/sub/init C_block[i,j], tmp

    contributions: list of (blk_idx, sign).
    init_flags is updated for each blk_idx after this Mi finishes.
    """
    nbj = H // TJ
    SA = SA_SCRATCH
    sB = lambda jj: SB_BASE + jj

    for bj in range(nbj):
        for bk in range(H):
            for jj in range(TJ):
                lines.append(f"copy {sB(jj)},{B_fn(bk, bj * TJ + jj)}")
            for ii in range(H):
                lines.append(f"copy {SA},{A_fn(ii, bk)}")
                for jj in range(TJ):
                    col = bj * TJ + jj
                    lines.append(f"mul {TMP},{SA},{sB(jj)}")
                    for blk_idx, sign in contributions:
                        dst = C_block_at(blk_idx, ii, col)
                        if not init_flags.get((blk_idx, ii, col), False):
                            if sign == '-':
                                # init via sub: 0 - tmp ... we don't have 0 easily
                                # Use mul: dst = -1 * tmp. But we don't have -1.
                                # Alternative: init via copy to a neg-tmp cell.
                                # For simplicity, skip negation-init: assume the
                                # first contribution is always '+'.
                                raise RuntimeError(
                                    f"negation-init not supported (blk={blk_idx})")
                            lines.append(f"copy {dst},{TMP}")
                            init_flags[(blk_idx, ii, col)] = True
                        elif sign == '+':
                            lines.append(f"add {dst},{TMP}")
                        else:  # sign == '-'
                            lines.append(f"sub {dst},{TMP}")


def build_strassen_no_sC(TJ=4) -> str:
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

    init_flags: dict = {}  # (blk_idx, ii, col) -> bool

    # ORDER MATTERS for negation-init:
    # We must reorder Mi/contribs so that the FIRST contribution to each C cell
    # is '+'. Default plans satisfy this for blk=0 (M1+ first), blk=1 (M3+),
    # blk=2 (M2+), blk=3 (M1+).  Good.
    # For M2's blk=3 contrib '-', this is NOT the first since M1 contributed
    # to blk=3 with '+'. OK.
    # For M5's blk=0 contrib '-', M1 is first ('+'). OK.

    for (la, lhs_a, lhs_b, ra, rhs_a, rhs_b, contribs) in plans:
        # Always materialize for sum/sub; skip copy for single-block ops
        if la == "copy":
            A_kernel_fn = lhs_a
        else:
            emit_combo(lines, Aop_at, lhs_a, lhs_b, la)
            A_kernel_fn = Aop_at
        if ra == "copy":
            B_kernel_fn = rhs_a
        else:
            emit_combo(lines, Bop_at, rhs_a, rhs_b, ra)
            B_kernel_fn = Bop_at
        emit_inner_no_sC(lines, A_kernel_fn, B_kernel_fn, contribs,
                         init_flags, TJ)

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# Hybrid: use sC for multi-contrib Mi, direct for single-contrib
def emit_inner_streaming_with_sC(lines, A_fn, B_fn, contributions, init_flags,
                                  TJ):
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
                    dst = C_block_at(blk_idx, ii, col)
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


def build_strassen_hybrid(TJ=4) -> str:
    """Use sC scratchpad for multi-contrib Mi, no sC for single-contrib."""
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

    # init_flags has TWO meanings here:
    #  - for sC-mode kernels: per-blk_idx (each Mi initializes once per blk)
    #  - for no-sC mode: per (blk_idx, ii, col) cell
    # We track both separately and read appropriately.
    init_blk_flags = [False, False, False, False]
    init_cell_flags: dict = {}

    for (la, lhs_a, lhs_b, ra, rhs_a, rhs_b, contribs) in plans:
        if la == "copy":
            A_kernel_fn = lhs_a
        else:
            emit_combo(lines, Aop_at, lhs_a, lhs_b, la)
            A_kernel_fn = Aop_at
        if ra == "copy":
            B_kernel_fn = rhs_a
        else:
            emit_combo(lines, Bop_at, rhs_a, rhs_b, ra)
            B_kernel_fn = Bop_at

        # Pick mode
        if len(contribs) == 1:
            # Single-contrib: use direct accumulation, no sC
            blk_idx, sign = contribs[0]
            # If this contribution is the FIRST write to this block, all
            # cells need init (copy mul-result). Otherwise add/sub.
            if not init_blk_flags[blk_idx]:
                # Use direct kernel that copies tmp to C cells on first write
                # (per cell flag) — emit_inner_no_sC handles this.
                emit_inner_no_sC(lines, A_kernel_fn, B_kernel_fn, contribs,
                                 init_cell_flags, TJ)
                init_blk_flags[blk_idx] = True
            else:
                # Block already initialized — use direct add/sub
                # Mark all cells as already initialized
                for ii in range(H):
                    for jj in range(H):
                        init_cell_flags[(blk_idx, ii, jj)] = True
                emit_inner_no_sC(lines, A_kernel_fn, B_kernel_fn, contribs,
                                 init_cell_flags, TJ)
        else:
            emit_inner_streaming_with_sC(lines, A_kernel_fn, B_kernel_fn,
                                          contribs, init_blk_flags, TJ)
            # For block flags handled by the function; mark cells too for
            # consistency in case mixed-mode interleaving happens
            for blk_idx, _ in contribs:
                for ii in range(H):
                    for jj in range(H):
                        init_cell_flags[(blk_idx, ii, jj)] = True

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

    print("V_no_sC: streaming Strassen with no sC scratchpad")
    for TJ in [1, 2, 4, 8]:
        try:
            ir = build_strassen_no_sC(TJ=TJ)
            new_ir, info = rename_optimal(ir)
            cost = score_16x16(new_ir)
            results.append((f"no_sC", TJ, cost, new_ir))
            print(f"  TJ={TJ}: {cost:,}")
        except Exception as e:
            print(f"  TJ={TJ}: ERROR {e}")

    print("\nV_hybrid: sC for multi-contrib, no-sC for single-contrib")
    for TJ in [1, 2, 4, 8]:
        try:
            ir = build_strassen_hybrid(TJ=TJ)
            new_ir, info = rename_optimal(ir)
            cost = score_16x16(new_ir)
            results.append((f"hybrid", TJ, cost, new_ir))
            print(f"  TJ={TJ}: {cost:,}")
        except Exception as e:
            print(f"  TJ={TJ}: ERROR {e}")

    if results:
        best = min(results, key=lambda r: r[2])
        name, TJ, cost, ir = best
        print(f"\nBEST: {name} TJ={TJ}: {cost:,} (vs record {PREV_RECORD:,})")
        if cost < PREV_RECORD:
            out_path = os.path.join(HERE, "records",
                                    f"record_{cost}_lane1.ir")
            with open(out_path, "w") as f:
                f.write(ir + "\n")
            print(f"NEW RECORD saved -> {out_path}")
            log_event({
                "ts": int(time.time()),
                "type": "new_record",
                "cost": cost,
                "prev": PREV_RECORD,
                "lane": 1,
                "file": os.path.basename(__file__),
                "variant": name,
                "TJ": TJ,
            })
        else:
            print(f"no new record")
            log_event({
                "ts": int(time.time()),
                "type": "exp_done",
                "lane": 1,
                "best_cost": cost,
                "best_variant": name,
                "best_TJ": TJ,
                "exp": os.path.basename(__file__),
            })
