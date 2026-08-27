"""Lane 1: Streaming Strassen with C=A aliasing and direct-read for copy-Mi.

Three variants tried (all sweep TJ in {1,2,4,8}):
  V1: baseline streaming Strassen with materialized A_OP / B_OP for every Mi
      (this is the same as exp_0_strassen_v8; sanity-check baseline 100,683).
  V2: alias C with A — C output addresses == A input addresses.  Renamer then
      sees a single 256-cell region with 5 reads each (4 derivations + 1 exit)
      instead of separate A (4 reads) + C (1 read) regions.  Predicted ~3.4k.
  V3: skip materializing the operand for "copy"-Mi cases (M2 reads B11 directly,
      M3 reads A11 directly, M4 reads A22, M5 reads B22).  This avoids 64 copy
      writes per such Mi (4 of them) but adds direct bulk reads inside the
      inner kernel — usually a wash, but worth measuring.
  V4: V2 + V3 combined.

If any beats 73602, save as records/record_<cost>_lane1.ir.
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
C_BASE = 120_000  # only used when alias_C_with_A=False

A_OP = 60_000
B_OP = 60_064

SA_SCRATCH = 1_000
TMP        = 1_001
SB_BASE    = 1_002
SC_BASE    = 1_010


def A_at(i, j): return A_BASE + i * N + j
def B_at(i, j): return B_BASE + i * N + j
def A_blk(bi, bj, ii, jj): return A_at(bi * H + ii, bj * H + jj)
def B_blk(bi, bj, ii, jj): return B_at(bi * H + ii, bj * H + jj)


def Aop_at(ii, jj): return A_OP + ii * H + jj
def Bop_at(ii, jj): return B_OP + ii * H + jj


# ---------------------------------------------------------------------------
# Helpers parameterized on alias and the C-block address function
# ---------------------------------------------------------------------------

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


def emit_inner_streaming(lines, A_fn, B_fn, contributions, init_flags,
                         C_block_at, TJ):
    """sa-cache style 8x8 inner kernel; output streams directly into C blocks
    as additive (or subtractive, or initial copy) contributions.
    """
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


def build_strassen_ir(TJ=4, alias_C_with_A=False, skip_copy_materialize=False):
    """Build streaming Strassen IR.

    Args:
      TJ: inner kernel jj-tile size (1, 2, 4, 8)
      alias_C_with_A: if True, C output addresses == A input addresses
                     (the 256 A cells are reused as C accumulators).
      skip_copy_materialize: if True, for Mi where one operand is a single
                     block (no add/sub), do not materialize a separate A_OP
                     or B_OP — read directly from the original bulk in the
                     inner kernel.
    """
    inputs = ([A_at(i, j) for i in range(N) for j in range(N)] +
              [B_at(i, j) for i in range(N) for j in range(N)])

    if alias_C_with_A:
        # C[i,j] lives at the same address as A[i,j].  After all derivations
        # of A are done (every Mi that uses A's cells as part of an add/sub),
        # the inner kernel writes the FIRST contribution into the A address,
        # which is allowed because the dependency is sequential per Mi — by
        # the time we start writing to C[i,j], all reads of A[i,j] have
        # already happened in the materialization step earlier in THIS Mi.
        # CAUTION: A cells are read across Mi (each Mi materializes its own
        # derived operand from original A bulk).  We must finish all Mi's
        # derivation phase BEFORE writing to a C/A-aliased cell.  But the
        # streaming pattern interleaves derivation+inner per Mi.  So if M1
        # writes to C11 (= A11 in alias) during its inner kernel, then M3's
        # later derivation of A11 reads the WRONG value.
        #
        # SOLUTION: do all 7 derivations FIRST (no, derivations need A_OP
        # space which we want to reuse).  Alternative: alias only the
        # specific C cells whose A-alias has finished its last read.
        #
        # Simpler solution: alias C with A only after all Mi computations
        # finish their bulk-A reads.  In v8, every Mi reads bulk-A at the
        # *materialization* step (via the lhs_a/lhs_b lambdas).  That
        # happens at the START of each Mi.  After M7 materialization, no
        # more bulk-A reads are needed — the inner kernels then just write
        # to C blocks.  But each Mi's inner kernel writes to C blocks
        # immediately after that Mi's materialization, so M1's inner
        # kernel writes to C11+C22 BEFORE M3 ever reads A11.
        #
        # Therefore: we cannot naively alias C=A while keeping the
        # streaming order intact.  We need to either:
        # (a) defer all C-writes until after M7, OR
        # (b) use a different alias scheme (e.g., alias C with B-cells
        #     whose final read happens earlier).
        #
        # Simpler approach we'll try: alias C with A but only for cells
        # that A12/A21 (the cells used by only 2 Mi each — M5/M7 for A12,
        # M2/M6 for A21).  After both their Mi materializations finish,
        # those A cells are dead and can host C cells.
        #
        # Even more conservative: hand-craft the schedule so that:
        # 1. Do M1..M7 materializations FIRST, then run all 7 inner kernels.
        #    Cost: 14 derived 8x8 operands materialized = 14*64 = 896 cells.
        #    That's exactly the v3-style flat layout that gave 100,683+.
        # OR
        # 2. Run a different streaming order: M5 then M7 (A12 dead), M2 then
        #    M6 (A21 dead), M1 then M4 (A11+A22 used in three Mi total),
        #    M3 (A11 reread)... messy and most A cells get a 5th read.
        raise NotImplementedError("alias_C_with_A=True needs schedule rework")

    C_base = C_BASE
    def C_at(i, j): return C_base + i * N + j

    def C_block_at(blk_idx, ii, jj):
        if blk_idx == 0: return C_at(ii, jj)
        if blk_idx == 1: return C_at(ii, H + jj)
        if blk_idx == 2: return C_at(H + ii, jj)
        if blk_idx == 3: return C_at(H + ii, H + jj)
        raise ValueError(blk_idx)

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
        # Choose A operand source for inner kernel
        if la == "copy":
            if skip_copy_materialize:
                A_kernel_fn = lhs_a  # read directly from bulk
            else:
                emit_copy_block(lines, Aop_at, lhs_a)
                A_kernel_fn = Aop_at
        else:
            emit_combo(lines, Aop_at, lhs_a, lhs_b, la)
            A_kernel_fn = Aop_at

        if ra == "copy":
            if skip_copy_materialize:
                B_kernel_fn = rhs_a
            else:
                emit_copy_block(lines, Bop_at, rhs_a)
                B_kernel_fn = Bop_at
        else:
            emit_combo(lines, Bop_at, rhs_a, rhs_b, ra)
            B_kernel_fn = Bop_at

        emit_inner_streaming(lines, A_kernel_fn, B_kernel_fn, contribs,
                             init_flags, C_block_at, TJ)

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Variant: defer all 7 inner kernels until after all 14 materializations
# Then alias C with A is sound: the 256 A cells are dead by Mi loop start.
# But the 14 derived 8x8 operands cost 896 cells.
# ---------------------------------------------------------------------------

def build_strassen_ir_deferred(TJ=4, alias_C_with_A=True):
    """Materialize all 14 derived operands first, THEN run 7 inner kernels.

    With deferred-inner, by the time the first inner kernel runs, NO bulk-A
    cell will be read again — so we can alias C=A safely.

    Layout of derived operands: 14 distinct 8x8 regions = 14*64 = 896 cells.
    But the renamer will assign optimal addresses, so layout overhead =
    sum of (read_count[op_cell] * cost(rank)) = each op cell is read 8 times
    (once per inner kernel iteration in column k) → it's the inner kernel
    that reads the operands the most.
    """
    inputs = ([A_at(i, j) for i in range(N) for j in range(N)] +
              [B_at(i, j) for i in range(N) for j in range(N)])

    if alias_C_with_A:
        def C_at(i, j): return A_at(i, j)
    else:
        def C_at(i, j): return C_BASE + i * N + j

    outputs = [C_at(i, j) for i in range(N) for j in range(N)]

    def C_block_at(blk_idx, ii, jj):
        if blk_idx == 0: return C_at(ii, jj)
        if blk_idx == 1: return C_at(ii, H + jj)
        if blk_idx == 2: return C_at(H + ii, jj)
        if blk_idx == 3: return C_at(H + ii, H + jj)
        raise ValueError(blk_idx)

    lines = [",".join(map(str, inputs))]

    A11 = lambda ii, jj: A_blk(0, 0, ii, jj)
    A12 = lambda ii, jj: A_blk(0, 1, ii, jj)
    A21 = lambda ii, jj: A_blk(1, 0, ii, jj)
    A22 = lambda ii, jj: A_blk(1, 1, ii, jj)
    B11 = lambda ii, jj: B_blk(0, 0, ii, jj)
    B12 = lambda ii, jj: B_blk(0, 1, ii, jj)
    B21 = lambda ii, jj: B_blk(1, 0, ii, jj)
    B22 = lambda ii, jj: B_blk(1, 1, ii, jj)

    # 14 derived operand regions, each 64 cells, packed sequentially
    # starting at A_OP.  Each gets a distinct slot.
    OP_BASE = 60_000
    derived = {}
    for k in range(14):
        derived[k] = lambda ii, jj, k=k: OP_BASE + k * 64 + ii * H + jj

    # Mi i (1-indexed) → (lhs_kind, lhs_a, lhs_b, rhs_kind, rhs_a, rhs_b,
    #                     contribs, lhs_idx_or_None, rhs_idx_or_None)
    # We give each (non-copy) derived operand a slot 0..13.
    # Copy operands read directly from bulk (skip_copy_materialize=True).
    #
    # For deferred mode, we MUST materialize EVERY operand because the
    # inner kernels are run AFTER all materializations, and (in the alias
    # case) by then bulk-A cells host C output values (overwritten).
    # So even "copy" operands need an A_OP/B_OP buffer.
    #
    # Wait — that's only for A-side.  B is not aliased, so we can read B
    # directly.  Similarly, for the A-side: only operands that depend on
    # A (which is everything on the A-side of each Mi) need to be
    # materialized BEFORE we start writing to A=C.

    # For simplicity, materialize everything: 14 derived 8x8 operands.
    plans = [
        # (la, lhs_a, lhs_b, ra, rhs_a, rhs_b, contribs)
        ("add", A11, A22, "add", B11, B22, [(0, '+'), (3, '+')]),  # M1
        ("add", A21, A22, "copy", B11, None, [(2, '+'), (3, '-')]),  # M2
        ("copy", A11, None, "sub", B12, B22, [(1, '+'), (3, '+')]),  # M3
        ("copy", A22, None, "sub", B21, B11, [(0, '+'), (2, '+')]),  # M4
        ("add", A11, A12, "copy", B22, None, [(0, '-'), (1, '+')]),  # M5
        ("sub", A21, A11, "add", B11, B12, [(3, '+')]),  # M6
        ("sub", A12, A22, "add", B21, B22, [(0, '+')]),  # M7
    ]

    # Phase 1: materialize all 14 (or 10) derived operands.
    A_kernel_fns = []
    B_kernel_fns = []
    slot = 0
    for (la, lhs_a, lhs_b, ra, rhs_a, rhs_b, contribs) in plans:
        if la == "copy":
            # MUST materialize on A side (alias case) so that bulk A is
            # safe to overwrite
            d = derived[slot]; slot += 1
            emit_copy_block(lines, d, lhs_a)
            A_kernel_fns.append(d)
        else:
            d = derived[slot]; slot += 1
            emit_combo(lines, d, lhs_a, lhs_b, la)
            A_kernel_fns.append(d)
        if ra == "copy":
            # B not aliased; can read directly from bulk in inner kernel.
            B_kernel_fns.append(rhs_a)
        else:
            d = derived[slot]; slot += 1
            emit_combo(lines, d, rhs_a, rhs_b, ra)
            B_kernel_fns.append(d)

    n_derived = slot  # number of operand slots actually used
    assert slot <= 14

    # Phase 2: run 7 inner kernels, each writing into C blocks.
    init_flags = [False, False, False, False]
    for idx, (la, lhs_a, lhs_b, ra, rhs_a, rhs_b, contribs) in enumerate(plans):
        emit_inner_streaming(lines, A_kernel_fns[idx], B_kernel_fns[idx],
                             contribs, init_flags, C_block_at, TJ)

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines), n_derived


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def log_event(d):
    with open(os.path.join(HERE, "events.jsonl"), "a") as f:
        f.write(json.dumps(d) + "\n")


def log_token(d):
    with open(os.path.join(HERE, "token_log.jsonl"), "a") as f:
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

    print("=" * 60)
    print("V1: streaming (baseline, no alias, no skip-copy)")
    for TJ in [1, 2, 4, 8]:
        ir = build_strassen_ir(TJ=TJ, alias_C_with_A=False,
                              skip_copy_materialize=False)
        try:
            new_ir, info = rename_optimal(ir)
            cost = score_16x16(new_ir)
            results.append(("V1", TJ, cost, new_ir))
            print(f"  TJ={TJ}: {cost:,}")
        except Exception as e:
            print(f"  TJ={TJ}: ERROR {e}")

    print("\nV3: streaming with skip-copy-materialize")
    for TJ in [1, 2, 4, 8]:
        ir = build_strassen_ir(TJ=TJ, alias_C_with_A=False,
                              skip_copy_materialize=True)
        try:
            new_ir, info = rename_optimal(ir)
            cost = score_16x16(new_ir)
            results.append(("V3", TJ, cost, new_ir))
            print(f"  TJ={TJ}: {cost:,}")
        except Exception as e:
            print(f"  TJ={TJ}: ERROR {e}")

    print("\nV5 (deferred, alias C=A):")
    for TJ in [1, 2, 4, 8]:
        try:
            ir, n_derived = build_strassen_ir_deferred(TJ=TJ,
                                                      alias_C_with_A=True)
            new_ir, info = rename_optimal(ir)
            cost = score_16x16(new_ir)
            results.append((f"V5-defAlias", TJ, cost, new_ir))
            print(f"  TJ={TJ}: {cost:,} (n_derived={n_derived})")
        except Exception as e:
            print(f"  TJ={TJ}: ERROR {e}")

    print("\nV6 (deferred, no alias):")
    for TJ in [1, 2, 4, 8]:
        try:
            ir, n_derived = build_strassen_ir_deferred(TJ=TJ,
                                                      alias_C_with_A=False)
            new_ir, info = rename_optimal(ir)
            cost = score_16x16(new_ir)
            results.append((f"V6-defNoAlias", TJ, cost, new_ir))
            print(f"  TJ={TJ}: {cost:,} (n_derived={n_derived})")
        except Exception as e:
            print(f"  TJ={TJ}: ERROR {e}")

    print("\n" + "=" * 60)
    if results:
        best = min(results, key=lambda r: r[2])
        name, TJ, cost, ir = best
        print(f"BEST: {name} TJ={TJ}: {cost:,} (vs record {PREV_RECORD:,})")
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
            print(f"no new record (best {cost:,} > {PREV_RECORD:,})")
            log_event({
                "ts": int(time.time()),
                "type": "exp_done",
                "lane": 1,
                "best_cost": cost,
                "best_variant": name,
                "best_TJ": TJ,
                "exp": os.path.basename(__file__),
            })
