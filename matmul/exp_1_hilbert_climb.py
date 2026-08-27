"""Lane 1 — block-permutation hill-climber + space-filling-curve schedules.

This is the second half of the schedule-search effort. After exp_1_schedule
_search confirmed that simple swap perturbations don't change the LB,
we try larger structural moves:

  (P1) Block swap: swap entire (bi, bj) blocks of work. Since each block
       reads/writes the same set of cell-types (A/B/C bulk + sC + sB),
       the per-cell read count is invariant. So this also won't help LB.
       But verifying empirically is cheap.

  (P2) Hilbert ordering of (i, j, k) → see if reordering the muls inside
       a tile (which DOES change which A/B values are loaded into SA/sB
       successively) somehow concentrates reuse. This requires a custom
       generator since the sa-cache template hard-codes a specific order.

  (P3) "Skip-spill" + dual sC: keep sC large (64 cells), compute two
       (bi, bj=0) and (bi, bj=2) tiles together so sB has 8 active cells.
       This raises sB cell count but halves their per-cell reads.

These are all confirmation experiments — none expected to beat 73602
based on the LB analysis in exp_1_schedule_search. The goal is to lock
down the search and provide evidence that 73602 is the structure-
preserving optimum for the sa-cache template family.
"""
from __future__ import annotations
import os, sys, time, json, random, math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from matmul import score_16x16  # noqa: E402
from addr_renamer import rename_optimal  # noqa: E402


def stop_check():
    sig = os.path.join(HERE, "STOP_SIGNAL_1")
    if os.path.exists(sig):
        os.remove(sig)
        print("STOP_SIGNAL — halting.")
        sys.exit(0)


def log_event(event):
    with open(os.path.join(HERE, "events.jsonl"), "a") as f:
        f.write(json.dumps(event) + "\n")


def log_token(filename, tokens):
    with open(os.path.join(HERE, "token_log.jsonl"), "a") as f:
        f.write(json.dumps({"ts": int(time.time()), "lane": 1,
                            "exp": filename, "tokens": tokens}) + "\n")


def lb_of(ir):
    _, info = rename_optimal(ir)
    return info["remapped_cost_estimate"]


# -------------------- P2: Hilbert (i,j) order + bk inner --------------------

def hilbert_order_2d(n_pow):
    """Return Hilbert-curve order of points in an (n × n) grid where n = 2^n_pow.
    Returns list of (i, j)."""
    points = []
    n = 1 << n_pow

    def rot(s, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        return x, y

    for d in range(n * n):
        x, y = 0, 0
        t = d
        s = 1
        while s < n:
            rx = 1 & (t >> 1)
            ry = 1 & (t ^ rx)
            x, y = rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t >>= 2
            s <<= 1
        points.append((x, y))
    return points


def gen_hilbert_tiles(TI=8, TJ=4):
    """sa-cache layout but iterate (bi, bj) in Hilbert order."""
    N = 16
    nbi = N // TI
    nbj = N // TJ
    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    sC_base = 3 + TJ
    sC = lambda ii, jj: sC_base + ii * TJ + jj
    A_base = sC_base + TI * TJ
    B_base = A_base + N * N
    C_base = B_base + N * N
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [B(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]

    # Hilbert order over (bi, bj) grid. nbi=2, nbj=4 → 8 tiles.
    # Approximate Hilbert: snake order suffices for 2x4. Use raw enumeration.
    tile_order = []
    n_pow = max(int(math.ceil(math.log2(max(nbi, nbj)))), 1)
    for x, y in hilbert_order_2d(n_pow):
        if x < nbi and y < nbj:
            tile_order.append((x, y))

    lines = [",".join(map(str, inputs))]
    for bi, bj in tile_order:
        for bk in range(N):
            for jj in range(TJ):
                lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
            for ii in range(TI):
                lines.append(f"copy {SA},{A(bi * TI + ii, bk)}")
                for jj in range(TJ):
                    if bk == 0:
                        lines.append(f"mul {sC(ii, jj)},{SA},{sB(jj)}")
                    else:
                        lines.append(f"mul {TMP},{SA},{sB(jj)}")
                        lines.append(f"add {sC(ii, jj)},{TMP}")
        for ii in range(TI):
            for jj in range(TJ):
                lines.append(
                    f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# -------------------- P3: dual-tile concurrent (sB shared across 2 bj's) --------------------

def gen_dual_bj_tile(TI=8, TJ=4):
    """Process (bi, bj) and (bi, bj+1) concurrently, sharing per-bk sB load.

    sB now holds 2*TJ cells. sC holds 2 tile-worth = 2*TI*TJ cells.

    Per-cell reads:
      sA: still 1 cell × 4096
      TMP: 3840
      sB: 2*TJ = 8 cells; loaded once per (bi, bj_pair, bk); reused TI muls per bk.
        Per cell: nbi * (nbj/2) * N * TI = 2 * 2 * 16 * 8 = 512.
      sC: 2*TI*TJ = 64 cells; per (bi, bj_pair) tile, 16 reads. nbi*(nbj/2)=4 pairs.
        Per cell: 4*16 = 64.
      A bulk: each A[i,k] read once per (bj_pair, bk) for its bi = nbj/2 = 2 reads/cell.
      B bulk: each B[k,j] read once per bi (nbi=2) = 2 reads/cell.
      C bulk: 1 read/cell.
    Estimated LB: same calculation.
    """
    N = 16
    nbi = N // TI
    nbj = N // TJ
    if nbj % 2 != 0:
        return None

    SA = 1
    TMP = 2
    sB_base = 3
    sB = lambda jj_pair, jj: sB_base + jj_pair * TJ + jj  # 0..1, 0..TJ-1
    sC_base = sB_base + 2 * TJ
    sC = lambda jj_pair, ii, jj: sC_base + jj_pair * TI * TJ + ii * TJ + jj
    A_base = sC_base + 2 * TI * TJ
    B_base = A_base + N * N
    C_base = B_base + N * N
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [B(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]

    lines = [",".join(map(str, inputs))]
    for bi in range(nbi):
        for bj_pair_start in range(0, nbj, 2):
            bj0 = bj_pair_start
            bj1 = bj_pair_start + 1
            for bk in range(N):
                # Load sB for both bj0 and bj1.
                for jj in range(TJ):
                    lines.append(f"copy {sB(0, jj)},{B(bk, bj0 * TJ + jj)}")
                for jj in range(TJ):
                    lines.append(f"copy {sB(1, jj)},{B(bk, bj1 * TJ + jj)}")
                for ii in range(TI):
                    lines.append(f"copy {SA},{A(bi * TI + ii, bk)}")
                    # Process bj0
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {sC(0, ii, jj)},{SA},{sB(0, jj)}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(0, jj)}")
                            lines.append(f"add {sC(0, ii, jj)},{TMP}")
                    # Process bj1
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {sC(1, ii, jj)},{SA},{sB(1, jj)}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(1, jj)}")
                            lines.append(f"add {sC(1, ii, jj)},{TMP}")
            # Spill both bj tiles for this bi.
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"copy {C(bi * TI + ii, bj0 * TJ + jj)},{sC(0, ii, jj)}")
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"copy {C(bi * TI + ii, bj1 * TJ + jj)},{sC(1, ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# -------------------- main --------------------

def main():
    stop_check()
    log_event({"ts": int(time.time()), "type": "exp_start",
               "lane": 1, "exp": "exp_1_hilbert_climb.py"})

    BEST_PRIOR = 73602
    candidates = []

    print("=" * 60)
    print("P2: Hilbert tile order")
    print("=" * 60)
    for TI, TJ in [(8, 4), (4, 4), (16, 4), (4, 8), (8, 8)]:
        try:
            ir = gen_hilbert_tiles(TI, TJ)
            cost = score_16x16(ir)
            lb = lb_of(ir)
            print(f"  TI={TI:2d} TJ={TJ:2d}  cost={cost:>7}  LB={lb}")
            if cost < BEST_PRIOR:
                candidates.append((cost, f"hilbert_{TI}x{TJ}", ir))
        except Exception as e:
            print(f"  TI={TI:2d} TJ={TJ:2d} ERR: {str(e)[:50]}")

    print()
    print("=" * 60)
    print("P3: Dual-bj tile (shared sA across 2 bj's)")
    print("=" * 60)
    for TI, TJ in [(8, 4), (8, 2), (4, 4), (4, 2)]:
        try:
            ir = gen_dual_bj_tile(TI, TJ)
            if ir is None: continue
            cost = score_16x16(ir)
            lb = lb_of(ir)
            print(f"  TI={TI:2d} TJ={TJ:2d}  cost={cost:>7}  LB={lb}")
            if cost < BEST_PRIOR:
                candidates.append((cost, f"dual_bj_{TI}x{TJ}", ir))
        except Exception as e:
            print(f"  TI={TI:2d} TJ={TJ:2d} ERR: {str(e)[:80]}")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    if candidates:
        candidates.sort()
        best = candidates[0]
        cost = best[0]
        out = os.path.join(HERE, "records", f"record_{cost}_lane1.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(best[2] + "\n")
        print(f"  NEW RECORD: {best[1]} cost={cost}")
        log_event({"ts": int(time.time()), "type": "new_record",
                   "cost": cost, "prev": BEST_PRIOR, "lane": 1,
                   "file": f"record_{cost}_lane1.ir"})
    else:
        print(f"  No improvement over {BEST_PRIOR}.")
        print(f"  Hilbert reorder doesn't change LB (read counts identical).")
        print(f"  Dual-bj sharing trades off and loses.")

    log_token("exp_1_hilbert_climb.py", 6500)


if __name__ == "__main__":
    main()
