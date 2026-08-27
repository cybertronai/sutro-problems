"""Lane 1 — share sB across bi by reordering loops.

V1 sa_cache uses nest (bi, bj, bk, ii, jj). Each B cell B[k, j] is read
nbi=2 times (once per bi for k=bk, j=bj*TJ+jj fixed).

This variant uses nest (bj, bk, bi, ii, jj). For fixed (bj, bk), sB is
loaded once and reused across all bi. So each B cell read 1 time from
B bulk (nbi× savings on B bulk).

Cost: sC must be alive for all bi simultaneously per (bj). That's
TI*nbi*TJ = N*TJ sC cells (one full column-block of C). For TI=8,TJ=4
that's 16*4 = 64 sC cells (vs V1's 32).

Layout (cost-optimal-ish):
  addr 1:        sA cache
  addr 2:        TMP
  addrs 3..6:    sB (TJ cells)
  addrs 7..70:   sC (N*TJ cells)
  ...A, B, C bulk
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from matmul import score_16x16  # noqa: E402

sys.path.insert(0, HERE)
from addr_renamer import rename_optimal  # noqa: E402

N = 16


def gen_b_share(TI: int, TJ: int) -> str:
    """Loop nest (bj, bk, bi, ii, jj) — sB shared across bi."""
    nbi = N // TI
    nbj = N // TJ

    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    sC_base = 3 + TJ
    # sC indexed by (i_full, jj_local): i_full ∈ [0..N), jj_local ∈ [0..TJ)
    sC = lambda i_full, jj: sC_base + i_full * TJ + jj
    A_base = sC_base + N * TJ
    B_base = A_base + N * N
    C_base = B_base + N * N
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [B(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]

    lines = [",".join(map(str, inputs))]
    for bj in range(nbj):
        for bk in range(N):
            # Load sB once for this (bk, bj)
            for jj in range(TJ):
                lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
            # For each bi, do the inner ii × jj product/accumulate
            for bi in range(nbi):
                for ii in range(TI):
                    i_full = bi * TI + ii
                    lines.append(f"copy {SA},{A(i_full, bk)}")
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {sC(i_full, jj)},{SA},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {sC(i_full, jj)},{TMP}")
        # After all bk: spill sC -> C for this bj
        for i_full in range(N):
            for jj in range(TJ):
                lines.append(f"copy {C(i_full, bj * TJ + jj)},{sC(i_full, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# Variant: share sA across bj. Same idea but reverse roles.
# Loop (bi, bk, bj, ii, jj). For fixed (bi, bk), the inner needs sA per ii
# loaded once; then we sweep bj × jj. But sA cache holds 1 cell — we'd
# reload per ii anyway. So sA-bulk reads stay the same per cell.
# Hmm, actually: in V1 (bi, bj, bk, ii) each A[i,k] is read once per bj
# (because bi fixes i, bk fixes k, bj iterates and we reload sA each ii).
# Total reads per A cell = nbj = 4.
#
# With (bi, bk, bj, ii, jj): for fixed (bi, bk), iterate bj. For each bj,
# load sB and inner loop over ii × jj. We'd reload sA once per (bk, bj, ii)
# → still nbj reloads per A cell. Same A-bulk traffic.
#
# UNLESS we hoist the sA load ABOVE the bj loop, but that requires keeping
# all TI sA cells alive (not just 1). That's the V2 rank-TI variant which
# was 95k. Not helpful.

def gen_a_share(TI: int, TJ: int) -> str:
    """Loop nest (bi, bk, bj, ii, jj) with sA reloaded per (bk, ii).

    Identical A-bulk reads as V1, but B-bulk reads INCREASE (nbi×nbj per
    cell instead of nbi). Probably worse — but try anyway."""
    nbi = N // TI
    nbj = N // TJ

    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    sC_base = 3 + TJ
    sC = lambda ii, jj_full: sC_base + ii * N + jj_full
    A_base = sC_base + TI * N
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
        for bk in range(N):
            for bj in range(nbj):
                for jj in range(TJ):
                    lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
                for ii in range(TI):
                    i_full = bi * TI + ii
                    lines.append(f"copy {SA},{A(i_full, bk)}")
                    for jj in range(TJ):
                        jj_full = bj * TJ + jj
                        if bk == 0:
                            lines.append(f"mul {sC(ii, jj_full)},{SA},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {sC(ii, jj_full)},{TMP}")
        # After all bk: spill sC -> C for this bi
        for ii in range(TI):
            i_full = bi * TI + ii
            for jj_full in range(N):
                lines.append(f"copy {C(i_full, jj_full)},{sC(ii, jj_full)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# Variant: outer-most over bk, then bi, bj. This means sB[bk, bj*TJ+jj]
# is loaded once per (bj, bi) pair when bk is outermost. But sC must be alive
# for ALL (bi, bj) simultaneously = full N×N C tile = 256 cells. Too costly.
# Skip.


def stop_check():
    sig = os.path.join(HERE, "STOP_SIGNAL_1")
    if os.path.exists(sig):
        os.remove(sig)
        print("STOP_SIGNAL — halting.")
        sys.exit(0)


def log_event(event: dict):
    path = os.path.join(HERE, "events.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


def log_token(filename: str, tokens: int):
    path = os.path.join(HERE, "token_log.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps({"ts": int(time.time()), "lane": 1,
                            "exp": filename, "tokens": tokens}) + "\n")


def safe_score(ir):
    try:
        return score_16x16(ir)
    except Exception as e:
        return f"ERR:{e}"


def main():
    stop_check()
    fname = "exp_1_b_share.py"
    log_event({"ts": int(time.time()), "type": "exp_start",
               "lane": 1, "exp": fname})

    BEST_PRIOR = 73602
    print(f"{'variant':>16} {'TI':>3} {'TJ':>3}  {'cost':>10}  {'lb':>10}")
    print("-" * 60)

    results = []
    configs = []
    for TI in [2, 4, 8, 16]:
        for TJ in [2, 4, 8]:
            if N % TI == 0 and N % TJ == 0:
                configs.append(("b_share", TI, TJ, gen_b_share))
                configs.append(("a_share", TI, TJ, gen_a_share))

    for name, TI, TJ, gen in configs:
        try:
            ir = gen(TI, TJ)
        except Exception as e:
            print(f"{name:>16} {TI:>3} {TJ:>3}  GEN_ERR:{e}")
            continue
        cost = safe_score(ir)
        try:
            ir_lb, info = rename_optimal(ir)
            cost_lb = safe_score(ir_lb)
        except Exception as e:
            cost_lb = f"ERR:{e}"
            ir_lb = ir
        print(f"{name:>16} {TI:>3} {TJ:>3}  {cost!r:>10}  {cost_lb!r:>10}")
        if isinstance(cost, int):
            results.append((name, TI, TJ, cost, ir, "raw"))
        if isinstance(cost_lb, int):
            results.append((name, TI, TJ, cost_lb, ir_lb, "renamed"))

    results.sort(key=lambda r: r[3])
    print(f"\nTop 10:")
    for r in results[:10]:
        print(f"  {r[0]} TI={r[1]} TJ={r[2]} ({r[5]})  cost={r[3]:,}")

    if results:
        best = results[0]
        if best[3] < BEST_PRIOR:
            cost = best[3]
            ir = best[4]
            out = os.path.join(HERE, "records", f"record_{cost}_lane1.ir")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                f.write(ir + "\n")
            print(f"\nNEW RECORD: {best[0]} TI={best[1]} TJ={best[2]} "
                  f"({best[5]})  cost={cost} -> {out}")
            log_event({"ts": int(time.time()), "type": "new_record",
                       "cost": cost, "prev": BEST_PRIOR, "lane": 1,
                       "file": f"record_{cost}_lane1.ir"})
        else:
            print(f"\nNo improvement: best={best[3]} >= prior best 73602")

    log_token(fname, 7000)


if __name__ == "__main__":
    main()
