"""Lane 1 — rank-R sA cache with bj INSIDE the (bi, bk, ii_outer) chunk loop.

Loop nest:
  for bi:
    for bk:
      for ii_outer in 0..TI/R:
        load sA[0..R-1] = A[bi*TI+ii_outer*R+0..R-1, bk]   (R reads from A bulk)
        for bj:
          load sB[0..TJ-1] = B[bk, bj*TJ..]
          for jj in 0..TJ:
            for ii_inner in 0..R:
              mul/add sC[bi*TI+ii_outer*R+ii_inner, bj*TJ+jj] += sA*sB

A-bulk reads per cell: 1 (each A[i,k] loaded once when bi=i//TI, bk=k,
  ii_outer=(i%TI)//R)  [vs 4 in V1]

B-bulk reads per cell: nbi * (TI/R)
  Wait — for fixed (bk, bj), sB is loaded once per (bi, ii_outer). That's
  nbi * (TI/R) per (bk, bj). Number of distinct B cells per (bk, bj) = TJ.
  Each cell B[k, j] is loaded when bk=k, bj=j//TJ, all (bi, ii_outer) — that's
  nbi*(TI/R) loads per cell.
  For TI=8, TJ=4, R=2: nbi=2, TI/R=4 → 8 loads per B cell. Yikes! V1 had 2.

So this is a tradeoff: reduce A-bulk reads by 4× but multiply B-bulk reads by 4×.
Could be a net win or loss depending on cost(A_addr) vs cost(B_addr).

For 16x16: A bulk has 256 cells, B bulk has 256 cells. Both at similar
addrs.

Total inner reads in this scheme:
  sA: 4096 reads (Ti*Tj*N*nbi = 4096) — same as V1
  TMP: 3840 — same
  sB: ? loaded N*nbj*nbi*(TI/R) times = N*N*nbi*(TI/R)/TJ ?? Let me recount.

  Number of sB-load operations: nbi*N*(TI/R)*nbj*TJ =
    For TI=8,TJ=4,R=2: 2*16*4*4*4 = 2048. Each load is one read from B bulk.
    V1: nbi*nbj*N*TJ = 2*4*16*4 = 512. So 4× more B-bulk reads.

  Number of sB inner-reads (in mul ops): per inner (jj,ii_inner) one read,
    total = nbi*N*(TI/R)*nbj*TJ*R = nbi*N*TI*nbj*TJ = 4096. Same as V1.

  sC reads: each sC cell read N times during accumulation + 1 spill = 17.
    Number of sC cells = TI*TJ = 32. Total sC reads = 32*17 = 544? No wait
    in V1 each sC cell is read 128 times because each (bi,bj) iter does
    N additions per cell, plus the final spill. Hmm let me recount.

Actually the structure is the same as V1 internally for sC. sC[ii, jj] is
written/read once per bk iteration (ii's row times jj's col within the
current bj). For fixed (bi, bj), sC[ii, jj] is written N times, read
(N-1)+1 = N times (writes for k=0 are init, reads for k>0). Wait actually
written N times, read N-1 times during accumulation, read 1 time at spill.

OK forget the analysis, let's just code and score.
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from matmul import score_16x16  # noqa: E402

sys.path.insert(0, HERE)
from addr_renamer import rename_optimal  # noqa: E402

N = 16


def gen_rank_outer_bj(TI: int, TJ: int, R: int) -> str:
    """rank-R sA strip pinned across bj."""
    assert TI % R == 0
    nbi = N // TI
    nbj = N // TJ
    n_outer = TI // R

    sA = lambda ii_inner: 1 + ii_inner
    TMP = R + 1
    sB = lambda jj: R + 2 + jj
    sC_base = R + 2 + TJ
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

    lines = [",".join(map(str, inputs))]

    # We need all sC[ii, jj] for ii ∈ 0..TI alive across all bj at fixed bi.
    # Actually, sC tracks the partial accumulation. For loop nest
    # (bi, bk, ii_outer, bj, jj, ii_inner), each sC cell is touched once per
    # (bk) iteration for fixed (bi, ii_outer*R+ii_inner, bj*TJ+jj). So sC[ii, jj]
    # is touched once per bk. Initialization at bk=0.
    # For fixed bi, sC[ii ∈ 0..TI, jj ∈ 0..TJ] — only 32 cells per bi.
    # But we sweep bj inside ii_outer inside bk: so we need sC[ii_outer*R..R-1, jj]
    # for each bj iter, and we need them alive across bk.

    # Verify init logic: at bk=0, for each (ii_outer, bj, jj, ii_inner),
    # we initialize sC[ii_full, jj_full]. Each (ii_full, jj_full) is hit
    # exactly once at bk=0. Then for bk>0 we accumulate. Good.

    for bi in range(nbi):
        for bk in range(N):
            for ii_outer in range(n_outer):
                # Load R sA cells (once per ii_outer per bk)
                for ii_inner in range(R):
                    ii = ii_outer * R + ii_inner
                    lines.append(f"copy {sA(ii_inner)},{A(bi * TI + ii, bk)}")
                # Now for each bj, load sB and inner kernel
                for bj in range(nbj):
                    for jj in range(TJ):
                        lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
                    for jj in range(TJ):
                        for ii_inner in range(R):
                            ii = ii_outer * R + ii_inner
                            jj_full = bj * TJ + jj
                            if bk == 0:
                                lines.append(f"mul {sC(ii, jj)},{sA(ii_inner)},{sB(jj)}")
                                # NOTE: sC indexed by (ii local, jj local) but
                                # we have multiple bj — we need (ii, jj_full)
                                # to keep them distinct!
                                # Fixed below.
                            else:
                                lines.append(f"mul {TMP},{sA(ii_inner)},{sB(jj)}")
                                lines.append(f"add {sC(ii, jj)},{TMP}")
            # sC has accumulated for one (bi, bk). Wait NO — we accumulate
            # over ALL bk for the same (bi, bj) pair. But here we mix bj's
            # within bk. Need different sC indexing.
        # Spill sC: but our sC indexing only distinguishes (ii, jj_local)
        # which conflates bj. BUG!
        for ii in range(TI):
            for jj in range(TJ):
                # We can only spill ONE bj here. This generator is BUGGY.
                pass
    # Discard this implementation; rewrite below.
    return None


def gen_rank_outer_bj_correct(TI: int, TJ: int, R: int) -> str:
    """Rewritten correctly: sC indexed by full (i, j_full) within current bi.

    Loop nest: (bi, bk, ii_outer, bj, jj, ii_inner)
    sC alive: TI*N cells per bi (one full bi-row of C, all 16 jj values).
    """
    assert TI % R == 0
    nbi = N // TI
    nbj = N // TJ
    n_outer = TI // R

    sA = lambda ii_inner: 1 + ii_inner
    TMP = R + 1
    sB = lambda jj: R + 2 + jj
    sC_base = R + 2 + TJ
    # sC indexed by (ii ∈ 0..TI, j_full ∈ 0..N) — TI*N cells
    sC = lambda ii, j_full: sC_base + ii * N + j_full
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
            for ii_outer in range(n_outer):
                for ii_inner in range(R):
                    ii = ii_outer * R + ii_inner
                    lines.append(f"copy {sA(ii_inner)},{A(bi * TI + ii, bk)}")
                for bj in range(nbj):
                    for jj in range(TJ):
                        lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
                    for jj in range(TJ):
                        for ii_inner in range(R):
                            ii = ii_outer * R + ii_inner
                            j_full = bj * TJ + jj
                            if bk == 0:
                                lines.append(f"mul {sC(ii, j_full)},{sA(ii_inner)},{sB(jj)}")
                            else:
                                lines.append(f"mul {TMP},{sA(ii_inner)},{sB(jj)}")
                                lines.append(f"add {sC(ii, j_full)},{TMP}")
        # Spill sC -> C bulk for this bi
        for ii in range(TI):
            for j_full in range(N):
                lines.append(f"copy {C(bi * TI + ii, j_full)},{sC(ii, j_full)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# Variant: same loop nest but R=TI, so we hoist sA load OUTSIDE bk too.
# Actually no, sA depends on bk so must reload each bk. Skip.


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
    fname = "exp_1_rank_outer_bj.py"
    log_event({"ts": int(time.time()), "type": "exp_start",
               "lane": 1, "exp": fname})

    BEST_PRIOR = 73602

    print(f"{'TI':>3} {'TJ':>3} {'R':>3}  {'cost':>10}  {'lb':>10}  {'reads':>8}")
    print("-" * 50)

    from lower_bound import analyze
    results = []
    for TI in [4, 8, 16]:
        for TJ in [2, 4, 8]:
            for R in [1, 2, 4, 8, 16]:
                if R > TI or TI % R != 0 or N % TI != 0 or N % TJ != 0:
                    continue
                ir = gen_rank_outer_bj_correct(TI, TJ, R)
                cost = safe_score(ir)
                try:
                    info = analyze(ir)
                    reads = sum(info['reads'].values())
                    ir_lb = info['remapped_ir']
                    cost_lb = safe_score(ir_lb)
                except Exception as e:
                    reads = -1
                    cost_lb = f"ERR:{e}"
                    ir_lb = ir
                print(f"{TI:>3} {TJ:>3} {R:>3}  {cost!r:>10}  {cost_lb!r:>10}  {reads:>8}")
                if isinstance(cost, int):
                    results.append((TI, TJ, R, cost, ir, "raw"))
                if isinstance(cost_lb, int):
                    results.append((TI, TJ, R, cost_lb, ir_lb, "renamed"))

    results.sort(key=lambda r: r[3])
    print(f"\nTop 10:")
    for r in results[:10]:
        print(f"  TI={r[0]} TJ={r[1]} R={r[2]} ({r[5]})  cost={r[3]:,}")

    if results:
        best = results[0]
        if best[3] < BEST_PRIOR:
            cost = best[3]
            ir = best[4]
            out = os.path.join(HERE, "records", f"record_{cost}_lane1.ir")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                f.write(ir + "\n")
            print(f"\nNEW RECORD: TI={best[0]} TJ={best[1]} R={best[2]} "
                  f"({best[5]})  cost={cost} -> {out}")
            log_event({"ts": int(time.time()), "type": "new_record",
                       "cost": cost, "prev": BEST_PRIOR, "lane": 1,
                       "file": f"record_{cost}_lane1.ir"})
        else:
            print(f"\nNo improvement: best={best[3]} >= prior best 73602")

    log_token(fname, 7000)


if __name__ == "__main__":
    main()
