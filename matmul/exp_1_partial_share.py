"""Lane 1 — partial b_share / partial a_share.

Idea: in V1 each A cell is read 4 times (nbj loads). Reduce by chunking
the bj loop into groups of CJ bj's, sharing sA across each chunk:

  for bi:
    for bj_chunk in 0..nbj/CJ:
      for bk:
        for ii in 0..TI:
          load sA = A[bi*TI+ii, bk]   (1 read per chunk per ii per bk)
          for bj_local in 0..CJ:
            load sB once per (bk, bj_local)
            for jj: mul/add sC[ii, bj_local*TJ+jj]

A-bulk reads per cell: nbj/CJ (vs 4 in V1).
B-bulk reads per cell: nbi*1 = nbi (same as V1, since each B[k,j] is loaded
  once per (bi, bj_chunk_containing_j) = nbi times).
sC cells: TI * (CJ*TJ) (vs TI*TJ in V1).

For TI=8, TJ=4, nbj=4:
  CJ=1: V1, sA reloaded per bj. sC=32. A reads/cell=4. B reads/cell=2.
  CJ=2: sA shared across 2 bj's. sC=64. A reads/cell=2. B reads/cell=2.
  CJ=4: full b_share. sC=128. A reads/cell=1. B reads/cell=2... wait
        actually CJ=nbj means one bj_chunk, so within that one chunk
        we already iterate bk and load sB once per (bk, bj_local) = nbj
        times. Each B cell loaded once per bi = nbi times. Confirmed.

Symmetrically: chunk bi into groups of CI bi's, share sB across CI bi's.
  V1: CI=1, sC=TI*TJ.
  CI=nbi: full b_share, sC=N*TJ.

Combined: chunk both → still doable but sC = (CI*TI) * (CJ*TJ).

For 16x16 the params: TI∈{2,4,8,16}, TJ∈{2,4,8,16}, CI ∈ {1..nbi}, CJ ∈ {1..nbj}.
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from matmul import score_16x16  # noqa: E402

sys.path.insert(0, HERE)
from addr_renamer import rename_optimal  # noqa: E402

N = 16


def gen_partial_share(TI: int, TJ: int, CI: int, CJ: int) -> str:
    """Loop nest:
      for bi_outer:
        for bj_outer:
          for bk:
            for bi_inner in 0..CI:
              for ii in 0..TI:
                load sA = A[(bi_outer*CI+bi_inner)*TI+ii, bk]
                for bj_inner in 0..CJ:
                  if first ii of this iter, load sB[*] = B[bk, j_full..]
                  ... actually simpler: load sB once per (bk, bj_inner)
                      OUTSIDE the bi_inner loop.

    Wait, sB depends on (bk, j_full), where j_full = (bj_outer*CJ+bj_inner)*TJ+jj.
    To share sB across CI bi_inner values: load sB OUTSIDE bi_inner.

    Restructured:
      for bi_outer (size nbi/CI):
        for bj_outer (size nbj/CJ):
          for bk:
            for bj_inner (size CJ):
              load sB[jj] = B[bk, (bj_outer*CJ+bj_inner)*TJ+jj] for jj
              for bi_inner (size CI):
                for ii in 0..TI:
                  load sA = A[i_full, bk]
                  for jj in 0..TJ:
                    mul/add sC[i_full, j_full] += sA*sB
          # spill sC for this (bi_outer, bj_outer)
    """
    nbi = N // TI
    nbj = N // TJ
    assert nbi % CI == 0
    assert nbj % CJ == 0
    NBI_outer = nbi // CI
    NBJ_outer = nbj // CJ

    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    sC_base = 3 + TJ
    # sC alive: CI*TI * CJ*TJ cells per (bi_outer, bj_outer)
    # indexed by (i_local ∈ 0..CI*TI, j_local ∈ 0..CJ*TJ)
    n_sC = (CI * TI) * (CJ * TJ)
    sC = lambda i_local, j_local: sC_base + i_local * (CJ * TJ) + j_local
    A_base = sC_base + n_sC
    B_base = A_base + N * N
    C_base = B_base + N * N
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [B(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]

    lines = [",".join(map(str, inputs))]
    for bi_outer in range(NBI_outer):
        for bj_outer in range(NBJ_outer):
            for bk in range(N):
                for bj_inner in range(CJ):
                    j_base = (bj_outer * CJ + bj_inner) * TJ
                    # Load sB
                    for jj in range(TJ):
                        lines.append(f"copy {sB(jj)},{B(bk, j_base + jj)}")
                    for bi_inner in range(CI):
                        i_base = (bi_outer * CI + bi_inner) * TI
                        for ii in range(TI):
                            i_full = i_base + ii
                            i_local = bi_inner * TI + ii
                            lines.append(f"copy {SA},{A(i_full, bk)}")
                            for jj in range(TJ):
                                j_local = bj_inner * TJ + jj
                                if bk == 0:
                                    lines.append(f"mul {sC(i_local, j_local)},{SA},{sB(jj)}")
                                else:
                                    lines.append(f"mul {TMP},{SA},{sB(jj)}")
                                    lines.append(f"add {sC(i_local, j_local)},{TMP}")
            # Spill sC for this (bi_outer, bj_outer)
            for bi_inner in range(CI):
                i_base = (bi_outer * CI + bi_inner) * TI
                for ii in range(TI):
                    i_full = i_base + ii
                    i_local = bi_inner * TI + ii
                    for bj_inner in range(CJ):
                        j_base = (bj_outer * CJ + bj_inner) * TJ
                        for jj in range(TJ):
                            j_full = j_base + jj
                            j_local = bj_inner * TJ + jj
                            lines.append(f"copy {C(i_full, j_full)},{sC(i_local, j_local)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# Variant: sA shared INSIDE bi_inner — i.e. swap loop order so for fixed
# (bi_outer, bj_outer, bk, bi_inner, ii) we do all bj_inner. Then sA is
# loaded once per (bk, bi_inner, ii) regardless of CJ. A-bulk reads per cell
# = nbj/CJ (since for each bj_outer iter, sA[i,k] loaded once per ii per bk
# and we have nbj/CJ bj_outer iters → nbj/CJ A-bulk reads per cell).
def gen_partial_share_a(TI: int, TJ: int, CI: int, CJ: int) -> str:
    """Loop swapped: bi_inner inside bk, bj_inner innermost.

      for bi_outer, bj_outer:
        for bk:
          for bi_inner:
            for ii:
              load sA
              for bj_inner:
                load sB
                for jj: mul/add

    This reloads sB per (bi_inner, ii) = CI*TI times per (bk, bj_outer). So
    B-bulk reads per cell = nbi/CI * CI*TI = nbi*TI / CJ ... hmm let me
    recount. For each B[k,j], it's loaded when bk=k, j is in current bj_outer's
    range (i.e., j//TJ in [bj_outer*CJ, (bj_outer+1)*CJ)). For each
    (bi_outer, bk, bi_inner, ii) iter inside that bj_outer, sB[k,j] is
    re-loaded. Total = nbi_outer * CI * TI = nbi * TI per B cell.
    For TI=8, nbi=2: 2*8=16 loads per B cell. V1 had 2. Disastrous.

    So this variant is WORSE for B. Skip; keep gen_partial_share.
    """
    return gen_partial_share(TI, TJ, CI, CJ)


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
    fname = "exp_1_partial_share.py"
    log_event({"ts": int(time.time()), "type": "exp_start",
               "lane": 1, "exp": fname})

    BEST_PRIOR = 73602
    from lower_bound import analyze

    print(f"{'TI':>3} {'TJ':>3} {'CI':>3} {'CJ':>3}  {'cost':>10}  {'lb':>10}  {'reads':>8}")
    print("-" * 55)

    results = []
    configs = []
    for TI in [2, 4, 8, 16]:
        for TJ in [2, 4, 8]:
            if N % TI != 0 or N % TJ != 0:
                continue
            nbi = N // TI
            nbj = N // TJ
            for CI in range(1, nbi + 1):
                for CJ in range(1, nbj + 1):
                    if nbi % CI != 0 or nbj % CJ != 0:
                        continue
                    configs.append((TI, TJ, CI, CJ))

    for TI, TJ, CI, CJ in configs:
        try:
            ir = gen_partial_share(TI, TJ, CI, CJ)
        except Exception as e:
            print(f"{TI:>3} {TJ:>3} {CI:>3} {CJ:>3}  GEN_ERR:{e}")
            continue
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
        print(f"{TI:>3} {TJ:>3} {CI:>3} {CJ:>3}  {cost!r:>10}  {cost_lb!r:>10}  {reads:>8}")
        if isinstance(cost, int):
            results.append((TI, TJ, CI, CJ, cost, ir, "raw"))
        if isinstance(cost_lb, int):
            results.append((TI, TJ, CI, CJ, cost_lb, ir_lb, "renamed"))

    results.sort(key=lambda r: r[4])
    print(f"\nTop 10:")
    for r in results[:10]:
        print(f"  TI={r[0]} TJ={r[1]} CI={r[2]} CJ={r[3]} ({r[6]})  cost={r[4]:,}")

    if results:
        best = results[0]
        if best[4] < BEST_PRIOR:
            cost = best[4]
            ir = best[5]
            out = os.path.join(HERE, "records", f"record_{cost}_lane1.ir")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                f.write(ir + "\n")
            print(f"\nNEW RECORD: TI={best[0]} TJ={best[1]} CI={best[2]} "
                  f"CJ={best[3]} ({best[6]})  cost={cost} -> {out}")
            log_event({"ts": int(time.time()), "type": "new_record",
                       "cost": cost, "prev": BEST_PRIOR, "lane": 1,
                       "file": f"record_{cost}_lane1.ir"})
        else:
            print(f"\nNo improvement: best={best[4]} >= prior best 73602")

    log_token(fname, 8000)


if __name__ == "__main__":
    main()
