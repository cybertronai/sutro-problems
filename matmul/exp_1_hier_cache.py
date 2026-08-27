"""Lane 1 — combine hierarchical loop order with rank-1 sA cache.

Hierarchical loop structure (asymmetric: A loaded 1×, B loaded nb×):
  for ib in range(nb):
    for k in range(N):
      load sA[ii] = A[ib*B+ii, k]  for ii in 0..B    (B reads of A bulk)
      for jb in range(nb):
        load sB[jj] = B[k, jb*B+jj]  for jj in 0..B   (B reads of B bulk)
        for ii in range(B):
          for jj in range(B):
            mul tmp, sA[ii], sB[jj]
            add sC[jb*B*B + ii*B + jj], tmp

A-bulk reads: nb * N * B = 16 * B per ib. Total = 16 * 16 = 256. ✓
B-bulk reads: nb * N * B * nb = N * B * nb^2 = 16 * 4 * 16 = 1024. ✓

Variant H1 (current hierarchical, baseline 80217):
  Layout: TMP@1, sA@2-5, sB@6-9, sC@10-73, B_bulk@74-329, A_bulk@330-585

Variant H2: rank-1 sA cache at addr 1
  Inner becomes:
    for ii: copy SA_cache, sA[ii]   (B*nb reads of sA strip per k)
            for jj: mul tmp, SA_cache, sB[jj]; add sC, tmp

  SA_cache reads = total inner mul-reads of sA = N*B*nb*B = 4096
  TMP reads = 3840 (unchanged)
  sA strip reads: B*nb*N = 256 reads spread over B cells = 256/B per cell
                  = 64 per cell for B=4 (4 cells), but each is read in inner
                  loop: copy SA_cache, sA[ii]: that happens B*nb*N times per cell?
                  Actually: each ii in B, in each (k, jb) iteration. So nb*N
                  per cell? Wait: per (ib, k, jb) we have B inner copies (one per ii).
                  Total per cell: nb * N * nb iterations / B cells * B = nb*nb*N = 256.
                  Oh wait that's not right either. Let me think.

  In hierarchical: sA[ii] in inner is read TIMES = B (for jj-loop) per (ib,k,jb).
  Total per ii per ib: nb*B*N. = 4*4*16 = 256. So sA strip cell read 256 times.

  In H2: sA[ii] in inner is read in `copy SA_cache, sA[ii]` once per (ib,k,jb,ii).
  Total per ii per ib: nb * N. = 4*16 = 64. Then SA_cache read in muls = B*nb*N
  per ii = 4*4*16 = 256, but this is shared across all ii, so addr 1 reads = 4096.

  So H2 sA strip reads = 64 per cell (= 4 cells × 64 = 256 total reads), at cheap
  addrs. Savings:
    H1: sA strip 1024 reads/cell × cost 2 ≈ avg 2.5 = 2500 per cell × 4 = 10240.
        Wait actually H1 reads from sA strip 1024 times per cell.
    Per the brief: "sA strip 1024 reads each, 4 cells".

  H2: sA strip 64 reads/cell × cost 2-3 ≈ 150 per cell × 4 = 600.
    SA_cache 4096 reads × cost 1 = 4096.

  H2 total scratch cost = 4096 (SA_cache) + 600 (strip) + tmp + sB + sC.
  H1 total = 10240 (strip) + tmp + sB + sC.

  Difference: H2 saves 10240 - 4096 - 600 = 5544.

  But we need an extra address for SA_cache. Add it at addr 1, push sA strip to 2-5,
  TMP to 6, sB to 7-10, sC to 11-74.

Variant H3: rank-1 sB cache at addr 1
  Symmetric mirror of H2. The hot-most reads are different — sA strip is loaded
  once per ib*k = 64 times across the ib outer block, while sB is reloaded
  per jb. So sB has more reload bandwidth. Hmm.

Variant H4: BOTH sA cache AND sB cache (rank-1 each).
  Inner: for ii: copy SA, sA[ii]; for jj: copy SB, sB[jj]; mul tmp, SA, SB; add sC, tmp
  But this means SB is reloaded per (ii,jj) which destroys the sB strip benefit.
  Probably worse. Skip.

Variant H5: H2 but with different block size B.
  Currently B=4. Try B=2 (smaller sC), B=8 (bigger sC).
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from matmul import score_16x16  # noqa: E402

N = 16


def gen_h1(B):
    """Hierarchical, baseline."""
    nb = N // B
    TMP = 1
    sA = lambda ii: 2 + ii
    sB = lambda jj: 2 + B + jj
    sC_base = 2 + 2*B
    sC = lambda jb, ii, jj: sC_base + jb*B*B + ii*B + jj
    B_base = sC_base + nb*B*B
    A_base = B_base + N*N
    C_base = A_base + N*N
    A = lambda i, k: A_base + i*N + k
    Bf = lambda k, j: B_base + k*N + j
    C = lambda i, j: C_base + i*N + j
    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]
    for ib in range(nb):
        for k in range(N):
            for ii in range(B):
                lines.append(f"copy {sA(ii)},{A(ib*B+ii, k)}")
            for jb in range(nb):
                for jj in range(B):
                    lines.append(f"copy {sB(jj)},{Bf(k, jb*B+jj)}")
                for ii in range(B):
                    for jj in range(B):
                        if k == 0:
                            lines.append(f"mul {sC(jb, ii, jj)},{sA(ii)},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{sA(ii)},{sB(jj)}")
                            lines.append(f"add {sC(jb, ii, jj)},{TMP}")
        for jb in range(nb):
            for ii in range(B):
                for jj in range(B):
                    lines.append(f"copy {C(ib*B+ii, jb*B+jj)},{sC(jb, ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def gen_h2(B):
    """Hierarchical + rank-1 sA cache at addr 1.

    Layout:
      addr 1: SA_cache    (4096 reads)
      addr 2: TMP         (3840 reads)
      addrs 3..2+B: sA strip (B cells, 64 reads each for B=4)
      addrs 3+B..2+2B: sB strip (B cells, 1024 reads each)
      addrs 3+2B..: sC (nb * B * B cells)
    """
    nb = N // B
    SA_C = 1
    TMP = 2
    sA = lambda ii: 3 + ii
    sB = lambda jj: 3 + B + jj
    sC_base = 3 + 2*B
    sC = lambda jb, ii, jj: sC_base + jb*B*B + ii*B + jj
    B_base = sC_base + nb*B*B
    A_base = B_base + N*N
    C_base = A_base + N*N
    A = lambda i, k: A_base + i*N + k
    Bf = lambda k, j: B_base + k*N + j
    C = lambda i, j: C_base + i*N + j
    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]
    for ib in range(nb):
        for k in range(N):
            for ii in range(B):
                lines.append(f"copy {sA(ii)},{A(ib*B+ii, k)}")
            for jb in range(nb):
                for jj in range(B):
                    lines.append(f"copy {sB(jj)},{Bf(k, jb*B+jj)}")
                for ii in range(B):
                    lines.append(f"copy {SA_C},{sA(ii)}")
                    for jj in range(B):
                        if k == 0:
                            lines.append(f"mul {sC(jb, ii, jj)},{SA_C},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{SA_C},{sB(jj)}")
                            lines.append(f"add {sC(jb, ii, jj)},{TMP}")
        for jb in range(nb):
            for ii in range(B):
                for jj in range(B):
                    lines.append(f"copy {C(ib*B+ii, jb*B+jj)},{sC(jb, ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def gen_h3(B):
    """Hierarchical + rank-1 sB cache at addr 1.

    But sB has 1024 reads/cell in H1 (very hot already). Caching sB into
    addr 1 means: copy SB_cache, sB[jj] inside (ib, k, jb, jj loop).
    The inner ii loop reads SB_cache repeatedly.

    Wait, in hierarchical inner: for ii: for jj: read sA[ii], sB[jj].
    Reordering: for jj: copy SB_cache, sB[jj]; for ii: read sA[ii], SB_cache.

    SB_cache reads in mul = N*B*nb*B = 4096 (same number).
    sB strip reads (copy SB_cache, sB[jj]) = N*B*nb per cell = 64. Same as
    sA strip in H2.

    So H3 is symmetric to H2. The difference: in H2 we cache sA which has
    256 reads/cell (low). In H3 we'd cache sB which has 1024 reads/cell (high).

    Hmm wait, in H2, sA strip cell reads (in inner mul) was 1024 per cell. After
    caching: sA cell read only by `copy SA_cache, sA[ii]` once per (ib,k,jb,ii) =
    nb * N = 64 reads per cell. So we COMPRESS 1024 reads/cell into 64+the cache.

    So H3 wouldn't help: sB strip is read 1024/cell currently. Caching reduces
    it to 64/cell. SAVES the same as H2!

    Are H2 and H3 mutually exclusive? Let's think. Could we have both?
    Inner: for ii: copy SA_cache, sA[ii]; for jj: copy SB_cache, sB[jj]; mul tmp...
    But then SB_cache is reloaded for each jj-inner, total reads = 4096 (mul reads)
    plus 4096 (copies) = 8192 reads at addr 1. But wait we only have 1 addr 1.

    Actually we can put SA_cache at addr 1 and SB_cache at addr... 2? But then TMP
    has to move. Let me think about this.
    """
    nb = N // B
    SB_C = 1
    TMP = 2
    sA = lambda ii: 3 + ii
    sB = lambda jj: 3 + B + jj
    sC_base = 3 + 2*B
    sC = lambda jb, ii, jj: sC_base + jb*B*B + ii*B + jj
    B_base = sC_base + nb*B*B
    A_base = B_base + N*N
    C_base = A_base + N*N
    A = lambda i, k: A_base + i*N + k
    Bf = lambda k, j: B_base + k*N + j
    C = lambda i, j: C_base + i*N + j
    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]
    for ib in range(nb):
        for k in range(N):
            for ii in range(B):
                lines.append(f"copy {sA(ii)},{A(ib*B+ii, k)}")
            for jb in range(nb):
                for jj in range(B):
                    lines.append(f"copy {sB(jj)},{Bf(k, jb*B+jj)}")
                for jj in range(B):
                    lines.append(f"copy {SB_C},{sB(jj)}")
                    for ii in range(B):
                        if k == 0:
                            lines.append(f"mul {sC(jb, ii, jj)},{sA(ii)},{SB_C}")
                        else:
                            lines.append(f"mul {TMP},{sA(ii)},{SB_C}")
                            lines.append(f"add {sC(jb, ii, jj)},{TMP}")
        for jb in range(nb):
            for ii in range(B):
                for jj in range(B):
                    lines.append(f"copy {C(ib*B+ii, jb*B+jj)},{sC(jb, ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def gen_h2_v2(B):
    """H2 but make outer loop go over jb instead of ib (so B is the
    'load-once' axis). Symmetric to standard hierarchical but with
    sB-strip cached at addr 1."""
    nb = N // B
    SB_C = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    sA = lambda ii: 3 + B + ii
    sC_base = 3 + 2*B
    sC = lambda ib, ii, jj: sC_base + ib*B*B + ii*B + jj
    A_base = sC_base + nb*B*B
    B_base = A_base + N*N
    C_base = B_base + N*N
    A = lambda i, k: A_base + i*N + k
    Bf = lambda k, j: B_base + k*N + j
    C = lambda i, j: C_base + i*N + j
    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]
    for jb in range(nb):
        for k in range(N):
            for jj in range(B):
                lines.append(f"copy {sB(jj)},{Bf(k, jb*B+jj)}")
            for ib in range(nb):
                for ii in range(B):
                    lines.append(f"copy {sA(ii)},{A(ib*B+ii, k)}")
                for jj in range(B):
                    lines.append(f"copy {SB_C},{sB(jj)}")
                    for ii in range(B):
                        if k == 0:
                            lines.append(f"mul {sC(ib, ii, jj)},{sA(ii)},{SB_C}")
                        else:
                            lines.append(f"mul {TMP},{sA(ii)},{SB_C}")
                            lines.append(f"add {sC(ib, ii, jj)},{TMP}")
        for ib in range(nb):
            for ii in range(B):
                for jj in range(B):
                    lines.append(f"copy {C(ib*B+ii, jb*B+jj)},{sC(ib, ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


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


def main():
    stop_check()
    log_event({"ts": int(time.time()), "type": "exp_start",
               "lane": 1, "exp": "exp_1_hier_cache.py"})

    print(f"{'gen':>10} {'B':>3}  {'cost':>10}")
    print("-" * 30)

    results = []
    for name, gen in [("h1", gen_h1), ("h2", gen_h2),
                      ("h3", gen_h3), ("h2_v2", gen_h2_v2)]:
        for B in (2, 4, 8):
            try:
                ir = gen(B)
                cost = score_16x16(ir)
                print(f"{name:>10} {B:>3}  {cost:>10}")
                results.append((name, B, cost, ir))
            except Exception as e:
                print(f"{name:>10} {B:>3}  ERR {e!s:.40}")

    results.sort(key=lambda r: r[2])
    print(f"\nTop 5:")
    for r in results[:5]:
        print(f"  {r[0]} B={r[1]}  cost={r[2]:,}")

    BEST_PRIOR = 73602
    best = results[0]
    if best[2] < BEST_PRIOR:
        cost = best[2]
        out = os.path.join(HERE, "records", f"record_{cost}_lane1.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(best[3] + "\n")
        print(f"\nNEW RECORD: {best[0]} B={best[1]}  cost={cost}")
        log_event({"ts": int(time.time()), "type": "new_record",
                   "cost": cost, "prev": BEST_PRIOR, "lane": 1,
                   "file": f"record_{cost}_lane1.ir"})
    else:
        print(f"\nNo improvement: best={best[2]} (vs 73602)")

    log_token("exp_1_hier_cache.py", 7000)


if __name__ == "__main__":
    main()
