"""Lane 1 — sa-cache template generalization sweep.

Three variants over (TI, TJ) with TI|16 and TJ|16:

  V1: rank-1 sA cache (single addr=1 cell, reloaded TI times per (bi,bk)
      slice). This is the current best template.
  V2: rank-TI sA strip (TI cells holding A[bi*TI:(bi+1)*TI, bk], loaded
      once per (bi,bk) and reused TJ*1 = TJ times in inner). This trades
      one cheap cell at addr 1 for TI cheap cells at addrs 1..TI.
  V3: rank-TI sA strip + rank-TJ sB strip — same as V2 but sB is loaded
      once per (bi,bj,bk) iteration and reused TI*TJ times. Actually the
      v1 template already loads sB once per bk inside the bj loop, so V3
      is more about layout. We try a variant where sB is moved to
      cheaper addresses than sA when sA is rank-TI.

For (TI=8, TJ=4) V1 we expect to recover 73602 exactly.
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # so `import matmul` works
from matmul import score_16x16  # noqa: E402

N = 16
DIVISORS = [1, 2, 4, 8, 16]


# ---------------------------------------------------------------------------
# V1: rank-1 sA cache
# ---------------------------------------------------------------------------

def gen_v1(TI: int, TJ: int) -> str:
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

    lines = [",".join(map(str, inputs))]
    for bi in range(nbi):
        for bj in range(nbj):
            for bk in range(N):
                # Load sB strip
                for jj in range(TJ):
                    lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
                # Inner: cache sA in single cell, multiply against sB
                for ii in range(TI):
                    lines.append(f"copy {SA},{A(bi * TI + ii, bk)}")
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {sC(ii, jj)},{SA},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {sC(ii, jj)},{TMP}")
            # Spill sC to C bulk
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# V2: rank-TI sA strip (TI cells), sB rank-TJ. sA at addrs 1..TI (cheap),
# sB at addrs TI+2..TI+1+TJ (still cheap-ish), tmp pushed but only used
# on accumulating mults.
#
# Loop nest:
#   for bi:
#     for bj:
#       for bk:
#         load sA[ii] = A[bi*TI+ii, bk]   for ii in 0..TI    (TI reads from A bulk)
#         load sB[jj] = B[bk, bj*TJ+jj]   for jj in 0..TJ    (TJ reads from B bulk)
#         for ii in 0..TI:
#           for jj in 0..TJ:
#             mul tmp,sA[ii],sB[jj]
#             add sC[ii,jj],tmp     (or direct mul into sC at bk==0)
#       spill sC -> C bulk
#
# This loads each A cell N/TI*N total times, and each B cell N/TI*N times,
# same as V1 actually for A. Let me think again... in V1 sA single cell is
# reloaded once per inner ii, so A[bi*TI+ii, bk] is loaded once per bj. With
# nbj iterations of bj, that's nbj reloads. So A bulk read count = nbj.
# For TI=8, TJ=4 that's nbj=4 reads per A cell, matching the doc.
#
# In V2, sA strip is loaded once per (bi, bj, bk), so A bulk read count =
# nbj as well (per A cell). Same. The difference is *where* sA lives.
# In V1, sA is at addr 1 (cost 1, 4096 reads). In V2 sA is at addrs 1..TI,
# read TJ*nbj*N = 16*nbj*TI/TI... hmm let me recount. Per inner: each sA[ii]
# read TJ times in (ii,jj) loop. Per bk: TI*TJ reads (each sA[ii] read TJ).
# Per bj: N*TI*TJ reads. Per bi: nbj*N*TI*TJ. Total over bi: nbi*nbj*N*TI*TJ
# = N*N*N = 4096 inner reads of sA (regardless of TI, TJ). So V2 uses 4096
# sA reads spread over TI cells = 4096/TI reads per sA cell.
#
# At TI=4: sA cells at addrs 1..4 (costs 1,2,2,2), reads per cell 1024.
# Total sA cost = 1024*(1+2+2+2) = 7168.
# vs V1 with TI=4: sA single cell, 4096 reads at cost 1 = 4096.
# So V1 is better for sA alone. But V2 saves on A bulk reads because we
# don't reload from A bulk on every inner iteration... wait, V1 does
# `copy SA, A[i,k]` for each ii inside the bk loop, so A bulk is read TI
# times per bk = TI*N per bj = TI*N*nbj per bi = 4096 / nbi total ≈ same.
# So bulk A read counts are identical. The only difference is sA scratchpad
# layout.

def gen_v2(TI: int, TJ: int) -> str:
    """rank-TI sA strip + rank-TJ sB strip."""
    nbi = N // TI
    nbj = N // TJ

    # Layout: sA at 1..TI, sB at TI+1..TI+TJ, tmp after, sC after that.
    # Hotter cells should go closer to addr 1.
    # sA reads: N*N*N / TI = 4096/TI per cell.
    # sB reads: N*N*N / TJ = 4096/TJ per cell.
    # tmp reads: 4096 - (cells where bk==0) = 4096 - 256 = 3840 reads.
    # sC reads: TI*TJ*nbi*nbj*N inner reads + spill = each cell read N times
    #           in mult phase plus 1 spill.
    # Hottest: tmp (3840), then sA-or-sB depending on TI/TJ, then sC, then bulk.

    sA = lambda ii: 1 + ii
    sB = lambda jj: 1 + TI + jj
    TMP = 1 + TI + TJ
    sC_base = TMP + 1
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
    for bi in range(nbi):
        for bj in range(nbj):
            for bk in range(N):
                for ii in range(TI):
                    lines.append(f"copy {sA(ii)},{A(bi * TI + ii, bk)}")
                for jj in range(TJ):
                    lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
                for ii in range(TI):
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {sC(ii, jj)},{sA(ii)},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{sA(ii)},{sB(jj)}")
                            lines.append(f"add {sC(ii, jj)},{TMP}")
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# V3: V2 layout but with TMP at addr 1 (hottest cell first), sA next,
# then sB, then sC. tmp has 3840 reads, which is most. This mirrors the
# sa_cache approach where tmp is at addr 2 but here we put tmp at 1.
# In V1 sA single cell @ addr 1 has 4096 reads, tmp @ addr 2 has 3840.
# In V3 tmp @ addr 1 has 3840 reads, sA strip somewhere else.

def gen_v3(TI: int, TJ: int) -> str:
    """tmp at addr 1, sA strip after, sB strip after, sC after."""
    nbi = N // TI
    nbj = N // TJ

    TMP = 1
    sA = lambda ii: 2 + ii
    sB = lambda jj: 2 + TI + jj
    sC_base = 2 + TI + TJ
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
    for bi in range(nbi):
        for bj in range(nbj):
            for bk in range(N):
                for ii in range(TI):
                    lines.append(f"copy {sA(ii)},{A(bi * TI + ii, bk)}")
                for jj in range(TJ):
                    lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
                for ii in range(TI):
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {sC(ii, jj)},{sA(ii)},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{sA(ii)},{sB(jj)}")
                            lines.append(f"add {sC(ii, jj)},{TMP}")
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main: sweep
# ---------------------------------------------------------------------------

def stop_check():
    sig = os.path.join(os.path.dirname(HERE), "matmul", "STOP_SIGNAL_1")
    sig2 = os.path.join(HERE, "STOP_SIGNAL_1")
    for s in (sig, sig2):
        if os.path.exists(s):
            os.remove(s)
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
    log_event({"ts": int(time.time()), "type": "exp_start",
               "lane": 1, "exp": "exp_1_tile_sweep_v1.py"})

    print(f"{'V':>3} {'TI':>3} {'TJ':>3}  {'cost':>10}")
    print("-" * 30)

    results = []
    for variant_name, gen in [("V1", gen_v1), ("V2", gen_v2), ("V3", gen_v3)]:
        for TI in DIVISORS:
            for TJ in DIVISORS:
                ir = gen(TI, TJ)
                cost = safe_score(ir)
                print(f"{variant_name:>3} {TI:>3} {TJ:>3}  {cost!r:>10}")
                if isinstance(cost, int):
                    results.append((variant_name, TI, TJ, cost, ir))

    # Sanity: V1 (TI=8, TJ=4) should reproduce 73602
    v1_8_4 = next((r for r in results if r[:3] == ("V1", 8, 4)), None)
    if v1_8_4 is not None:
        assert v1_8_4[3] == 73602, f"V1(8,4) gave {v1_8_4[3]}, expected 73602"
        print(f"\nSanity check passed: V1(8,4) = 73602")

    results.sort(key=lambda r: r[3])
    print(f"\nTop 5:")
    for r in results[:5]:
        print(f"  {r[0]} TI={r[1]} TJ={r[2]}  cost={r[3]:,}")

    best = results[0]
    BEST_PRIOR = 73602
    if best[3] < BEST_PRIOR:
        cost = best[3]
        ir = best[4]
        out = os.path.join(HERE, "records", f"record_{cost}_lane1.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(ir + "\n")
        print(f"\nNEW RECORD: {best[0]} TI={best[1]} TJ={best[2]}  cost={cost} -> {out}")
        log_event({"ts": int(time.time()), "type": "new_record",
                   "cost": cost, "prev": BEST_PRIOR, "lane": 1,
                   "file": f"record_{cost}_lane1.ir"})
    else:
        print(f"\nNo improvement: best={best[3]} >= prior best 73602")

    log_token("exp_1_tile_sweep_v1.py", 8000)


if __name__ == "__main__":
    main()
