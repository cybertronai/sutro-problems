"""Lane 0 — share B across bi tiles to reduce total B bulk reads.

Schedule: outermost bj, then bk, then bi (innermost).
  - sB (TJ cells) loaded once per (bj, bk).
  - For each bi tile, accumulate into a *separate* sC tile (nbi tiles total).
  - After the bk sweep finishes for the bj column, spill all nbi sC tiles to C.

Total B bulk reads = nbj × N × TJ = 4*16*4 = 256 (each B cell read 1×, vs 2× in V1).
Total A bulk reads = nbj × N × nbi × TI = 4*16*2*8 = 1024 (same as V1).
sC scratchpad = nbi * TI * TJ = 2*8*4 = 64 cells (vs 32 in V1).

So we save 256 B-bulk reads (each at cost ~21 = 5440 saved) but add 32 sC cells.
Per V1 profile: sC at 7..38 averages ~4.5 cost, with 4096 reads → avg cost 5.06 per read.
With 64 sC cells, sC at 7..70 averages cost(7..70). Each cell still read N times in
its bk sweep. With nbi=2, total sC reads = 2 * (TI*TJ) * N * nbj = 2*32*16*4 = 4096
(same as V1!). But the 64 cells are at higher addrs.

Wait — does each sC tile get reused? In V1, sC is reused across (bi, bj) tiles. In
the new schedule, each (bi) tile keeps its own sC tile alive across the bk sweep
within bj, and spills/restarts per bj. So sC has nbi=2 tiles live at once.

Total sC reads = N * nbi * nbj * TI * TJ = 16*2*4*8*4 = 4096 (same).
But now stretched across 64 cells instead of 32, so addrs go up to 70 instead of 38.

Cost of sC reads:
  V1 (32 cells, addrs 7-38): sum_{a=7..38} cost(a) * 128 reads/cell
    cost(7-9)=3 (3 cells), cost(10-16)=4 (7), cost(17-25)=5 (9), cost(26-36)=6 (11),
    cost(37-38)=7 (2). Sum = 9+28+45+66+14 = 162. × 128 = 20,736 ✓

  New (64 cells, addrs 7-70): cost(7-9)=3 (3), cost(10-16)=4 (7), cost(17-25)=5 (9),
    cost(26-36)=6 (11), cost(37-49)=7 (13), cost(50-64)=8 (15), cost(65-70)=9 (6).
    Sum = 9+28+45+66+91+120+54 = 413. × 64 reads/cell = 26,432.
  Increase: 26432 - 20736 = +5696.

  Plus all higher cells (A, B, C bulk) shift up by 32. cost goes up some per read.

  Total saved on B bulk: 256 reads × avg cost shift. Let me compute roughly:
  V1 B bulk reads = 512 cost 10,738. Half the reads × similar cost = 5369 saved.
  But the BULK addrs ALL shift up by 32, so each remaining bulk read costs more.

  Net not obviously positive. Let me just code and measure.
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from matmul import score_16x16  # noqa: E402
from addr_renamer import rename_optimal  # noqa: E402

N = 16
BEST_PRIOR = 73602


def gen_b_share(TI, TJ):
    """Outer (bj, bk, bi) with persistent multi-tile sC."""
    nbi, nbj = N // TI, N // TJ
    SA, TMP = 1, 2
    sB = lambda jj: 3 + jj
    # nbi separate sC tiles, each TI*TJ cells
    sC_base = 3 + TJ
    sC = lambda bi, ii, jj: sC_base + bi * (TI * TJ) + ii * TJ + jj
    A_base = sC_base + nbi * TI * TJ
    B_base = A_base + N * N
    C_base = B_base + N * N
    A = lambda i, k: A_base + i * N + k
    Bf = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j
    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]

    for bj in range(nbj):
        for bk in range(N):
            # Load sB strip once per (bj, bk)
            for jj in range(TJ):
                lines.append(f"copy {sB(jj)},{Bf(bk, bj * TJ + jj)}")
            # Sweep all bi tiles, accumulating into their separate sC tiles
            for bi in range(nbi):
                for ii in range(TI):
                    lines.append(f"copy {SA},{A(bi * TI + ii, bk)}")
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {sC(bi, ii, jj)},{SA},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {sC(bi, ii, jj)},{TMP}")
        # End of bk sweep for this bj; spill all bi tiles' sC to C
        for bi in range(nbi):
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(bi, ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def gen_a_share(TI, TJ):
    """Mirror: outer (bi, bk, bj) with persistent multi-tile sC.
    sA loaded once per (bi, bk), shared across bj tiles.
    Wait — sA is a single cell in V1, already shared across jj. To 'share A
    across bj tiles', we need to reuse the SAME A[i,k] copy across nbj
    iterations of bj.

    Let's instead share A_strip (TI cells) once per (bi, bk):
      for bi: for bk: load A_strip; for bj: for ii: for jj: ...
    Each A cell read once per (bi, bk) = 1× total (no nbj multiplier!).
    A bulk reads = N * nbi * TI = 16 * 2 * 8 = 256 (vs 1024 in V1).

    sB strip per (bk, bj) reload = N * nbj * TJ = 16 * 4 * 4 = 256 reads of B
    per cell? Actually B is reloaded each (bi, bk, bj). With outer bi: B
    reloaded nbi*N*nbj times = 2*16*4 = 128 → reads = 128*TJ = 512 (same as V1).

    sC: must persist across bk for each (bi, bj). With (bi, bk, bj) order:
    bj is innermost so sC persists across bj only? No — we need sC to persist
    across bk for accumulation. With (bi, bk, bj), bk is OUTSIDE bj. So for
    each (bi, bj), sC is fresh; sC accumulates across bk for fixed (bi, bj).
    So sC must persist across all bk for each (bi, bj). With bk INSIDE this
    requires many sC tiles live simultaneously.

    OK actually if we go (bi, bk, bj), sC is over (bi, bj). For fixed bi, we
    need nbj sC tiles live across bk. Then spill at end of bi.

    sC tiles = nbj * TI * TJ = 4*8*4 = 128 cells.
    """
    nbi, nbj = N // TI, N // TJ
    SA = 1
    TMP = 2
    sA = lambda ii: 3 + ii  # TI cells
    sB = lambda jj: 3 + TI + jj  # TJ cells
    sC_base = 3 + TI + TJ
    sC = lambda bj, ii, jj: sC_base + bj * (TI * TJ) + ii * TJ + jj
    A_base = sC_base + nbj * TI * TJ
    B_base = A_base + N * N
    C_base = B_base + N * N
    A = lambda i, k: A_base + i * N + k
    Bf = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j
    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]

    for bi in range(nbi):
        for bk in range(N):
            # Load sA strip
            for ii in range(TI):
                lines.append(f"copy {sA(ii)},{A(bi * TI + ii, bk)}")
            for bj in range(nbj):
                # Load sB strip
                for jj in range(TJ):
                    lines.append(f"copy {sB(jj)},{Bf(bk, bj * TJ + jj)}")
                for ii in range(TI):
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {sC(bj, ii, jj)},{sA(ii)},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{sA(ii)},{sB(jj)}")
                            lines.append(f"add {sC(bj, ii, jj)},{TMP}")
        # Spill all bj tiles' sC for this bi
        for bj in range(nbj):
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(bj, ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def stop_check():
    sig = os.path.join(HERE, "STOP_SIGNAL_0")
    if os.path.exists(sig):
        os.remove(sig)
        print("STOP_SIGNAL — halting.")
        sys.exit(0)


def log_event(event):
    with open(os.path.join(HERE, "events.jsonl"), "a") as f:
        f.write(json.dumps(event) + "\n")


def log_token(filename, tokens):
    with open(os.path.join(HERE, "token_log.jsonl"), "a") as f:
        f.write(json.dumps({"ts": int(time.time()), "lane": 0,
                            "exp": filename, "tokens": tokens}) + "\n")


def main():
    stop_check()
    log_event({"ts": int(time.time()), "type": "exp_start",
               "lane": 0, "exp": "exp_0_b_share.py"})

    results = []

    print("\n=== B-share variants (outer bj, sC tiled by bi) ===")
    for TI, TJ in [(8, 4), (4, 4), (4, 8), (16, 4), (8, 8), (16, 2), (16, 8), (8, 2)]:
        try:
            ir = gen_b_share(TI, TJ)
            raw = score_16x16(ir)
            new_ir, _ = rename_optimal(ir)
            renamed = score_16x16(new_ir)
            tag = " <-- BEST" if renamed < BEST_PRIOR else ""
            print(f"  b_share TI={TI:>2} TJ={TJ:>2}: raw={raw:>6} renamed={renamed:>6}{tag}")
            results.append(("b_share", TI, TJ, raw, renamed, new_ir))
        except Exception as e:
            print(f"  b_share TI={TI} TJ={TJ}: ERR {e}")

    print("\n=== A-share variants (outer bi, sC tiled by bj) ===")
    for TI, TJ in [(8, 4), (4, 4), (4, 8), (16, 4), (8, 8), (16, 2), (16, 8), (8, 2)]:
        try:
            ir = gen_a_share(TI, TJ)
            raw = score_16x16(ir)
            new_ir, _ = rename_optimal(ir)
            renamed = score_16x16(new_ir)
            tag = " <-- BEST" if renamed < BEST_PRIOR else ""
            print(f"  a_share TI={TI:>2} TJ={TJ:>2}: raw={raw:>6} renamed={renamed:>6}{tag}")
            results.append(("a_share", TI, TJ, raw, renamed, new_ir))
        except Exception as e:
            print(f"  a_share TI={TI} TJ={TJ}: ERR {e}")

    results.sort(key=lambda r: r[4])
    print("\nTop 5:")
    for r in results[:5]:
        print(f"  {r[0]} TI={r[1]} TJ={r[2]}: raw={r[3]} renamed={r[4]}")

    if results and results[0][4] < BEST_PRIOR:
        cost = results[0][4]
        out = os.path.join(HERE, "records", f"record_{cost}_lane0.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(results[0][5] + "\n")
        print(f"\nNEW RECORD cost={cost}")
        log_event({"ts": int(time.time()), "type": "new_record",
                   "cost": cost, "prev": BEST_PRIOR, "lane": 0,
                   "file": f"record_{cost}_lane0.ir"})
    else:
        print(f"\nNo improvement: best={results[0][4] if results else 'N/A'} vs {BEST_PRIOR}")

    log_token("exp_0_b_share.py", 4000)


if __name__ == "__main__":
    main()
