"""Lane 0 — asymmetric tiles to create non-uniform per-cell reads.

Idea: split the C output into multiple regions, each computed with a different
(TI, TJ) tile shape. This changes per-cell read counts:
  - A cells in tile-region with smaller TJ are read MORE (more bj iters).
  - A cells in tile-region with larger TJ are read LESS.

If we cluster heavily-read A cells in one region and rarely-read in another,
the renamer can place them accordingly: heavy ones at cheap addrs, light at
expensive addrs. With a steep cost gradient, this could pay off.

Specifically: if some A cells get 8 reads and others 2 reads, the 8-read cells
go before the 2-read cells in the renamed address ordering.

Compute the read multiset under various asymmetric splits and check which beats
the uniform 73602.

Asymmetric split idea: split C horizontally into two row-bands.
  Band 0 (rows 0..R-1): tile (TI_0, TJ_0). A_band0 read N/TJ_0 times each.
  Band 1 (rows R..15): tile (TI_1, TJ_1). A_band1 read N/TJ_1 times each.
  B columns are shared, but B is read separately in each band:
    B in band 0 read N/TI_0 * R times? Let me think.

Actually the band split changes things. Let me think structurally.

Given the schedule "compute C[i,j] for all (i,j) using sa-cache", the per-A-cell
reads = nbj_for_that_band, per-B-cell reads = number of bands × nbi_per_band.

Hmm let's just code two row-bands and see.

Variant A: Band 0 = rows 0..7 with (TI=8, TJ=4), Band 1 = rows 8..15 with (TI=8, TJ=8).
  Band 0: nbi=1, nbj=4. Each A_band0 cell read 4×.
  Band 1: nbi=1, nbj=2. Each A_band1 cell read 2×.
  B: in band 0, each cell read TI=8 reads / TJ=4 cells per (bk, bj) — actually B reads:
     band 0: nbi*nbj*TJ B-loads × N = 1*4*4*16 = 256? No wait, let me recount.
     Per (bk, bj): TJ B-loads. Per band: nbj * N * TJ = 4 * 16 * 4 = 256 B-loads.
     But each B[k,j] is read 1 time per band (only in its bj tile). So per cell: 1.
     Across both bands: 2 reads per B cell? Wait, both bands cover all B (since B is N×N
     and used by all rows of A). Each band's bj sweep covers all 16 j values.
     Band 0 (TJ=4): bj in [0..3] tiles, sweeping all 16 cols. 4 bj iter × N bk = 64 (bk, bj)
     pairs. Each B[k,j] read once per (bk, bj) where j is in that bj's tile. So each B[k,j]
     read 1 time in band 0. Plus 1 time in band 1. Total 2 — same as V1.

Hmm so making TJ different between bands doesn't change B read count. But A read count IS
different per band.

Variant B: split BOTH by row band AND by column band.
  Band 00: rows 0..R, cols 0..C, tile (TI_a, TJ_a). A_band00 cells read nbj_a times.
  Band 01: rows 0..R, cols C..15, tile (TI_b, TJ_b). A_band00 cells read again here!

Wait — the rows in 'band 0' (rows 0..R) need to compute C for ALL columns. So band 0 covers
columns 0..15 too. We can't split A's row reads per column-band without computing the same
C[i,j] twice.

OK so band 0 = rows 0..R, ALL cols. Then within band 0 we do sa-cache with (TI_0, TJ_0).
Band 0 covers nbj_0 = 16/TJ_0 j-tiles, so each A_band0 cell read nbj_0 times.

Band split is the only freedom. Let me code it.
"""
from __future__ import annotations
import os, sys, time, json, math
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from matmul import score_16x16  # noqa: E402
from addr_renamer import rename_optimal  # noqa: E402

N = 16
BEST_PRIOR = 73602


def gen_two_band(R, TI0, TJ0, TI1, TJ1):
    """Split C into rows 0..R-1 (band 0) and R..15 (band 1).
    Band 0 uses (TI0, TJ0). Band 1 uses (TI1, TJ1).

    Bands share the same A, B inputs but use separate sC tiles.
    """
    if R % TI0 != 0:
        return None
    if (16 - R) % TI1 != 0:
        return None
    if 16 % TJ0 != 0 or 16 % TJ1 != 0:
        return None

    SA, TMP = 1, 2
    # Both bands' inner loops overlap in scratchpad use.
    # Layout: SA(1), TMP(2), sB strip (max(TJ0, TJ1) cells), sC tile (max(TI*TJ) cells), bulk.
    sB = lambda jj: 3 + jj
    sB_size = max(TJ0, TJ1)
    sC_base = 3 + sB_size
    sC_size = max(TI0 * TJ0, TI1 * TJ1)
    sC = lambda ii, jj, TJ: sC_base + ii * TJ + jj  # within current band's TJ

    A_base = sC_base + sC_size
    B_base = A_base + N * N
    C_base = B_base + N * N
    A = lambda i, k: A_base + i * N + k
    Bf = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]

    def emit_band(row_start, row_end, TI, TJ):
        rows = row_end - row_start
        nbi = rows // TI
        nbj = N // TJ
        for bi in range(nbi):
            for bj in range(nbj):
                for bk in range(N):
                    for jj in range(TJ):
                        lines.append(f"copy {sB(jj)},{Bf(bk, bj * TJ + jj)}")
                    for ii in range(TI):
                        global_i = row_start + bi * TI + ii
                        lines.append(f"copy {SA},{A(global_i, bk)}")
                        for jj in range(TJ):
                            if bk == 0:
                                lines.append(f"mul {sC(ii, jj, TJ)},{SA},{sB(jj)}")
                            else:
                                lines.append(f"mul {TMP},{SA},{sB(jj)}")
                                lines.append(f"add {sC(ii, jj, TJ)},{TMP}")
                for ii in range(TI):
                    for jj in range(TJ):
                        global_i = row_start + bi * TI + ii
                        lines.append(f"copy {C(global_i, bj * TJ + jj)},{sC(ii, jj, TJ)}")

    emit_band(0, R, TI0, TJ0)
    emit_band(R, N, TI1, TJ1)
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
               "lane": 0, "exp": "exp_0_asym_tiles.py"})

    # Sweep some band splits
    results = []
    splits = []
    for R in [4, 8, 12]:
        for TI0 in [1, 2, 4, 8]:
            if R % TI0 != 0:
                continue
            for TJ0 in [1, 2, 4, 8, 16]:
                for TI1 in [1, 2, 4, 8, 16]:
                    if (16 - R) % TI1 != 0:
                        continue
                    for TJ1 in [1, 2, 4, 8, 16]:
                        splits.append((R, TI0, TJ0, TI1, TJ1))

    print(f"Trying {len(splits)} splits...")
    print(f"{'R':>3} {'TI0':>3} {'TJ0':>3} {'TI1':>3} {'TJ1':>3}  {'raw':>6}  {'renamed':>7}")
    print("-" * 55)
    for R, TI0, TJ0, TI1, TJ1 in splits:
        try:
            ir = gen_two_band(R, TI0, TJ0, TI1, TJ1)
            if ir is None:
                continue
            raw = score_16x16(ir)
            new_ir, _ = rename_optimal(ir)
            renamed = score_16x16(new_ir)
            results.append((R, TI0, TJ0, TI1, TJ1, raw, renamed, new_ir))
            if renamed < BEST_PRIOR + 2000 or renamed < 75000:
                tag = " <-- BEST" if renamed < BEST_PRIOR else ""
                print(f"  {R:>3} {TI0:>3} {TJ0:>3} {TI1:>3} {TJ1:>3}  {raw:>6}  {renamed:>7}{tag}")
        except Exception as e:
            pass

    results.sort(key=lambda r: r[6])
    print("\nTop 10 by renamed cost:")
    for r in results[:10]:
        print(f"  R={r[0]} TI0={r[1]} TJ0={r[2]} TI1={r[3]} TJ1={r[4]}: "
              f"raw={r[5]} renamed={r[6]}")

    if results and results[0][6] < BEST_PRIOR:
        cost = results[0][6]
        out = os.path.join(HERE, "records", f"record_{cost}_lane0.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(results[0][7] + "\n")
        print(f"\nNEW RECORD cost={cost}")
        log_event({"ts": int(time.time()), "type": "new_record",
                   "cost": cost, "prev": BEST_PRIOR, "lane": 0,
                   "file": f"record_{cost}_lane0.ir"})
    else:
        print(f"\nNo improvement. best={results[0][6]} vs {BEST_PRIOR}")

    log_token("exp_0_asym_tiles.py", 4500)


if __name__ == "__main__":
    main()
