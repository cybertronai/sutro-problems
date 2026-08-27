"""Lane 1 — rank-r sA cache sweep.

The current best (sa_cache_16x16.ir, cost=73602) caches a single sA cell at
addr 1 (read 4096 times). The hypothesis: cache r ≪ TI A-cells in addrs
1..r, sharing the inner jj-loop work across r rows. We sweep
rank r ∈ {1, 2, 4, 8} for (TI=8, TJ=4) and also try (TI=16, TJ=4).

For each rank r, the inner kernel becomes:

  for ii_outer in range(TI/r):
    for ii_inner in range(r):
      copy sA[ii_inner], A[bi*TI + ii_outer*r + ii_inner, bk]
    for jj in range(TJ):
      for ii_inner in range(r):
        # accumulate sC[ii_outer*r + ii_inner, jj] += sA[ii_inner] * sB[jj]

Layout (addrs):
  sA[0..r-1]     -> 1..r              (cost 1 for r=1; 1,2 for r=2; 1,2,2,2 for r=4; 1..3 for r=8)
  TMP            -> r+1
  sB[0..TJ-1]    -> r+2..r+1+TJ
  sC[0..TI*TJ-1] -> r+2+TJ..r+1+TJ+TI*TJ
  A bulk
  B bulk
  C bulk

Note: since arithmetic and writes are free, and reads pay, the only effect of
rank-r vs rank-1 is the address layout shift. But rank-1 keeps sA at the
single cheapest address (1) with all 4096 sA-reads paid at cost-1 (= 4096).
With rank-r, sA reads are distributed: r cells each read 4096/r times, at
addrs 1..r with various costs. For r=2: cells at addrs 1,2 with costs 1,2;
total sA cost = (4096/2)*1 + (4096/2)*2 = 2048+4096 = 6144 (vs 4096 for
rank-1, so +2048 worse). But TMP moves from addr 2 to addr 3 (still cost 2),
sB moves from addrs 3..6 to addrs 4..7 (costs 2,2,3,3 vs 2,3,3,3). Hmm
similar.

So rank-r doesn't naively help. The savings could come from a re-ordering of
how A is loaded — e.g., if we change the loop nest so sA loads are amortized.
We'll first implement the straight rank-r and see if there's an unexpected
win, then try variants.
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # so `import matmul` works
from matmul import score_16x16  # noqa: E402

sys.path.insert(0, HERE)
from addr_renamer import rename_optimal  # noqa: E402

N = 16


def gen_rank(TI: int, TJ: int, R: int) -> str:
    """Generate IR with rank-R sA cache, tile (TI,TJ).

    Requires R | TI and TI | N and TJ | N.
    """
    assert TI % R == 0
    assert N % TI == 0
    assert N % TJ == 0
    nbi = N // TI
    nbj = N // TJ
    n_outer = TI // R

    sA = lambda ii_inner: 1 + ii_inner          # 1..R
    TMP = R + 1                                 # R+1
    sB = lambda jj: R + 2 + jj                  # R+2..R+1+TJ
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
    for bi in range(nbi):
        for bj in range(nbj):
            for bk in range(N):
                # Load sB strip (always TJ cells per bk)
                for jj in range(TJ):
                    lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
                # Inner over ii_outer chunks of size R
                for ii_outer in range(n_outer):
                    # Load R sA cells
                    for ii_inner in range(R):
                        ii = ii_outer * R + ii_inner
                        lines.append(f"copy {sA(ii_inner)},{A(bi * TI + ii, bk)}")
                    # Multiply: for each jj, for each ii_inner in chunk
                    for jj in range(TJ):
                        for ii_inner in range(R):
                            ii = ii_outer * R + ii_inner
                            if bk == 0:
                                lines.append(f"mul {sC(ii, jj)},{sA(ii_inner)},{sB(jj)}")
                            else:
                                lines.append(f"mul {TMP},{sA(ii_inner)},{sB(jj)}")
                                lines.append(f"add {sC(ii, jj)},{TMP}")
            # Spill sC to C bulk
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Variant: rank-r sA cache + reordered inner so sB loaded only once per
# ii_outer chunk if we hoist it. But sB only depends on bk, jj — so it's
# already loaded once per bk. No change.
#
# Different variant: hoist sB load OUT of the bk loop?  No, sB depends on bk.
#
# Real variant: rank-r with bk-loop INSIDE the ii_outer loop. That means we
# materialize sA for one chunk of R rows and accumulate over ALL bk for that
# chunk. This requires keeping all TI*TJ sC alive (which we already do) and
# doing TI/R outer chunks; for each chunk we sweep bk=0..N-1. Then sA gets
# loaded N times per chunk (once per bk). Total sA loads from A bulk =
# (TI/R)*N*R = TI*N per (bi,bj). Same as before. So it's the same A traffic.
# But the order changes inner-to-outer, which only affects scheduling not
# total reads.
# ---------------------------------------------------------------------------

def gen_rank_swapped(TI: int, TJ: int, R: int) -> str:
    """rank-R but with ii_outer loop OUTSIDE bk so sA chunks are pinned.

    Note: this needs sC to accumulate over bk for each chunk, just like the
    standard order.
    """
    assert TI % R == 0
    nbi = N // TI
    nbj = N // TJ
    n_outer = TI // R

    sA = lambda ii_inner: 1 + ii_inner
    TMP = R + 1
    sB = lambda jj: R + 2 + jj
    sC_base = R + 2 + TJ
    # Only need TI*TJ sC cells but in this variant we only need R*TJ at a time
    # Still allocate TI*TJ for simplicity; only R*TJ are hot per ii_outer chunk.
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
            for ii_outer in range(n_outer):
                for bk in range(N):
                    # Load R sA cells
                    for ii_inner in range(R):
                        ii = ii_outer * R + ii_inner
                        lines.append(f"copy {sA(ii_inner)},{A(bi * TI + ii, bk)}")
                    # Load sB strip
                    for jj in range(TJ):
                        lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
                    for jj in range(TJ):
                        for ii_inner in range(R):
                            ii = ii_outer * R + ii_inner
                            if bk == 0:
                                lines.append(f"mul {sC(ii, jj)},{sA(ii_inner)},{sB(jj)}")
                            else:
                                lines.append(f"mul {TMP},{sA(ii_inner)},{sB(jj)}")
                                lines.append(f"add {sC(ii, jj)},{TMP}")
            # Spill sC -> C
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


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
    fname = "exp_1_rank_v1.py"
    log_event({"ts": int(time.time()), "type": "exp_start",
               "lane": 1, "exp": fname})

    BEST_PRIOR = 73602
    print(f"{'variant':>20} {'TI':>3} {'TJ':>3} {'R':>3}  {'cost':>10}  {'lb':>10}")
    print("-" * 72)

    results = []
    configs = []
    # rank-r sweep at (TI=8, TJ=4)
    for R in [1, 2, 4, 8]:
        configs.append(("rank_v1", 8, 4, R, gen_rank))
        configs.append(("rank_swap", 8, 4, R, gen_rank_swapped))
    # rank-r at (TI=16, TJ=4) with R in {1,2,4,8,16}
    for R in [1, 2, 4, 8, 16]:
        configs.append(("rank_v1", 16, 4, R, gen_rank))
        configs.append(("rank_swap", 16, 4, R, gen_rank_swapped))
    # rank-r at (TI=4, TJ=4)
    for R in [1, 2, 4]:
        configs.append(("rank_v1", 4, 4, R, gen_rank))
        configs.append(("rank_swap", 4, 4, R, gen_rank_swapped))
    # rank-r at (TI=8, TJ=2) and (TI=8, TJ=8)
    for R in [1, 2, 4, 8]:
        configs.append(("rank_v1", 8, 2, R, gen_rank))
        configs.append(("rank_v1", 8, 8, R, gen_rank))

    for name, TI, TJ, R, gen in configs:
        try:
            ir = gen(TI, TJ, R)
        except AssertionError:
            continue
        cost = safe_score(ir)
        # Compute structure-preserving lower bound via renamer
        try:
            ir_lb, info = rename_optimal(ir)
            lb = info["remapped_cost_estimate"]
            cost_lb = safe_score(ir_lb)
        except Exception as e:
            lb = -1
            cost_lb = f"ERR:{e}"
            ir_lb = ir
        print(f"{name:>20} {TI:>3} {TJ:>3} {R:>3}  {cost!r:>10}  {cost_lb!r:>10}")
        if isinstance(cost, int):
            results.append((name, TI, TJ, R, cost, ir, "raw"))
        if isinstance(cost_lb, int):
            results.append((name, TI, TJ, R, cost_lb, ir_lb, "renamed"))

    results.sort(key=lambda r: r[4])
    print(f"\nTop 10:")
    for r in results[:10]:
        print(f"  {r[0]} TI={r[1]} TJ={r[2]} R={r[3]} ({r[6]})  cost={r[4]:,}")

    if results:
        best = results[0]
        if best[4] < BEST_PRIOR:
            cost = best[4]
            ir = best[5]
            out = os.path.join(HERE, "records", f"record_{cost}_lane1.ir")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                f.write(ir + "\n")
            print(f"\nNEW RECORD: {best[0]} TI={best[1]} TJ={best[2]} "
                  f"R={best[3]} ({best[6]})  cost={cost} -> {out}")
            log_event({"ts": int(time.time()), "type": "new_record",
                       "cost": cost, "prev": BEST_PRIOR, "lane": 1,
                       "file": f"record_{cost}_lane1.ir"})
        else:
            print(f"\nNo improvement: best={best[4]} >= prior best 73602")

    log_token(fname, 9000)


if __name__ == "__main__":
    main()
