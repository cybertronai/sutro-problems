"""Lane 1 — explore loop orderings of the rank-1 sA-cache template.

Original V1 has loop order (bi, bj, bk, jj-load-sB, ii, jj-mul). Try other
permutations to see if we can reduce inner-loop overhead, sB reload count,
or sC spill count.

Variants:
  L1: (bi, bj, bk) — original  [jj-load-sB inside bk, ii inside bk]
  L2: (bj, bi, bk) — swap bi/bj
  L3: (bi, bk, bj) — bk in middle, bj inner
  L4: (bk, bi, bj) — bk outer
  L5: (bj, bk, bi) — bk middle, bi inner
  L6: (bk, bj, bi)

For each (bi, bj, bk), the tile sC must be freshly accumulated once. So
sC must persist for the duration of the (bi, bj) pair × all bk values.
This means sC scratch must remain in registers across all bk for that tile.

Loop orders that put bk inside (bi, bj) keep sC alive only for one tile at
a time (small scratchpad). Loop orders that put bk outside need separate sC
tiles per (bi, bj) tile — a much bigger scratchpad.

So the main interesting variants are L1, L2 (bk innermost). They differ
only in iteration order, not in scratch usage. Let's try.

Also: try sweeping which sub-loop is innermost (ii vs jj vs sB-load).
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from matmul import score_16x16  # noqa: E402

N = 16


def gen_L1(TI, TJ):
    """Original order (bi, bj, bk)."""
    nbi, nbj = N // TI, N // TJ
    SA, TMP = 1, 2
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
                    lines.append(f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def gen_L2(TI, TJ):
    """Order (bj, bi, bk) — swap outer bi/bj."""
    nbi, nbj = N // TI, N // TJ
    SA, TMP = 1, 2
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
    for bj in range(nbj):
        for bi in range(nbi):
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
                    lines.append(f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def gen_inner_swap(TI, TJ):
    """Swap inner: ii outer, jj inner -> jj outer, ii inner.
    sA single-cell only works if A varies in the innermost loop. If we
    reorder so ii is innermost, we'd need sA to be a strip indexed by ii.
    So instead: cache sB single-cell, sA strip rank-TI.
    """
    nbi, nbj = N // TI, N // TJ
    SB1, TMP = 1, 2
    sA = lambda ii: 3 + ii
    sC_base = 3 + TI
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
                # Load sA strip
                for ii in range(TI):
                    lines.append(f"copy {sA(ii)},{A(bi * TI + ii, bk)}")
                for jj in range(TJ):
                    lines.append(f"copy {SB1},{B(bk, bj * TJ + jj)}")
                    for ii in range(TI):
                        if bk == 0:
                            lines.append(f"mul {sC(ii, jj)},{sA(ii)},{SB1}")
                        else:
                            lines.append(f"mul {TMP},{sA(ii)},{SB1}")
                            lines.append(f"add {sC(ii, jj)},{TMP}")
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def gen_inner_swap_dual_cache(TI, TJ):
    """Same idea: keep both sA and sB as scratch strips, but cache the
    individual element in addr 1 by rotating which one gets cached.

    For (TI, TJ): sA strip at addrs (TI cells), sB strip at addrs (TJ cells),
    cell-cache CACHE at addr 1, used in inner-most.
    Inner: for ii: copy CACHE,sA[ii]; for jj: mul tmp,CACHE,sB[jj]; add sC.
    This is the same as L1 where sA is strip and we cache one element.

    Layout: addr 1 = CACHE (4096 reads), addr 2 = tmp (3840 reads),
            sB strip at 3..2+TJ, sA strip at 3+TJ..2+TJ+TI, sC after.
    sA strip is much less hot than sB strip in this scheme:
      sA cells: each loaded into CACHE once per ii per bk per bj per bi
                = 4096/TI inner copies for sA strip read; total sA strip
                reads = 4096 / TI per cell? No wait: in inner, sA[ii] is
                read once per (bi, bj, bk, ii) = nbi*nbj*N reads per cell
                over all ii? Actually sA[ii] is only read in `copy CACHE,
                sA[ii]` once per ii per bk per bj per bi. Total sA strip
                reads per cell = nbj * N = 4*16 = 64 (TI=8, TJ=4). Same
                count as sB strip but different read-cost shape.

    Hmm — actually, in L1, we directly do `copy SA, A(bi*TI+ii, bk)`. That
    reads from A bulk. So sA-strip caching adds an extra layer: load A bulk
    -> sA strip; then copy sA strip cell -> CACHE. That's 2 reads of
    something hot vs 1 read of A bulk.

    BUT: A bulk reads cost more than sA-strip reads! A bulk (256 cells)
    sits at addrs 39..294 → costs sqrt(39)..sqrt(294) ≈ 7..18. sA-strip at
    addrs 3+TJ..2+TJ+TI sits very cheap.

    In V1: A bulk read TI*N*nbj per cell = 4 per cell, total = 4 * sum_cost(A bulk).
    With strip: A bulk read TI per (bi,bk) = N*nbi reads of TI cells, but
    each A cell read only nbj times (once per bj). Wait same count for A bulk.

    Difference: in V1, `copy SA, A(...)` is the only A bulk read. With strip,
    A bulk -> sA strip happens once per (bi, bk), but inner uses sA strip
    not A bulk. So A bulk read TI * N * nbi * nbj /TI = N * nbi * nbj = 64 reads per
    bi,bk  hmm let me actually recount.

    Actually in V1: `copy SA, A(bi*TI+ii, bk)` is inside (bi, bj, bk, ii).
    Total A bulk reads = nbi * nbj * N * TI = nbi*nbj*N*TI. Each A[i,k] cell
    is read nbj times. nbj=4 for TJ=4. So 4 reads per A cell. Cost = 4 *
    sum(sqrt(A_addrs)).

    In strip variant: `copy sA[ii], A(bi*TI+ii, bk)` is inside (bi, bj, bk, ii).
    Same count! Each A cell read nbj times. So bulk A cost same.

    But scratchpad rearrangement: sA strip occupies TI cells, costing extra
    addresses for sB and sC. Net loss.

    So strip-A doesn't help. The single-cache-cell trick is optimal because
    A is touched in O(1) memory.
    """
    return None  # Not implementing — same as V2.


def gen_L1_split_init(TI, TJ):
    """L1 but pull out the bk=0 special case as a separate initialization
    pass. This lets the bk>=1 loop omit the if/else branch. Doesn't change
    instruction count, just emit cleanliness — should produce identical IR.
    """
    nbi, nbj = N // TI, N // TJ
    SA, TMP = 1, 2
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
            # bk=0: mul into sC (no read of sC)
            for jj in range(TJ):
                lines.append(f"copy {sB(jj)},{B(0, bj * TJ + jj)}")
            for ii in range(TI):
                lines.append(f"copy {SA},{A(bi * TI + ii, 0)}")
                for jj in range(TJ):
                    lines.append(f"mul {sC(ii, jj)},{SA},{sB(jj)}")
            for bk in range(1, N):
                for jj in range(TJ):
                    lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
                for ii in range(TI):
                    lines.append(f"copy {SA},{A(bi * TI + ii, bk)}")
                    for jj in range(TJ):
                        lines.append(f"mul {TMP},{SA},{sB(jj)}")
                        lines.append(f"add {sC(ii, jj)},{TMP}")
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
               "lane": 1, "exp": "exp_1_loop_orders.py"})

    tests = []
    for TI in (4, 8, 16):
        for TJ in (2, 4, 8):
            tests.append((TI, TJ))

    print(f"{'gen':>20} {'TI':>3} {'TJ':>3}  {'cost':>10}")
    print("-" * 50)
    results = []
    for name, gen in [("L1", gen_L1), ("L2", gen_L2),
                      ("inner_swap", gen_inner_swap),
                      ("L1_split_init", gen_L1_split_init)]:
        for TI, TJ in tests:
            try:
                ir = gen(TI, TJ)
                cost = score_16x16(ir)
                print(f"{name:>20} {TI:>3} {TJ:>3}  {cost:>10}")
                results.append((name, TI, TJ, cost, ir))
            except Exception as e:
                print(f"{name:>20} {TI:>3} {TJ:>3}  ERR {e}")

    results.sort(key=lambda r: r[3])
    print(f"\nTop 5:")
    for r in results[:5]:
        print(f"  {r[0]} TI={r[1]} TJ={r[2]}  cost={r[3]:,}")

    BEST_PRIOR = 73602
    best = results[0]
    if best[3] < BEST_PRIOR:
        cost = best[3]
        out = os.path.join(HERE, "records", f"record_{cost}_lane1.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(best[4] + "\n")
        print(f"\nNEW RECORD: {best[0]} TI={best[1]} TJ={best[2]}  cost={cost}")
        log_event({"ts": int(time.time()), "type": "new_record",
                   "cost": cost, "prev": BEST_PRIOR, "lane": 1,
                   "file": f"record_{cost}_lane1.ir"})

    log_token("exp_1_loop_orders.py", 6000)


if __name__ == "__main__":
    main()
