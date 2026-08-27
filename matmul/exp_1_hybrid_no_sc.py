"""Lane 1 — eliminate sC scratchpad, write directly to C.

Same loop as V1 but no separate sC accumulator — instead read/write C
addresses directly. C is placed at the cheapest possible addresses (right
after sB).

Layout (TI, TJ):
  addr 1:           SA cache
  addr 2:           tmp
  addrs 3..2+TJ:    sB strip
  addrs 3+TJ..258+TJ: C bulk (256 cells, output)
  addrs 259+TJ..514+TJ: A bulk
  addrs 515+TJ..770+TJ: B bulk

This avoids the sC spill but pays 16x reads on the now-far C addresses.
We expect this to be worse, but let's measure.

Variant H_DIRECT: same as above.
Variant H_INTERLEAVE: lay out C cells in tile-major order (so within each
  (bi,bj) tile, the cells are contiguous and start at a cheap addr). This
  means tile 0 cells are cheap, tile 1 cells less cheap, etc.

Variant H_PARTIAL: keep ONE tile's worth of C in cheap scratch (like sC)
  but instead of always spilling, only spill at the END of all tiles. So
  the cheap area is sC[0..31] for tile (0,0), used for 16 bk iterations,
  then spilled. SAME AS V1. Skip.
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from matmul import score_16x16  # noqa: E402

N = 16


def gen_h_direct(TI, TJ):
    """No sC scratchpad."""
    nbi, nbj = N // TI, N // TJ
    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    C_base = 3 + TJ
    C = lambda i, j: C_base + i*N + j
    A_base = C_base + N*N
    A = lambda i, k: A_base + i*N + k
    B_base = A_base + N*N
    Bf = lambda k, j: B_base + k*N + j
    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]
    for bi in range(nbi):
        for bj in range(nbj):
            for bk in range(N):
                for jj in range(TJ):
                    lines.append(f"copy {sB(jj)},{Bf(bk, bj*TJ+jj)}")
                for ii in range(TI):
                    lines.append(f"copy {SA},{A(bi*TI+ii, bk)}")
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {C(bi*TI+ii, bj*TJ+jj)},{SA},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {C(bi*TI+ii, bj*TJ+jj)},{TMP}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def gen_h_interleave(TI, TJ):
    """No sC, but C cells laid out in tile-major order."""
    nbi, nbj = N // TI, N // TJ
    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    # C[i, j] mapped to tile-major: tile_idx = (bi, bj), cell = (ii, jj)
    # addr = C_base + tile_idx * (TI*TJ) + ii*TJ + jj
    C_base = 3 + TJ

    def C(i, j):
        bi, ii = divmod(i, TI)
        bj, jj = divmod(j, TJ)
        tile_idx = bi * nbj + bj
        return C_base + tile_idx * (TI*TJ) + ii*TJ + jj

    A_base = C_base + N*N
    A = lambda i, k: A_base + i*N + k
    B_base = A_base + N*N
    Bf = lambda k, j: B_base + k*N + j

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]
    for bi in range(nbi):
        for bj in range(nbj):
            for bk in range(N):
                for jj in range(TJ):
                    lines.append(f"copy {sB(jj)},{Bf(bk, bj*TJ+jj)}")
                for ii in range(TI):
                    lines.append(f"copy {SA},{A(bi*TI+ii, bk)}")
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {C(bi*TI+ii, bj*TJ+jj)},{SA},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {C(bi*TI+ii, bj*TJ+jj)},{TMP}")
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
               "lane": 1, "exp": "exp_1_hybrid_no_sc.py"})

    print(f"{'gen':>15} {'TI':>3} {'TJ':>3}  {'cost':>10}")
    print("-" * 35)

    results = []
    for name, gen in [("h_direct", gen_h_direct), ("h_interleave", gen_h_interleave)]:
        for TI in (1, 2, 4, 8, 16):
            for TJ in (1, 2, 4, 8, 16):
                try:
                    ir = gen(TI, TJ)
                    cost = score_16x16(ir)
                    print(f"{name:>15} {TI:>3} {TJ:>3}  {cost:>10}")
                    results.append((name, TI, TJ, cost, ir))
                except Exception as e:
                    print(f"{name:>15} {TI:>3} {TJ:>3}  ERR {e!s:.40}")

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
    else:
        print(f"\nNo improvement: best={best[3]}")

    log_token("exp_1_hybrid_no_sc.py", 5500)


if __name__ == "__main__":
    main()
