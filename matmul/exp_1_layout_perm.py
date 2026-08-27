"""Lane 1 — try permutations of the hot scratchpad layout.

Hot cells for V1 (TI=8, TJ=4):
  - SA single cell: 4096 reads
  - tmp: 3840 reads
  - sB[0..3] strip: 1024 reads each (4096 total for the 4 cells)
  - sC[0..31]: 128 reads each (4096 total accumulator reads + 256 spills)

Total reads from hot cells (top 6 cells, addrs 1..6):
   addr-cost vector if filled in greedy-cost order:
   addr 1: cost 1
   addr 2-4: cost 2
   addr 5-9: cost 3

So 9 cells fit in cost ≤ 3 zone. Currently used: SA(1), tmp(2), sB(3..6) =
6 cells. Then sC starts at 7..38 (cost 3..7).

Permutations:
  P1 (current): SA@1, tmp@2, sB@3..6, sC@7..38
  P2: SA@1, tmp@2, sC[part]@3..6, sB@7..10  — moves sB to costlier
  P3: tmp@1, SA@2, sB@3..6, sC@7..38       — sA<->tmp swap
  P4: SA@1, sB[0]@2, tmp@3, sB[1..3]@4..6, sC@7..38
  P5: SA@1, tmp@2, sB@3..6, sC@7..38, but compress sC by reusing
      cells across (bi,bj)? Already done.

Also: try splitting sC into hot half (high read count) and cold half. But
all sC cells have the same read count (128). So no benefit.

Idea: put SOME sC cells in cheap addrs and rotate. e.g., the first 4
sC cells go at addrs 7..10 (cost 3..4) and last 28 at addrs 11..38 (cost
4..7). The order of (ii,jj) determines who's where.

Most interesting: shrinking the inner mul-add cost.
  Current: mul tmp,SA,sB[jj] = 1+2 = 3
           add sC[ii,jj],tmp = cost(sC) + 2

  Idea: 3-op add instead of 2-op:
    mul tmp,SA,sB[jj]                    # 3
    add sC[ii,jj],sC[ii,jj],tmp          # cost(sC)+2 (same)
    add tmp_bank[k%2], sC[ii,jj], tmp    # alternate accumulator?

Hmm, the structure (mul, add) is fundamentally 3+cost(sC)+2 = cost(sC)+5
per accumulation. We can't reduce this unless sC reads are cheaper or
fewer.

Let's just sweep a few promising permutations.
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from matmul import score_16x16  # noqa: E402

N = 16


def gen_p1(TI, TJ):
    """Original: SA@1, tmp@2, sB@3..2+TJ, sC@3+TJ..."""
    return _gen(TI, TJ, sa_addr=1, tmp_addr=2, sb_base=3, sc_base=3+TJ)


def gen_p3(TI, TJ):
    """tmp@1, SA@2, sB@3..2+TJ, sC@3+TJ..."""
    return _gen(TI, TJ, sa_addr=2, tmp_addr=1, sb_base=3, sc_base=3+TJ)


def gen_p4(TI, TJ):
    """SA@1, sB@2..1+TJ, tmp@2+TJ, sC@3+TJ..."""
    return _gen(TI, TJ, sa_addr=1, tmp_addr=2+TJ, sb_base=2, sc_base=3+TJ)


def gen_p5(TI, TJ):
    """sB@1..TJ, SA@TJ+1, tmp@TJ+2, sC@TJ+3..."""
    return _gen(TI, TJ, sa_addr=TJ+1, tmp_addr=TJ+2, sb_base=1, sc_base=TJ+3)


def gen_p6(TI, TJ):
    """SA@1, tmp@2, sB[0]@3, sC tile starts at 4 interleaved with sB.
    Actually let's interleave: SA@1, tmp@2, sB[0]@3, sB[1]@4, sC[0..3]@5..8,
    sB[2]@9, sB[3]@10, sC[4..]@11... that's complicated. Skip.

    Instead: SA@1, sB strip @ 2..1+TJ, tmp @ 2+TJ, sC @ 3+TJ...
    """
    return _gen(TI, TJ, sa_addr=1, tmp_addr=1+TJ+1, sb_base=2, sc_base=3+TJ)


def _gen(TI, TJ, sa_addr, tmp_addr, sb_base, sc_base):
    nbi, nbj = N // TI, N // TJ
    SA = sa_addr
    TMP = tmp_addr
    sB = lambda jj: sb_base + jj
    sC = lambda ii, jj: sc_base + ii * TJ + jj
    A_base = sc_base + TI * TJ
    B_base = A_base + N * N
    C_base = B_base + N * N
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j

    # Validate uniqueness
    used = {SA, TMP} | {sB(j) for j in range(TJ)} | {sC(i,j) for i in range(TI) for j in range(TJ)}
    assert len(used) == 2 + TJ + TI*TJ, f"address collision in {(SA, TMP, sb_base, sc_base)}"

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
               "lane": 1, "exp": "exp_1_layout_perm.py"})

    print(f"{'P':>5} {'TI':>3} {'TJ':>3}  {'cost':>10}")
    print("-" * 30)

    results = []
    for name, gen in [("P1", gen_p1), ("P3", gen_p3),
                      ("P4", gen_p4), ("P5", gen_p5),
                      ("P6", gen_p6)]:
        for TI in (4, 8, 16):
            for TJ in (2, 4, 8):
                try:
                    ir = gen(TI, TJ)
                    cost = score_16x16(ir)
                    print(f"{name:>5} {TI:>3} {TJ:>3}  {cost:>10}")
                    results.append((name, TI, TJ, cost, ir))
                except AssertionError as e:
                    print(f"{name:>5} {TI:>3} {TJ:>3}  COLLIDE")
                except Exception as e:
                    print(f"{name:>5} {TI:>3} {TJ:>3}  ERR {e!s:.40}")

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

    log_token("exp_1_layout_perm.py", 5500)


if __name__ == "__main__":
    main()
