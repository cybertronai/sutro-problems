"""Lane 1: full A=C alias with deferred writeback for stranded cells.

Combines finaladd + alias-chain + deferred writeback for C[bi=0, non-last-bj]:
- C[bi=0, last-bj]: finaladd writes to A[bi=0, jj=0..3] (32 cells).
- C[bi=0, non-last-bj]: at (bi=0, bj<last, bk=N-1), do NOT finaladd. Instead:
    mul TMP, sA, sB[jj]; add sC[ii,jj], TMP  (regular accumulate)
    Then after bk loop, copy sC values to a HOLD buffer (96 cells).
- C[bi=1, non-last-bj]: finaladd writes to A[bi=0, k=4..15] (96 cells, alias chain).
- C[bi=1, last-bj]: finaladd writes to A[bi=1, jj=0..3] (32 cells).
- After bi=1 done: A[bi=1, k=4..15] is free. Copy 96 hold cells to those addrs.

Layout target:
  1: SA, 2: TMP, 3-6: sB, 7-38: active sC (32 cells, 123 reads each).
  39-... aliased A=C (256 cells, 5 reads each).
  ... B (256 cells, 2 reads).
  ... hold (96 cells, 1 read each).

Predicted cost ~71,735 (worse than current 69,697).
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matmul  # noqa: E402
from addr_renamer import rename_optimal  # noqa: E402

LANE = 1
STOP_FILE = os.path.join(HERE, f"STOP_SIGNAL_{LANE}")


def check_stop():
    if os.path.exists(STOP_FILE):
        try: os.remove(STOP_FILE)
        except OSError: pass
        print("STOP_SIGNAL — halting.")
        sys.exit(0)


def log_event(d):
    d.setdefault("ts", time.time())
    with open(os.path.join(HERE, "events.jsonl"), "a") as f:
        f.write(json.dumps(d) + "\n")


def log_tokens(exp, tokens):
    with open(os.path.join(HERE, "token_log.jsonl"), "a") as f:
        f.write(json.dumps({
            "ts": time.time(), "lane": LANE, "exp": exp, "tokens": tokens
        }) + "\n")


def generate(N=16, TI=8, TJ=4):
    nbi = N // TI
    nbj = N // TJ
    assert nbi == 2

    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    sC = lambda ii, jj: 7 + ii * TJ + jj
    HOLD_BASE = 7 + TI * TJ
    # 96 hold cells: index by (bj, ii, jj) for bi=0, bj=0..nbj-2.
    hold = lambda bj, ii, jj: HOLD_BASE + bj * TI * TJ + ii * TJ + jj

    A_base = HOLD_BASE + (nbj - 1) * TI * TJ
    B_base = A_base + N * N
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j

    def C(i, j):
        bi = i // TI
        bj = j // TJ
        if bj == nbj - 1:
            jj = j - (nbj - 1) * TJ
            return A(i, jj)
        else:
            if bi == 0:
                # FULL ALIAS: target A[bi=1, k=TJ..N-1]
                # Map: (i=bi*TI+ii, j=bj*TJ+jj_in) for bi=0, bj<nbj-1, ii=0..TI-1
                # → A[1*TI + ii, TJ + j]. j = bj*TJ + jj_in ranges 0..(nbj-1)*TJ-1.
                ai = TI + (i % TI)  # bi=1's row
                ak = TJ + j
                return A(ai, ak)
            else:
                # bi=1, non-last: chain to A[bi=0, k=TJ..N-1]
                ai = (bi - 1) * TI + (i % TI)
                ak = TJ + j
                return A(ai, ak)

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [B(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]

    lines = [",".join(map(str, inputs))]

    # bi=0
    bi = 0
    for bj in range(nbj):
        for bk in range(N):
            for jj in range(TJ):
                lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
            for ii in range(TI):
                lines.append(f"copy {SA},{A(bi * TI + ii, bk)}")
                for jj in range(TJ):
                    if bk == 0:
                        lines.append(f"mul {sC(ii, jj)},{SA},{sB(jj)}")
                    elif bk == N - 1:
                        if bj == nbj - 1:
                            # last-bj: finaladd to A[bi=0, jj]
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            c_addr = C(bi * TI + ii, bj * TJ + jj)
                            lines.append(f"add {c_addr},{sC(ii, jj)},{TMP}")
                        else:
                            # non-last-bj: do regular accumulate (don't finaladd)
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {sC(ii, jj)},{TMP}")
                    else:
                        lines.append(f"mul {TMP},{SA},{sB(jj)}")
                        lines.append(f"add {sC(ii, jj)},{TMP}")
        # End of (bi=0, bj): copy sC to hold[bj] if non-last-bj.
        if bj < nbj - 1:
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"copy {hold(bj, ii, jj)},{sC(ii, jj)}")

    # bi=1
    bi = 1
    for bj in range(nbj):
        for bk in range(N):
            for jj in range(TJ):
                lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
            for ii in range(TI):
                lines.append(f"copy {SA},{A(bi * TI + ii, bk)}")
                for jj in range(TJ):
                    if bk == 0:
                        lines.append(f"mul {sC(ii, jj)},{SA},{sB(jj)}")
                    elif bk == N - 1:
                        # finaladd for all bj's (alias chain)
                        lines.append(f"mul {TMP},{SA},{sB(jj)}")
                        c_addr = C(bi * TI + ii, bj * TJ + jj)
                        lines.append(f"add {c_addr},{sC(ii, jj)},{TMP}")
                    else:
                        lines.append(f"mul {TMP},{SA},{sB(jj)}")
                        lines.append(f"add {sC(ii, jj)},{TMP}")

    # End of bi=1: copy hold[bj=0..nbj-2] to A[bi=1, k=TJ..N-1] addresses.
    # Mapping: hold[bj, ii, jj] → A[bi=1, ii, TJ + bj*TJ + jj].
    for bj in range(nbj - 1):
        for ii in range(TI):
            for jj in range(TJ):
                # The C address in the alias map for (bi=0, j = bj*TJ+jj):
                #   ai = TI + ii (bi=1's row)
                #   ak = TJ + j = TJ + bj*TJ + jj
                # → A[TI+ii, TJ+bj*TJ+jj]
                target = A(TI + ii, TJ + bj * TJ + jj)
                lines.append(f"copy {target},{hold(bj, ii, jj)}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


if __name__ == "__main__":
    check_stop()
    log_event({"type": "exp_start", "lane": LANE, "exp": "exp_1_full_alias_finaladd.py"})
    print(f"[{time.strftime('%H:%M:%S')}] starting full_alias_finaladd")

    ir = generate()
    cost = matmul.score_16x16(ir)
    print(f"  raw = {cost:,}")

    renamed, info = rename_optimal(ir)
    cost_r = matmul.score_16x16(renamed)
    print(f"  renamed = {cost_r:,}  (renamer estimate {info['remapped_cost_estimate']:,})")

    BEST = min(cost, cost_r)
    BEST_IR = renamed if cost_r < cost else ir
    print(f"  best = {BEST:,}")

    PREV = 69697
    if BEST < PREV:
        out = os.path.join(HERE, "records", f"record_{BEST}_lane{LANE}.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(BEST_IR + "\n")
        print(f"  NEW RECORD! saved {out}")
        log_event({
            "type": "new_record", "cost": BEST, "prev": PREV,
            "lane": LANE, "file": out,
        })
    else:
        print(f"  no improvement vs {PREV:,}")

    log_event({
        "type": "exp_done", "lane": LANE,
        "exp": "exp_1_full_alias_finaladd.py",
        "best": BEST,
    })
    log_tokens("exp_1_full_alias_finaladd.py", 5500)
