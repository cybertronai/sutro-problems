"""Lane 0 v7: alias-earlier-bi + write final accumulator directly to C.

Replace the writeback copy at end of (bi,bj) with a 3-arg add at bk=N-1
that writes the final accumulator into the C address. Saves 1 sC read
per (ii, jj) per (bi, bj) iteration.

Schedule:
  for bk in 0..N-1:
    [bj's bk loop loads sB once]
    for ii in 0..TI-1:
      copy SA, A(bi*TI+ii, bk)
      for jj in 0..TJ-1:
        if bk == 0:
          mul sC[ii,jj], SA, sB[jj]      # init
        elif bk == N-1:
          mul tmp, SA, sB[jj]
          add C(bi*TI+ii, bj*TJ+jj), sC[ii,jj], tmp   # FINAL: write C
        else:
          mul tmp, SA, sB[jj]
          add sC[ii,jj], tmp
  # No separate writeback; the bk=N-1 step did it.
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

STOP_FILE = os.path.join(HERE, "STOP_SIGNAL_0")


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
            "ts": time.time(), "lane": 0, "exp": exp, "tokens": tokens
        }) + "\n")


def generate_alias_finaladd(N=16, TI=8, TJ=4):
    nbi = N // TI
    nbj = N // TJ
    assert nbi == 2

    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    sC = lambda ii, jj: 7 + ii * TJ + jj

    A_base = 7 + TI * TJ
    B_base = A_base + N * N
    C_base = B_base + N * N
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j
    C_bulk = lambda i, j: C_base + i * N + j

    def C(i, j):
        bi = i // TI
        bj = j // TJ
        if bj == nbj - 1:
            jj = j - (nbj - 1) * TJ
            return A(i, jj)
        if bi >= 1:
            ai = (bi - 1) * TI + (i % TI)
            ak = TJ + j  # j here is actual column 0..(nbj-1)*TJ-1
            return A(ai, ak)
        return C_bulk(i, j)

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
                        elif bk == N - 1:
                            # Final: write directly into C(bi*TI+ii, bj*TJ+jj)
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            c_addr = C(bi * TI + ii, bj * TJ + jj)
                            lines.append(f"add {c_addr},{sC(ii, jj)},{TMP}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {sC(ii, jj)},{TMP}")
            # No writeback — the bk=N-1 add did it.
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


if __name__ == "__main__":
    check_stop()
    log_event({"type": "exp_start", "lane": 0, "exp": "exp_0_alias_finaladd.py"})
    print(f"[{time.strftime('%H:%M:%S')}] starting exp_0_alias_finaladd")

    ir = generate_alias_finaladd()
    cost = matmul.score_16x16(ir)
    print(f"  raw = {cost:,}")

    renamed, info = rename_optimal(ir)
    cost_r = matmul.score_16x16(renamed)
    print(f"  renamed = {cost_r:,}  (renamer estimate {info['remapped_cost_estimate']:,})")

    BEST = min(cost, cost_r)
    BEST_IR = renamed if cost_r < cost else ir
    print(f"  best = {BEST:,}")

    PREV = 70993
    if BEST < PREV:
        out = os.path.join(HERE, "records", f"record_{BEST}_lane0.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(BEST_IR + "\n")
        print(f"  NEW RECORD! saved {out}")
        log_event({
            "type": "new_record", "cost": BEST, "prev": PREV,
            "lane": 0, "file": out,
        })
    else:
        print(f"  no improvement vs {PREV:,}")

    log_tokens("exp_0_alias_finaladd.py", 4500)
