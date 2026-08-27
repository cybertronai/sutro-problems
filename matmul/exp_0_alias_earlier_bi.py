"""Lane 0 v5: alias C onto A using earlier-bi-row trick.

For nbi=2 with TI=8, TJ=4:
  - C[bi=0, last-bj-cols (j>=12)] aliases A[bi=0, jj] for jj=0..3
    (last-bj writeback during bi=0; A[bi=0, *] is fully read by then? NO!
    bi=0's last-bj writeback happens at end of (bi=0, bj=3). At that point,
    A[bi=0, *] reads are done — but bi=1 hasn't started yet. So safe.)
    Wait, A[bi=0, *] is used in bi=0's bk loops. Once we exit bi=0, we move
    to bi=1 which uses A[bi=1, *]. So A[bi=0] is fully done after bi=0's
    last bj. Yes, alias works.
  - C[bi=1, last-bj-cols] aliases A[bi=1, jj] for jj=0..3 (same trick)
  - C[bi=1, non-last-bj-cols] aliases A[bi=0, ...] (since bi=0 is done)

Address mapping for the 256 A cells:
  A[bi=0, jj=0..3] (i.e., A[i=0..7, k=0..3]): 32 cells, aliased to
    C[bi=0, j=12..15] (last-bj-cols of bi=0).
  A[bi=0, k=4..15]: 96 cells. Aliased to C[bi=1, j=0..11] (non-last-bj-cols).
    Map: A[i, k] for i=0..7, k=4..15 -> C[i+8, k-4]? Need 96 distinct cells
    on each side. C[bi=1, j=0..11] = C[i=8..15, j=0..11] = 8×12 = 96. Match.
  A[bi=1, jj=0..3] (i.e., A[i=8..15, k=0..3]): 32 cells, aliased to
    C[bi=1, j=12..15].
  A[bi=1, k=4..15]: 96 cells, NOT aliased (no later bi to receive their C).

C cells not aliased: C[bi=0, j=0..11] = 8×12 = 96 cells -> bulk-C.

Schedule notes:
  - C[bi=0, last-bj]: writeback at end of (bi=0, bj=3) into A[bi=0, jj] addrs.
    Constraint: A[bi=0, k=0..3] last read at (bi=0, bj=3, bk=3). That's during
    THIS very bj iteration. Specifically `copy SA, A(bi=0*TI+ii, bk)` for
    bk=0..3 happens before the writeback (which is after bk loop). Safe.
  - C[bi=1, last-bj]: at (bi=1, bj=3) writeback. A[bi=1, k=0..3] last read
    at bi=1, bj=3, bk=3. Safe (writeback after bk).
  - C[bi=1, j=0..11]: bj=0,1,2 writebacks during bi=1. Each writeback writes
    into A[bi=0, ?] — A[bi=0] is fully dead since bi=1 doesn't touch it. Safe.
  - C[bi=0, j=0..11]: bj=0,1,2 during bi=0. Goes to bulk-C. (Can't alias
    onto A[bi=1] because bi=1 hasn't been read yet — those cells are LIVE
    inputs. Overwriting them would break correctness.)

Layout target (renamer-optimal):
  addr 1: SA (4096)
  addr 2: TMP (3840)
  addrs 3-6: sB (1024 × 4)
  addrs 7-38: active sC (128 × 32)
  addrs 39-198: A=C aliased (5 × 160)
  addrs 199-294: A non-aliased (4 × 96)
  addrs 295-550: B (2 × 256)
  addrs 551-646: bulk-C (1 × 96)

Predicted: ~70,993.
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


def generate_alias_earlier_bi(N=16, TI=8, TJ=4):
    nbi = N // TI
    nbj = N // TJ
    assert nbi == 2, "this generator hardcoded for nbi=2"

    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    sC = lambda ii, jj: 7 + ii * TJ + jj             # 32 cells: 7..38

    A_base = 7 + TI * TJ                              # 39
    B_base = A_base + N * N                           # 295
    C_base = B_base + N * N                           # 551
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j
    C_bulk = lambda i, j: C_base + i * N + j

    def C(i, j):
        bi = i // TI
        bj = j // TJ
        # Case 1: last-bj cols for any bi -> A[bi-row, jj]
        if bj == nbj - 1:
            jj = j - (nbj - 1) * TJ
            return A(i, jj)
        # Case 2: bi == nbi-1 (=1) and bj < nbj-1 -> A[bi=0, ?]
        # Map C[i, j] for i=8..15, j=0..11 to A[i-8, j+4]:
        # That is, A[ai=0..7, ak=4..15], i.e., A[bi=0, k=4..15].
        # Wait: 96 distinct A cells in A[bi=0, k=4..15] = 8 rows × 12 cols.
        # We need 96 distinct C cells in C[bi=1, j=0..11] = 8 rows × 12 cols.
        # Map: C[8+ai, j] -> A[ai, j+4] for ai=0..7, j=0..11.
        if bi == nbi - 1 and bj < nbj - 1:
            ai = i - TI            # 0..7
            ak = j + TJ            # 4..15  (j=0..11 maps to ak=4..15)
            return A(ai, ak)
        # Case 3: bi == 0 and bj < nbj-1 -> bulk-C
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
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {sC(ii, jj)},{TMP}")
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(
                        f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(ii, jj)}"
                    )
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


if __name__ == "__main__":
    check_stop()
    log_event({"type": "exp_start", "lane": 0, "exp": "exp_0_alias_earlier_bi.py"})
    print(f"[{time.strftime('%H:%M:%S')}] starting exp_0_alias_earlier_bi")

    ir = generate_alias_earlier_bi()
    cost = matmul.score_16x16(ir)
    print(f"  raw = {cost:,}")

    renamed, info = rename_optimal(ir)
    cost_r = matmul.score_16x16(renamed)
    print(f"  renamed = {cost_r:,}  (renamer estimate {info['remapped_cost_estimate']:,})")

    BEST = min(cost, cost_r)
    BEST_IR = renamed if cost_r < cost else ir
    print(f"  best = {BEST:,}")

    PREV = 72350
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

    log_tokens("exp_0_alias_earlier_bi.py", 5000)
