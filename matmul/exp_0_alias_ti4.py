"""Lane 0 v6: alias-earlier-bi with smaller tiles TI=4, TJ=4 -> nbi=4, nbj=4.

With nbi=4, all C cells except C[bi=0, non-last-bj] and C[bi=last, non-last-bj]
no wait... with nbi=4 we have 4 bi's. Earlier-bi aliasing puts C[bi=k]
non-last-bj cells onto A[bi'<k]. So C[bi=0, non-last-bj] has no recipient
(no bi'<0); it must go to bulk-C. C[bi=k>=1, non-last-bj] aliases to A[bi=k-1].

A[bi=last=3, k=4..15] has no source from bi=4+ (doesn't exist), so
48 cells stranded with 4 reads each.

Cell budget:
  Active sC: TI*TJ = 16 cells. Reads: per (bi, bj): TI mul-init + TI*(N-1) adds + writeback.
    Wait recompute. For each (bi, bj) with TI=4, TJ=4:
    bk=0: TI*TJ mul-inits (write only, no read of sC).
    bk=1..N-1 (15 iters): TI*TJ adds (read sC). 15 × 16 = 240 reads.
    Writeback: TI*TJ copies (read sC). 16 reads.
    Per (bi, bj): 256 reads of sC, spread over 16 cells -> 16 reads/cell.
    Across nbi*nbj=16 (bi,bj) iterations: 16 × 16 = 256 reads/cell.

  TMP: per (bi, bj): for each bk=1..N-1, TI*TJ muls into tmp + TI*TJ reads
       by add. So per bk-step: TI*TJ writes and TI*TJ reads. TMP read =
       TI*TJ reads per bk × 15 bks = 240 per (bi,bj). × 16 (bi,bj) = 3840.
  SA: per inner mul, 1 read of SA. Inner muls per (bi,bj): TI*TJ*N = 256.
       × 16 (bi,bj) = 4096.
  sB: per inner mul, 1 read. 4096 / TJ = 1024 reads per sB cell. (4 cells.)

  A=C aliased: 208 cells × 5 reads
  A non-alias: 48 cells × 4 reads
  B: 256 cells × 2 reads
  bulk-C: 48 cells × 1 read

Total cells: 1 (SA) + 1 (TMP) + 4 (sB) + 16 (sC) + 256 (A) + 256 (B) + 48 (bulk-C) = 582

Layout (renamer-optimal):
  1: SA (4096)
  2: TMP (3840)
  3-6: sB (1024 × 4)
  7-22: sC (256 × 16)
  23-230: A=C (5 × 208)
  231-278: A non-alias (4 × 48)
  279-534: B (2 × 256)
  535-582: bulk-C (1 × 48)
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


def generate_alias_chain(N=16, TI=4, TJ=4):
    """C[bi, *] aliases:
        last-bj cols -> A[bi, jj] (jj = 0..TJ-1)
        non-last-bj cols (j=0..(nbj-1)*TJ-1 = 0..N-TJ-1) -> A[bi-1, k=TJ..N-1]
                                                          if bi>=1
        else -> bulk-C
    """
    nbi = N // TI
    nbj = N // TJ

    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    sC = lambda ii, jj: 7 + ii * TJ + jj             # TI*TJ = 16 cells

    A_base = 7 + TI * TJ                              # 23
    B_base = A_base + N * N                           # 279
    C_base = B_base + N * N                           # 535
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j
    C_bulk = lambda i, j: C_base + i * N + j

    def C(i, j):
        bi = i // TI
        bj = j // TJ
        if bj == nbj - 1:
            # last-bj cols -> A[bi, jj=0..TJ-1]
            jj = j - (nbj - 1) * TJ
            return A(i, jj)
        else:
            # non-last-bj cols
            if bi >= 1:
                # alias to A[bi-1, k=TJ..N-1]
                # Map (i, j) where i = bi*TI + ii, j = bj*TJ + jj_in (bj<nbj-1).
                # Linear index of (bi, ii, j): we have nbi*TI = N rows
                # but bi starts from 1.
                # Available cells in A[bi-1]: TI rows × (N-TJ) cols = TI*(N-TJ).
                # Slots for bi: bj=0..nbj-2 gives (nbj-1)*TI*TJ cells
                # = (nbj-1) * TI * TJ. With TJ=4, nbj=4: 3*4*4=48. Matches.
                ai = (bi - 1) * TI + (i % TI)        # row in A
                ak = TJ + j                           # col in A: TJ + (bj*TJ + jj_in)
                # j ranges 0..(nbj-1)*TJ-1 = 0..N-TJ-1. ak = TJ..N-1. ✓
                return A(ai, ak)
            else:
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
    log_event({"type": "exp_start", "lane": 0, "exp": "exp_0_alias_ti4.py"})
    print(f"[{time.strftime('%H:%M:%S')}] starting exp_0_alias_ti4")

    # Verify safety: must check that for each writeback, the target A
    # address is no longer being read.
    # bi=k writes C[bi=k, non-last-bj] to A[bi=k-1, *]. At that moment,
    # we are inside bi=k loop (bk loop done for current bj). bi=k-1 is
    # complete. ✓
    # bi=k writes C[bi=k, last-bj] to A[bi=k, jj=0..TJ-1]. At that moment,
    # we just finished bi=k's last bj's bk loop. A[bi=k, *] reads happen
    # ONLY within bi=k's (bj, bk, ii) iterations. bj=last is the LAST bj,
    # so all reads of A[bi=k, *] (across bj=0..last) are now done. ✓

    ir = generate_alias_chain()
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

    log_tokens("exp_0_alias_ti4.py", 5500)
