"""Lane 1: alias-chain + finaladd, parameterized by (TI, TJ).

Goal: reduce sC size from 32 to {16, 8, 4, ...} so bulk addrs start lower
(cheaper bulk reads). Cost-vs-tile-size tradeoff:
  smaller sC -> sC-cells cost less, bulk-reads cost less,
                BUT A reads = nbj=N/TJ × 256 (more if TJ small),
                BUT B reads = nbi=N/TI × 256 (more if TI small).

Aliasing schema (alias-chain): for each bi >= 1,
  C[bi, last-bj-cols]  -> A[bi, jj=0..TJ-1]              (32 cells/bi)
  C[bi, non-last-bj]   -> A[bi-1, k=TJ..N-1]            (TI*(N-TJ) per bi)
For bi=0:
  last-bj-cols  -> A[0, jj=0..TJ-1]
  non-last-bj   -> bulk-C  (no recipient since no bi=-1)

A[bi=last, k=TJ..N-1] = TI*(N-TJ) cells stranded, never aliased.

Finaladd trick: at bk=N-1, write final accumulator directly to C-address via
3-arg add: `add C, sC, tmp`. Saves 1 sC-read per (ii,jj) per (bi,bj).
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


def generate(N, TI, TJ):
    assert N % TI == 0 and N % TJ == 0
    nbi = N // TI
    nbj = N // TJ

    SA = 1
    TMP = 2
    sB_base = 3
    sB = lambda jj: sB_base + jj                       # TJ cells
    sC_base = sB_base + TJ                              # 3+TJ
    sC = lambda ii, jj: sC_base + ii * TJ + jj         # TI*TJ cells

    A_base = sC_base + TI * TJ
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
        else:
            if bi >= 1:
                ai = (bi - 1) * TI + (i % TI)
                ak = TJ + j
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
                        elif bk == N - 1:
                            # FINAL: write directly to C
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            c_addr = C(bi * TI + ii, bj * TJ + jj)
                            lines.append(f"add {c_addr},{sC(ii, jj)},{TMP}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {sC(ii, jj)},{TMP}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def run_variant(N, TI, TJ, prev_best):
    try:
        ir = generate(N, TI, TJ)
    except AssertionError:
        return None
    raw = matmul.score_16x16(ir) if N == 16 else matmul.score_4x4(ir)
    renamed, info = rename_optimal(ir)
    cost_r = matmul.score_16x16(renamed) if N == 16 else matmul.score_4x4(renamed)
    best = min(raw, cost_r)
    best_ir = renamed if cost_r < raw else ir
    return {
        "TI": TI, "TJ": TJ, "raw": raw, "renamed": cost_r,
        "best": best, "best_ir": best_ir,
    }


if __name__ == "__main__":
    check_stop()
    log_event({"type": "exp_start", "lane": LANE, "exp": "exp_1_minsc_alias_finaladd.py"})
    print(f"[{time.strftime('%H:%M:%S')}] starting minsc_alias_finaladd")

    N = 16
    PREV = 69697

    # Try all valid (TI, TJ) divisor pairs of N=16. sC = TI*TJ.
    candidates = [
        (TI, TJ) for TI in [1, 2, 4, 8, 16] for TJ in [1, 2, 4, 8, 16]
    ]
    # Sort by sC ascending for printing.
    candidates.sort(key=lambda p: (p[0] * p[1], p[0]))

    best_overall = None
    best_overall_ir = None
    results = []

    for TI, TJ in candidates:
        check_stop()
        r = run_variant(N, TI, TJ, PREV)
        if r is None:
            continue
        results.append(r)
        flag = ""
        if best_overall is None or r["best"] < best_overall:
            best_overall = r["best"]
            best_overall_ir = r["best_ir"]
            flag = " *"
        print(f"  TI={TI:2d} TJ={TJ:2d} sC={TI*TJ:3d}  "
              f"raw={r['raw']:>7,}  renamed={r['renamed']:>7,}  "
              f"best={r['best']:>7,}{flag}")

    print(f"\n  best_overall = {best_overall:,}  vs prev {PREV:,}")
    if best_overall < PREV:
        out = os.path.join(HERE, "records", f"record_{best_overall}_lane{LANE}.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(best_overall_ir + "\n")
        print(f"  NEW RECORD! saved {out}")
        log_event({
            "type": "new_record", "cost": best_overall, "prev": PREV,
            "lane": LANE, "file": out,
        })
    else:
        print(f"  no improvement vs {PREV:,}")

    log_event({
        "type": "exp_done", "lane": LANE,
        "exp": "exp_1_minsc_alias_finaladd.py",
        "best": best_overall, "n_variants": len(results),
    })
    log_tokens("exp_1_minsc_alias_finaladd.py", 5500)
