"""Lane 0 v4: mixed strategy — alias C onto A for the LAST bi only.

For bi < nbi-1: standard sa_cache (C goes to bulk-C, except last-bj-cols
                aliased to A[bi-row, jj] as in partial_last_bj).
For bi == nbi-1: ALL C[last_bi_row, *] aliased to A[last_bi_row, *].
                 Frozen sC needed for bj < nbj-1 within this bi.

Aliased cells per bi:
  bi < nbi-1: TI × TJ = 32 cells  (last bj only) -> 5 reads each
  bi == nbi-1: TI × N = 8 × 16 = 128 cells (whole row) -> 5 reads each

Total aliased: (nbi-1) × 32 + TI × N = 1×32 + 128 = 160 cells × 5 reads
Non-aliased A: 256 - 160 = 96 cells × 4 reads
Non-aliased C: 256 - 160 = 96 cells × 1 read
B: 256 cells × 2 reads
Frozen sC (for bi=last only, nbj-1=3 blocks of TI*TJ=32): 96 cells × 1 read
Active sC: 32 cells × 128 reads
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


def generate_mixed_alias(N=16, TI=8, TJ=4):
    nbi = N // TI
    nbj = N // TJ

    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    sC_active = lambda ii, jj: 7 + ii * TJ + jj         # 32 cells
    # Frozen sC for last bi only: (nbj-1) blocks
    sC_frozen_base = 7 + TI * TJ                         # 39
    sC_frozen = lambda bj, ii, jj: sC_frozen_base + bj * (TI * TJ) + ii * TJ + jj
    A_base = sC_frozen_base + (nbj - 1) * TI * TJ        # 39 + 96 = 135
    B_base = A_base + N * N                              # 391
    C_base = B_base + N * N                              # 647 (only used for non-aliased C)
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j
    C_bulk = lambda i, j: C_base + i * N + j

    def C(i, j):
        bi = i // TI
        bj = j // TJ
        # Alias all of last-bi-row onto A[last_bi_row, *]
        if bi == nbi - 1:
            return A(i, j)
        # For bi < nbi-1, alias only the last-bj-cols onto A[i, jj]
        if bj == nbj - 1:
            jj = j - (nbj - 1) * TJ
            return A(i, jj)
        # Otherwise, bulk-C
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
                            lines.append(f"mul {sC_active(ii, jj)},{SA},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {sC_active(ii, jj)},{TMP}")
            # End of (bi, bj) writeback
            if bi == nbi - 1 and bj < nbj - 1:
                # Last bi, but not last bj: defer (write to frozen sC).
                for ii in range(TI):
                    for jj in range(TJ):
                        lines.append(
                            f"copy {sC_frozen(bj, ii, jj)},{sC_active(ii, jj)}"
                        )
            else:
                # All other cases: write directly. C() handles aliasing.
                for ii in range(TI):
                    for jj in range(TJ):
                        lines.append(
                            f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC_active(ii, jj)}"
                        )
        # End of bi: if last bi, dump frozen sC blocks to A addresses.
        if bi == nbi - 1:
            for bj in range(nbj - 1):
                for ii in range(TI):
                    for jj in range(TJ):
                        lines.append(
                            f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC_frozen(bj, ii, jj)}"
                        )
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


if __name__ == "__main__":
    check_stop()
    log_event({"type": "exp_start", "lane": 0, "exp": "exp_0_alias_mixed.py"})
    print(f"[{time.strftime('%H:%M:%S')}] starting exp_0_alias_mixed")

    ir = generate_mixed_alias()
    cost = matmul.score_16x16(ir)
    print(f"  raw = {cost:,}")

    renamed, info = rename_optimal(ir)
    cost_r = matmul.score_16x16(renamed)
    print(f"  renamed = {cost_r:,}  (renamer estimate {info['remapped_cost_estimate']:,})")

    BEST = min(cost, cost_r)
    BEST_IR = renamed if cost_r < cost else ir
    print(f"  best = {BEST:,}")

    PREV = 72350  # current Lane 0 record
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

    log_tokens("exp_0_alias_mixed.py", 4500)
