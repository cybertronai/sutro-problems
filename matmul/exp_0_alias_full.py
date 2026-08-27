"""Lane 0 v3: full alias of C onto A, with frozen-sC for deferred writebacks.

Schedule:
  for bi in range(nbi):
    for bj in range(nbj):
      [bk-loop accumulates into active sC]
      if bj < nbj-1:
        copy active sC into frozen[bj]   # 32 cells written, will be read once
      else:  # bj == nbj-1
        copy active sC directly to A[bi-row, last-bj-cols]  # alias!
    # End of bi: write frozen sC blocks to A addresses
    for bj in 0..nbj-2:
      copy frozen[bj] -> A[bi-row, bj-cols]

Layout target (renamer-optimal):
  addr 1: SA (4096 reads)
  addr 2: TMP (3840)
  addrs 3-6: sB (1024 each)
  addrs 7-38: active sC (128 reads each, 32 cells)
  addrs 39-294: A=C aliased (5 reads each, 256 cells)
  addrs 295-550: B (2 reads each, 256 cells)
  addrs 551-646: frozen sC (1 read each, 96 cells)

Predicted cost: ~56,993.
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


def generate_full_alias(N=16, TI=8, TJ=4):
    nbi = N // TI
    nbj = N // TJ

    # Initial layout (will be remapped by renamer)
    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    sC_active = lambda ii, jj: 7 + ii * TJ + jj         # 32 cells
    # Frozen sC: (nbj-1) blocks of TI*TJ each
    sC_frozen_base = 7 + TI * TJ                         # 39
    sC_frozen = lambda bj, ii, jj: sC_frozen_base + bj * (TI * TJ) + ii * TJ + jj
    A_base = sC_frozen_base + (nbj - 1) * TI * TJ
    B_base = A_base + N * N
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j

    def C(i, j):
        return A(i, j)  # full alias

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
            # End of (bi, bj): handle writeback.
            if bj < nbj - 1:
                # Copy active sC -> frozen[bj]
                for ii in range(TI):
                    for jj in range(TJ):
                        lines.append(
                            f"copy {sC_frozen(bj, ii, jj)},{sC_active(ii, jj)}"
                        )
            else:
                # bj == nbj-1: A[bi-row, *] is fully consumed. Write directly to A.
                # The C cells for this bj are C[bi*TI..bi*TI+TI-1, (nbj-1)*TJ..N-1].
                for ii in range(TI):
                    for jj in range(TJ):
                        j_full = bj * TJ + jj
                        lines.append(
                            f"copy {C(bi * TI + ii, j_full)},{sC_active(ii, jj)}"
                        )
        # End of bi: write all frozen sC blocks to A addresses.
        for bj in range(nbj - 1):
            for ii in range(TI):
                for jj in range(TJ):
                    j_full = bj * TJ + jj
                    lines.append(
                        f"copy {C(bi * TI + ii, j_full)},{sC_frozen(bj, ii, jj)}"
                    )
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def generate_full_alias_4x4():
    return generate_full_alias(N=4, TI=2, TJ=2)


if __name__ == "__main__":
    check_stop()
    log_event({"type": "exp_start", "lane": 0, "exp": "exp_0_alias_full.py"})
    print(f"[{time.strftime('%H:%M:%S')}] starting exp_0_alias_full")

    # 4x4 sanity
    ir4 = generate_full_alias_4x4()
    cost4 = matmul.score_4x4(ir4)
    print(f"  4x4 full-alias cost = {cost4:,}")

    # 16x16
    ir16 = generate_full_alias()
    cost16 = matmul.score_16x16(ir16)
    print(f"  16x16 raw = {cost16:,}")

    renamed, info = rename_optimal(ir16)
    cost_r = matmul.score_16x16(renamed)
    print(f"  16x16 renamed = {cost_r:,}  (renamer estimate {info['remapped_cost_estimate']:,})")

    BEST = min(cost16, cost_r)
    BEST_IR = renamed if cost_r < cost16 else ir16
    print(f"  best = {BEST:,}")

    PREV = 73602
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

    log_tokens("exp_0_alias_full.py", 4000)
