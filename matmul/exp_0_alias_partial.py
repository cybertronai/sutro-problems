"""Lane 0 experiment v2: alias C onto A *partially*.

Keep sa_cache schedule as-is. For bj < nbj-1, write C to its bulk addrs.
For bj = nbj-1, A[bi*TI..bi*TI+TI-1, *] has been fully read (all bj iterations
done), so we can write C[bi*TI..bi*TI+TI-1, *] into A's addresses for that
bi-row's chunk for that bj.

But wait: the C cells for bj < nbj-1 use C-bulk addresses. The C cells for
bj = nbj-1 use A addresses. So aliasing only saves the bj=nbj-1 chunk.

Better strategy: alias the C cells of the LAST bj (bj=nbj-1) for ALL bi
onto the A cells of the SAME bi-row. This way:
  C[bi*TI..bi*TI+TI-1, (nbj-1)*TJ..N-1] aliases A[bi*TI..bi*TI+TI-1, ?].
There are TI rows of TJ columns = TI*TJ cells per bi. Need same count.
A[bi*TI..bi*TI+TI-1, *] has TI*N cells. C bj=nbj-1 has TI*TJ cells. So
TI*TJ <= TI*N — fits comfortably. Map to first TJ A columns of that bi-row.

For bi=0..nbi-1: C[bi*TI+ii, (nbj-1)*TJ+jj] aliases A[bi*TI+ii, jj].

Let's count: nbi * TI * TJ = 2 * 8 * 4 = 64 cells aliased (out of 256 C cells,
out of 256 A cells). 64 cells × 4 reads (A) + 1 (C output) = 5 reads/cell.
Other 192 A cells: 4 reads. Other 192 C cells: 1 read.
B: 256 cells × 2 reads.

Renamer-optimal placement after merging:
  64 cells × 5 reads (A & C aliased)
  192 cells × 4 reads (A non-aliased)
  256 cells × 2 reads (B)
  192 cells × 1 read (C non-aliased)

Plus scratchpad cells (sA, tmp, sB, sC) untouched.

Total: 64+192+256+192 = 704 distinct bulk cells (vs 768 in baseline).
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


def generate_partial_alias(N=16, TI=8, TJ=4):
    """sa_cache schedule with C[bi-row, last-bj-cols] aliased to A.

    For (bi, bj=nbj-1) writeback, use A[bi*TI+ii, jj] addresses (first TJ
    columns of A's bi-row). Safe because by the time we write, all A reads
    for bi-row are done (bj=nbj-1 is the last bj).
    """
    nbi = N // TI
    nbj = N // TJ

    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    sC = lambda ii, jj: 7 + ii * TJ + jj
    A_base = 7 + TI * TJ          # 39
    B_base = A_base + N * N       # 295
    C_base = B_base + N * N       # 551
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j

    def C(i, j):
        # Alias when j is in the last bj column block.
        if j >= (nbj - 1) * TJ:
            # Map to first TJ columns of A's row i.
            jj = j - (nbj - 1) * TJ
            return A(i, jj)
        return C_base + i * N + j

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


def generate_partial_alias_all_bj(N=16, TI=8, TJ=4):
    """Alias EVERY bj's C-row writeback into A addresses, but only after
    all bj iterations done for that bi.

    Per-bj separate sC scratchpad (so sC doesn't get overwritten across bj).
    Then at end of bi, copy each per-bj sC to A addresses.

    Actually we already tried this in exp_0_alias_ca.py: cost 89,886.
    """
    pass  # see exp_0_alias_ca


if __name__ == "__main__":
    check_stop()
    log_event({"type": "exp_start", "lane": 0, "exp": "exp_0_alias_partial.py"})
    print(f"[{time.strftime('%H:%M:%S')}] starting exp_0_alias_partial")

    ir = generate_partial_alias()
    cost = matmul.score_16x16(ir)
    print(f"  partial alias (last-bj only) cost = {cost:,}  (baseline 73,602)")

    renamed_ir, info = rename_optimal(ir)
    cost_r = matmul.score_16x16(renamed_ir)
    print(f"  + renamer = {cost_r:,}  (renamer estimate {info['remapped_cost_estimate']:,})")

    BEST = min(cost, cost_r)
    BEST_IR = renamed_ir if cost_r < cost else ir
    print(f"  best = {BEST:,}")

    if BEST < 73602:
        out = os.path.join(HERE, "records", f"record_{BEST}_lane0.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(BEST_IR + "\n")
        print(f"  NEW RECORD! saved {out}")
        log_event({
            "type": "new_record", "cost": BEST, "prev": 73602,
            "lane": 0, "file": out,
        })
    else:
        print(f"  no improvement")

    log_tokens("exp_0_alias_partial.py", 3000)
