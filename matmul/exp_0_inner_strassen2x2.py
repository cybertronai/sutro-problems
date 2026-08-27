"""Lane 0: 2x2-Strassen INNER kernel inside the 69,697 sa-cache schedule.

Goal: Verify (or refute) the brief's prediction that placing a 2x2-Strassen
substructure at the innermost (ii, jj) level of the sa-cache schedule loses
to the naive contraction.

Baseline (sa-cache + final-add fusion + alias-output-with-input): 69,697
(records/record_69697_lane0.ir).

Inside that schedule, the inner contraction (per bk) is:
  for ii in 0..TI-1:
    copy SA, A(bi*TI+ii, bk)
    for jj in 0..TJ-1:
      mul tmp, SA, sB[jj]
      add sC[ii,jj], tmp     (or initialize at bk=0, write-to-C at bk=N-1)

For 2x2 inner-Strassen we'd need to split the (ii, jj) pair into 2x2
sub-blocks. BUT — note the contraction is over (i, j) in C, not over k.
Strassen's bilinear identity speeds up matrix-matrix multiplication
(saves muls when both matrices have algebraic structure aligned). At the
inner (ii, jj) level we are computing TI*TJ rank-1 updates from one A and
one B-row — *not* a matrix product. There's no 2x2 Strassen-applicable
structure here.

So we must reshape: instead of the 1D inner contraction over k, group
contiguous 2x2 (i, j) blocks of C and 2x2 (i, k) blocks of A and 2x2
(k, j) blocks of B, then use 2x2-Strassen as the inner-most matmul.

That means the schedule per (bi, bj) becomes:
  for bk in range(N // 2):           # k-block
    for ii in 0..TI//2:                # i-block (2 rows)
      load 2x2 A panel: a11, a12, a21, a22
      for jj in 0..TJ//2:              # j-block (2 cols)
        load 2x2 B panel: b11, b12, b21, b22
        compute 7 Strassen mults via derived ops
        update sC[2x2 block]

For TI=8, TJ=4 we'd have 4 i-blocks * 2 j-blocks = 8 panels per bk.
Per panel: 7 mults + 18 derived ops; vs naive: 8 mults + 4 adds (of which
each mul-add is fused into 2-op `add sC, tmp`).

Total per (bi, bj, bk) panel:
  Naive: 8 muls (2 reads each) + 8 adds (2 reads each) = 32 reads from sA/sB/tmp/sC
  Strassen 2x2: 7 muls + 4 derived A-ops + 4 derived B-ops + 7 mults of derived
                + 4 sC accumulation ops with derived combos
  ~58 reads vs 32 — clearly worse.

Implementing nonetheless to confirm empirically.

Schedule:
  Hold 4 sA cells (a11..a22), 4 sB cells, 4 derived-A cells (m1A..m4A),
  4 derived-B cells, 4 sC cells per panel.
  TJ=4, TI=8 → 2 panels in jj × 4 panels in ii = 8 panels per bk per (bi, bj).
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
        try:
            os.remove(STOP_FILE)
        except OSError:
            pass
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


def generate_inner_strassen2x2(N=16):
    """16x16 with 2x2 inner-Strassen at the innermost matmul block.

    Outer loops: tile sizes TI=2, TJ=2, TK=2 — every 2x2 sub-block uses
    Strassen. Total: 8 * 8 * 8 = 512 Strassen panels (was 4096 muls naive).
    Each panel: 7 muls + ~18 derived ops + 4 sC updates.

    Layout:
      addr 1: SA pool start (we'll use 1..16 for scratch)
      addr 17..16+N*N : A inputs
      addr 17+N*N..16+2*N*N : B inputs
      addr 17+2*N*N..16+3*N*N : C outputs
    """
    # Scratch addresses (kept hot)
    a11, a12, a21, a22 = 1, 2, 3, 4
    b11, b12, b21, b22 = 5, 6, 7, 8
    # Strassen derived M's (results)
    m1, m2, m3, m4, m5, m6, m7 = 9, 10, 11, 12, 13, 14, 15
    # Two scratch cells for derived combos in mults
    dA, dB = 16, 17
    # tmp for accumulation
    tmp = 18
    # Persistent C accumulator — full 16x16
    sC_BASE = 19  # 256 cells, addrs 19..274

    A_BASE = sC_BASE + N * N            # 275
    B_BASE = A_BASE + N * N             # 531
    C_BASE = B_BASE + N * N             # 787 (output)

    A = lambda i, k: A_BASE + i * N + k
    B = lambda k, j: B_BASE + k * N + j
    C = lambda i, j: C_BASE + i * N + j
    sC = lambda i, j: sC_BASE + i * N + j

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [B(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]

    lines = [",".join(map(str, inputs))]

    # Iterate over 2x2 panels of C (bi, bj) and 2-step k blocks (bk)
    for bi in range(N // 2):
        for bj in range(N // 2):
            for bk in range(N // 2):
                # Load A 2x2 panel
                lines.append(f"copy {a11},{A(2*bi+0, 2*bk+0)}")
                lines.append(f"copy {a12},{A(2*bi+0, 2*bk+1)}")
                lines.append(f"copy {a21},{A(2*bi+1, 2*bk+0)}")
                lines.append(f"copy {a22},{A(2*bi+1, 2*bk+1)}")
                # Load B 2x2 panel
                lines.append(f"copy {b11},{B(2*bk+0, 2*bj+0)}")
                lines.append(f"copy {b12},{B(2*bk+0, 2*bj+1)}")
                lines.append(f"copy {b21},{B(2*bk+1, 2*bj+0)}")
                lines.append(f"copy {b22},{B(2*bk+1, 2*bj+1)}")

                # Strassen 2x2 mults
                # m1 = (a11+a22)(b11+b22)
                lines.append(f"add {dA},{a11},{a22}")
                lines.append(f"add {dB},{b11},{b22}")
                lines.append(f"mul {m1},{dA},{dB}")
                # m2 = (a21+a22) b11
                lines.append(f"add {dA},{a21},{a22}")
                lines.append(f"mul {m2},{dA},{b11}")
                # m3 = a11 (b12-b22)
                lines.append(f"sub {dB},{b12},{b22}")
                lines.append(f"mul {m3},{a11},{dB}")
                # m4 = a22 (b21-b11)
                lines.append(f"sub {dB},{b21},{b11}")
                lines.append(f"mul {m4},{a22},{dB}")
                # m5 = (a11+a12) b22
                lines.append(f"add {dA},{a11},{a12}")
                lines.append(f"mul {m5},{dA},{b22}")
                # m6 = (a21-a11)(b11+b12)
                lines.append(f"sub {dA},{a21},{a11}")
                lines.append(f"add {dB},{b11},{b12}")
                lines.append(f"mul {m6},{dA},{dB}")
                # m7 = (a12-a22)(b21+b22)
                lines.append(f"sub {dA},{a12},{a22}")
                lines.append(f"add {dB},{b21},{b22}")
                lines.append(f"mul {m7},{dA},{dB}")

                # Combine into sC (accumulating over bk)
                # c11 = m1 + m4 - m5 + m7
                # c12 = m3 + m5
                # c21 = m2 + m4
                # c22 = m1 - m2 + m3 + m6
                if bk == 0:
                    # First k-block: initialize sC via 3-arg writes.
                    # c11: tmp = m1 + m4; tmp = tmp - m5; sC11 = tmp + m7
                    lines.append(f"add {tmp},{m1},{m4}")
                    lines.append(f"sub {tmp},{m5}")
                    lines.append(f"add {sC(2*bi+0, 2*bj+0)},{tmp},{m7}")
                    # c12 = m3 + m5
                    lines.append(f"add {sC(2*bi+0, 2*bj+1)},{m3},{m5}")
                    # c21 = m2 + m4
                    lines.append(f"add {sC(2*bi+1, 2*bj+0)},{m2},{m4}")
                    # c22 = m1 - m2 + m3 + m6
                    lines.append(f"sub {tmp},{m1},{m2}")
                    lines.append(f"add {tmp},{m3}")
                    lines.append(f"add {sC(2*bi+1, 2*bj+1)},{tmp},{m6}")
                else:
                    # Accumulate
                    lines.append(f"add {tmp},{m1},{m4}")
                    lines.append(f"sub {tmp},{m5}")
                    lines.append(f"add {tmp},{m7}")
                    lines.append(f"add {sC(2*bi+0, 2*bj+0)},{tmp}")

                    lines.append(f"add {tmp},{m3},{m5}")
                    lines.append(f"add {sC(2*bi+0, 2*bj+1)},{tmp}")

                    lines.append(f"add {tmp},{m2},{m4}")
                    lines.append(f"add {sC(2*bi+1, 2*bj+0)},{tmp}")

                    lines.append(f"sub {tmp},{m1},{m2}")
                    lines.append(f"add {tmp},{m3}")
                    lines.append(f"add {tmp},{m6}")
                    lines.append(f"add {sC(2*bi+1, 2*bj+1)},{tmp}")

    # Write sC -> C
    for i in range(N):
        for j in range(N):
            lines.append(f"copy {C(i, j)},{sC(i, j)}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


if __name__ == "__main__":
    check_stop()
    log_event({"type": "exp_start", "lane": 0, "exp": "exp_0_inner_strassen2x2.py"})
    print(f"[{time.strftime('%H:%M:%S')}] starting exp_0_inner_strassen2x2")

    ir = generate_inner_strassen2x2()
    raw = matmul.score_16x16(ir)
    print(f"  raw = {raw:,}")
    renamed, info = rename_optimal(ir)
    cost_r = matmul.score_16x16(renamed)
    print(f"  renamed = {cost_r:,}  (renamer estimate {info['remapped_cost_estimate']:,})")

    BEST = min(raw, cost_r)
    BEST_IR = renamed if cost_r < raw else ir
    print(f"  best = {BEST:,}")

    PREV = 69697
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
        print(f"  no improvement vs {PREV:,} (delta = +{BEST - PREV:,})")
        log_event({
            "type": "no_improvement", "cost": BEST, "prev": PREV,
            "lane": 0, "exp": "exp_0_inner_strassen2x2.py",
        })

    log_tokens("exp_0_inner_strassen2x2.py", 6000)
