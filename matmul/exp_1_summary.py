"""Lane 1 — final summary of explored variants.

All variants tried:
  - V1: rank-1 sA cache + rank-TJ sB strip (the current best template)
  - V2: rank-TI sA strip + rank-TJ sB strip
  - V3: tmp@1 instead of sA@1
  - L2: swap bi/bj
  - inner_swap: rank-1 sB cache + rank-TI sA strip
  - hierarchical (h1): asymmetric reload, B near origin (cost 80217)
  - h2: hierarchical + rank-1 sA cache
  - h3: hierarchical + rank-1 sB cache
  - dot product: per-cell dot product with full row caching
  - block-B: cache full B column strip (N*TJ cells)
  - hybrid no-sC: write directly to C, no sC scratchpad

Result: V1(TI=8, TJ=4) = 73602 remains the best. The lower-bound tool
confirms this is the address-optimal layout for that algorithm structure.
To improve further requires a structurally different algorithm (e.g.,
fewer mul ops via Strassen-like tricks, but Strassen overhead dominates
at this scale).

Key insights for next-lane explorers:
1. The 73602 cost has dominant contributors: sC (28%), A bulk (18%), B bulk (15%).
2. Total reads = 17,920. To reduce cost, we must reduce TOTAL reads (i.e.,
   fewer ops), since current address layout is provably optimal.
3. Strassen reduces muls but adds many adds; net read count grows at this N.
4. The (TI=8, TJ=4) shape is optimal because:
     - Larger TI doesn't reduce A reads (A read 1024 times regardless of TI).
     - Larger TJ doesn't reduce B reads (B read 512 times regardless of TJ).
     - Smaller (TI, TJ) increases sC reuse but enlarges per-cell read count
       which doesn't change total reads — but spreads them across MORE cells
       (in the (16,16) case sC has 256 cells, very expensive).
   So the real optimization knob is sC SIZE (TI*TJ), where smaller is
   better up to the point where sB or sA loading needs change.
   Empirically (8,4)=32 cells is optimal.
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from matmul import score_16x16  # noqa: E402

N = 16


def gen_v1(TI, TJ):
    """The current best template: rank-1 sA cache + sB strip."""
    nbi, nbj = N // TI, N // TJ
    SA, TMP = 1, 2
    sB = lambda jj: 3 + jj
    sC_base = 3 + TJ
    sC = lambda ii, jj: sC_base + ii * TJ + jj
    A_base = sC_base + TI * TJ
    B_base = A_base + N * N
    C_base = B_base + N * N
    A = lambda i, k: A_base + i * N + k
    Bf = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j
    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]
    for bi in range(nbi):
        for bj in range(nbj):
            for bk in range(N):
                for jj in range(TJ):
                    lines.append(f"copy {sB(jj)},{Bf(bk, bj * TJ + jj)}")
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
               "lane": 1, "exp": "exp_1_summary.py"})

    print("Lane 1 summary:")
    print("  V1(TI=8, TJ=4) =", score_16x16(gen_v1(8, 4)))
    print("  Lower bound for V1(8,4): 73602 (confirmed by lower_bound.py)")
    print()
    print("Best result: 73,602 — same as prior best record. No improvement.")
    print()
    print("Variants explored (none beat 73602):")
    print("  V1 grid sweep — all (TI,TJ) with TI|16, TJ|16: best (8,4)=73602")
    print("  V2 rank-TI sA strip — all worse (extra hot cells)")
    print("  V3 tmp@1 — 73858 (since sA hotter than tmp)")
    print("  Loop reorderings — equivalent")
    print("  Hierarchical (h1) — 80217")
    print("  Hierarchical + sA-cache (h2) — 82892")
    print("  Hierarchical + sB-cache (h3) — 80588")
    print("  Dot-product variants — 141k+")
    print("  Block-B caching — 88k+")
    print("  Hybrid no-sC (direct C write) — 87918+")

    log_token("exp_1_summary.py", 1500)


if __name__ == "__main__":
    main()
