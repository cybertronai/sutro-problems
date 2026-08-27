"""Lane 1 — block-cache-B variants.

Cache a vertical strip of B (N rows × TJ cols) in cheap scratch. Then iterate
over rows of A. For each A row, cache it; do TJ dot products against the
strip; spill to C.

Variant BB1: cache full B-strip (16 × TJ = 16*TJ cells).
  for bj in nbj:
    cache B[0:N, bj*TJ:(bj+1)*TJ]  -- N*TJ cells
    for i in N:
      cache A[i, 0:N] -- N cells
      for jj in TJ:
        dot = mul(A_row[0], sB[0,jj])
        for bk in 1..15:
          dot += A_row[bk] * sB[bk, jj]
        write C[i, bj*TJ+jj] = dot

  B bulk reads: each cell loaded 1×.
  A bulk reads: each cell loaded nbj×.
  ACC reads: 1 per inner dot product (spill).
  ACC2 reads: N-1 = 15 per inner dot product.
  A_row[bk]: read TJ times per i (one per jj). So per addr: TJ*N reads = 4*16 = 64 reads.
              Across 16 different A_rows... wait, A_row[bk] is overwritten each i. Total reads
              per address = TJ * N (per i, addr read TJ times) × N (loaded N times) = N*N*TJ = 1024.
              Wait, for each i we load A_row once (N=16 reads of A_row addrs from A bulk).
              Then for each jj=0..TJ, we read each A_row[bk] once. So per i, A_row[bk] cell is
              read TJ times. Across N i's, per A_row cell: N*TJ reads. With nbj iterations of bj,
              wait we're inside the bj loop.

  Let me redo. Loop: for bj: for i: cache A_row; for jj: dot.
    Per (bj, i, jj): A_row[bk] read once per bk. Total A_row reads per cell = TJ (per (bj,i)) × N i's × nbj bj's = TJ * N * nbj = 4*16*4 = 256? Hmm with TJ=4 nbj=4: 4*16*4 = 256 per A_row addr.

  sB[bk, jj] (block-B): each cell read once per (i, jj) with the matching jj index.
    Per cell sB[bk, jj]: read once per i. Total reads = N (i) × nbj (bj) = 16*4 = 64 reads per cell.
    For nbj=4 outer iterations, the strip is freshly loaded each time, but the N*TJ scratch
    addresses are reused. So per addr: 64 reads.

  Ahh let me reconsider this. For TJ=4, sB has 64 cells = 16*4. They sit at e.g., addrs 4..67.
  Each cell read 64 times across the bj-loop body. Address 4 cost = 2, addr 67 cost = 9.

OK let me just code it.
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from matmul import score_16x16  # noqa: E402

N = 16


def gen_bb1(TJ):
    """Block-B caching."""
    nbj = N // TJ
    ACC = 1
    ACC2 = 2
    sA = lambda bk: 3 + bk             # 3..2+N = 3..18
    sB = lambda bk, jj: 19 + bk*TJ + jj  # 19..19+N*TJ-1
    C_base = 19 + N*TJ
    C = lambda i, j: C_base + i*N + j
    A_base = C_base + N*N
    A = lambda i, k: A_base + i*N + k
    B_base = A_base + N*N
    Bf = lambda k, j: B_base + k*N + j

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]

    for bj in range(nbj):
        # Cache B strip
        for bk in range(N):
            for jj in range(TJ):
                lines.append(f"copy {sB(bk, jj)},{Bf(bk, bj*TJ+jj)}")
        for i in range(N):
            # Cache A row
            for bk in range(N):
                lines.append(f"copy {sA(bk)},{A(i, bk)}")
            for jj in range(TJ):
                # Dot product into ACC
                lines.append(f"mul {ACC},{sA(0)},{sB(0, jj)}")
                for bk in range(1, N):
                    lines.append(f"mul {ACC2},{sA(bk)},{sB(bk, jj)}")
                    lines.append(f"add {ACC},{ACC2}")
                lines.append(f"copy {C(i, bj*TJ+jj)},{ACC}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def gen_bb2(TJ, TI):
    """Block-B + block-A: cache A row block, B column block.
    for bi: for bj: cache A_row block (TI*N), cache B_strip (N*TJ)
              for ii in TI: for jj in TJ: dot product
    """
    nbi = N // TI
    nbj = N // TJ
    ACC = 1
    ACC2 = 2
    sA = lambda ii, bk: 3 + ii*N + bk  # TI*N cells starting at 3
    sB = lambda bk, jj: 3 + TI*N + bk*TJ + jj  # N*TJ cells
    sc_base = 3 + TI*N + N*TJ
    C_base = sc_base
    C = lambda i, j: C_base + i*N + j
    A_base = C_base + N*N
    A = lambda i, k: A_base + i*N + k
    B_base = A_base + N*N
    Bf = lambda k, j: B_base + k*N + j

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]

    for bi in range(nbi):
        for bj in range(nbj):
            # Cache A block (TI rows of N elements each)
            for ii in range(TI):
                for bk in range(N):
                    lines.append(f"copy {sA(ii, bk)},{A(bi*TI+ii, bk)}")
            # Cache B strip (N rows of TJ cols)
            for bk in range(N):
                for jj in range(TJ):
                    lines.append(f"copy {sB(bk, jj)},{Bf(bk, bj*TJ+jj)}")
            # Compute TI*TJ output cells
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"mul {ACC},{sA(ii, 0)},{sB(0, jj)}")
                    for bk in range(1, N):
                        lines.append(f"mul {ACC2},{sA(ii, bk)},{sB(bk, jj)}")
                        lines.append(f"add {ACC},{ACC2}")
                    lines.append(f"copy {C(bi*TI+ii, bj*TJ+jj)},{ACC}")
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
               "lane": 1, "exp": "exp_1_block_b.py"})

    print(f"{'gen':>10} {'TI':>3} {'TJ':>3}  {'cost':>10}")
    print("-" * 35)

    results = []
    for TJ in (1, 2, 4, 8, 16):
        try:
            ir = gen_bb1(TJ)
            cost = score_16x16(ir)
            print(f"{'bb1':>10} {'-':>3} {TJ:>3}  {cost:>10}")
            results.append(("bb1", 0, TJ, cost, ir))
        except Exception as e:
            print(f"{'bb1':>10} {'-':>3} {TJ:>3}  ERR {e!s:.40}")

    for TI in (1, 2, 4, 8, 16):
        for TJ in (1, 2, 4, 8, 16):
            try:
                ir = gen_bb2(TJ, TI)
                cost = score_16x16(ir)
                print(f"{'bb2':>10} {TI:>3} {TJ:>3}  {cost:>10}")
                results.append(("bb2", TI, TJ, cost, ir))
            except Exception as e:
                print(f"{'bb2':>10} {TI:>3} {TJ:>3}  ERR {e!s:.40}")

    results.sort(key=lambda r: r[3])
    print(f"\nTop 5:")
    for r in results[:5]:
        print(f"  {r[0]} TI={r[1]} TJ={r[2]}  cost={r[3]:,}")

    BEST_PRIOR = 73602
    best = results[0]
    if best[3] < BEST_PRIOR:
        cost = best[3]
        out = os.path.join(HERE, "records", f"record_{cost}_lane1.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(best[4] + "\n")
        print(f"\nNEW RECORD: {best[0]} TI={best[1]} TJ={best[2]}  cost={cost}")
        log_event({"ts": int(time.time()), "type": "new_record",
                   "cost": cost, "prev": BEST_PRIOR, "lane": 1,
                   "file": f"record_{cost}_lane1.ir"})
    else:
        print(f"\nNo improvement: best={best[3]}")

    log_token("exp_1_block_b.py", 7000)


if __name__ == "__main__":
    main()
