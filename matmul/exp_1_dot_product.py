"""Lane 1 — dot-product based variants.

Structure: for each i, cache A[i,:] in cheap scratch. For each j, cache B[:,j] in
cheap scratch. Compute dot product into a register (cheap addr 1), spill to C.

Variant DP1: Outer i, inner j. A_row cached per i (16 cells), B_col cached per j (16 cells).
  A bulk: each cell read 1× (load A_row once per i).
  B bulk: each cell read N=16× (load B_col once per (i,j) pair → N times per j-col).

  Memory layout:
    addr 1: ACC (accumulator, also tmp)
    addrs 2..17: A_row (16 cells)
    addrs 18..33: B_col (16 cells)
    addrs 34..289: C bulk (output)
    addrs 290..545: A bulk
    addrs 546..801: B bulk

  Read cost analysis:
    ACC: written N*N=256 times, read N*N + N*N*(N-1)/?=... Let me think.
    Per (i,j): mul ACC, A_row[0], B_col[0]; then for bk=1..15: mul tmp, A_row[bk], B_col[bk]; add ACC, tmp.
    But ACC and tmp are separate. We have only 1 cheap addr. Hmm.

    Alternative without tmp:
      mul ACC, A_row[0], B_col[0]
      for bk=1..15:
        mul ACC2, A_row[bk], B_col[bk]
        add ACC, ACC2
    ACC and ACC2 both at cheap addrs.

  Reads per (i,j):
    ACC reads: 15 (in add) + 1 (spill to C) = 16
    ACC2 reads: 15 (in add)
    A_row reads: 16 (one per bk, in mul)
    B_col reads: 16

  Across all (i,j) = 256:
    ACC: 16*256 = 4096 reads
    ACC2: 15*256 = 3840 reads
    A_row[bk]: read once per (i,j), so 256 reads per A_row cell across all j of one i.
              Per i, A_row[bk] is read N=16 times. ✓
              Total A_row reads = 16 cells × N (per i) × N (rows) = 4096. Hmm wait
              that's per cell across all i? No, A_row is reloaded each i.
              Per i: A_row[bk] is read N times (once per j). 16 cells × 16 reads = 256 reads per i.
              Across N=16 i's: 16 cells × 16 j's = 256 total per i × 16 i's = 4096 reads of A_row.

              But A_row cells reuse same addresses! So address 2 sees 16 reads × 16 i's = 256 reads.
              Same for all 16 A_row addrs. Total = 16 addrs × 256 reads = 4096 reads. ✓

    B_col[bk]: per (i,j): read once. Per j: N reads. Per (i,j): 1 read. Total = 256 reads of B_col strip.
              Per address: 256/16 = 16 reads per B_col cell? No wait, B_col is loaded per j and reused
              across all i? No — in inner i, j: B_col is loaded fresh inside the (i,j) loop.

              Hmm actually loop structure: for i: for j: load B_col, compute. So B_col is loaded
              16*16 = 256 times (once per (i,j)). Each load = 16 cells. So B bulk reads = 256*16 = 4096.
              Each B cell read = 4096/256 = 16 times. ✓

              B_col scratch: per (i,j): each B_col[bk] read once in mul. So 256 reads per cell? No,
              total B_col scratch reads = 256 (i,j pairs) × 16 (bk per pair, but only 1 read of each cell per pair) = 256*1 per cell = 256.
              Wait that's not right either. Per (i,j), in inner: each B_col[bk] read once. So per (i,j),
              B_col[0] read 1 time, ..., B_col[15] read 1 time. Per cell = 1 read per (i,j) = 256 reads
              across all (i,j). With 16 cells, total = 16*256 = 4096 reads.

              Per ADDRESS (since B_col reuses 16 addrs across all (i,j)): each addr sees 256 reads.

  Now layout costs:
    ACC @ addr 1: 4096 × 1 = 4096
    ACC2 @ addr 2: 3840 × 2 = 7680
    A_row @ 3..18: 256 reads × addr-cost. cost(3,4)=2,2; cost(5..9)=3 (5 cells); cost(10..16)=4 (7 cells);
                   cost(17,18)=5,5. Sum = 4+15+28+10 = 57. Times 256 = 14592.
    B_col @ 19..34: cost(19..25)=5 (7 cells); cost(26..34)=6 (9 cells). Sum = 35+54 = 89. Times 256 = 22784.
    A bulk @ 35..290: cost(35,36)=6; cost(37..49)=7 (13); cost(50..64)=8 (15); cost(65..81)=9 (17);
                     cost(82..100)=10 (19); cost(101..121)=11 (21); cost(122..144)=12 (23);
                     cost(145..169)=13 (25); cost(170..196)=14 (27); cost(197..225)=15 (29);
                     cost(226..256)=16 (31); cost(257..290)=17 (34). Sum complex.
                     Each cell read 1× → cost = sum_costs ≈ 13*36 = ... let me compute exactly later.
    B bulk @ 291..546: even more expensive, each read 16×.
    C bulk @ 547..802: each read 1×.

  This is going to be expensive due to B bulk reads * 16.

Variant DP2: cache only A_row, do mul-add directly from B_bulk.
Variant DP3: outer j, inner i. Then A is reloaded N times, B is reloaded once.
  But which axis to put close to origin?

Let's just code and measure.
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from matmul import score_16x16  # noqa: E402

N = 16


def gen_dp1():
    """Outer i, inner j. A_row cached, B_col cached."""
    ACC = 1
    ACC2 = 2
    sA = lambda bk: 3 + bk             # 3..18
    sB = lambda bk: 19 + bk            # 19..34
    C_base = 35
    C = lambda i, j: C_base + i*N + j
    A_base = C_base + N*N              # 291
    A = lambda i, k: A_base + i*N + k
    B_base = A_base + N*N              # 547
    Bf = lambda k, j: B_base + k*N + j
    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]
    for i in range(N):
        # Load A_row
        for bk in range(N):
            lines.append(f"copy {sA(bk)},{A(i, bk)}")
        for j in range(N):
            # Load B_col
            for bk in range(N):
                lines.append(f"copy {sB(bk)},{Bf(bk, j)}")
            # Dot product
            lines.append(f"mul {ACC},{sA(0)},{sB(0)}")
            for bk in range(1, N):
                lines.append(f"mul {ACC2},{sA(bk)},{sB(bk)}")
                lines.append(f"add {ACC},{ACC2}")
            lines.append(f"copy {C(i, j)},{ACC}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def gen_dp2():
    """Outer i, inner j. A_row cached only; B read directly from bulk."""
    ACC = 1
    ACC2 = 2
    sA = lambda bk: 3 + bk             # 3..18
    C_base = 19
    C = lambda i, j: C_base + i*N + j
    A_base = C_base + N*N              # 275
    A = lambda i, k: A_base + i*N + k
    B_base = A_base + N*N              # 531
    Bf = lambda k, j: B_base + k*N + j
    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]
    for i in range(N):
        for bk in range(N):
            lines.append(f"copy {sA(bk)},{A(i, bk)}")
        for j in range(N):
            lines.append(f"mul {ACC},{sA(0)},{Bf(0, j)}")
            for bk in range(1, N):
                lines.append(f"mul {ACC2},{sA(bk)},{Bf(bk, j)}")
                lines.append(f"add {ACC},{ACC2}")
            lines.append(f"copy {C(i, j)},{ACC}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def gen_dp3():
    """Outer j, inner i. B_col cached. (Symmetric of dp1 but B in cheap.)"""
    ACC = 1
    ACC2 = 2
    sB = lambda bk: 3 + bk
    sA = lambda bk: 19 + bk
    C_base = 35
    C = lambda i, j: C_base + i*N + j
    A_base = C_base + N*N
    A = lambda i, k: A_base + i*N + k
    B_base = A_base + N*N
    Bf = lambda k, j: B_base + k*N + j
    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]
    for j in range(N):
        for bk in range(N):
            lines.append(f"copy {sB(bk)},{Bf(bk, j)}")
        for i in range(N):
            for bk in range(N):
                lines.append(f"copy {sA(bk)},{A(i, bk)}")
            lines.append(f"mul {ACC},{sA(0)},{sB(0)}")
            for bk in range(1, N):
                lines.append(f"mul {ACC2},{sA(bk)},{sB(bk)}")
                lines.append(f"add {ACC},{ACC2}")
            lines.append(f"copy {C(i, j)},{ACC}")
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
               "lane": 1, "exp": "exp_1_dot_product.py"})

    print(f"{'gen':>6}  {'cost':>10}")
    print("-" * 22)

    results = []
    for name, gen in [("dp1", gen_dp1), ("dp2", gen_dp2), ("dp3", gen_dp3)]:
        try:
            ir = gen()
            cost = score_16x16(ir)
            print(f"{name:>6}  {cost:>10}")
            results.append((name, cost, ir))
        except Exception as e:
            print(f"{name:>6}  ERR {e}")

    results.sort(key=lambda r: r[1])
    BEST_PRIOR = 73602
    best = results[0]
    if best[1] < BEST_PRIOR:
        cost = best[1]
        out = os.path.join(HERE, "records", f"record_{cost}_lane1.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(best[2] + "\n")
        print(f"\nNEW RECORD: {best[0]}  cost={cost}")
        log_event({"ts": int(time.time()), "type": "new_record",
                   "cost": cost, "prev": BEST_PRIOR, "lane": 1,
                   "file": f"record_{cost}_lane1.ir"})
    else:
        print(f"\nNo improvement: best={best[1]}")

    log_token("exp_1_dot_product.py", 5000)


if __name__ == "__main__":
    main()
