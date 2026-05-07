"""16×16 matmul submission: C↔A address aliasing + final-add fusion.

Two structure-preserving tricks on top of the sa-cache schedule (TI=8, TJ=4):

1. **Alias C output cells with A input cells.** The scorer requires input
   addresses to be distinct from each other, but does NOT require output
   addresses to be distinct from inputs — only that the output address holds
   the right value at exit. After all reads of A[i,k] are done, that cell is
   dead and can host a C value. With (bi, bj, bk) loop order, 160 of the 256
   C cells can alias onto A cells without changing the schedule:

   * j in last bj-block (bj=nbj−1, j=12..15): A[i, j−12] is read 4×, then
     immediately receives C[i, j] before any further reads. (4 cells per
     row × 16 rows = 64 cells.)
   * bi=1, j in non-last bj-block (j=0..11): by the time bi=1 starts, all
     of A[bi=0, k=4..15] is dead; we route those C cells onto A[bi=0]'s
     dead slots. (8 rows × 12 cols = 96 cells.)
   * Remaining 96 cells (bi=0, j=0..11) need fresh bulk addresses — they
     are written at the same time their A counterparts are still being
     read in the bi=1 phase, so aliasing is not possible.

2. **Final-add fusion.** Each accumulating step is `mul tmp, sA, sB; add sC,
   tmp` — two ops, four reads. At bk=N−1 the result will be copied to C.
   Replace the final pair with `mul tmp, sA, sB; add C, sC, tmp` (3-operand
   add): the result is written directly into C, eliminating the separate
   sC→C copy. Saves one sC read per (ii, jj, bi, bj) = 256 reads at sC's
   cost ≈ 5 each ≈ 1.3k cost.

Layout (greedy by descending read count, provably optimal among address
permutations because cost(addr)=⌈√addr⌉ is non-decreasing):

  addr 1         sA cache         (4,096 reads × cost 1 = 4,096)
  addr 2         tmp              (3,840 reads × cost 2 = 7,680)
  addrs 3..6     sB strip          (1,024 reads each × cost 2-3)
  addrs 7..38    sC accumulator    (120 reads each × cost 3-7)
  addrs 39..198  A aliased with C  (5 reads each × cost 7-15)
  addrs 199..294 A non-aliased     (4 reads each × cost 15-18)
  addrs 295..550 B input           (2 reads each × cost 18-24)
  addrs 551..646 C non-aliased     (1 read at exit × cost 24-26)

Total cost: 69,697.
"""
from __future__ import annotations


def generate_aliased_16x16() -> str:
    N, TI, TJ = 16, 8, 4
    nbi, nbj = N // TI, N // TJ
    assert nbi == 2

    # ── 1. Build the op list with symbolic cell labels (no addresses yet). ──
    label_id: dict = {}
    def lbl(key):
        i = label_id.get(key)
        if i is None:
            i = len(label_id)
            label_id[key] = i
        return i

    SA  = lbl("sA")
    TMP = lbl("tmp")
    sB  = lambda jj: lbl(("sB", jj))
    sC  = lambda ii, jj: lbl(("sC", ii, jj))
    A   = lambda i, k: lbl(("A", i, k))
    B   = lambda k, j: lbl(("B", k, j))
    C_bulk = lambda i, j: lbl(("Cbulk", i, j))

    def C(i, j):
        bi, bj = i // TI, j // TJ
        if bj == nbj - 1:
            return A(i, j - (nbj - 1) * TJ)        # alias onto A's first TJ cols
        if bi >= 1:
            return A((bi - 1) * TI + (i % TI), TJ + j)  # alias onto bi=0's dead A
        return C_bulk(i, j)                        # fresh bulk slot

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [B(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]

    ops: list = []
    for bi in range(nbi):
        for bj in range(nbj):
            for bk in range(N):
                for jj in range(TJ):
                    ops.append(("copy", (sB(jj), B(bk, bj * TJ + jj))))
                for ii in range(TI):
                    ops.append(("copy", (SA, A(bi * TI + ii, bk))))
                    for jj in range(TJ):
                        if bk == 0:
                            ops.append(("mul", (sC(ii, jj), SA, sB(jj))))
                        elif bk == N - 1:
                            ops.append(("mul", (TMP, SA, sB(jj))))
                            ops.append(("add",
                                (C(bi * TI + ii, bj * TJ + jj),
                                 sC(ii, jj), TMP)))
                        else:
                            ops.append(("mul", (TMP, SA, sB(jj))))
                            ops.append(("add", (sC(ii, jj), TMP)))

    # ── 2. Count reads per cell (writes are free). ──
    reads: dict = {}
    def bump(c, n=1):
        reads[c] = reads.get(c, 0) + n
    for op, cells in ops:
        if op == "copy":
            _, src = cells
            bump(src)
        elif op in ("add", "sub", "mul"):
            if len(cells) == 3:        # `op dest, s1, s2` reads s1 and s2
                _, s1, s2 = cells
                bump(s1); bump(s2)
            else:                      # `op dest, src` ≡ `op dest, dest, src`
                dest, src = cells
                bump(dest); bump(src)
    for o in outputs:
        bump(o)                        # exit reads

    # ── 3. Greedy address assignment: hottest cell → lowest addr. ──
    # Tie-break by label id (deterministic).
    sorted_cells = sorted(reads, key=lambda c: (-reads[c], c))
    addr = {c: i + 1 for i, c in enumerate(sorted_cells)}

    # ── 4. Emit IR. ──
    def fmt(cells):
        return ",".join(str(addr[c]) for c in cells)

    lines = [",".join(str(addr[c]) for c in inputs)]
    for op, cells in ops:
        lines.append(f"{op} {fmt(cells)}")
    lines.append(",".join(str(addr[c]) for c in outputs))
    return "\n".join(lines)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matmul import score_16x16  # noqa: E402

    ir = generate_aliased_16x16()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "aliased_16x16.ir")
    with open(out_path, "w") as f:
        f.write(ir + "\n")
    cost = score_16x16(ir)
    print(f"aliased_16x16.ir  cost={cost:,}")
    assert cost == 69_697, cost
