"""Compute renamer-optimal cost for hypothetical layouts.

Compare:
  Layout A (current 69,697): 1, 1, 4, 32, 64, 192, 256, 96 cells with reads
                              4096, 3840, 1024, 120, 5, 4, 2, 1.
  Layout B (alias C[bi=0,non-last-bj] onto B): drop 96×1 bulk-C, add 96×3 B+C.
  Other variants too.
"""
from __future__ import annotations

import math


def renamer_cost(groups):
    """groups = [(count, reads_per_cell), ...]. Returns total cost.

    Cells with higher reads/cell get cheaper addrs. Sort descending by reads.
    """
    groups_sorted = sorted(groups, key=lambda g: -g[1])
    total = 0
    addr = 1
    for count, reads in groups_sorted:
        for _ in range(count):
            cost = math.isqrt(addr - 1) + 1
            total += reads * cost
            addr += 1
    return total


def fmt_layout(name, groups):
    cost = renamer_cost(groups)
    total_reads = sum(c * r for c, r in groups)
    total_cells = sum(c for c, _ in groups)
    print(f"  {name}:  cells={total_cells:4d}  reads={total_reads:7,}  cost={cost:7,}")
    return cost


if __name__ == "__main__":
    print("Hypothetical layouts (renamer cost):")
    print()

    # Layout A: current 69,697 (TI=8, TJ=4, alias-chain finaladd).
    # Actual read counts: 1×4096, 1×3840, 4×1024, 32×120, 160×5, 96×4, 256×2, 96×1.
    A = [(1, 4096), (1, 3840), (4, 1024), (32, 120),
         (160, 5), (96, 4), (256, 2), (96, 1)]
    fmt_layout("A: current 69,697 (8x4)", A)

    # Layout B: alias C[bi=0,non-last-bj] (96 cells × 1 read) onto B (256 cells
    # × 2 reads). Result: 96 B-cells become B+C (3 reads each), 160 stay B (2 reads).
    # Eliminate the 96×1 bulk-C cells from layout.
    B = [(1, 4096), (1, 3840), (4, 1024), (32, 120),
         (160, 5), (96, 4), (96, 3), (160, 2)]
    fmt_layout("B: alias bi=0 non-last-bj onto B (no extra cost)", B)

    # Layout B2: same but with frozen scratch overhead.
    # If we buffer 96 cells in frozen-sC, each frozen cell has 1 read (write-once,
    # read-once for the final copy-to-B-addr). 96 extra cells with 1 read.
    # But we ALSO eliminate the 96 bulk-C cells from layout (replaced by
    # frozen-sC + B+C alias). Hmm, let's think:
    # - 96 B+C cells (3 reads): B(2) + C-output(1).
    # - 96 frozen-sC cells (1 read each): write at bk=N-1 finaladd-style... wait
    #   we can't do finaladd into frozen cells AND have them be read once for
    #   copy-to-B-addr. Actually frozen cell read ONCE (the final copy), plus
    #   the C-output read happens at the B-addr (aliased). But we need to
    #   COPY frozen->B-addr, which is 1 read of frozen + 1 write of B-addr.
    # - The C-output read at B-addr is aliased.
    # Wait, that's still adding 96 cells. So total cells go up by 96.
    B2 = [(1, 4096), (1, 3840), (4, 1024), (32, 120),
          (64, 5), (192, 4), (96, 3), (160, 2), (96, 1)]
    fmt_layout("B2: B-alias + 96 frozen-sC cells", B2)

    # Layout C: a more aggressive variant - increase A=C alias scope.
    # E.g., 128 cells aliased (5 reads each) instead of 64.
    C = [(1, 4096), (1, 3840), (4, 1024), (32, 120),
         (128, 5), (128, 4), (256, 2), (96, 1)]
    fmt_layout("C: 128 A=C aliases (hypothetical)", C)

    # Layout D: ultimate alias - all 256 C onto A.
    # Total: 1+1+4+32+256(A=C, 5 reads)+256(B, 2 reads)
    # But all A cells now have 4 (A reads) + 1 (C output) = 5 reads each.
    D = [(1, 4096), (1, 3840), (4, 1024), (32, 120),
         (256, 5), (256, 2)]
    fmt_layout("D: full A=C alias, no bulk-C, no frozen", D)

    # Layout E: same as A but sC moved to addr 1 region (replace SA).
    # SA could be eliminated if we read A directly... but it's already 4096 reads,
    # so it's at addr 1 (cost 1). No improvement possible there.
    # Wait, what if we eliminate TMP by using 3-arg add directly?
    # Each mul-add cycle becomes: mul tmp, A, B; add sC, sC, tmp.
    # That's still 2 ops, 2 reads of tmp/sC per cycle. Same.
    # We can't directly avoid TMP without extra cells.

    # Layout F: outer-product with full sB row.
    # sB=N cells (16), each read TI×nbi times per k = TI×N per k = 16×16 = 256
    # over all k. Wait recount: for each k (16 k's), for each i (16 i's),
    # each sB[j] is read once. So sB[j] reads = N×N = 256 per cell. 16 cells × 256 = 4096.
    # SA = 4096 (1 cell).
    # TMP = 3840 (1 cell).
    # sC = full 256 (one per C[i,j]). Each read 15 times (k=1..15) + final = 15-16. Then 1 more for exit.
    #   Actually if we use bulk-C as the accumulation cell directly (no separate sC):
    #   each C cell is read 15 times. C lives at low addrs 19+. Plus 1 exit read.
    F = [(1, 4096), (1, 3840), (16, 256), (256, 16), (256, 1)]
    fmt_layout("F: outer-product full sB, C bulk accum", F)

    # Layout TI4TJ4: TI=4, TJ=4, finaladd, alias-chain.
    # SA=4096, TMP=3840, sB(4)=1024, sC(16)=240, A=C(208)=5, strandedA(48)=4,
    # B(256)=4, bulk-C(48)=1.
    Ti4Tj4 = [(1, 4096), (1, 3840), (4, 1024), (16, 240),
              (208, 5), (48 + 256, 4), (48, 1)]
    fmt_layout("TI=4,TJ=4 alias finaladd", Ti4Tj4)

    # Layout TI4TJ4 + B-alias (bi=0 non-last to B):
    # 48 cells alias C[bi=0, non-last] onto B (4 reads B + 1 C-out = 5).
    # Remaining B: 256-48 = 208 cells × 4 reads.
    # No bulk-C left.
    Ti4Tj4_B = [(1, 4096), (1, 3840), (4, 1024), (16, 240),
                (208, 5), (48 + 0, 4), (48, 5), (208, 4)]
    fmt_layout("TI=4,TJ=4 + B-alias", Ti4Tj4_B)

    # Layout D2: full A=C alias + 96 frozen cells.
    # 256 cells × 5 reads (full A=C alias) + 256 B × 2 + 96 frozen × 1.
    D2 = [(1, 4096), (1, 3840), (4, 1024), (32, 120),
          (256, 5), (256, 2), (96, 1)]
    fmt_layout("D2: full A=C alias + 96 frozen", D2)

    # Layout D3: full A=C alias + 32 frozen (recycled buffer).
    # If we can do bj-by-bj processing reusing 32 frozen cells, we save cells.
    # But each frozen cell is read multiple times then? Hmm...
    # Actually if 32 frozen cells are reused, they're written 3 times each (3
    # bj's) and read 3 times each (one read per bj for copy out). So 32×3 = 96 reads.
    # Total reads same as 96×1 = 96. Just fewer cells.
    D3 = [(1, 4096), (1, 3840), (4, 1024), (32, 120),
          (256, 5), (256, 2), (32, 3)]
    fmt_layout("D3: full A=C alias + 32 recycled-frozen (3 reads each)", D3)

    # Layout E_full: full A=C alias with 96 hold cells for deferred writeback.
    # Active sC: 32 cells × 123 reads/cell. Hold: 96 × 1.
    # SA=4096, TMP=3840, sB=1024×4, sC=123×32, A=C=5×256, B=2×256, hold=1×96.
    Efull = [(1, 4096), (1, 3840), (4, 1024), (32, 123),
             (256, 5), (256, 2), (96, 1)]
    fmt_layout("E_full: full A=C alias + 96 hold cells", Efull)

    # Layout E_partial: same idea but partial — alias just 32 of the 96 stranded
    # to save sC reads. Actually we don't have a partial version at smaller cost...

    # Hmm, let's think: hold cells are 96 × 1 read. They effectively replace the
    # 96 bulk-C cells in layout A (also 96 × 1). But the active sC went from 120
    # reads/cell to 123 reads/cell (96 extra reads for hold-copy).
    # Plus C-output reads: in layout A, 96 C-output reads on bulk-C addrs (cost ~25).
    # In E_full, 96 C-output reads on aliased A addrs (already counted in 5 reads/cell).
    # The 96 cells that became aliased had NO bulk-C cost before (they ARE A cells in
    # layout A). They had 4 reads (A only). Now have 5 reads (A + C-out).
    # So: 96 cells get +1 read each (4→5). Active sC gets +96/32=+3 reads/cell.
    # But 96 bulk-C cells (1 read each) replaced by 96 hold cells (1 read each). Same.

    # Layout G: outer-product with sB cache + alias C onto A.
    # Same idea as F but C aliased onto A (256 cells with 4 input + 16 accum + 1 output reads = 21).
    # A is read once per (i, k) = 1 read. So A = 1 read each (256 cells).
    # sC accumulation: each C[i,j] accumulated 15 times. With C aliased to A,
    # C address has 1 (A read) + 15 (accum) + 1 (output) = 17 reads. But A is read FIRST, then
    # C accum overwrites? Need to check.
    G = [(1, 4096), (1, 3840), (16, 256), (256, 17)]
    fmt_layout("G: outer-product alias", G)
