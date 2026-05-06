"""Closed-form cost calculator for parameterized matmul schedules.

Each schedule decomposes into a list of memory regions, each described by
``(name, n_cells, reads_per_cell)``. The total program cost is the sum of
``reads * shell_cost(addr)`` over every read of every cell, where addresses
are assigned greedily: regions sorted by descending reads-per-cell, packed
contiguously starting at addr 1. (This greedy packing is optimal because
shell_cost is monotone non-decreasing in addr.)

To validate the model, ``schedule_asym_outer_i(16, 4, 4)`` reproduces
hierarchical_16x16's cost of 80,217 to the byte.

Run as a script to sweep tile sizes and report the best schedule.
"""
from __future__ import annotations
import math
from typing import List, Tuple, Optional

Region = Tuple[str, int, int]  # (name, n_cells, reads_per_cell)


def shell_cost(addr: int) -> int:
    return math.isqrt(addr - 1) + 1


def shell_sum(a: int, b: int) -> int:
    """Σ_{addr=a..b} ceil(sqrt(addr)). Closed-form via shell counting."""
    if a > b:
        return 0
    # cost(addr) = k for addr in [(k-1)^2 + 1, k^2]. Walk shells.
    total = 0
    addr = a
    while addr <= b:
        k = shell_cost(addr)
        shell_end = k * k  # last addr in shell k
        end = min(b, shell_end)
        total += k * (end - addr + 1)
        addr = end + 1
    return total


def pack_cost(regions: List[Region]) -> Tuple[int, list]:
    """Greedy address assignment: highest reads/cell at lowest addrs."""
    nonzero = [r for r in regions if r[1] > 0]
    sorted_r = sorted(nonzero, key=lambda r: (-r[2], r[0]))
    addr = 1
    total = 0
    breakdown = []
    for name, ncells, reads in sorted_r:
        a, b = addr, addr + ncells - 1
        s = shell_sum(a, b)
        c = reads * s
        breakdown.append((name, a, b, ncells, reads, s, c))
        total += c
        addr = b + 1
    return total, breakdown


# ---------------------------------------------------------------------------
# Schedule families. Each returns a region list or None if shape invalid.
# ---------------------------------------------------------------------------

def schedule_asym_outer_i(n: int, Ti: int, Tj: int) -> Optional[List[Region]]:
    """Outer over i-blocks of Ti rows. For each (ib, k): load Ti A-cells →
    sA. For each jb: load Tj B-cells → sB, do Ti*Tj muls accumulating into
    sC[jb, ii, jj]. sC has Ti*n cells, reused across all ib (rewritten by
    k=0 init mul). Matches hierarchical_16x16 when Ti=Tj=4.

    Reads/cell:
      TMP=1 cell, n^3-n^2 reads (one per non-init add)
      sA=Ti cells, n^3/Ti reads each
      sB=Tj cells, n^3/Tj reads each
      sC=Ti*n cells, n^2/Ti reads each (across n_ib re-uses)
      B_bulk: n/Ti reloads per cell (once per ib)
      A_bulk: 1 (each cell loaded once into sA over the program)
      C_bulk: 1 (output read at exit)
    """
    if n % Ti or n % Tj:
        return None
    return [
        ('TMP',    1,        n**3 - n**2),
        ('sA',     Ti,       n**3 // Ti),
        ('sB',     Tj,       n**3 // Tj),
        ('sC',     Ti * n,   n * n // Ti),
        ('B_bulk', n * n,    n // Ti),
        ('A_bulk', n * n,    1),
        ('C_bulk', n * n,    1),
    ]


def schedule_asym_outer_j(n: int, Ti: int, Tj: int) -> Optional[List[Region]]:
    """Mirror of outer_i: A is reloaded n_jb times instead of B. Equivalent
    cost to outer_i with (Ti, Tj) swapped, kept here for clarity."""
    if n % Ti or n % Tj:
        return None
    return [
        ('TMP',    1,        n**3 - n**2),
        ('sA',     Ti,       n**3 // Ti),
        ('sB',     Tj,       n**3 // Tj),
        ('sC',     Tj * n,   n * n // Tj),
        ('A_bulk', n * n,    n // Tj),
        ('B_bulk', n * n,    1),
        ('C_bulk', n * n,    1),
    ]


def schedule_hold_A_block(n: int, Ti: int, Tj: int) -> Optional[List[Region]]:
    """Outer over ib. At ib start, preload the full ib-block of A (Ti rows
    × n cols = Ti*n cells) into scratchpad. Each A scratch cell is read
    once per (jb) at the matching k = n_jb times. So sA cells: Ti*n cells,
    each read n_jb*Tj = n times — but n_jb*Tj = n always, so reads/cell =
    n. Total sA reads = Ti*n*n = Ti*n^2. Spread over Ti*n cells → 16/cell."""
    if n % Ti or n % Tj:
        return None
    return [
        ('TMP',    1,        n**3 - n**2),
        ('sA',     Ti * n,   n),
        ('sB',     Tj,       n**3 // Tj),
        ('sC',     Ti * n,   n * n // Ti),
        ('B_bulk', n * n,    n // Ti),
        ('A_bulk', n * n,    1),
        ('C_bulk', n * n,    1),
    ]


def schedule_outer_product(n: int) -> List[Region]:
    """Pure outer-product over k. sA = column k of A (n cells), sB = row k
    of B (n cells), sC = full C (n^2 cells). Each k: load sA (n cells from
    A_bulk), load sB (n cells from B_bulk), do n^2 muls into sC."""
    return [
        ('TMP',    1,        n**3 - n**2),
        ('sA',     n,        n * n),     # per k, sA(ii) read n times; n k's → n^2
        ('sB',     n,        n * n),
        ('sC',     n * n,    n),         # n-1 adds + 1 final copy = n
        ('A_bulk', n * n,    1),
        ('B_bulk', n * n,    1),
        ('C_bulk', n * n,    1),
    ]


def schedule_decoupled(n: int, Tio: int, Tjo: int,
                       Tii: int, Tji: int) -> Optional[List[Region]]:
    """Decoupled 2-level: outer super-block (Tio × Tjo), inner sub-tile
    (Tii × Tji) within. Generalizes schedule_2level_singleB (Tii=Tio,
    Tji=1). Loop nest is (bi_o, bj_o) outermost, then k, then inner sub-
    tiles (ib_in over Tio/Tii, jb_in over Tjo/Tji), then ii × jj inner.

    Reuse pattern: sA = Tii cells (one inner-i row block, reloaded per
    (bi_o, bj_o, k, ib_in)). sB = Tji cells (reloaded per (bi_o, bj_o, k,
    ib_in, jb_in)). sC = Tio*Tjo cells (super-block, reused across outer
    iters).

    A_bulk reloads = n / Tjo (each A cell loaded once per outer-j block).
    B_bulk reloads = n / Tii (each B cell loaded once per inner-i block,
    summed across all outer iters).
    """
    if (n % Tio or n % Tjo or Tio % Tii or Tjo % Tji
            or Tii < 1 or Tji < 1):
        return None
    return [
        ('TMP',    1,           n**3 - n**2),
        ('sA',     Tii,         n**3 // Tii),
        ('sB',     Tji,         n**3 // Tji),
        ('sC',     Tio * Tjo,   n**3 // (Tio * Tjo)),
        ('A_bulk', n * n,       n // Tjo),
        ('B_bulk', n * n,       n // Tii),
        ('C_bulk', n * n,       1),
    ]


def schedule_2level_singleB(n: int, Tio: int, Tjo: int) -> Optional[List[Region]]:
    """2-level hierarchy: outer iterates super-blocks of size Tio × Tjo over
    C. Inside each super-block, inner asymmetric matmul holds Tio rows of A
    in scratchpad (sA) and uses a single sB cell at addr 1 (Tj_inner=1).
    sC sized to the super-block (Tio*Tjo cells, reused across outer iters).

    Reduces to asym_outer_i(n, Tio, 1) when Tjo = n (no outer-j blocking).
    Reproduces the reported 73,602 cost at (Tio=4, Tjo=8): smaller sC
    region (32 cells) wins over the (4,1) family's larger 64-cell sC,
    despite paying 2 A-reloads instead of 1.
    """
    if n % Tio or n % Tjo:
        return None
    n_ibo = n // Tio
    n_jbo = n // Tjo
    return [
        ('TMP',    1,           n**3 - n**2),
        ('sA',     Tio,         n**3 // Tio),
        ('sB',     1,           n**3),
        ('sC',     Tio * Tjo,   16 * n_ibo * n_jbo),  # 16 reads/cell per outer iter
        ('A_bulk', n * n,       n_jbo),               # reloaded once per outer-j block
        ('B_bulk', n * n,       n_ibo),               # reloaded once per outer-i block
        ('C_bulk', n * n,       1),
    ]


def schedule_full_C_in_scratch(n: int, Tj: int) -> Optional[List[Region]]:
    """Like outer-product but partial: sB is Tj cells (not full row), so
    the j-dim is blocked while k stays outer. sC is still full C. Loads B
    once per (k, jb) — one full B traversal."""
    if n % Tj:
        return None
    return [
        ('TMP',    1,        n**3 - n**2),
        ('sA',     1,        n**3),  # single A cell pinned per (k, ii)
        ('sB',     Tj,       n**3 // Tj),
        ('sC',     n * n,    n),
        ('A_bulk', n * n,    1),
        ('B_bulk', n * n,    1),
        ('C_bulk', n * n,    1),
    ]


def main():
    n = 16
    divs = [1, 2, 4, 8, 16]
    rows = []

    print("=" * 72)
    print(f"asymmetric outer-i, n={n}")
    print(f"{'Ti':>3} {'Tj':>3}  {'cost':>10}")
    for Ti in divs:
        for Tj in divs:
            r = schedule_asym_outer_i(n, Ti, Tj)
            if r is None:
                continue
            c, _ = pack_cost(r)
            rows.append(('asym_outer_i', Ti, Tj, c))
            print(f"{Ti:>3} {Tj:>3}  {c:>10,}")

    print("=" * 72)
    print(f"hold-A-block (preload Ti*n A cells per ib), n={n}")
    print(f"{'Ti':>3} {'Tj':>3}  {'cost':>10}")
    for Ti in divs:
        for Tj in divs:
            r = schedule_hold_A_block(n, Ti, Tj)
            if r is None:
                continue
            c, _ = pack_cost(r)
            rows.append(('hold_A_block', Ti, Tj, c))
            print(f"{Ti:>3} {Tj:>3}  {c:>10,}")

    print("=" * 72)
    print(f"2-level single-sB (outer Tio × Tjo super-block, inner Tj=1), n={n}")
    print(f"{'Tio':>3} {'Tjo':>3}  {'cost':>10}")
    for Tio in divs:
        for Tjo in divs:
            r = schedule_2level_singleB(n, Tio, Tjo)
            if r is None:
                continue
            c, _ = pack_cost(r)
            rows.append(('2level_singleB', Tio, Tjo, c))
            print(f"{Tio:>3} {Tjo:>3}  {c:>10,}")

    print("=" * 72)
    print("outer-product (no blocking)")
    c, _ = pack_cost(schedule_outer_product(n))
    print(f"  cost = {c:,}")
    rows.append(('outer_product', n, n, c))

    print("=" * 72)
    print("full-C-in-scratch (k outer, sA single cell, sB Tj cells)")
    for Tj in divs:
        r = schedule_full_C_in_scratch(n, Tj)
        if r is None:
            continue
        c, _ = pack_cost(r)
        rows.append(('full_C_in_scratch', n, Tj, c))
        print(f"  Tj={Tj:>2}  cost = {c:,}")

    print("=" * 72)
    print("decoupled 2-level (Tio, Tjo, Tii, Tji), top 15 by cost")
    decoupled = []
    for Tio in divs:
        for Tjo in divs:
            for Tii in divs:
                for Tji in divs:
                    r = schedule_decoupled(n, Tio, Tjo, Tii, Tji)
                    if r is None:
                        continue
                    c, _ = pack_cost(r)
                    decoupled.append((Tio, Tjo, Tii, Tji, c))
    decoupled.sort(key=lambda x: x[4])
    print(f"{'Tio':>3} {'Tjo':>3} {'Tii':>3} {'Tji':>3}  {'cost':>10}")
    for Tio, Tjo, Tii, Tji, c in decoupled[:15]:
        rows.append((f'decoupled', f'{Tio}×{Tjo}', f'{Tii}×{Tji}', c))
        print(f"{Tio:>3} {Tjo:>3} {Tii:>3} {Tji:>3}  {c:>10,}")

    rows.sort(key=lambda r: r[3])
    print("=" * 72)
    print("Top 10:")
    print(f"{'family':<22} {'Ti':>3} {'Tj':>3}  {'cost':>10}")
    for fam, Ti, Tj, c in rows[:10]:
        print(f"{fam:<22} {Ti:>3} {Tj:>3}  {c:>10,}")

    # Validate: hierarchical_16x16 = asym_outer_i(16, 4, 4) = 80,217
    c44, bd = pack_cost(schedule_asym_outer_i(16, 4, 4))
    assert c44 == 80_217, f"validation failed: got {c44}, expected 80,217"
    print("=" * 72)
    print(f"validation: asym_outer_i(16,4,4) = {c44:,}  (matches hierarchical_16x16 ✓)")
    print("breakdown:")
    for name, a, b, nc, reads, s, cost in bd:
        print(f"  {name:<8} addrs {a:>4}..{b:<4}  {nc:>4} cells × {reads:>5} reads × Σcost={s:>5}  = {cost:>8,}")


if __name__ == '__main__':
    main()
