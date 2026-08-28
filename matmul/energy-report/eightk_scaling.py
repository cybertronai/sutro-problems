"""Reproduce the Sutro 8192^3 movement-score extrapolation.

This script uses the challenge's cost per source read,
``ceil(sqrt(address))``, and the literal address layout of the generalized
``sa_cache`` schedule.  It does not model arithmetic, writes, time, memory
capacity, HBM, control, clocking, or leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt


def shell_cost(address: int) -> int:
    """Return ceil(sqrt(address)) for a positive integer address."""
    if address < 1:
        raise ValueError("addresses start at 1")
    return isqrt(address - 1) + 1


def shell_prefix(last_address: int) -> int:
    """Return sum(ceil(sqrt(a)) for a in 1..last_address) in O(1)."""
    if last_address <= 0:
        return 0
    q = isqrt(last_address)
    return (
        q * (q + 1) * (4 * q - 1) // 6
        + (last_address - q * q) * (q + 1)
    )


def shell_sum(first_address: int, last_address: int) -> int:
    return shell_prefix(last_address) - shell_prefix(first_address - 1)


@dataclass(frozen=True)
class ScoreBreakdown:
    a_loads: int
    b_loads: int
    c_exit: int
    sa_reads: int
    sb_reads: int
    tmp_reads: int
    sc_reads: int

    @property
    def total(self) -> int:
        return sum(vars(self).values())


def sa_cache_score(n: int, tile_i: int, tile_j: int) -> ScoreBreakdown:
    """Exact score of the literal generalized sa_cache address layout."""
    if n % tile_i or n % tile_j:
        raise ValueError("tile dimensions must divide n")

    sa = 1
    tmp = 2
    sb_first, sb_last = 3, 2 + tile_j
    sc_first = sb_last + 1
    sc_last = sc_first + tile_i * tile_j - 1
    a_first, a_last = sc_last + 1, sc_last + n * n
    b_first, b_last = a_last + 1, a_last + n * n
    c_first, c_last = b_last + 1, b_last + n * n

    block_count = (n // tile_i) * (n // tile_j)
    return ScoreBreakdown(
        a_loads=(n // tile_j) * shell_sum(a_first, a_last),
        b_loads=(n // tile_i) * shell_sum(b_first, b_last),
        c_exit=shell_sum(c_first, c_last),
        sa_reads=n**3 * shell_cost(sa),
        sb_reads=block_count * n * tile_i * shell_sum(sb_first, sb_last),
        tmp_reads=n * n * (n - 1) * shell_cost(tmp),
        sc_reads=block_count * n * shell_sum(sc_first, sc_last),
    )


def baseline_score(n: int) -> int:
    """Exact score of the repository's naive square-matmul layout."""
    matrix_cells = n * n
    tmp = 3 * matrix_cells + 1
    return (
        n * shell_sum(1, matrix_cells)
        + n * shell_sum(matrix_cells + 1, 2 * matrix_cells)
        + n * shell_sum(2 * matrix_cells + 1, 3 * matrix_cells)
        + matrix_cells * (n - 1) * shell_cost(tmp)
    )


def divisors(n: int) -> list[int]:
    return [value for value in range(1, n + 1) if n % value == 0]


def main() -> None:
    # Small-size validation against checked-in scored IR.
    assert baseline_score(16) == 340_704
    assert sa_cache_score(16, 8, 4).total == 73_602

    n = 8192
    candidates = (
        (sa_cache_score(n, ti, tj).total, ti, tj)
        for ti in divisors(n)
        for tj in divisors(n)
    )
    score, tile_i, tile_j = min(candidates)
    breakdown = sa_cache_score(n, tile_i, tile_j)

    print(f"n={n:,}; MACs={n**3:,}; conventional ops={2*n**3:,}")
    print(f"best sa_cache tile: Ti={tile_i}, Tj={tile_j}")
    print(f"score: {score:,}; energy at 1 fJ/unit: {score * 1e-15:.15f} J")
    paid_reads = (
        3 * n**3                  # SA, sB, and sC reads
        + n * n * (n - 1)        # TMP reads
        + n**3 // tile_j          # bulk A loads
        + n**3 // tile_i          # bulk B loads
        + n * n                   # output exit reads
    )
    print(f"paid reads: {paid_reads:,}")
    print("breakdown (score units):")
    for name, value in vars(breakdown).items():
        print(f"  {name:10s} {value:>20,}  {value * 1e-15:.12f} J")
    naive = baseline_score(n)
    print(f"naive: {naive:,}; {naive * 1e-15:.12f} J")
    print(f"improvement: {naive / score:.9f}x")


if __name__ == "__main__":
    main()
