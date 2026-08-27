"""Sanity check: what's the best 8x8 sa-cache cost?"""
from __future__ import annotations

import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def _cost(addr):
    return math.isqrt(addr - 1) + 1


def estimate_sa_cache_cost(N, TI, TJ):
    """Estimate the renamed-optimal cost of an N×N sa-cache.

    Layout (read-count-descending):
      addr 1: SA scratch — read TI*TJ*N*nbi*nbj = N**3 reads
      addr 2: TMP — read TI*TJ*(N-1)*nbi*nbj = (N-1)*N**2 reads (-N**2)
      addrs 3..(2+TJ): sB — TJ cells × TI*N*nbi reads each = TI*N*nbi/TJ wait...

    Actually let me derive more carefully.

    Loop: for bi in nbi, bj in nbj, bk in N:
      for jj in TJ: copy sB[jj], B[bk, bj*TJ+jj]   (TJ B reads per (bi, bj, bk))
      for ii in TI:
        copy SA, A[bi*TI+ii, bk]                    (1 A read per (bi,bj,bk,ii))
        for jj in TJ:
          if bk==0: mul sC[ii,jj], SA, sB[jj]      (1 SA + 1 sB read)
          else:    mul tmp, SA, sB[jj]; add sC[ii,jj], tmp  (1 SA + 1 sB + 1 tmp + 1 sC reads)
      after bk loop: copy C[..],sC[..]              (1 sC read per (ii,jj) per (bi,bj))

    Reads per cell type:
      SA: 1 cell, total reads = TI*TJ*N*nbi*nbj = N**3 reads
      TMP: 1 cell, total reads = TI*TJ*(N-1)*nbi*nbj = (N-1)*N**2
      sB: TJ cells, each read TI*N*nbi*nbj/TJ... no per cell:
          sB[jj] is read in mul once per (bi, bj, bk, ii) = TI*N*nbi*nbj reads.
      sC: TI*TJ cells, each accumulated (N-1) times from add(sC, tmp), plus 1 flush = N reads.
          But across nbi*nbj outer iterations, sC is reused, so per cell:
          (N-1)*nbi*nbj add reads + nbi*nbj flush reads = N*nbi*nbj reads.
      A bulk: N*N cells, each read by 'copy SA' once per (bj iteration covering its bi).
              For A[i, k]: read at (bi, bj, bk=k, ii=i-bi*TI).  Across all bj: nbj reads.
      B bulk: N*N cells, each read at (bi, bj, bk=k, jj=j-bj*TJ).  Across all bi: nbi reads.
      C bulk: N*N cells, each read once at exit.

    Total cells: 1 + 1 + TJ + TI*TJ + N*N (A) + N*N (B) + N*N (C).
    """
    nbi = N // TI
    nbj = N // TJ
    if N % TI or N % TJ:
        return None
    cells_reads = []
    cells_reads.append(("SA", 1, N**3))
    cells_reads.append(("TMP", 1, (N-1) * N**2))
    cells_reads.append(("sB", TJ, TI * N * nbi * nbj))
    cells_reads.append(("sC", TI * TJ, N * nbi * nbj))
    cells_reads.append(("A", N*N, nbj))
    cells_reads.append(("B", N*N, nbi))
    cells_reads.append(("C", N*N, 1))

    # Build sorted list of (reads_per_cell, count) and assign greedy addresses.
    expanded = []
    for name, count, rpc in cells_reads:
        for _ in range(count):
            expanded.append(rpc)
    expanded.sort(reverse=True)
    total = 0
    for addr, rpc in enumerate(expanded, start=1):
        total += _cost(addr) * rpc
    return total


print("N, TI, TJ -> est_cost")
for N in [16]:
    for TI in [1, 2, 4, 8, 16]:
        for TJ in [1, 2, 4, 8, 16]:
            c = estimate_sa_cache_cost(N, TI, TJ)
            if c is not None:
                print(f"  N={N} TI={TI} TJ={TJ}: {c:,}")
