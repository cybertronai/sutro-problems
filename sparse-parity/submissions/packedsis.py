"""Packed SIS-walk generator for the mask32 sparse-parity bands.

Same algorithm as mp.generate_sis_mask (static information set, weight-
ordered null-space walk, weight-k capture) but with GF(2) bits packed
8-per-cell so row XORs cost 5 ops instead of 33, and with the walk state
kept as (w_S packed 3 bytes, coefficient vector packed 2 bytes):

    w = (w_S, a)   because the free-part of  s0 + sum a_j knob_j  is a.

The walk keeps the S-part of the current vector in ROW order (bit r = the
coordinate whose pivot row is r), which kills the pivot-order gather: s0
and the knob vectors read straight off the RREF table columns, and the
scatter to column order happens once at the end.
"""
from __future__ import annotations

import os
import sys
from itertools import combinations
from random import Random

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import mask_sparse_parity as mp  # noqa: E402


def _flip_schedule(G, cap, g2=None):
    """Visits: empty, all singles, then pairs.  If g2 is given, only pairs
    within the first g2 knobs (a cheap partial cap-2)."""
    sets = [()]
    for w in range(1, cap + 1):
        if w == 2 and g2 is not None:
            sets.extend(combinations(range(g2), 2))
        else:
            sets.extend(combinations(range(G), w))
    flips, prev = [], frozenset()
    for cur in sets:
        cur_s = frozenset(cur)
        flips.append((sorted(cur_s.symmetric_difference(prev)), len(cur)))
        prev = cur_s
    return flips


def _layout2(prefix, walk, inputs, outputs):
    """Two-phase staged layout (same idea as mp.stage_walk_layout but
    generic): frequency-sorted addressing per phase, walk re-addressed
    from 1 into the dead prefix range, copy bridge for live-ins."""

    def scan(lines):
        cnt, uni = {}, set()
        for l in lines:
            op, rest = l.split(" ", 1)
            if op == "set":
                uni.add(int(rest.split(",")[0]))
                continue
            if op == "cmp":
                rest = rest.rsplit(",", 1)[0]
            parts = [int(x) for x in rest.split(",")]
            uni.update(parts)
            for a in parts[1:]:
                cnt[a] = cnt.get(a, 0) + 1
        return cnt, uni

    pc, pu = scan(prefix)
    wc, wu = scan(walk)
    for a in outputs:
        wc[a] = wc.get(a, 0) + 1          # final read of each output cell

    pre_map = {a: i + 1 for i, a in
               enumerate(sorted(pu, key=lambda a: (-pc.get(a, 0), a)))}
    walk_order = sorted(wu, key=lambda a: (-wc.get(a, 0), a))
    walk_map = {a: i + 1 for i, a in enumerate(walk_order)}

    def remap(lines, m):
        out = []
        for l in lines:
            op, rest = l.split(" ", 1)
            if op == "set":
                d, v = rest.split(",")
                out.append(f"set {m[int(d)]},{v}")
                continue
            if op == "cmp":
                d, a, b, p = rest.split(",")
                out.append(f"cmp {m[int(d)]},{m[int(a)]},{m[int(b)]},{p}")
                continue
            parts = [int(x) for x in rest.split(",")]
            out.append(f"{op} {m[parts[0]]},"
                       + ",".join(str(m[x]) for x in parts[1:]))
        return out

    lines = [",".join(str(pre_map[a]) for a in inputs)]
    lines += remap(prefix, pre_map)
    # Bridge live-ins across the two address spaces.  Copies must not
    # clobber a prefix cell another copy still needs: emit greedily any
    # copy whose destination is not a pending source; break cycles with
    # one scratch cell above both address ranges.
    live = [a for a in walk_order if a in pu]
    pending = {a: (pre_map[a], walk_map[a]) for a in live}
    scratch = max(max(pre_map.values()), max(walk_map.values())) + 1
    bridge = []
    while pending:
        srcs = {s for s, _ in pending.values()}
        ready = [a for a, (s, d) in pending.items() if d not in srcs]
        if not ready:
            a = next(iter(pending))
            s, d = pending.pop(a)
            bridge.append(f"copy {scratch},{s}")
            pending[a] = (scratch, d)
            continue
        for a in ready:
            s, d = pending.pop(a)
            bridge.append(f"copy {d},{s}")
    lines += bridge
    lines += remap(walk, walk_map)
    lines.append(",".join(str(walk_map[a]) for a in outputs))
    return "\n".join(lines)


SECTIONS = []
def generate_packed_sis(cap=2, seed=0, n=32, m=18, k=5, layout=True, g2=None):
    G = n - m
    rng = Random(seed)
    # Cluster the free columns into cols 0..15 (harmless; any 14-subset is
    # equivalent under column symmetry) so different seeds stay diverse.
    F = sorted(rng.sample(range(2 * 8), G))
    S = sorted(set(range(n)) - set(F))
    Spos = {c: j for j, c in enumerate(S)}
    Fpos = {c: j for j, c in enumerate(F)}
    NB = n // 8          # 4 bytes per packed X row
    YB = NB              # y byte index in M row
    SB = (m + 7) // 8    # 3 bytes for 18 S-order bits
    AB = (G + 7) // 8    # 2 bytes for 14 F-order bits

    a_ = 1
    def alloc(sz=1):
        nonlocal a_
        base = a_; a_ += sz; return base

    # ---- constants ----
    ZERO = alloc(); ONE = alloc(); NEG = alloc()
    POW2 = [alloc() for _ in range(8)]
    M55 = alloc(); M33 = alloc(); M0F = alloc()
    ROW = [alloc() for _ in range(m)]
    KREM = {sz: alloc() for sz in range(0, cap + 1)}
    # ---- RREF state ----
    Mc = [[alloc() for _ in range(NB + 1)] for _ in range(m)]   # packed rows
    used = [alloc() for _ in range(m)]
    PIV = [alloc() for _ in range(m)]       # pivot row per S column
    PR = [alloc() for _ in range(NB + 1)]   # pivot row copy
    # ---- readoff ----
    S0B = [alloc() for _ in range(SB)]                          # s0 packed
    CB = [[alloc() for _ in range(SB)] for _ in range(G)]       # colvec per knob
    # ---- walk state ----
    WS = [alloc() for _ in range(SB)]
    A = [alloc() for _ in range(AB)]
    OUTS = [alloc() for _ in range(SB)]
    OUTA = [alloc() for _ in range(AB)]
    BITR = [alloc() for _ in range(m)]      # extracted OUTS bits (unscatter)
    OUT = [alloc() for _ in range(n)]
    # ---- temps ----
    (T, T2, BIT, NU, ELIG, ISF, ISM, ISO, DOX, MASK,
     P1, P2, P3, P4, P5, P6, ACC, OK) = (alloc() for _ in range(18))
    # ---- inputs (declared last in alloc, first in text) ----
    Xc = [[alloc() for _ in range(n)] for _ in range(m)]
    yc = [alloc() for _ in range(m)]

    inputs = [Xc[i][c] for i in range(m) for c in range(n)] + \
             [yc[i] for i in range(m)]
    outputs = OUT[:]

    pre, wk = [], []
    def pe(s): pre.append(s)
    def we(s): wk.append(s)
    SECTIONS.append(("const", len(pre)))

    # ================= prefix: constants, pack, RREF, readoff ==========
    pe(f"set {ZERO},0"); pe(f"set {ONE},1"); pe(f"set {NEG},-1")
    for p in range(8):
        pe(f"set {POW2[p]},{1 << p}")
    pe(f"set {M55},85"); pe(f"set {M33},51"); pe(f"set {M0F},15")
    for r in range(m):
        pe(f"set {ROW[r]},{r}")
    for sz, cell in KREM.items():
        pe(f"set {cell},{k - sz}")

    SECTIONS.append(("pack_start", len(pre)))
    # pack: M[i] = [X row i (4 bytes) | y_i (byte 4, bit 0)]
    for i in range(m):
        for b in range(NB):
            pe(f"copy {Mc[i][b]},{Xc[i][8 * b]}")
            for p in range(1, 8):
                pe(f"mul {T},{Xc[i][8 * b + p]},{POW2[p]}")
                pe(f"add {Mc[i][b]},{Mc[i][b]},{T}")
        pe(f"copy {Mc[i][YB]},{yc[i]}")

    for r in range(m):
        pe(f"set {used[r]},0")

    SECTIONS.append(("rref_start", len(pre)))
    # RREF over S columns, static column order.  Any eligible row works as
    # pivot, so the find keeps overwriting PIV/PR with each eligible row
    # (last one wins) -- no first-found tracking.  Rows are marked used
    # during elimination via the pivot-row match.
    for jj in range(m):
        c = S[jj]; cb, cp = c // 8, c % 8
        pe(f"set {PIV[jj]},{m}")
        for b in range(NB + 1):
            pe(f"set {PR[b]},0")
        for r in range(m):
            pe(f"and {T},{Mc[r][cb]},{POW2[cp]}")       # truthy iff bit
            pe(f"sub {NU},{used[r]},{ONE}")             # 0xFF iff free
            pe(f"and {ELIG},{T},{NU}")                  # truthy iff cand
            pe(f"select {PIV[jj]},{ELIG},{ROW[r]},{PIV[jj]}")
            for b in range(NB + 1):
                pe(f"select {PR[b]},{ELIG},{Mc[r][b]},{PR[b]}")
        for r in range(m):
            pe(f"and {T},{Mc[r][cb]},{POW2[cp]}")
            pe(f"cmp {ISM},{PIV[jj]},{ROW[r]},eq")
            pe(f"or {used[r]},{used[r]},{ISM}")         # mark pivot used
            pe(f"sub {ISO},{ISM},{ONE}")                # 0xFF iff other
            pe(f"and {DOX},{T},{ISO}")                  # truthy iff flip
            pe(f"select {MASK},{DOX},{NEG},{ZERO}")     # 0xFF or 0
            for b in range(NB + 1):
                pe(f"and {T2},{PR[b]},{MASK}")
                pe(f"xor {Mc[r][b]},{Mc[r][b]},{T2}")

    # No gather: the walk keeps the S-part in ROW order (bit r = the
    # coordinate whose pivot row is r).  Popcount is order-invariant, and
    # the scatter to column order happens once, after the walk.
    SECTIONS.append(("s0_start", len(pre)))
    # s0 packed in row order: bit r = y-byte of row r (always a clean 0/1)
    for b in range(SB):
        pe(f"set {S0B[b]},0")
    for r in range(m):
        pe(f"mul {T},{Mc[r][YB]},{POW2[r % 8]}")
        pe(f"add {S0B[r // 8]},{S0B[r // 8]},{T}")

    SECTIONS.append(("colvec_start", len(pre)))
    # knob column vectors in row order: CB[f] bit r = bit f of M row r
    for fi, f in enumerate(F):
        fb, fp = f // 8, f % 8
        for b in range(SB):
            pe(f"set {CB[fi][b]},0")
        for r in range(m):
            rp = r % 8
            if fp == 7:
                # POW2[7] reads as -128; normalize the bit first
                pe(f"and {T},{Mc[r][fb]},{POW2[fp]}")
                pe(f"cmp {T},{T},{ZERO},ne")
                if rp:
                    pe(f"mul {T},{T},{POW2[rp]}")
            else:
                pe(f"and {T},{Mc[r][fb]},{POW2[fp]}")
                if rp > fp:
                    pe(f"mul {T},{T},{POW2[rp - fp]}")
                elif rp < fp:
                    pe(f"div {T},{T},{POW2[fp - rp]}")
            pe(f"add {CB[fi][r // 8]},{CB[fi][r // 8]},{T}")

    SECTIONS.append(("prefix_end", len(pre)))
    # ================= walk ===========================================
    for b in range(SB):
        we(f"copy {WS[b]},{S0B[b]}")
    for b in range(AB):
        we(f"set {A[b]},0")
        we(f"set {OUTA[b]},0")
    for b in range(SB):
        we(f"set {OUTS[b]},0")

    def capture(sz):
        for b in range(SB):
            if b == SB - 1 and m % 8:
                # top byte holds only m % 8 bits; popcount of a <=2-bit
                # value is (x & 1) + (x >> 1)
                assert m % 8 == 2
                we(f"and {P2},{WS[b]},{ONE}")
                we(f"div {P3},{WS[b]},{POW2[1]}")
                we(f"add {P5},{P2},{P3}")
            else:
                we(f"div {P1},{WS[b]},{POW2[1]}")
                we(f"and {P1},{P1},{M55}")
                we(f"sub {P2},{WS[b]},{P1}")
                we(f"and {P3},{P2},{M33}")
                we(f"div {P4},{P2},{POW2[2]}")
                we(f"and {P4},{P4},{M33}")
                we(f"add {P5},{P3},{P4}")
                we(f"div {P6},{P5},{POW2[4]}")
                we(f"add {P5},{P5},{P6}")
                we(f"and {P5},{P5},{M0F}")               # popcount <= 8
            if b == 0:
                we(f"copy {ACC},{P5}")
            else:
                we(f"add {ACC},{ACC},{P5}")
        we(f"cmp {OK},{ACC},{KREM[sz]},eq")
        for b in range(SB):
            we(f"select {OUTS[b]},{OK},{WS[b]},{OUTS[b]}")
        for b in range(AB):
            we(f"select {OUTA[b]},{OK},{A[b]},{OUTA[b]}")

    capture(0)
    for flips, sz in _flip_schedule(G, cap, g2):
        for j in flips:
            for b in range(SB):
                we(f"xor {WS[b]},{WS[b]},{CB[j][b]}")
            we(f"xor {A[j // 8]},{A[j // 8]},{POW2[j % 8]}")
        capture(sz)

    # unscatter row-order outs to column order.
    # F columns: bit Fpos[c] of OUTA.  S columns: S column j lives in the
    # row r with PIV[j] == r: extract every OUTS row bit once, then an
    # 18-way select per column.
    for c in range(n):
        if c in Fpos:
            j = Fpos[c]
            we(f"and {T},{OUTA[j // 8]},{POW2[j % 8]}")
            we(f"cmp {OUT[c]},{T},{ZERO},ne")
    for r in range(m):
        we(f"and {BITR[r]},{OUTS[r // 8]},{POW2[r % 8]}")   # truthy = bit
    for c in range(n):
        if c not in Spos:
            continue
        j = Spos[c]
        we(f"set {OUT[c]},0")
        for r in range(m):
            we(f"cmp {ISM},{PIV[j]},{ROW[r]},eq")
            we(f"select {OUT[c]},{ISM},{BITR[r]},{OUT[c]}")
        we(f"cmp {OUT[c]},{OUT[c]},{ZERO},ne")

    if not layout:
        lines = [",".join(map(str, inputs))] + pre + wk + \
                [",".join(map(str, outputs))]
        return "\n".join(lines)
    return _layout2(pre, wk, inputs, outputs)


if __name__ == "__main__":
    import sys, time
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    g2 = int(sys.argv[3]) if len(sys.argv) > 3 else None
    t = time.time()
    ir = generate_packed_sis(cap=cap, seed=seed, g2=g2)
    res = mp.evaluate_mask(ir)
    print(f"packed_sis cap={cap} seed={seed} g2={g2}: cost={res.cost} "
          f"recovery={res.recovery:.4f} lines={ir.count(chr(10)) + 1} "
          f"({time.time() - t:.1f}s)")
