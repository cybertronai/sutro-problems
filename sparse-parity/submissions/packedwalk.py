"""packedwalk: packed-bit static information-set walk for the mask32 bands.

Same algorithm as mask_sparse_parity.generate_sis_mask (static info set,
branchless GF(2) Gauss-Jordan on [X_S | y | X_F], coefficient-weight-ordered
null-space walk, weight==k capture), recompiled onto BIT-PACKED rows:

  * each 33-bit augmented row lives in 5 cells (8 bits per cell);
  * pivot-row extraction is fused into the pivot search as packed
    select-chains (5 chains instead of 33 scalar ones);
  * elimination is 5 (and+xor) pairs per row instead of 33, trimmed to the
    not-yet-pivoted cell range;
  * the walk state w, the captured output and every knob vector are packed
    into 4 cells, so a basis flip costs 4 xors and a capture costs a SWAR
    popcount (nibble-combined across the 4 cells) + cmp + 4 selects;
  * outputs are unpacked once at the end.

The GF(2) math is operation-for-operation equivalent to the reference, so
recovery is bit-identical for the same info set / cap / seed; only the energy
changes.  Layout is a generic multi-phase frequency renumber with
clobber-safe bridge copies (phases: load+RREF | readoff | walk+unpack).

numpy-free generator; scoring via mask_sparse_parity.
"""
from __future__ import annotations

import math
import os
import sys
from itertools import combinations
from random import Random
from typing import Dict, List, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import mask_sparse_parity as mp  # noqa: E402

MASK55 = 85   # 0x55
MASK33 = 51   # 0x33
MASK0F = 15   # 0x0F


# ---------------------------------------------------------------------------
# generic staged layout: per-phase frequency-sorted addressing
# ---------------------------------------------------------------------------

def _split_op(l: str):
    parts = l.split(" ", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def renumber_phases(ir: str, split_idxs: Sequence[int]) -> str:
    """Give each phase its own frequency-optimal addressing from addr 1.

    ``split_idxs`` are body-line indices where new phases begin.  Cells whose
    value crosses a boundary are preserved by bridge copies emitted at the
    top of the consuming phase, ordered so no bridge copy clobbers the source
    of another (cycles broken through one scratch slot).  Cells whose first
    access in a phase is a pure write need no bridge.
    """
    lines = ir.splitlines()
    header, body, footer = lines[0], lines[1:-1], lines[-1]
    bounds = [0] + sorted(split_idxs) + [len(body)]
    phases = [body[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]

    def reads_of(l: str) -> List[int]:
        op, rest = _split_op(l)
        if op == "set":
            return []
        if rest.endswith(",eq") or rest.endswith(",ne"):
            rest = rest[:-3]
        args = [int(x) for x in rest.split(",")]
        if op in ("copy", "not", "abs"):
            return [args[1]]
        if op in ("select", "cmp"):
            return args[1:]
        if len(args) == 2:                      # in-place binary: dest is read
            return args
        return args[1:]

    def dest_of(l: str) -> int:
        return int(_split_op(l)[1].split(",")[0])

    out: List[str] = [""]
    last: Dict[int, int] = {}
    for p, seg in enumerate(phases):
        cnt: Dict[int, int] = {}
        uni: set = set()
        first_read: set = set()
        seen: set = set()
        for l in seg:
            op, _ = _split_op(l)
            rd = reads_of(l)
            dst = dest_of(l)
            uni.add(dst)
            for a in rd:
                uni.add(a)
                cnt[a] = cnt.get(a, 0) + 1
            for a in [dst] + rd:
                if a not in seen:
                    seen.add(a)
                    if a in rd:
                        first_read.add(a)
        if p == len(phases) - 1:
            for x in footer.split(","):
                a = int(x)
                uni.add(a)
                cnt[a] = cnt.get(a, 0) + 1
                if a not in seen:
                    seen.add(a)
                    first_read.add(a)
        order = sorted(uni, key=lambda a: (-cnt.get(a, 0), a))
        amap = {a: i + 1 for i, a in enumerate(order)}

        if p > 0:
            bridges = [(last[a], amap[a]) for a in order
                       if a in first_read and a in last]
            # copy dst,src overwrites cell dst, which may still be needed as
            # another pending copy's source: emit copies whose dst is no
            # pending source first; break cycles through a scratch slot that
            # collides with neither phase's addresses nor any bridge end.
            pending = dict(bridges)              # src(old) -> dst(new)
            hi = max([len(order)] + [a for ab in bridges for a in ab]
                     + [max(last.values(), default=0)])
            scratch = hi + 1
            while pending:
                srcs = set(pending)
                ready = [(s, d) for s, d in pending.items() if d not in srcs]
                if not ready:
                    s, d = next(iter(pending.items()))
                    out.append(f"copy {scratch},{s}")
                    del pending[s]
                    pending[scratch] = d
                    continue
                for s, d in ready:
                    out.append(f"copy {d},{s}")
                    del pending[s]

        for l in seg:
            op, rest = _split_op(l)
            suffix = ""
            if rest.endswith(",eq") or rest.endswith(",ne"):
                rest, suffix = rest[:-3], rest[-3:]
            args = rest.split(",")
            dst = amap[int(args[0])]
            if op == "set":
                out.append(f"set {dst},{args[1]}")
            else:
                srcs = ",".join(str(amap[int(x)]) for x in args[1:])
                out.append(f"{op} {dst},{srcs}{suffix}")
        for a in order:
            last[a] = amap[a]
        if p == 0:
            out[0] = ",".join(str(amap[int(x)]) for x in header.split(","))

    out.append(",".join(str(last[int(x)]) for x in footer.split(",")))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# packed siswalk generator
# ---------------------------------------------------------------------------

def generate_packed_sis(
    n_sets: int = 1,
    cap: int = 2,
    *,
    seed: int = 0,
    triple_knobs: int | None = None,
    spec: mp.Spec = mp.MASK32,
) -> tuple:
    """Packed siswalk.  Returns (ir_text, [readoff_split, walk_split]).

    ``triple_knobs`` = T additionally visits weight-3 coefficient sets over
    the first T knobs (on top of the weight<=cap walk), a cheap recovery
    boost: +C(T,3) visits.
    """
    n, m, k = spec.n_bits, spec.m_train, spec.k_secret
    G = n - m
    n_aug = n + 1                      # [X_S (m) | y (1) | X_F (G)]
    NC = (n_aug + 7) // 8              # packed cells per row (5 for mask32)
    WN = (n + 7) // 8                  # packed cells per n-bit vector (4)
    if not 1 <= n_sets:
        raise ValueError("n_sets must be >= 1")
    if not 0 <= cap <= G:
        raise ValueError(f"cap must be in [0, {G}]")
    rng = Random(seed)
    isets = [sorted(rng.sample(range(n), m)) for _ in range(n_sets)]

    a = 1

    def alloc(sz):
        nonlocal a
        base = a
        a += sz
        return base

    W_base = alloc(n_sets * WN)        # packed running solution per set
    OUTP_base = alloc(WN)              # packed captured output
    FB_base = alloc(n_sets * G * WN)   # packed knob vectors
    WSUM = alloc(1)
    OK = alloc(1)
    M_base = alloc(n_sets * m * NC)    # packed augmented tables
    PR_base = alloc(NC)                # packed pivot row
    SEL_base = alloc(NC)               # packed selected row (readoff)
    PC_base = alloc(WN)                # popcount scratch
    pivot_base = alloc(n_sets * m)
    used_base = alloc(m)
    FIRST_base = alloc(m)
    MATCH_base = alloc(m)
    ROW_base = alloc(m)
    BITC_base = alloc(8)               # constants 1<<j
    ZERO = alloc(1)
    ONE = alloc(1)
    M_VAL = alloc(1)
    K_VAL = alloc(1)
    C55 = alloc(1)
    C33 = alloc(1)
    C0F = alloc(1)
    C2 = alloc(1)
    C4 = alloc(1)
    C16 = alloc(1)
    t1 = alloc(1)
    t2 = alloc(1)
    t3 = alloc(1)
    t4 = alloc(1)
    OUT_base = alloc(n)                # unpacked final outputs
    X_tr_base = alloc(n * m)
    y_tr_base = alloc(m)

    W_at = lambda s, cell: W_base + s * WN + cell
    OUTP_at = lambda cell: OUTP_base + cell
    FB_at = lambda s, j, cell: FB_base + (s * G + j) * WN + cell
    M_at = lambda s, r, cell: M_base + (s * m + r) * NC + cell
    PR_at = lambda cell: PR_base + cell
    SEL_at = lambda cell: SEL_base + cell
    PC_at = lambda cell: PC_base + cell
    pivot_at = lambda s, c: pivot_base + s * m + c
    used_at = lambda r: used_base + r
    FIRST_at = lambda r: FIRST_base + r
    MATCH_at = lambda r: MATCH_base + r
    ROW_at = lambda r: ROW_base + r
    BITC = lambda j: BITC_base + j
    OUT_at = lambda c: OUT_base + c
    X_at = lambda i, c: X_tr_base + i * n + c
    y_at = lambda i: y_tr_base + i

    inputs = [X_at(i, c) for i in range(m) for c in range(n)] + [
        y_at(i) for i in range(m)
    ]
    lines = [",".join(map(str, inputs))]
    emit = lines.append

    # ---- phase 0 constants --------------------------------------------------
    emit(f"set {ZERO},0")
    emit(f"set {ONE},1")
    emit(f"set {M_VAL},{m}")
    for j in range(8):
        emit(f"set {BITC(j)},{1 << j}")
    for r in range(m):
        emit(f"set {ROW_at(r)},{r}")

    # ---- phase 0: load packed + RREF per info set ---------------------------
    for si, S in enumerate(isets):
        free_cols = [c for c in range(n) if c not in S]

        def src_of(r, p, S=S, free_cols=free_cols):
            # bit position p of the packed augmented row r
            if p < m:
                return X_at(r, S[p])
            if p == m:
                return y_at(r)
            return X_at(r, free_cols[p - (m + 1)])

        # load + pack
        for r in range(m):
            for cell in range(NC):
                ps = [p for p in range(cell * 8, min(cell * 8 + 8, n_aug))]
                if not ps:
                    emit(f"set {M_at(si, r, cell)},0")
                    continue
                emit(f"mul {M_at(si, r, cell)},{src_of(r, ps[0])},{BITC(ps[0] % 8)}")
                for p in ps[1:]:
                    emit(f"mul {t1},{src_of(r, p)},{BITC(p % 8)}")
                    emit(f"or {M_at(si, r, cell)},{t1}")

        # RREF over the m info-set columns (Gauss-Jordan, static pivot order,
        # pivot-row extraction fused into the search; cells below col//8 are
        # dead and skipped)
        for r in range(m):
            emit(f"set {used_at(r)},0")
        for col in range(m):
            lo = col // 8
            cm = BITC(col % 8)
            emit(f"copy {t4},{M_VAL}")                  # piv_i
            emit(f"copy {t3},{ZERO}")                   # found
            for cl in range(lo, NC):
                emit(f"set {PR_at(cl)},0")
            for r in range(m):
                emit(f"and {t1},{M_at(si, r, lo)},{cm}")        # bit (0/2^j)
                emit(f"cmp {t1},{t1},{ZERO},ne")                # bit01
                emit(f"xor {t2},{used_at(r)},{ONE}")            # not_used
                emit(f"and {t1},{t1},{t2}")                     # eligible
                emit(f"select {t2},{t3},{ZERO},{t1}")           # is_first
                emit(f"copy {FIRST_at(r)},{t2}")
                for cl in range(lo, NC):
                    emit(f"select {PR_at(cl)},{t2},{M_at(si, r, cl)},{PR_at(cl)}")
                emit(f"select {t4},{t2},{ROW_at(r)},{t4}")      # piv_i
                emit(f"or {used_at(r)},{t2}")
                emit(f"or {t3},{t1}")                           # found
            emit(f"copy {pivot_at(si, col)},{t4}")
            for r in range(m):
                emit(f"xor {t2},{FIRST_at(r)},{ONE}")           # is_other
                emit(f"sub {t2},{ZERO},{t2}")                   # mask_other 0/-1
                emit(f"and {t1},{M_at(si, r, lo)},{cm}")        # bit (nonzero)
                emit(f"select {t1},{t1},{t2},{ZERO}")           # elim mask
                for cl in range(lo, NC):
                    emit(f"and {t3},{PR_at(cl)},{t1}")
                    emit(f"xor {M_at(si, r, cl)},{t3}")

    # ---- phase 1: read off s0 (y column) and the G knob vectors -------------
    readoff_split = len(lines) - 1

    # readoff constants (re-set: free, and avoids bridging)
    emit(f"set {ZERO},0")
    for j in range(8):
        emit(f"set {BITC(j)},{1 << j}")
    for r in range(m):
        emit(f"set {ROW_at(r)},{r}")

    y_cell, y_bit = m // 8, m % 8
    knob_cells = sorted({(m + 1 + j) // 8 for j in range(G)} | {y_cell})
    for si, S in enumerate(isets):
        for cell in range(WN):
            emit(f"set {W_at(si, cell)},0")
        free_cols = [c for c in range(n) if c not in S]
        for j, f in enumerate(free_cols):
            for cell in range(WN):
                static_bit = (1 << (f % 8)) if f // 8 == cell else 0
                emit(f"set {FB_at(si, j, cell)},{static_bit}")
        for jj, c in enumerate(S):
            for r in range(m):
                emit(f"cmp {MATCH_at(r)},{pivot_at(si, jj)},{ROW_at(r)},eq")
            for cell in knob_cells:
                emit(f"set {SEL_at(cell)},0")
                for r in range(m):
                    emit(f"select {SEL_at(cell)},{MATCH_at(r)},{M_at(si, r, cell)},{SEL_at(cell)}")
            # s0 bit at output position c
            emit(f"and {t1},{SEL_at(y_cell)},{BITC(y_bit)}")
            emit(f"cmp {t1},{t1},{ZERO},ne")
            emit(f"mul {t1},{t1},{BITC(c % 8)}")
            emit(f"or {W_at(si, c // 8)},{t1}")
            # knob bits at output position c
            for j in range(G):
                bit = m + 1 + j
                emit(f"and {t1},{SEL_at(bit // 8)},{BITC(bit % 8)}")
                emit(f"cmp {t1},{t1},{ZERO},ne")
                emit(f"mul {t1},{t1},{BITC(c % 8)}")
                emit(f"or {FB_at(si, j, c // 8)},{t1}")

    # ---- phase 2: walk + unpack ---------------------------------------------
    walk_split = len(lines) - 1

    # walk constants
    emit(f"set {ZERO},0")
    emit(f"set {K_VAL},{k}")
    emit(f"set {C55},{MASK55}")
    emit(f"set {C33},{MASK33}")
    emit(f"set {C0F},{MASK0F}")
    emit(f"set {C2},2")
    emit(f"set {C4},4")
    emit(f"set {C16},16")
    for j in range(8):
        emit(f"set {BITC(j)},{1 << j}")

    def capture(si):
        # SWAR popcount of the 4 packed w cells, nibble-combined:
        # step1 (per-2-bit counts) and step2 (per-nibble counts) per cell,
        # then add cells pairwise (nibble sums <= 8, no overflow) and finish.
        for cell in range(WN):
            emit(f"div {t1},{W_at(si, cell)},{C2}")
            emit(f"and {t1},{t1},{C55}")
            emit(f"sub {PC_at(cell)},{W_at(si, cell)},{t1}")
        for cell in range(WN):
            emit(f"and {t1},{PC_at(cell)},{C33}")
            emit(f"div {t3},{PC_at(cell)},{C4}")
            emit(f"and {t3},{t3},{C33}")
            emit(f"add {PC_at(cell)},{t1},{t3}")
        emit(f"add {PC_at(0)},{PC_at(1)}")
        emit(f"add {PC_at(2)},{PC_at(3)}")
        emit(f"and {t1},{PC_at(0)},{C0F}")
        emit(f"div {t3},{PC_at(0)},{C16}")
        emit(f"add {t1},{t1},{t3}")
        emit(f"and {t2},{PC_at(2)},{C0F}")
        emit(f"div {t3},{PC_at(2)},{C16}")
        emit(f"add {t2},{t2},{t3}")
        emit(f"add {WSUM},{t1},{t2}")
        emit(f"cmp {OK},{WSUM},{K_VAL},eq")
        for cell in range(WN):
            emit(f"select {OUTP_at(cell)},{OK},{W_at(si, cell)},{OUTP_at(cell)}")

    # visit schedule: weight<=cap over all G knobs, plus weight-3 sets over
    # the first triple_knobs knobs if requested
    coef_sets: List[tuple] = [()]
    for w in range(1, cap + 1):
        coef_sets.extend(combinations(range(G), w))
    if triple_knobs:
        coef_sets.extend(combinations(range(triple_knobs), 3))
    flips_list: List[List[int]] = []
    prev = frozenset()
    for cur in coef_sets:
        cur_s = frozenset(cur)
        flips_list.append(sorted(cur_s.symmetric_difference(prev)))
        prev = cur_s

    for cell in range(WN):
        emit(f"set {OUTP_at(cell)},0")
    for si in range(n_sets):
        capture(si)
        for flips in flips_list[1:]:   # first transition is empty (s0 again)
            for j in flips:
                for cell in range(WN):
                    emit(f"xor {W_at(si, cell)},{FB_at(si, j, cell)}")
            capture(si)

    # ---- unpack outputs ------------------------------------------------------
    for c in range(n):
        emit(f"and {t1},{OUTP_at(c // 8)},{BITC(c % 8)}")
        emit(f"cmp {OUT_at(c)},{t1},{ZERO},ne")

    lines.append(",".join(str(OUT_at(c)) for c in range(n)))
    ir = "\n".join(lines)
    if len(lines) > mp.OP_CAP:
        raise ValueError(
            f"packed siswalk IR has {len(lines) - 2:,} ops, over the {mp.OP_CAP:,} cap"
        )
    return ir, [readoff_split, walk_split]


def generate(n_sets: int = 1, cap: int = 2, *, seed: int = 0,
             triple_knobs: int | None = None) -> str:
    """Packed siswalk with staged three-phase layout."""
    ir, splits = generate_packed_sis(n_sets, cap, seed=seed,
                                     triple_knobs=triple_knobs)
    return renumber_phases(ir, splits)


if __name__ == "__main__":
    import time

    for label, kw in [
        ("n_sets=1 cap=2", dict(n_sets=1, cap=2)),
        ("n_sets=1 cap=2 T=8", dict(n_sets=1, cap=2, triple_knobs=8)),
        ("n_sets=1 cap=3", dict(n_sets=1, cap=3)),
        ("n_sets=2 cap=2", dict(n_sets=2, cap=2)),
    ]:
        t0 = time.time()
        ir = generate(**kw)
        res = mp.evaluate_mask(ir)
        print(
            f"packed siswalk {label:>18}: cost={res.cost:>9,} "
            f"recovery={res.recovery:.4f} lines={len(ir.splitlines()):,} "
            f"[{time.time() - t0:.1f}s]"
        )
