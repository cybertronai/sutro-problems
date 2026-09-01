"""Attack generator for the sparse-parity 100% band.

Design (vs the weightscan5 record, cost 12,461,610):

* Septet packing, row-major: each of the 18 rows of [X|y] (33 bits)
  lives in 5 cells of 7 bits.  Values stay in [0,127] so ``div`` by
  2/4/16 is an unsigned shift and ``and`` masks behave.  GF(2) row ops
  become 5-cell XORs instead of 33 scalar ops.
* Gauss-Jordan RREF with the same dynamic pivoting as the record
  (identical coverage: fails only on rank-deficient draws, ~2^-14),
  but one pivot costs 18 rows x ~14 ops instead of ~103.
* The walk runs in ROW coordinates: z (3 septets, 18 row-bits) holds
  the pivot-part of the current solution; the free part is the
  coefficient vector itself (known statically per visit).
  weight(w) = popcount(z) + |a_t|, tested against a constant cell.
* Capture records only the coefficient vector: at most one visit fires
  (unique identifiability), so ``xor cap[j], OK`` for j in a_t
  (statically known) rebuilds cap = a_t exactly when OK fires.  No
  per-visit output selects at all.
* The full 32-bit mask is reconstructed once at the end:
  zz = ycol ^ sum_j cap[j] * A[j] (row coords), then each output cell
  selects between the zz bit at its pivot row and cap at its free rank.
* Weight<=cap visits follow a revolving-door order (2 flips per
  transition inside a level) instead of lexicographic (~2.58 flips).
"""
from __future__ import annotations

from itertools import combinations

N, M, K = 32, 18, 5
G = N - M                              # 14 free variables w.h.p.
P = 5                                  # septets per row (33 bits: 7*4+5)
S = 3                                  # septets per column-slice (18 bits)


def _csept(c: int):
    """(septet, bit) of column c within a packed row."""
    return divmod(c, 7)


def _revdoor(n: int, k: int):
    """All k-subsets of range(n); consecutive ones differ by one swap."""
    if k == 0:
        return [()]
    if k == n:
        return [tuple(range(n))]
    a = _revdoor(n - 1, k)
    b = _revdoor(n - 1, k - 1)
    return a + [s + (n - 1,) for s in reversed(b)]


def walk_sets(cap: int, order: str = "revdoor"):
    sets: list[tuple] = [()]
    for w in range(1, cap + 1):
        if order == "revdoor":
            sets.extend(_revdoor(G, w))
        elif order == "lex":
            sets.extend(combinations(range(G), w))
        else:
            raise ValueError(order)
    return sets


def generate_septwalk(weight_cap: int = 5, order: str = "revdoor") -> str:
    sets = walk_sets(weight_cap, order)

    # ---------------- address allocation -------------------------------
    a = 1
    def alloc(sz=1):
        nonlocal a
        base = a
        a += sz
        return base

    # hot walk state first
    z_base = alloc(S)                      # running solution, row coords
    t1 = alloc(1); t2 = alloc(1); t3 = alloc(1); t4 = alloc(1)
    v2 = alloc(1); v3 = alloc(1)
    pc_base = alloc(S)
    WSUM = alloc(1); OK = alloc(1)
    cap_base = alloc(G)
    A_base = alloc(G * S)                  # knob columns (row coords)
    UM_base = alloc(S)                     # used-row mask (septets)
    # constants
    ZERO = alloc(1); ONE = alloc(1); TWO = alloc(1); FOUR = alloc(1)
    C85 = alloc(1); C51 = alloc(1); C15 = alloc(1); C127 = alloc(1)
    BP = [alloc(1) for _ in range(7)]      # 1,2,4,8,16,32,64
    V = [alloc(1) for _ in range(14)]      # 0..13 (JC + weight targets)
    C18 = alloc(1)
    ROWC = [alloc(1) for _ in range(M)]    # 0..17
    # RREF state
    T_base = alloc(M * P)                  # packed rows of [X|y]
    PR_base = alloc(P)                     # pivot row buffer
    pivstore_base = alloc(N)               # pivot row per column (18 = free)
    free_base = alloc(N)
    rank_base = alloc(N)
    used_base = alloc(M)
    piv = alloc(1); found = alloc(1)
    bit = alloc(1); nu = alloc(1); elig = alloc(1); frst = alloc(1)
    isc = alloc(1); do = alloc(1); msk = alloc(1)
    fs = alloc(1); pw = alloc(1); acc = alloc(1)
    match = alloc(1); b01 = alloc(1); bp = alloc(1); cp = alloc(1)
    out_base = alloc(N)
    zz_base = alloc(S)
    YC_base = alloc(S)                     # masked y column (row coords)
    # inputs last
    X_base = alloc(N * M)
    y_base = alloc(M)

    z_at = lambda s: z_base + s
    pc_at = lambda s: pc_base + s
    cap_at = lambda j: cap_base + j
    A_at = lambda j, s: A_base + j * S + s
    UM_at = lambda s: UM_base + s
    T_at = lambda r, p: T_base + r * P + p
    PR_at = lambda p: PR_base + p
    X_at = lambda r, c: X_base + r * N + c
    y_at = lambda r: y_base + r

    inputs = [X_at(r, c) for r in range(M) for c in range(N)] + [
        y_at(r) for r in range(M)
    ]
    lines = [",".join(map(str, inputs))]
    emit = lines.append

    # ---------------- constants ----------------------------------------
    for cell, val in [(ZERO, 0), (ONE, 1), (TWO, 2), (FOUR, 4),
                      (C85, 85), (C51, 51), (C15, 15), (C127, 127),
                      (C18, 18)]:
        emit(f"set {cell},{val}")
    for i in range(7):
        emit(f"set {BP[i]},{1 << i}")
    for i in range(14):
        emit(f"set {V[i]},{i}")
    for r in range(M):
        emit(f"set {ROWC[r]},{r}")
        emit(f"set {used_base + r},0")

    # ---------------- load + pack (rows of [X|y] into septets) ---------
    for r in range(M):
        for p in range(P):
            hi = min(7 * p + 6, N)         # bit 32 of septet 4 is y
            lo = 7 * p
            def bitcell(b, r=r):
                return y_at(r) if b == N else X_at(r, b)
            emit(f"copy {T_at(r, p)},{bitcell(hi)}")
            for b in range(hi - 1, lo - 1, -1):
                emit(f"mul {T_at(r, p)},{TWO}")
                emit(f"add {T_at(r, p)},{bitcell(b)}")

    # ---------------- RREF (Gauss-Jordan, dynamic pivoting) ------------
    for c in range(N):
        cs, cb = _csept(c)
        # pivot search: first unused row with bit set in column c
        emit(f"copy {piv},{C18}")
        emit(f"copy {found},{ZERO}")
        for r in range(M):
            emit(f"and {bit},{T_at(r, cs)},{BP[cb]}")
            emit(f"xor {nu},{used_base + r},{ONE}")
            emit(f"select {elig},{bit},{nu},{ZERO}")
            emit(f"select {frst},{found},{ZERO},{elig}")
            emit(f"select {piv},{frst},{ROWC[r]},{piv}")
            emit(f"or {used_base + r},{frst}")
            emit(f"or {found},{elig}")
        emit(f"copy {pivstore_base + c},{piv}")
        emit(f"cmp {free_base + c},{piv},{C18},eq")
        # pivot row buffer (all zero when column is free)
        for p in range(P):
            emit(f"copy {PR_at(p)},{ZERO}")
            for r in range(M):
                emit(f"cmp {isc},{piv},{ROWC[r]},eq")
                emit(f"select {PR_at(p)},{isc},{T_at(r, p)},{PR_at(p)}")
        # eliminate column c from every other row with a set bit
        for r in range(M):
            emit(f"cmp {isc},{piv},{ROWC[r]},ne")
            emit(f"and {bit},{T_at(r, cs)},{BP[cb]}")
            emit(f"select {do},{bit},{isc},{ZERO}")
            emit(f"and {do},{found}")
            emit(f"sub {msk},{ZERO},{do}")          # 0 or -1
            for p in range(P):
                emit(f"and {t1},{PR_at(p)},{msk}")
                emit(f"xor {T_at(r, p)},{t1}")

    # ---------------- free-column ranks + used-row mask -----------------
    emit(f"copy {rank_base},{ZERO}")
    for c in range(1, N):
        emit(f"add {rank_base + c},{rank_base + c - 1},{free_base + c - 1}")
    for s in range(S):
        emit(f"copy {UM_at(s)},{ZERO}")
        for r in range(7 * s, min(7 * s + 7, M)):
            emit(f"select {t1},{used_base + r},{BP[r - 7 * s]},{ZERO}")
            emit(f"or {UM_at(s)},{t1}")

    # ---------------- knob columns A[j] = column f_j (row coords) -------
    for j in range(G):
        # fs = septet of f_j, pw = its bit power (weighted sums over the
        # unique column c with rank[c]==j and free[c]).
        emit(f"copy {fs},{ZERO}")
        emit(f"copy {pw},{ZERO}")
        for c in range(N):
            cs, cb = _csept(c)
            emit(f"cmp {isc},{rank_base + c},{V[j]},eq")
            emit(f"and {match},{isc},{free_base + c}")
            emit(f"mul {t1},{match},{V[cs]}")
            emit(f"add {fs},{t1}")
            emit(f"mul {t1},{match},{BP[cb]}")
            emit(f"add {pw},{t1}")
        for s in range(S):
            emit(f"copy {A_at(j, s)},{ZERO}")
        for r in range(M):
            s, b = divmod(r, 7)
            emit(f"copy {acc},{T_at(r, 0)}")
            for p in range(1, P):
                emit(f"cmp {isc},{fs},{V[p]},eq")
                emit(f"select {acc},{isc},{T_at(r, p)},{acc}")
            emit(f"and {t1},{acc},{pw}")
            emit(f"select {b01},{t1},{BP[b]},{ZERO}")
            emit(f"or {A_at(j, s)},{b01}")
        for s in range(S):
            emit(f"and {A_at(j, s)},{UM_at(s)}")

    # ---------------- masked y column (row coords), prefix --------------
    ys, yb = _csept(N)
    for s in range(S):
        emit(f"copy {YC_base + s},{ZERO}")
        for r in range(7 * s, min(7 * s + 7, M)):
            emit(f"and {bit},{T_at(r, ys)},{BP[yb]}")
            emit(f"select {b01},{bit},{used_base + r},{ZERO}")
            emit(f"mul {t1},{b01},{BP[r - 7 * s]}")
            emit(f"or {YC_base + s},{t1}")

    # ================= WALK PHASE BOUNDARY ==============================
    # Everything below re-addresses from 1 (see stage_septwalk_layout).
    # Every cell read below is either written below or bridged:
    #   YC (3), A (42), pivstore/free/rank (96).
    boundary = len(lines)

    # walk-phase constants (set ops are free)
    for cell, val in [(ZERO, 0), (ONE, 1), (TWO, 2), (FOUR, 4),
                      (C85, 85), (C51, 51), (C15, 15)]:
        emit(f"set {cell},{val}")
    for i in range(7):
        emit(f"set {BP[i]},{1 << i}")
    for i in range(14):
        emit(f"set {V[i]},{i}")
    for r in range(M):
        emit(f"set {ROWC[r]},{r}")
    for j in range(G):
        emit(f"set {cap_at(j)},0")
    for s in range(S):
        emit(f"copy {z_at(s)},{YC_base + s}")

    # ---------------- popcount + capture --------------------------------
    def weight_test(target: int):
        """Emit OK = (popcount(z) == target), specialized for small targets."""
        if target == 0:
            # popcount(z) == 0  <=>  z == 0
            emit(f"or {t1},{z_at(0)},{z_at(1)}")
            emit(f"or {t1},{z_at(2)}")
            emit(f"cmp {OK},{t1},{ZERO},eq")
            return
        if target == 1:
            # popcount(z) == 1  <=>  exactly one nonzero septet, all
            # septets being "zero or a power of two" (x & (x-1) == 0).
            emit(f"copy {v2},{ONE}")                # alle: all e_s so far
            emit(f"copy {v3},{ZERO}")               # cntnz
            for s in range(S):
                x = z_at(s)
                emit(f"sub {t1},{x},{ONE}")
                emit(f"and {t1},{x}")               # 0 iff x is 0 or pow2
                emit(f"cmp {t1},{t1},{ZERO},eq")         # e_s
                emit(f"and {v2},{t1}")
                emit(f"cmp {t1},{x},{ZERO},ne")     # nz_s
                emit(f"add {v3},{t1}")
            emit(f"cmp {v3},{v3},{ONE},eq")
            emit(f"and {OK},{v3},{v2}")
            return
        # SWAR popcount, tail merged across septets:
        # v3_s = lo_s + 16*hi_s, sum has no nibble carry (lo sum <= 12).
        for s in range(S):
            x = z_at(s)
            emit(f"div {t1},{x},{TWO}")
            emit(f"and {t2},{t1},{C85}")
            emit(f"sub {v2},{x},{t2}")
            emit(f"and {t3},{v2},{C51}")
            emit(f"div {t4},{v2},{FOUR}")
            emit(f"and {t4},{C51}")
            emit(f"add {pc_at(s)},{t3},{t4}")       # v3_s
        emit(f"add {WSUM},{pc_at(0)},{pc_at(1)}")
        emit(f"add {WSUM},{pc_at(2)}")              # T = sum lo + 16*sum hi
        emit(f"and {t1},{WSUM},{C15}")
        emit(f"div {t2},{WSUM},{BP[4]}")            # >> 4
        emit(f"add {WSUM},{t1},{t2}")
        emit(f"cmp {OK},{WSUM},{V[target]},eq")

    def capture(cur):
        weight_test(K - len(cur))                   # target = 5 - |a_t| >= 0
        for j in cur:
            emit(f"xor {cap_at(j)},{OK}")

    # ---------------- the walk ------------------------------------------
    capture(())
    prev: frozenset = frozenset()
    for cur_t in sets[1:]:
        cur = frozenset(cur_t)
        for j in sorted(cur.symmetric_difference(prev)):
            for s in range(S):
                emit(f"xor {z_at(s)},{A_at(j, s)}")
        capture(cur_t)
        prev = cur

    # ---------------- final reconstruction ------------------------------
    for s in range(S):
        emit(f"copy {zz_base + s},{YC_base + s}")
    for j in range(G):
        emit(f"sub {t1},{ZERO},{cap_at(j)}")        # 0 or -1
        for s in range(S):
            emit(f"and {t2},{A_at(j, s)},{t1}")
            emit(f"xor {zz_base + s},{t2}")
    for c in range(N):
        # pivot-coordinate bit: zz[pivstore[c]] as 0/1
        emit(f"copy {bp},{ZERO}")
        for r in range(M):
            s, b = divmod(r, 7)
            emit(f"cmp {isc},{pivstore_base + c},{ROWC[r]},eq")
            emit(f"and {t1},{zz_base + s},{BP[b]}")
            emit(f"select {b01},{t1},{ONE},{ZERO}")
            emit(f"select {bp},{isc},{b01},{bp}")
        # free-coordinate bit: cap[rank[c]]
        emit(f"copy {cp},{ZERO}")
        for j in range(G):
            emit(f"cmp {isc},{rank_base + c},{V[j]},eq")
            emit(f"select {cp},{isc},{cap_at(j)},{cp}")
        emit(f"select {out_base + c},{free_base + c},{cp},{bp}")

    lines.append(",".join(str(out_base + c) for c in range(N)))
    bridged = (
        [YC_base + s for s in range(S)]
        + [A_at(j, s) for j in range(G) for s in range(S)]
        + [pivstore_base + c for c in range(N)]
        + [free_base + c for c in range(N)]
        + [rank_base + c for c in range(N)]
    )
    return "\n".join(lines), boundary, bridged


def stage_septwalk_layout(ir_boundary_bridged) -> str:
    """Two-phase layout: prefix gets its own frequency-optimal addresses,
    the walk+final phase re-addresses from 1 into the dead prefix range,
    bridged by scratch-mediated copies (hazard-free)."""
    ir, boundary, bridged = ir_boundary_bridged
    lines = ir.splitlines()
    inputs_d, outputs_d = lines[0], lines[-1]
    prefix, walk = lines[1:boundary], lines[boundary:-1]

    def split_op(l):
        parts = l.split(" ", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    def universe_reads(seg):
        cnt, uni = {}, set()
        for l in seg:
            op, rest = split_op(l)
            if op == "set":
                uni.add(int(rest.split(",")[0]))
                continue
            rest2 = rest.replace(",eq", "").replace(",ne", "")
            args = [int(x) for x in rest2.split(",")]
            uni.add(args[0])
            for x in args[1:]:
                uni.add(x)
                cnt[x] = cnt.get(x, 0) + 1
        return cnt, uni

    pc, pu = universe_reads(prefix)
    wc, wu = universe_reads(walk)
    for a in bridged:                       # bridge copies read the prefix side
        pc[a] = pc.get(a, 0) + 1
    for x in inputs_d.split(","):           # input cells live in the prefix
        pu.add(int(x))
    for x in outputs_d.split(","):          # outputs are read once, walk side
        wc[int(x)] = wc.get(int(x), 0) + 1

    pre_map = {a: i + 1 for i, a in enumerate(
        sorted(pu, key=lambda a: (-pc.get(a, 0), a)))}
    walk_map = {a: i + 1 for i, a in enumerate(
        sorted(wu, key=lambda a: (-wc.get(a, 0), a)))}

    def remap(seg, mapping):
        out = []
        for l in seg:
            op, rest = split_op(l)
            suffix = ""
            if rest.endswith(",eq") or rest.endswith(",ne"):
                rest, suffix = rest[:-3], rest[-3:]
            args = rest.split(",")
            dst = mapping[int(args[0])]
            if op == "set":
                out.append(f"{op} {dst},{args[1]}{suffix}")
                continue
            srcs = ",".join(str(mapping[int(x)]) for x in args[1:])
            out.append(f"{op} {dst},{srcs}{suffix}")
        return out

    out = [",".join(str(pre_map[int(x)]) for x in inputs_d.split(","))]
    out += remap(prefix, pre_map)
    # two-round bridging: walk_map dests may collide with not-yet-bridged
    # prefix addresses, so stage every value through a private scratch slot.
    scratch0 = len(pre_map) + len(walk_map) + 1
    for i, a in enumerate(bridged):
        out.append(f"copy {scratch0 + i},{pre_map[a]}")
    for i, a in enumerate(bridged):
        out.append(f"copy {walk_map[a]},{scratch0 + i}")
    out += remap(walk, walk_map)
    out.append(",".join(str(walk_map[int(x)]) for x in outputs_d.split(",")))
    return "\n".join(out)


def generate_staged(weight_cap: int = 5, order: str = "revdoor") -> str:
    return stage_septwalk_layout(generate_septwalk(weight_cap, order))


if __name__ == "__main__":
    ir, b, br = generate_septwalk()
    print("lines:", len(ir.splitlines()) + 1, "boundary:", b, "bridged:", len(br))
