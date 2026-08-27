"""Lane 0 — exploit non-uniform per-cell A/B reads via input layout.

Plan from explorer brief:
1. Profile per-cell read counts of current best (sa-cache TI=8,TJ=4) — confirm
   they are uniform (4× per A cell, 2× per B cell).
2. Try alternative loop nests (bj,bi,bk), (bi,bk,bj), (bk,bi,bj) etc. — confirm
   that within the sa-cache template they all give uniform per-cell counts.
3. Try a *partial cache* scheme that creates non-uniform reads:
   - Pre-load some rows of A into low scratchpad addresses (one bulk read each).
   - Inner loop uses cached rows (free) for those rows and bulk reads for others.
   - This makes some A bulk cells get read once (cached) and others 4× (bulk).
4. Pipe every IR through addr_renamer.rename_optimal before scoring.

We expect (per the brief) that this confirms 73602 is locally optimal under
input-layout-only changes, but we test concrete variants.
"""
from __future__ import annotations
import os, sys, time, json, math
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from matmul import score_16x16  # noqa: E402
from addr_renamer import rename_optimal  # noqa: E402

N = 16
BEST_PRIOR = 73602


def cost_of(addr):
    return math.isqrt(addr - 1) + 1


# ---------- Profile helpers ----------
def count_reads(ir):
    """Return Counter of per-address read counts."""
    text = ir.replace(";", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    output_addrs = [int(x) for x in lines[-1].split(",")]
    reads = Counter()
    for ln in lines[1:-1]:
        head, _, rest = ln.partition(" ")
        ops = [int(x) for x in rest.split(",")]
        if head == "copy":
            dest, src = ops
            reads[src] += 1
        else:
            if len(ops) == 3:
                dest, s1, s2 = ops
            else:
                dest, s2 = ops
                s1 = dest
            reads[s1] += 1
            reads[s2] += 1
    for a in output_addrs:
        reads[a] += 1
    return reads


def get_input_addrs(ir):
    """Return list of declared input addresses (first line)."""
    lines = ir.replace(";", "\n").splitlines()
    return [int(x) for x in lines[0].strip().split(",")]


def per_cell_read_summary(ir):
    """Return dicts {A_addr: count}, {B_addr: count} based on the input layout
    of an sa-cache style IR. Inputs are 256 A then 256 B."""
    inputs = get_input_addrs(ir)
    assert len(inputs) == 512, f"Expected 512 inputs, got {len(inputs)}"
    A_addrs = inputs[:256]
    B_addrs = inputs[256:]
    reads = count_reads(ir)
    a_reads = [reads.get(a, 0) for a in A_addrs]
    b_reads = [reads.get(b, 0) for b in B_addrs]
    return a_reads, b_reads


# ---------- IR generators ----------
def gen_v1(TI, TJ, order="bi_bj_bk"):
    """sa-cache template with selectable outer loop order over (bi,bj,bk).
    Inner is always (jj-load-sB, ii, jj-mul). Returns IR string."""
    nbi, nbj = N // TI, N // TJ
    SA, TMP = 1, 2
    sB = lambda jj: 3 + jj
    sC_base = 3 + TJ
    sC = lambda ii, jj: sC_base + ii * TJ + jj
    A_base = sC_base + TI * TJ
    B_base = A_base + N * N
    C_base = B_base + N * N
    A = lambda i, k: A_base + i * N + k
    Bf = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j
    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]

    iter_lists = {
        "bi_bj_bk": [(bi, bj, bk) for bi in range(nbi) for bj in range(nbj) for bk in range(N)],
        "bj_bi_bk": [(bi, bj, bk) for bj in range(nbj) for bi in range(nbi) for bk in range(N)],
        "bi_bk_bj": [(bi, bj, bk) for bi in range(nbi) for bk in range(N) for bj in range(nbj)],
        "bj_bk_bi": [(bi, bj, bk) for bj in range(nbj) for bk in range(N) for bi in range(nbi)],
    }
    # For (bi, bk, bj) where bk varies in middle, sC must persist outside bk —
    # different (bi, bj) tiles need separate sC accumulators. So this would
    # require nbi*nbj separate sC tiles in memory. Skip.
    # For now we only support bk-innermost orders.
    if order not in ("bi_bj_bk", "bj_bi_bk"):
        return None

    # iterate (bi, bj) outer pair, then bk inside.
    if order == "bi_bj_bk":
        outer = [(bi, bj) for bi in range(nbi) for bj in range(nbj)]
    else:
        outer = [(bi, bj) for bj in range(nbj) for bi in range(nbi)]

    for bi, bj in outer:
        for bk in range(N):
            for jj in range(TJ):
                lines.append(f"copy {sB(jj)},{Bf(bk, bj * TJ + jj)}")
            for ii in range(TI):
                lines.append(f"copy {SA},{A(bi * TI + ii, bk)}")
                for jj in range(TJ):
                    if bk == 0:
                        lines.append(f"mul {sC(ii, jj)},{SA},{sB(jj)}")
                    else:
                        lines.append(f"mul {TMP},{SA},{sB(jj)}")
                        lines.append(f"add {sC(ii, jj)},{TMP}")
        for ii in range(TI):
            for jj in range(TJ):
                lines.append(f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def gen_partial_A_cache(TI, TJ, n_cached_rows):
    """sa-cache (TI,TJ) but pre-cache `n_cached_rows` rows of A into a
    scratchpad strip below the bulk. Inner loop uses cached rows directly
    (free address) and bulk-reads for non-cached rows.

    Cached rows: cells at low addrs sit at TJ+TI*TJ+3+offset .. (small).
    Wait — to make cached cells have FEWER reads than bulk cells, we need:
    - cached cell: 1 read (the load) + many uses in scratchpad space (free).
    - non-cached cell: 4 reads (bulk).

    So the cached cells go at LOW addrs (very cheap, but few reads). And the
    non-cached A cells go at HIGH addrs but MANY reads. The renamer will then
    sort: hot bulk cells go before low-read cached cells, but cached cells
    are already in scratchpad at low addr. After renaming, cells with most
    reads get lowest addr.

    Critically: we save `(4-1) * 16 = 48` reads of A bulk per cached row
    (each cell of cached row would have been read 4 times, now read 1 time).
    But we ADD 16 reads of the cache strip per cell, but the cache strip is
    in scratchpad (low addr). So net: trade `3 reads from bulk-cell @ A bulk
    addr` for `3 reads from cache-cell @ scratchpad addr`. WIN if the cache
    addr is cheaper than the bulk addr!

    Inputs: same flat layout.
    """
    nbi, nbj = N // TI, N // TJ
    SA, TMP = 1, 2
    sB = lambda jj: 3 + jj

    # Cache strip: n_cached_rows × N cells
    cache_base = 3 + TJ
    cacheA = lambda i, k: cache_base + i * N + k  # i in [0, n_cached_rows)
    n_cache = n_cached_rows * N

    sC_base = cache_base + n_cache
    sC = lambda ii, jj: sC_base + ii * TJ + jj
    A_base = sC_base + TI * TJ
    B_base = A_base + N * N
    C_base = B_base + N * N
    A = lambda i, k: A_base + i * N + k
    Bf = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]

    # Pre-load n_cached_rows rows of A into the cache strip
    for i in range(n_cached_rows):
        for k in range(N):
            lines.append(f"copy {cacheA(i, k)},{A(i, k)}")

    # Helper: get A operand — cache if i < n_cached_rows else bulk
    def Aop(i, k):
        if i < n_cached_rows:
            return cacheA(i, k)
        return A(i, k)

    # Use the standard sa-cache schedule with bi/bj/bk outer
    for bi in range(nbi):
        for bj in range(nbj):
            for bk in range(N):
                for jj in range(TJ):
                    lines.append(f"copy {sB(jj)},{Bf(bk, bj * TJ + jj)}")
                for ii in range(TI):
                    src_i = bi * TI + ii
                    lines.append(f"copy {SA},{Aop(src_i, bk)}")
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {sC(ii, jj)},{SA},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {sC(ii, jj)},{TMP}")
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def gen_partial_B_cache(TI, TJ, n_cached_cols):
    """Pre-cache n_cached_cols columns of B into scratchpad, similar idea.

    Each B cell normally read nbi=N/TI times. By caching, cell read once
    from bulk, and many times from scratchpad cache (low addr). For
    n_cached_cols < N, only some B columns are cached.
    """
    nbi, nbj = N // TI, N // TJ
    SA, TMP = 1, 2
    sB = lambda jj: 3 + jj

    cache_base = 3 + TJ
    # cache cols: n_cached_cols columns × N rows (k varies)
    cacheB = lambda k, j: cache_base + j * N + k  # j in [0, n_cached_cols)
    n_cache = n_cached_cols * N

    sC_base = cache_base + n_cache
    sC = lambda ii, jj: sC_base + ii * TJ + jj
    A_base = sC_base + TI * TJ
    B_base = A_base + N * N
    C_base = B_base + N * N
    A = lambda i, k: A_base + i * N + k
    Bf = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [Bf(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]

    # Pre-load cached B columns
    for j in range(n_cached_cols):
        for k in range(N):
            lines.append(f"copy {cacheB(k, j)},{Bf(k, j)}")

    def Bop(k, j):
        if j < n_cached_cols:
            return cacheB(k, j)
        return Bf(k, j)

    for bi in range(nbi):
        for bj in range(nbj):
            for bk in range(N):
                for jj in range(TJ):
                    src_j = bj * TJ + jj
                    lines.append(f"copy {sB(jj)},{Bop(bk, src_j)}")
                for ii in range(TI):
                    lines.append(f"copy {SA},{A(bi * TI + ii, bk)}")
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {sC(ii, jj)},{SA},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {sC(ii, jj)},{TMP}")
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# ---------- Helpers ----------
def stop_check():
    sig = os.path.join(HERE, "STOP_SIGNAL_0")
    if os.path.exists(sig):
        os.remove(sig)
        print("STOP_SIGNAL — halting.")
        sys.exit(0)


def log_event(event):
    with open(os.path.join(HERE, "events.jsonl"), "a") as f:
        f.write(json.dumps(event) + "\n")


def log_token(filename, tokens):
    with open(os.path.join(HERE, "token_log.jsonl"), "a") as f:
        f.write(json.dumps({"ts": int(time.time()), "lane": 0,
                            "exp": filename, "tokens": tokens}) + "\n")


def score_with_renamer(ir):
    """Run IR through addr_renamer and score the renamed version."""
    new_ir, info = rename_optimal(ir)
    cost = score_16x16(new_ir)
    return cost, new_ir, info


# ---------- Main ----------
def main():
    stop_check()
    log_event({"ts": int(time.time()), "type": "exp_start",
               "lane": 0, "exp": "exp_0_nonuniform_reads.py"})

    print("=" * 60)
    print("Step 1: Profile per-cell A/B read counts of V1(TI=8,TJ=4)")
    print("=" * 60)
    ir = gen_v1(8, 4)
    cost = score_16x16(ir)
    a_reads, b_reads = per_cell_read_summary(ir)
    print(f"  V1(8,4) cost={cost}")
    print(f"  A per-cell reads: min={min(a_reads)}, max={max(a_reads)}, "
          f"unique values={sorted(set(a_reads))}")
    print(f"  B per-cell reads: min={min(b_reads)}, max={max(b_reads)}, "
          f"unique values={sorted(set(b_reads))}")
    print(f"  -> reads are uniform within A and within B (as expected)")

    print()
    print("=" * 60)
    print("Step 2: Try alternative loop nests")
    print("=" * 60)
    results = []
    for order in ["bi_bj_bk", "bj_bi_bk"]:
        for TI, TJ in [(8, 4), (4, 8), (16, 4), (4, 16), (8, 8), (16, 2), (2, 16)]:
            ir = gen_v1(TI, TJ, order=order)
            if ir is None:
                continue
            raw_cost = score_16x16(ir)
            renamed_cost, _, info = score_with_renamer(ir)
            a_reads, b_reads = per_cell_read_summary(ir)
            results.append(("v1", order, TI, TJ, raw_cost, renamed_cost, ir))
            tag = " <-- BEST" if renamed_cost < BEST_PRIOR else ""
            uniform = (len(set(a_reads)) == 1 and len(set(b_reads)) == 1)
            print(f"  {order} TI={TI:>2} TJ={TJ:>2}: raw={raw_cost:>6} "
                  f"renamed={renamed_cost:>6}  A={set(a_reads)} B={set(b_reads)}{tag}")

    print()
    print("=" * 60)
    print("Step 3: Partial-A-cache (non-uniform A reads)")
    print("=" * 60)
    print("  pre-cache `n_cached_rows` rows of A; rest read from bulk")
    for n_cached in [0, 1, 2, 4, 8, 12, 16]:
        for TI, TJ in [(8, 4)]:
            try:
                ir = gen_partial_A_cache(TI, TJ, n_cached)
                raw_cost = score_16x16(ir)
                renamed_cost, _, _ = score_with_renamer(ir)
                a_reads, b_reads = per_cell_read_summary(ir)
                a_distrib = sorted(Counter(a_reads).items())
                results.append(("pA", "n=" + str(n_cached), TI, TJ, raw_cost, renamed_cost, ir))
                tag = " <-- BEST" if renamed_cost < BEST_PRIOR else ""
                print(f"  n={n_cached:>2} TI={TI} TJ={TJ}: raw={raw_cost:>6} "
                      f"renamed={renamed_cost:>6}  A_count_distrib={a_distrib}{tag}")
            except Exception as e:
                print(f"  n={n_cached} TI={TI} TJ={TJ}: ERR {e}")

    print()
    print("=" * 60)
    print("Step 4: Partial-B-cache (non-uniform B reads)")
    print("=" * 60)
    for n_cached in [0, 1, 2, 4, 8, 12, 16]:
        for TI, TJ in [(8, 4)]:
            try:
                ir = gen_partial_B_cache(TI, TJ, n_cached)
                raw_cost = score_16x16(ir)
                renamed_cost, _, _ = score_with_renamer(ir)
                a_reads, b_reads = per_cell_read_summary(ir)
                b_distrib = sorted(Counter(b_reads).items())
                results.append(("pB", "n=" + str(n_cached), TI, TJ, raw_cost, renamed_cost, ir))
                tag = " <-- BEST" if renamed_cost < BEST_PRIOR else ""
                print(f"  n={n_cached:>2} TI={TI} TJ={TJ}: raw={raw_cost:>6} "
                      f"renamed={renamed_cost:>6}  B_count_distrib={b_distrib}{tag}")
            except Exception as e:
                print(f"  n={n_cached} TI={TI} TJ={TJ}: ERR {e}")

    print()
    print("=" * 60)
    print("Summary — Top 8 by renamed cost:")
    print("=" * 60)
    results.sort(key=lambda r: r[5])
    for r in results[:8]:
        print(f"  {r[0]:>4} {str(r[1]):>10} TI={r[2]} TJ={r[3]}: "
              f"raw={r[4]:>6} renamed={r[5]:>6}")

    best = results[0]
    if best[5] < BEST_PRIOR:
        cost = best[5]
        # Re-rename to get final IR
        renamed_ir, _ = rename_optimal(best[6])
        out = os.path.join(HERE, "records", f"record_{cost}_lane0.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(renamed_ir + "\n")
        print(f"\nNEW RECORD: {best[0]} {best[1]} TI={best[2]} TJ={best[3]}  cost={cost}")
        log_event({"ts": int(time.time()), "type": "new_record",
                   "cost": cost, "prev": BEST_PRIOR, "lane": 0,
                   "file": f"record_{cost}_lane0.ir"})
    else:
        print(f"\nNo improvement: best={best[5]} vs prior {BEST_PRIOR}")

    log_token("exp_0_nonuniform_reads.py", 4500)


if __name__ == "__main__":
    main()
