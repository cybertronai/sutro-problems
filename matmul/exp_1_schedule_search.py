"""Lane 1 — schedule-space hill-climbing + structural perturbations.

Brief asks for SA / beam search over schedules with the address renamer as
inner solver. Key observation: the structure-preserving lower bound (LB) is
purely a function of per-cell read counts. Adjacent independent-instruction
swaps preserve every cell's read count, so they cannot move the LB. To
reduce cost, perturbations must change *which cells* hold intermediates or
how many times each cell is read.

This experiment runs three concrete searches:

  (S1) Swap-only hill-climb on sa_cache_16x16.ir — sanity check that the
       LB stays at 73602 regardless of permutation. Confirms the
       theoretical claim and saves wasted future search effort.

  (S2) "TMP-distribution" perturbation: split the single TMP cell into
       k cells (each holding partial-sums for a subset of muls). Doesn't
       change total read count, but spreads them across multiple low
       addresses. Within one sqrt-shell (addrs 1, 2-4, 5-9, ...) cost is
       constant, so this is mostly equivalent. Confirms or refutes.

  (S3) "Direct-to-C-tile" perturbation: accumulate directly into sC slots
       without TMP, using a 3-operand FMA-emulator: write the new product
       into a fresh cell, then sub-and-add via two ops. (Spoiler: Dally v0
       has no FMA; the cost of emulating it equals or exceeds TMP.)

  (S4) "Loop-block fusion" perturbation: run two adjacent (bi, bj=j) and
       (bi, bj=j+1) tiles together with a shared sB load. Saves nothing
       structurally (sB[jj] already loaded once per (bi,bj,bk)), and has
       to be carefully implemented to actually share — most naive fusions
       just reorder.

If any variant gives LB < 73602, save and report.
"""
from __future__ import annotations
import os, sys, time, json, random, math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from matmul import score_16x16  # noqa: E402
from addr_renamer import rename_optimal, count_reads  # noqa: E402


# -------------------- harness boilerplate --------------------

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


def lb_of_ir(ir):
    """Return structure-preserving lower bound."""
    _, info = rename_optimal(ir)
    return info["remapped_cost_estimate"]


# -------------------- IR parser/printer --------------------

def parse_ir(ir):
    lines = [ln.strip() for ln in ir.replace(";", "\n").splitlines() if ln.strip()]
    inputs = lines[0]
    outputs = lines[-1]
    body = []
    for ln in lines[1:-1]:
        head, _, rest = ln.partition(" ")
        ops = [int(x) for x in rest.split(",")]
        body.append((head, ops))
    return inputs, body, outputs


def emit_ir(inputs, body, outputs):
    out = [inputs]
    for op, ops in body:
        out.append(f"{op} {','.join(map(str, ops))}")
    out.append(outputs)
    return "\n".join(out)


# -------------------- S1: swap-only hill-climb --------------------

def reads_set(op, ops):
    """Return set of cells read by this instruction."""
    if op == "copy":
        # copy dest, src — reads src
        return {ops[1]}
    # add/sub/mul: 2- or 3-operand
    if len(ops) == 3:
        return {ops[1], ops[2]}
    return {ops[0], ops[1]}  # 2-operand: reads dest + src


def writes_set(op, ops):
    """Return set of cells written by this instruction."""
    return {ops[0]}


def can_swap(a, b):
    """Two adjacent instructions a, b can swap if no data hazard."""
    op_a, ops_a = a
    op_b, ops_b = b
    r_a = reads_set(op_a, ops_a)
    w_a = writes_set(op_a, ops_a)
    r_b = reads_set(op_b, ops_b)
    w_b = writes_set(op_b, ops_b)
    # RAW: b reads what a writes
    if w_a & r_b: return False
    # WAR: b writes what a reads
    if r_a & w_b: return False
    # WAW: both write the same cell
    if w_a & w_b: return False
    return True


def s1_swap_only_climb(ir, max_iters=100, seed=0):
    """Try random valid adjacent swaps; report LB after each."""
    random.seed(seed)
    inputs, body, outputs = parse_ir(ir)
    n = len(body)
    base_lb = lb_of_ir(emit_ir(inputs, body, outputs))
    print(f"  Initial LB: {base_lb}  ({n} instructions)")

    successes = 0
    samples = 0
    last_lb = base_lb
    for it in range(max_iters):
        i = random.randint(0, n - 2)
        if can_swap(body[i], body[i + 1]):
            samples += 1
            # Apply swap, check correctness via score, check LB.
            body[i], body[i + 1] = body[i + 1], body[i]
            new_ir = emit_ir(inputs, body, outputs)
            try:
                cost = score_16x16(new_ir)
            except Exception as e:
                # Revert; swap was unsafe (rare due to liveness subtleties).
                body[i], body[i + 1] = body[i + 1], body[i]
                continue
            new_lb = lb_of_ir(new_ir)
            if new_lb != last_lb:
                successes += 1
                print(f"  iter={it} swap@{i}: LB {last_lb} -> {new_lb}")
                last_lb = new_lb
            # Always keep the swap (for diversity in the random walk).
    print(f"  Swap samples: {samples}, LB-changing swaps: {successes}")
    print(f"  Final LB: {last_lb}")
    return last_lb


# -------------------- S2: TMP-distribution variants --------------------

def gen_sa_cache_with_tmp_split(k_tmp):
    """Same as sa_cache, but use k_tmp distinct TMP cells in round-robin
    instead of one. Total TMP reads = 3840 split into k_tmp bins of
    3840/k_tmp.

    Layout: sA at 1, TMPs at 2..1+k_tmp, sB at 2+k_tmp..1+k_tmp+TJ, ...
    """
    N = 16
    TI, TJ = 8, 4
    nbi = N // TI
    nbj = N // TJ

    SA = 1
    tmp_base = 2
    TMPs = [tmp_base + i for i in range(k_tmp)]
    sB_base = tmp_base + k_tmp
    sB = lambda jj: sB_base + jj
    sC_base = sB_base + TJ
    sC = lambda ii, jj: sC_base + ii * TJ + jj
    A_base = sC_base + TI * TJ
    B_base = A_base + N * N
    C_base = B_base + N * N
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [B(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]

    lines = [",".join(map(str, inputs))]
    counter = 0
    for bi in range(nbi):
        for bj in range(nbj):
            for bk in range(N):
                for jj in range(TJ):
                    lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
                for ii in range(TI):
                    lines.append(f"copy {SA},{A(bi * TI + ii, bk)}")
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {sC(ii, jj)},{SA},{sB(jj)}")
                        else:
                            t = TMPs[counter % k_tmp]
                            counter += 1
                            lines.append(f"mul {t},{SA},{sB(jj)}")
                            lines.append(f"add {sC(ii, jj)},{t}")
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# -------------------- S3: B-row caching for two adjacent bi tiles --------------------

def gen_sa_cache_double_bi():
    """Schedule reordered to (bj, bk, bi) so sB is shared across BOTH bi
    tiles for fixed (bj, bk). This halves B-bulk reads (each B[k,j] read
    once instead of twice). Trade-off: sC must hold both bi tiles
    simultaneously (64 cells instead of 32).

    Layout:
      addr 1: SA cache
      addr 2: TMP
      addrs 3..6: sB strip (TJ=4 cells)
      addrs 7..70: sC big tile (64 cells, holds both bi tiles)
      addrs 71..326: A bulk (256)
      addrs 327..582: B bulk (256)
      addrs 583..838: C bulk (256)
    """
    N = 16
    TI, TJ = 8, 4
    nbi = N // TI  # 2
    nbj = N // TJ  # 4

    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    sC_base = 3 + TJ
    # sC index: bi (2) × ii (TI) × jj (TJ) = 64 cells
    sC = lambda bi, ii, jj: sC_base + bi * (TI * TJ) + ii * TJ + jj
    A_base = sC_base + nbi * TI * TJ  # 71
    B_base = A_base + N * N           # 327
    C_base = B_base + N * N           # 583
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [B(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]

    lines = [",".join(map(str, inputs))]
    for bj in range(nbj):
        for bk in range(N):
            # Load sB once per (bj, bk) — shared across both bi tiles.
            for jj in range(TJ):
                lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
            for bi in range(nbi):
                for ii in range(TI):
                    lines.append(f"copy {SA},{A(bi * TI + ii, bk)}")
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {sC(bi, ii, jj)},{SA},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {sC(bi, ii, jj)},{TMP}")
        # Spill all bi tiles for this bj.
        for bi in range(nbi):
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(
                        f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(bi, ii, jj)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# -------------------- S4: A-row prefetch across adjacent ii --------------------

def gen_sa_cache_prefetch_ii():
    """Standard (bi, bj, bk) loop, but precompute TI A-elements into a
    cheap-addr A-strip BEFORE the bk loop, instead of reloading SA each
    inner iteration. Trade: A bulk reads × N (if loaded inside bk loop)
    vs strip-cache reads × TI*N (inside inner ii*jj).

    Layout (TI=8, TJ=4):
      addr 1: SA cache (single hot cell)
      addr 2: TMP
      addrs 3..6: sB strip (4)
      addrs 7..14: sA strip (8 cells, holds A[bi*TI+ii, bk] for all ii,
                  for fixed bk)
      addrs 15..46: sC tile (32)
      addrs 47..302: A bulk
      addrs 303..558: B bulk
      addrs 559..814: C bulk

    Read counts:
      sA strip: 8 cells × N (loaded once per bk) × ... per cell
        For each bk: load TI A's into strip. Then inner uses strip.
        Inner read of strip: copy SA, sA[ii] = TI reads per (bi, bj, bk)
        per cell? Actually sA[ii] is read once per (bi, bj, bk). Total
        strip reads per cell = nbi * nbj * N = 2 * 4 * 16 = 128.
      But SA cache is also added: 4096 reads.
      A bulk: each A[i,k] loaded once per (bi, bj, bk) where bi covers it
        = 1 per bj = 4 reads/cell. Same as before. NO IMPROVEMENT.

    This is structurally identical to current sa-cache but with an extra
    indirection. Skip — won't help.
    """
    return None  # Confirmed not implementing.


# -------------------- main --------------------

def main():
    stop_check()
    log_event({"ts": int(time.time()), "type": "exp_start",
               "lane": 1, "exp": "exp_1_schedule_search.py"})

    BEST_PRIOR = 73602
    sa_cache = open(os.path.join(HERE, "submissions/sa_cache_16x16.ir")).read()

    print("=" * 60)
    print("S1: swap-only hill-climb (sanity check)")
    print("=" * 60)
    s1_lb = s1_swap_only_climb(sa_cache, max_iters=200, seed=42)
    if s1_lb != BEST_PRIOR:
        print(f"  ANOMALY: swaps changed LB. Investigate.")

    print()
    print("=" * 60)
    print("S2: TMP-distribution variants (k=1, 2, 4, 8, 16)")
    print("=" * 60)
    s2_results = []
    for k in (1, 2, 4, 8, 16):
        ir = gen_sa_cache_with_tmp_split(k)
        try:
            cost = score_16x16(ir)
        except Exception as e:
            print(f"  k_tmp={k:2d} ERR: {e}")
            continue
        lb = lb_of_ir(ir)
        s2_results.append((cost, k, ir))
        print(f"  k_tmp={k:2d}  cost={cost:>7}  LB={lb}")

    print()
    print("=" * 60)
    print("S3: (bj, bk, bi) reorder with sB shared across both bi tiles")
    print("=" * 60)
    ir3 = gen_sa_cache_double_bi()
    try:
        cost3 = score_16x16(ir3)
        lb3 = lb_of_ir(ir3)
        print(f"  cost={cost3}  LB={lb3}")
    except Exception as e:
        print(f"  ERR: {e}")
        cost3 = lb3 = None
        ir3 = None

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Prior best: 73602")
    print(f"  S1 swap-only: {s1_lb} (expected 73602)")
    if s2_results:
        s2_best = min(s2_results, key=lambda r: r[0])
        print(f"  S2 best (TMP-split): k={s2_best[1]} cost={s2_best[0]}")
    if cost3 is not None:
        print(f"  S3 (bj,bk,bi) reorder: cost={cost3}")

    # Identify any improvements
    candidates = []
    if s2_results:
        for cost, k, ir in s2_results:
            if cost < BEST_PRIOR:
                candidates.append((cost, f"s2_tmp{k}", ir))
    if cost3 is not None and cost3 < BEST_PRIOR:
        candidates.append((cost3, "s3_double_bi", ir3))

    if candidates:
        candidates.sort()
        best = candidates[0]
        cost = best[0]
        out = os.path.join(HERE, "records", f"record_{cost}_lane1.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(best[2] + "\n")
        print(f"\n  NEW RECORD: {best[1]} cost={cost}")
        log_event({"ts": int(time.time()), "type": "new_record",
                   "cost": cost, "prev": BEST_PRIOR, "lane": 1,
                   "file": f"record_{cost}_lane1.ir"})
    else:
        print(f"\n  No improvement over 73602.")
        print(f"  Conjecture strengthened: 73602 is near-optimal for")
        print(f"  the sa-cache schedule family.")

    log_token("exp_1_schedule_search.py", 9000)


if __name__ == "__main__":
    main()
