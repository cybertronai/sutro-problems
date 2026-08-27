"""Lane 1 — outer-product schedule.

For each k in 0..N-1:
  load sA[i] = A[i, k]  for i in 0..N (16 reads from A bulk)
  load sB[j] = B[k, j]  for j in 0..N (16 reads from B bulk)
  for i, j: sC[i, j] += sA[i] * sB[j]  (256 mults, accumulation reads sC)
After all k: spill sC -> C bulk.

Total inner-reads vs V1:
  V1 sa_cache total reads = 17920.
  Outer-product total reads = (sA: 4096) + (sB: 4096) + (sC: 4352)
                            + (TMP: 3840) + (A-bulk: 256) + (B-bulk: 256)
                            + (C exit: 256)
                            = 17152, lower than 17920.

But sC has 256 cells (vs V1's 32), pushing bulk to addrs ≥ 290. We do need
TMP+sA+sB+sC = 1+16+16+256 = 289 scratchpad cells before bulk starts.
A bulk at addrs 290..545; B bulk 546..801; C bulk 802..1057.

Cost estimate:
  TMP at addr 1: 3840*1 = 3840
  sA at addrs 2..17: 256 reads/cell × cost(2..17). cost(2)=2, cost(3,4)=2,
    cost(5..9)=3, cost(10..16)=4, cost(17)=5. Sum cost(2..17) = 2+2+2+
    3+3+3+3+3+4+4+4+4+4+4+4+5 = 54. sA cost ~ 256*54 = 13824. Hmm bad.

Actually 16 sA cells at addrs 2..17 is bad. Let's reorganize: put sC first
since it's read 17 times each but there are 256 of them; per-cell cost
penalty grows fast.

Optimal: sort by reads desc.
  TMP: 3840 reads
  sA[i], sB[j]: 256 reads each (32 cells total)
  sC[i,j]: 17 reads each (256 cells)
  A,B bulk: 1 read each
  C bulk: 1 exit read

So order: TMP, then 32 cells of sA+sB, then 256 sC cells. Total
scratchpad before bulk = 1+32+256 = 289.

Let's just generate it and let renamer sort it out.
"""
from __future__ import annotations
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from matmul import score_16x16  # noqa: E402

sys.path.insert(0, HERE)
from addr_renamer import rename_optimal  # noqa: E402

N = 16


def gen_outer_product() -> str:
    """Pure outer-product over k."""
    # Initial layout (will be re-optimized by renamer):
    TMP = 1
    sA = lambda i: 2 + i           # 2..17
    sB = lambda j: 2 + N + j       # 18..33
    sC_base = 2 + 2 * N            # 34
    sC = lambda i, j: sC_base + i * N + j  # 34..289
    A_base = sC_base + N * N       # 290
    B_base = A_base + N * N        # 546
    C_base = B_base + N * N        # 802
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j
    C = lambda i, j: C_base + i * N + j

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [B(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]

    lines = [",".join(map(str, inputs))]
    for k in range(N):
        # Load sA[*] = A[*, k]
        for i in range(N):
            lines.append(f"copy {sA(i)},{A(i, k)}")
        # Load sB[*] = B[k, *]
        for j in range(N):
            lines.append(f"copy {sB(j)},{B(k, j)}")
        # Outer product
        for i in range(N):
            for j in range(N):
                if k == 0:
                    lines.append(f"mul {sC(i, j)},{sA(i)},{sB(j)}")
                else:
                    lines.append(f"mul {TMP},{sA(i)},{sB(j)}")
                    lines.append(f"add {sC(i, j)},{TMP}")
    # Spill sC -> C bulk
    for i in range(N):
        for j in range(N):
            lines.append(f"copy {C(i, j)},{sC(i, j)}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def gen_outer_product_no_tmp() -> str:
    """Outer-product but use sC as accumulator directly (no TMP).

    This means sC is read 16+1 = 17 times: 1 init mul + 15 add + 1 spill.
    Wait, init mul writes (no read). The 15 subsequent mul-add: each does
    `mul tmp,sA,sB; add sC,tmp` (reads tmp twice for the add — read +
    written). Without tmp: `add sC, sA*sB` is not a single op; we'd have
    to fuse into mul-add but the IR has no fma.

    So we MUST use tmp. This variant just confirms.
    """
    return gen_outer_product()


def gen_outer_inplace_alt() -> str:
    """Variant: at k==0 we mul directly into sC (no read of sC). For k>0 we
    use tmp for the multiply then add into sC. Same as gen_outer_product."""
    return gen_outer_product()


def stop_check():
    sig = os.path.join(HERE, "STOP_SIGNAL_1")
    if os.path.exists(sig):
        os.remove(sig)
        print("STOP_SIGNAL — halting.")
        sys.exit(0)


def log_event(event: dict):
    path = os.path.join(HERE, "events.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


def log_token(filename: str, tokens: int):
    path = os.path.join(HERE, "token_log.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps({"ts": int(time.time()), "lane": 1,
                            "exp": filename, "tokens": tokens}) + "\n")


def safe_score(ir):
    try:
        return score_16x16(ir)
    except Exception as e:
        return f"ERR:{e}"


def main():
    stop_check()
    fname = "exp_1_outer_prod.py"
    log_event({"ts": int(time.time()), "type": "exp_start",
               "lane": 1, "exp": fname})

    BEST_PRIOR = 73602

    ir = gen_outer_product()
    cost = safe_score(ir)
    print(f"outer_product raw: {cost!r}")
    try:
        ir_lb, info = rename_optimal(ir)
        cost_lb = safe_score(ir_lb)
        print(f"outer_product renamed: {cost_lb!r}")
        # Show top reads
        from lower_bound import analyze
        r = analyze(ir)
        print(f"  total reads: {sum(r['reads'].values())}")
        print(f"  top 10 reads:")
        for cell, n in sorted(r['reads'].items(), key=lambda x: -x[1])[:10]:
            print(f"    cell={cell:4d}  reads={n}")
    except Exception as e:
        cost_lb = f"ERR:{e}"
        ir_lb = ir

    if isinstance(cost_lb, int) and cost_lb < BEST_PRIOR:
        out = os.path.join(HERE, "records", f"record_{cost_lb}_lane1.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(ir_lb + "\n")
        print(f"\nNEW RECORD: outer_product cost={cost_lb} -> {out}")
        log_event({"ts": int(time.time()), "type": "new_record",
                   "cost": cost_lb, "prev": BEST_PRIOR, "lane": 1,
                   "file": f"record_{cost_lb}_lane1.ir"})
    elif isinstance(cost, int) and cost < BEST_PRIOR:
        out = os.path.join(HERE, "records", f"record_{cost}_lane1.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(ir + "\n")
        print(f"\nNEW RECORD: outer_product (raw) cost={cost} -> {out}")
        log_event({"ts": int(time.time()), "type": "new_record",
                   "cost": cost, "prev": BEST_PRIOR, "lane": 1,
                   "file": f"record_{cost}_lane1.ir"})
    else:
        print(f"\nNo improvement: best cost {min(c for c in [cost, cost_lb] if isinstance(c, int))} >= 73602")

    log_token(fname, 5000)


if __name__ == "__main__":
    main()
