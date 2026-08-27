"""Lane 0 experiment: alias C output addresses with A input addresses.

The scorer requires input addresses to be distinct, but output addresses can
overlap with input addresses (only need correct value at exit). Since A is
fully consumed before C is written for a given bi-row (when ordered carefully),
we can write C[bi,*] directly into A[bi,*] addresses.

Best case: A=C aliasing saves 3,452 from the bulk-read budget vs sa_cache_16x16.

We start with a 4x4 sanity test, then scale to 16x16.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matmul  # noqa: E402
from addr_renamer import rename_optimal  # noqa: E402


# --- STOP_SIGNAL handling ---
STOP_FILE = os.path.join(HERE, "STOP_SIGNAL_0")


def check_stop():
    if os.path.exists(STOP_FILE):
        try:
            os.remove(STOP_FILE)
        except OSError:
            pass
        print("STOP_SIGNAL — halting.")
        sys.exit(0)


# --- event/token logging helpers ---
def log_event(d):
    d.setdefault("ts", time.time())
    with open(os.path.join(HERE, "events.jsonl"), "a") as f:
        f.write(json.dumps(d) + "\n")


def log_tokens(exp, tokens):
    with open(os.path.join(HERE, "token_log.jsonl"), "a") as f:
        f.write(json.dumps({
            "ts": time.time(), "lane": 0, "exp": exp, "tokens": tokens
        }) + "\n")


# ---------------------------------------------------------------------------
# Generator: sa-cache schedule with C=A aliasing.
#
# Key observation: in the sa_cache schedule, A[bi*TI..bi*TI+TI-1, *] is read
# across (bj=0..nbj-1) iterations for a given bi. The last A read for that bi
# block happens during (bi, bj=nbj-1). So C[bi*TI..bi*TI+TI-1, *] cannot be
# written into A's addresses until bj=nbj-1's accumulator (sC) is ready.
#
# Solution A (simplest): keep schedule, just rename C addresses to A addresses
#   at the IR level. C[i,j] := A[i,j] (same shape). The C-write to A's address
#   happens at end of (bi, bj) iteration via `copy C(...),{sC(...)}`. For
#   bj < nbj-1, A[bi*TI..bi*TI+TI-1, *] is still being read in subsequent bj
#   iterations -- so we CANNOT overwrite A here. Aliasing only safe at last bj.
#
# Solution B: write C into a dedicated C-bulk for bj < nbj-1, but for bj=nbj-1
#   write into A's addresses. Saves only the bj=nbj-1 chunk of C.
#
# Solution C (this experiment): defer ALL C writeback until after each bi's
#   FULL bj loop. We accumulate sC across bj iterations? No - sC must be
#   per-bj because each (bi, bj) computes a different Tj-wide column block.
#
# Solution D: keep nbj-many sC blocks per bi (Ti×Tj × nbj cells in scratchpad).
#   For 16x16 with Ti=8, Tj=4, nbj=4: 8*4*4 = 128 cells of sC scratchpad.
#   That pushes A bulk out from 39 to ~135. Cost: A's reads at higher addrs
#   may exceed savings.
#
# Solution E (cleanest): change the schedule from (bi, bj, bk) to (bi, bk, bj).
#   For each bi: for each bk: for each bj: do partial mul. Now sC must be
#   the FULL Ti × N width (8 × 16 = 128 cells). Worse.
#
# Solution F (what we do here): the schedule (bi, bj, bk) has order:
#       for bi:
#         for bj:
#           for bk: ... (uses A[bi*TI:bi*TI+TI, bk])
#           writeback sC -> C[bi*TI:bi*TI+TI, bj*TJ:bj*TJ+TJ]
#   A[bi*TI..bi*TI+TI-1, *] is read across all bj. We need ALL bj to complete
#   before overwriting A[bi*TI..]. So we must defer the C writeback until
#   after ALL (bi, bj=*) blocks are done.
#
#   Since each (bi, bj) writeback is into DIFFERENT C cells (different bj),
#   we can't accumulate them in sC. Need separate sC per bj OR write C to a
#   temporary location and only copy to A's addrs after all bj done.
#
# ACTUALLY simpler: just delay the C writeback. Write final C values
# directly from sC into A addrs at the end of bi's full loop. So we keep
# nbj separate sC arrays.
#
# Math check: sC scratchpad becomes Ti × Tj × nbj = 8 × 4 × 4 = 128 cells.
# Currently sC is Ti × Tj = 32 cells (addrs 7..38). With 128 cells, sC
# occupies addrs 7..134. A bulk shifts from 39 to 135.
#
# That's bad - A is read 4× per cell, so 256 cells × 4 reads = 1024 reads
# now hit higher addresses. Hmm.
#
# Better: write each sC out to its OWN per-bj temporary slot in
# scratchpad just before we'd overwrite A. Then once all bj iterations of
# this bi are done, copy from those temp slots into A's addresses.
# That's the same cost.
#
# IDEA: only the bj=nbj-1 case can directly write to A's addrs (since A's
# reads for bj=nbj-1 happen before the C writeback of bj=nbj-1). For
# bj < nbj-1, write to bulk-C addrs as before.
#
# Simpler IDEA: the renamer treats reads only - so if we have C (output) at
# ADDRESSES that coincide with A's addresses, then those cells get
# (read-count of A) + 1 (output read) = 5 reads. The renamer's "highest
# count gets cheapest addrs" assignment will naturally place them at
# 39..294. But the key is: the simulator must still produce correct values.
# The simulator allows aliasing as long as at exit, mem[output_addr]
# contains the correct C value. Since the writes happen in program order,
# we just need each write to A's address to come AFTER the last A read of
# that cell.
# ---------------------------------------------------------------------------

def generate_4x4_baseline_alias():
    """4×4 sa-cache with TI=2, TJ=2, with delayed C writeback into A addrs."""
    return _generate_alias(N=4, TI=2, TJ=2)


def _generate_alias(N, TI, TJ):
    """sa-cache + alias C onto A addresses.

    Strategy: defer C writeback. Use nbj separate sC blocks (one per bj).
    After all bj iterations for a given bi finish (so A[bi*TI..bi*TI+TI-1, *]
    is fully consumed), copy each per-bj sC into the A addresses.
    """
    nbi = N // TI
    nbj = N // TJ

    SA = 1
    TMP = 2
    sB = lambda jj: 3 + jj
    # sC is now per-bj: shape (nbj, TI, TJ) flat -> nbj*TI*TJ cells
    sC_base = 3 + TJ
    sC = lambda bj, ii, jj: sC_base + bj * (TI * TJ) + ii * TJ + jj

    A_base = sC_base + nbj * TI * TJ
    B_base = A_base + N * N
    # No C bulk: aliased onto A
    A = lambda i, k: A_base + i * N + k
    B = lambda k, j: B_base + k * N + j
    # C[i,j] aliases A[i,j]
    C = lambda i, j: A(i, j)

    inputs = ([A(i, k) for i in range(N) for k in range(N)] +
              [B(k, j) for k in range(N) for j in range(N)])
    outputs = [C(i, j) for i in range(N) for j in range(N)]

    lines = [",".join(map(str, inputs))]
    for bi in range(nbi):
        for bj in range(nbj):
            for bk in range(N):
                for jj in range(TJ):
                    lines.append(f"copy {sB(jj)},{B(bk, bj * TJ + jj)}")
                for ii in range(TI):
                    lines.append(f"copy {SA},{A(bi * TI + ii, bk)}")
                    for jj in range(TJ):
                        if bk == 0:
                            lines.append(f"mul {sC(bj, ii, jj)},{SA},{sB(jj)}")
                        else:
                            lines.append(f"mul {TMP},{SA},{sB(jj)}")
                            lines.append(f"add {sC(bj, ii, jj)},{TMP}")
        # After ALL bj for this bi: A[bi*TI..bi*TI+TI-1, *] is fully consumed.
        # Now safe to overwrite A's addresses with C's values.
        for bj in range(nbj):
            for ii in range(TI):
                for jj in range(TJ):
                    lines.append(
                        f"copy {C(bi * TI + ii, bj * TJ + jj)},{sC(bj, ii, jj)}"
                    )
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def generate_16x16_alias():
    return _generate_alias(N=16, TI=8, TJ=4)


# --- main ---
if __name__ == "__main__":
    check_stop()
    log_event({"type": "exp_start", "lane": 0, "exp": "exp_0_alias_ca.py"})
    print(f"[{time.strftime('%H:%M:%S')}] starting exp_0_alias_ca")

    # Sanity: 4×4 first
    ir_4 = generate_4x4_baseline_alias()
    cost_4 = matmul.score_4x4(ir_4)
    print(f"  4x4 alias cost = {cost_4:,}")

    # Compare to baseline 4x4 (no alias)
    from submissions import sa_cache_16x16 as _stub  # noqa: F401  (just to ensure path)

    # 16x16 alias
    ir_16 = generate_16x16_alias()
    cost_16 = matmul.score_16x16(ir_16)
    print(f"  16x16 alias cost = {cost_16:,}  (baseline 73,602)")

    # Run renamer
    renamed_ir, info = rename_optimal(ir_16)
    cost_renamed = matmul.score_16x16(renamed_ir)
    print(f"  16x16 alias + renamer = {cost_renamed:,}  "
          f"(renamer says {info['remapped_cost_estimate']:,})")

    BEST = min(cost_16, cost_renamed)
    BEST_IR = renamed_ir if cost_renamed < cost_16 else ir_16

    if BEST < 73602:
        out = os.path.join(HERE, "records", f"record_{BEST}_lane0.ir")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(BEST_IR + "\n")
        print(f"  NEW RECORD! saved {out}")
        log_event({
            "type": "new_record", "cost": BEST, "prev": 73602,
            "lane": 0, "file": out,
        })
    else:
        print(f"  no improvement (best={BEST:,}, prev=73,602)")

    # rough token estimate (for token log)
    log_tokens("exp_0_alias_ca.py", 4500)
