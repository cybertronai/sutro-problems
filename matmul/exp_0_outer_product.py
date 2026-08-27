"""Outer-product accumulation: no sC scratchpad; accumulate into C directly.

For each k in 0..15:
    For each i in 0..15:
        copy SA, A[i,k]
        For each j in 0..15:
            mul tmp, SA, B[k,j]
            if k == 0: copy C[i,j], tmp        # init
            else:      add C[i,j], tmp          # accumulate

This avoids the 32-cell sC region of sa-cache, but does more A loads
(SA reloaded per (i, k) = 256 reloads vs sa-cache's 8 reloads per (bk)
within Tj-strip × nbi×nbj outer = 8×4×2×16 = 1024 reloads).
"""
from __future__ import annotations

import os
import sys
import time
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from matmul import score_16x16  # noqa: E402
from addr_renamer import rename_optimal  # noqa: E402

N = 16

A_BASE = 100_000
B_BASE = 110_000
C_BASE = 120_000

SA = 1_000
TMP = 1_001


def A_at(i, j): return A_BASE + i * N + j
def B_at(i, j): return B_BASE + i * N + j
def C_at(i, j): return C_BASE + i * N + j


def build_ir() -> str:
    inputs = ([A_at(i, j) for i in range(N) for j in range(N)] +
              [B_at(i, j) for i in range(N) for j in range(N)])
    outputs = [C_at(i, j) for i in range(N) for j in range(N)]

    lines = [",".join(map(str, inputs))]

    for k in range(N):
        for i in range(N):
            lines.append(f"copy {SA},{A_at(i, k)}")
            for j in range(N):
                if k == 0:
                    lines.append(f"mul {C_at(i, j)},{SA},{B_at(k, j)}")
                else:
                    lines.append(f"mul {TMP},{SA},{B_at(k, j)}")
                    lines.append(f"add {C_at(i, j)},{TMP}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def build_ir_with_sB(NB) -> str:
    """Variant: cache B[k, j..j+NB-1] in sB before the j loop."""
    inputs = ([A_at(i, j) for i in range(N) for j in range(N)] +
              [B_at(i, j) for i in range(N) for j in range(N)])
    outputs = [C_at(i, j) for i in range(N) for j in range(N)]

    SB = lambda jj: 1_002 + jj
    lines = [",".join(map(str, inputs))]

    nbj = N // NB
    for bj in range(nbj):
        for k in range(N):
            for jj in range(NB):
                lines.append(f"copy {SB(jj)},{B_at(k, bj * NB + jj)}")
            for i in range(N):
                lines.append(f"copy {SA},{A_at(i, k)}")
                for jj in range(NB):
                    j = bj * NB + jj
                    if k == 0:
                        lines.append(f"mul {C_at(i, j)},{SA},{SB(jj)}")
                    else:
                        lines.append(f"mul {TMP},{SA},{SB(jj)}")
                        lines.append(f"add {C_at(i, j)},{TMP}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


if __name__ == "__main__":
    stop_path = os.path.join(HERE, "STOP_SIGNAL_0")
    if os.path.exists(stop_path):
        os.remove(stop_path)
        print("STOP_SIGNAL — halting.")
        sys.exit(0)

    with open(os.path.join(HERE, "events.jsonl"), "a") as f:
        f.write(json.dumps({
            "ts": int(time.time()),
            "type": "exp_start",
            "lane": 0,
            "exp": os.path.basename(__file__),
        }) + "\n")

    PREV_RECORD = 73602
    best_cost = None
    best_ir = None
    best_label = None

    # Plain outer product (no sB cache)
    ir = build_ir()
    new_ir, _ = rename_optimal(ir)
    cost = score_16x16(new_ir)
    print(f"  plain outer product (k,i,j): {cost:,}")
    if best_cost is None or cost < best_cost:
        best_cost, best_ir, best_label = cost, new_ir, "plain"

    # With sB cache for various NB sizes
    for NB in [2, 4, 8, 16]:
        ir = build_ir_with_sB(NB)
        new_ir, _ = rename_optimal(ir)
        cost = score_16x16(new_ir)
        print(f"  outer + sB cache NB={NB}: {cost:,}")
        if cost < best_cost:
            best_cost, best_ir, best_label = cost, new_ir, f"sB-NB={NB}"

    print(f"\nBest: {best_label} = {best_cost:,}")

    if best_cost is not None and best_cost < PREV_RECORD:
        out_path = os.path.join(HERE, "records", f"record_{best_cost}_lane0.ir")
        with open(out_path, "w") as f:
            f.write(best_ir + "\n")
        print(f"NEW RECORD: {best_cost} (saved to {out_path})")
        with open(os.path.join(HERE, "events.jsonl"), "a") as f:
            f.write(json.dumps({
                "ts": int(time.time()),
                "type": "new_record",
                "cost": best_cost,
                "prev": PREV_RECORD,
                "lane": 0,
                "file": os.path.basename(__file__),
            }) + "\n")
    else:
        print(f"no new record (current best {PREV_RECORD})")
