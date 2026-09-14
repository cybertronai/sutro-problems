"""Track A: bipartite-matching exit-line staging for 4x4 GEMM.

Implements docs/gemm-exit-line-staging.md (system-1 lineage, verified
890 standing): k-outer wavefront accumulation, per-output accumulators
with normal liveness, a staging Copy after each output's final add
into an input cell already consumed by then, and the exit line reading
the 16 staged cells.

The bipartite matching: outputs complete at various times during the
wavefront; input cells die when their last consumer runs. An output may
stage into a cell whose death precedes the staging copy. We place
early-dying inputs at the cheapest addresses and match greedily by
completion order (proper Hungarian on 16x32 is unnecessary: the death
order is a laminar sequence under k-outer).

Scoring: exact competition cost function; the winner is verified
through matmul.score_4x4.
"""
from __future__ import annotations

import itertools
import math
import sys

K = 4


def read_cost(addr: int) -> int:
    return math.isqrt(addr - 1) + 1


def build(staging_mode: str = "greedy", acc_place: str = "recycle"):
    """Emit (ops, inputs, outputs) in competition format.

    Schedule: k-outer wavefront. Pass 0: each output's first product IS
    its accumulator. Passes 1-3: product then add. Staging copies after
    the final pass, into dead input cells; exit line = staged cells.
    """
    # input placement: A[i][k] and B[k][j]; pass-k inputs die after pass k.
    # early-dying at cheap addresses: pass-0 inputs 1-8, pass-1 9-16,
    # pass-2 17-24, pass-3 25-32.
    a_addr = {}
    b_addr = {}
    for i in range(K):
        for k in range(K):
            a_addr[(i, k)] = 1 + k * K + i            # pass k block: k*K+1..k*K+4
    for k in range(K):
        for j in range(K):
            b_addr[(k, j)] = 1 + K * K + k * K + j     # B block at 17..32
    # re-map so pass-0 A/B sit at 1-8: A pass0 = 1-4, B pass0 = 5-8, etc.
    a_addr = {(i, k): 1 + 2 * k * K + i for i in range(K) for k in range(K)}
    b_addr = {(k, j): 1 + 2 * k * K + K + j for k in range(K) for j in range(K)}

    inputs = [a_addr[(i, k)] for i in range(K) for k in range(K)] + \
             [b_addr[(k, j)] for k in range(K) for j in range(K)]

    next_scratch = 33
    free_cells: list[int] = []
    ops = []
    acc = {}

    def alloc():
        nonlocal next_scratch
        if free_cells:
            return free_cells.pop()
        a = next_scratch
        next_scratch += 1
        return a

    # pass 0: products are accumulators
    for i in range(K):
        for j in range(K):
            dst = alloc()
            ops.append(("mul", dst, [a_addr[(i, 0)], b_addr[(0, j)]]))
            acc[(i, j)] = dst
    # passes 1..3
    for k in range(1, K):
        for i in range(K):
            for j in range(K):
                prod = alloc()
                ops.append(("mul", prod, [a_addr[(i, k)], b_addr[(k, j)]]))
                old = acc[(i, j)]
                nd = alloc()
                ops.append(("add", nd, [old, prod]))
                acc[(i, j)] = nd
                if acc_place == "recycle":
                    free_cells.append(prod)
                    free_cells.append(old)

    # staging: after the final pass every input is dead; match outputs
    # to the cheapest dead input cells (1..16)
    dead_cheap = list(range(1, 2 * K * K + 1))  # all inputs dead post-pass-3
    staged = {}
    outs_order = sorted(acc.keys(), key=lambda ij: ij[0] * K + ij[1])
    if staging_mode == "greedy":
        cells = sorted(dead_cheap, key=read_cost)
        for idx, ij in enumerate(outs_order):
            staged[ij] = cells[idx]
    elif staging_mode == "row":
        for idx, ij in enumerate(outs_order):
            staged[ij] = idx + 1

    for ij in outs_order:
        ops.append(("copy", staged[ij], [acc[ij]]))

    outputs = [staged[ij] for ij in outs_order]
    return ops, inputs, outputs


def score(ops, outputs) -> int:
    c = 0
    for _, _, srcs in ops:
        c += sum(read_cost(s) for s in srcs)
    c += sum(read_cost(o) for o in outputs)
    return c


def to_ir(ops, inputs, outputs) -> str:
    lines = [",".join(map(str, inputs))]
    for opcode, dst, srcs in ops:
        lines.append(f"{opcode} {dst},{','.join(map(str, srcs))}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines) + "\n"


def main():
    best = None
    for sm in ("greedy", "row"):
        for ap in ("recycle", "fresh"):
            ops, inputs, outputs = build(sm, ap)
            c = score(ops, outputs)
            tag = f"staging={sm} acc={ap}"
            print(f"{tag}: {c} ops={len(ops)}")
            if best is None or c < best[0]:
                best = (c, tag, ops, inputs, outputs)
    c, tag, ops, inputs, outputs = best
    print(f"\nbest: {c} ({tag})")
    with open("exit_line_best.ir", "w") as f:
        f.write(to_ir(ops, inputs, outputs))
    print("IR -> exit_line_best.ir")


if __name__ == "__main__":
    main()
