"""GEMM schedule search harness: enumerate accumulation schedules for the
4x4 Dally-model matmul competition, targeting the ~474 cost floor.

Floor decomposition (from structural analysis of the 681 record):
  - mul inputs: 128 reads, perfectly aliased best case = 328
  - add operands: 96 adds' reads at the cheapest live addresses = 96
  - outputs: 16 cells at the cheapest addresses = 50
  Total theoretical floor ~474; the record (681) shows a 207 gap that
  lives in add placement (actual 197 vs floor 96) and staging.

This harness enumerates schedules over three concrete dimensions:
  1. accumulation tree shape per output (left-linear, right-linear,
     balanced)
  2. product-to-add assignment order (which product enters the chain
     first: cell-reuse pressure depends on it)
  3. dead-cell address recycling (temporary operands consumed by an add
     return to the pool for later operations)

Scoring uses the exact competition cost function (read cost
ceil(sqrt(addr)); the accumulator 2-operand form reads dst as first
source). The best-found schedule is symbolically verified by
matmul.score_4x4 before its competition-format IR is written.

Run from the repository root:  python3 matmul/schedule_search.py [trials]
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field

try:
    from . import score_4x4
except ImportError:  # Direct execution: python3 matmul/schedule_search.py
    from matmul import score_4x4

K = 4
N_IN = 2 * K * K  # 32 inputs
INPUT_A = 1
INPUT_B = 1 + K * K


def read_cost(addr: int) -> int:
    return math.isqrt(addr - 1) + 1


@dataclass
class CellPool:
    """Address allocator that recycles dead temporary cells."""
    next_addr: int = N_IN + 1
    free: list[int] = field(default_factory=list)
    peak: int = N_IN

    def alloc(self) -> int:
        if self.free:
            return self.free.pop()
        a = self.next_addr
        self.next_addr += 1
        self.peak = max(self.peak, a)
        return a

    def release(self, addr: int) -> None:
        self.free.append(addr)


def build_schedule(
    tree: str,
    order: str,
    recycle: bool,
    rng: random.Random,
) -> tuple[list[tuple], list[int]]:
    """Emit (ops, outputs) where ops are (opcode, dst, [srcs...])."""
    if tree not in {"left", "right", "balanced"}:
        raise ValueError(f"unknown tree shape: {tree!r}")
    if order not in {"batched", "shuffled", "pipelined"}:
        raise ValueError(f"unknown assignment order: {order!r}")
    if order == "pipelined" and tree != "left":
        raise ValueError("pipelined order uses a left-linear tree")

    pool = CellPool()
    ops: list[tuple[str, int, list[int]]] = []

    # interleave: for each output cell, produce K products then reduce.
    # order controls whether all products for one output are made
    # before its adds (batched) or products+adds interleave across
    # outputs (pipelined = fewer live products).
    outputs: list[int] = []

    def emit_add(left: int, right: int) -> int:
        nd = pool.alloc()
        ops.append(("add", nd, [left, right]))
        if recycle:
            # Allocate the destination first so it cannot alias a source
            # before the operation has consumed both operands. The pool is
            # LIFO, so release the accumulator-side operand last for prompt
            # reuse while leaving other cheap cells available to later muls.
            pool.release(right)
            pool.release(left)
        return nd

    if order == "pipelined":
        # round-robin over outputs: each round makes one product per
        # output, accumulating immediately -> max K live products
        accs = [0] * (K * K)
        # first product per output: direct into accumulator cell
        for i in range(K):
            for j in range(K):
                a = INPUT_A + i * K
                b = INPUT_B + j
                dst = pool.alloc()
                ops.append(("mul", dst, [a, b]))
                accs[i * K + j] = dst
        for step in range(1, K):
            for i in range(K):
                for j in range(K):
                    a = INPUT_A + i * K + step
                    b = INPUT_B + step * K + j
                    prod = pool.alloc()
                    ops.append(("mul", prod, [a, b]))
                    old = accs[i * K + j]
                    accs[i * K + j] = emit_add(old, prod)
        outputs = accs
    else:
        # batched: all products for one output, then its add tree
        for i in range(K):
            for j in range(K):
                prods = []
                for k in range(K):
                    a = INPUT_A + i * K + k
                    b = INPUT_B + k * K + j
                    dst = pool.alloc()
                    ops.append(("mul", dst, [a, b]))
                    prods.append(dst)
                if order == "shuffled":
                    rng.shuffle(prods)
                if tree == "left":
                    acc = prods[0]
                    rest = prods[1:]
                elif tree == "right":
                    acc = prods[-1]
                    rest = list(reversed(prods[:-1]))
                else:  # balanced
                    cur = prods
                    while len(cur) > 1:
                        nxt = []
                        for x in range(0, len(cur) - 1, 2):
                            nxt.append(emit_add(cur[x], cur[x + 1]))
                        if len(cur) % 2:
                            nxt.append(cur[-1])
                        cur = nxt
                    acc = cur[0]
                    rest = []
                for p in rest:
                    acc = emit_add(acc, p)
                outputs.append(acc)

    return ops, outputs


def score(ops, outputs) -> int:
    cost = 0
    for _, dst, srcs in ops:
        cost += sum(read_cost(s) for s in srcs)
    cost += sum(read_cost(o) for o in outputs)
    return cost


def to_ir(ops, outputs) -> str:
    lines = [",".join(str(INPUT_A + i) for i in range(K * K))
             + "," + ",".join(str(INPUT_B + i) for i in range(K * K))]
    for opcode, dst, srcs in ops:
        lines.append(f"{opcode} {dst},{','.join(map(str, srcs))}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines) + "\n"


def main() -> None:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    rng = random.Random(2026_0901)
    best = None
    configurations = [
        (tree, order, recycle)
        for tree in ("left", "right", "balanced")
        for order in ("batched", "shuffled")
        for recycle in (False, True)
    ]
    configurations.extend(("left", "pipelined", recycle)
                          for recycle in (False, True))
    attempts = max(1, trials // len(configurations))
    for tree, order, recycle in configurations:
        for _ in range(attempts):
            ops, outs = build_schedule(tree, order, recycle, rng)
            c = score(ops, outs)
            if best is None or c < best[0]:
                best = (c, tree, order, recycle, ops, outs)
                print(f"new best {c} (tree={tree} order={order} recycle={recycle})")
    c, tree, order, recycle, ops, outs = best
    ir = to_ir(ops, outs)
    verified_cost = score_4x4(ir)
    if verified_cost != c:
        raise RuntimeError(
            f"cost mismatch: search reported {c}, scorer returned {verified_cost}")

    print(f"\nfinal best: {verified_cost}  tree={tree} order={order} recycle={recycle}")
    print(f"ops: {len(ops)}")
    with open("best_schedule.ir", "w") as f:
        f.write(ir)
    print("symbolically verified IR written to best_schedule.ir")


if __name__ == "__main__":
    main()
