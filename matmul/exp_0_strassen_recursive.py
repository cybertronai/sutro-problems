"""Recursive Strassen for 16x16 matmul, parameterized by base-case size.

Generates IR for C = A @ B by applying Strassen's 2x2 block algorithm
recursively until the block size hits ``base_n``, then computing each
remaining base_n x base_n product with a flat naive matmul kernel
(outer-product, sa-cache style).

Sweep depths:
   depth=1  base 8x8    (= strassen-outer-2x2; expected ~100k)
   depth=2  base 4x4    (49 base mults)
   depth=3  base 2x2    (343 base mults)
   depth=4  base 1x1    (2401 base mults)

Each derived (n/2)x(n/2) operand at every recursion level needs its own
addresses, so cell counts grow as ~10 * (1 + 4 + 16 + ... + 4^(d-1)) per
level — exponential. After building the IR we run rename_optimal so the
final cost reflects the optimal address permutation.
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


# ---------------------------------------------------------------------------
# Address allocator: hand out unique fresh addresses on demand.
# Final pass renames everything optimally so absolute values are irrelevant
# as long as they don't collide with input/output regions.
# ---------------------------------------------------------------------------

class AddrAlloc:
    def __init__(self, start: int):
        self.next = start

    def alloc(self, count: int) -> int:
        base = self.next
        self.next += count
        return base


# Reserve fixed regions for inputs/outputs; everything else is allocated.
A_BASE = 100_000
B_BASE = 110_000
C_BASE = 120_000
WORK_BASE = 200_000


def A_at(i, j): return A_BASE + i * N + j
def B_at(i, j): return B_BASE + i * N + j
def C_at(i, j): return C_BASE + i * N + j


# ---------------------------------------------------------------------------
# Base-case kernel: naive n×n matmul writing into dest cells.
# Uses outer-product loop ``(k, i, j)`` so we only cache one A-cell at
# a time (matches outer_product_4x4.py which scores 800 on 4x4).
# ---------------------------------------------------------------------------

def emit_base_matmul(lines, n, A_fn, B_fn, C_fn, alloc: AddrAlloc):
    """Emit naive n×n matmul C = A @ B (outer-product order).

    A_fn(i,j), B_fn(i,j), C_fn(i,j) -> address.
    """
    if n == 1:
        # Single mul instruction
        lines.append(f"mul {C_fn(0, 0)},{A_fn(0, 0)},{B_fn(0, 0)}")
        return

    SA = alloc.alloc(1)
    TMP = alloc.alloc(1)
    sB = alloc.alloc(n)        # n cells

    for k in range(n):
        for j in range(n):
            lines.append(f"copy {sB + j},{B_fn(k, j)}")
        for i in range(n):
            lines.append(f"copy {SA},{A_fn(i, k)}")
            for j in range(n):
                if k == 0:
                    lines.append(f"mul {C_fn(i, j)},{SA},{sB + j}")
                else:
                    lines.append(f"mul {TMP},{SA},{sB + j}")
                    lines.append(f"add {C_fn(i, j)},{TMP}")


# ---------------------------------------------------------------------------
# Recursive Strassen kernel
# ---------------------------------------------------------------------------

def emit_combo(lines, n, dest_fn, lhs_fn, rhs_fn, op):
    """Element-wise n×n combo: dest[i,j] = lhs[i,j] op rhs[i,j]."""
    for ii in range(n):
        for jj in range(n):
            lines.append(
                f"{op} {dest_fn(ii, jj)},{lhs_fn(ii, jj)},{rhs_fn(ii, jj)}"
            )


def emit_strassen(lines, n, A_fn, B_fn, C_fn, alloc: AddrAlloc, base_n: int):
    """Emit Strassen-recursive C = A @ B for n×n with n a power of 2.

    When n == base_n, fall through to a naive base-case matmul.
    Otherwise, perform one level of Strassen splitting into halves.
    """
    if n <= base_n:
        emit_base_matmul(lines, n, A_fn, B_fn, C_fn, alloc)
        return

    h = n // 2

    # Sub-block accessors (no extra storage; just shifted indices into A/B/C).
    def A11(i, j): return A_fn(i, j)
    def A12(i, j): return A_fn(i, h + j)
    def A21(i, j): return A_fn(h + i, j)
    def A22(i, j): return A_fn(h + i, h + j)

    def B11(i, j): return B_fn(i, j)
    def B12(i, j): return B_fn(i, h + j)
    def B21(i, j): return B_fn(h + i, j)
    def B22(i, j): return B_fn(h + i, h + j)

    def C11(i, j): return C_fn(i, j)
    def C12(i, j): return C_fn(i, h + j)
    def C21(i, j): return C_fn(h + i, j)
    def C22(i, j): return C_fn(h + i, h + j)

    # Allocate 10 derived h×h operands (5 from A, 5 from B) and 7 result tiles.
    cells = h * h

    def make_block(base):
        return lambda i, j: base + i * h + j

    der_A11pA22 = make_block(alloc.alloc(cells))
    der_A21pA22 = make_block(alloc.alloc(cells))
    der_A11pA12 = make_block(alloc.alloc(cells))
    der_A21mA11 = make_block(alloc.alloc(cells))
    der_A12mA22 = make_block(alloc.alloc(cells))

    der_B11pB22 = make_block(alloc.alloc(cells))
    der_B12mB22 = make_block(alloc.alloc(cells))
    der_B21mB11 = make_block(alloc.alloc(cells))
    der_B11pB12 = make_block(alloc.alloc(cells))
    der_B21pB22 = make_block(alloc.alloc(cells))

    M1 = make_block(alloc.alloc(cells))
    M2 = make_block(alloc.alloc(cells))
    M3 = make_block(alloc.alloc(cells))
    M4 = make_block(alloc.alloc(cells))
    M5 = make_block(alloc.alloc(cells))
    M6 = make_block(alloc.alloc(cells))
    M7 = make_block(alloc.alloc(cells))

    # Phase 1: derive 10 combos
    emit_combo(lines, h, der_A11pA22, A11, A22, "add")
    emit_combo(lines, h, der_A21pA22, A21, A22, "add")
    emit_combo(lines, h, der_A11pA12, A11, A12, "add")
    emit_combo(lines, h, der_A21mA11, A21, A11, "sub")
    emit_combo(lines, h, der_A12mA22, A12, A22, "sub")
    emit_combo(lines, h, der_B11pB22, B11, B22, "add")
    emit_combo(lines, h, der_B12mB22, B12, B22, "sub")
    emit_combo(lines, h, der_B21mB11, B21, B11, "sub")
    emit_combo(lines, h, der_B11pB12, B11, B12, "add")
    emit_combo(lines, h, der_B21pB22, B21, B22, "add")

    # Phase 2: 7 sub-products via recursion
    emit_strassen(lines, h, der_A11pA22, der_B11pB22, M1, alloc, base_n)
    emit_strassen(lines, h, der_A21pA22, B11,         M2, alloc, base_n)
    emit_strassen(lines, h, A11,         der_B12mB22, M3, alloc, base_n)
    emit_strassen(lines, h, A22,         der_B21mB11, M4, alloc, base_n)
    emit_strassen(lines, h, der_A11pA12, B22,         M5, alloc, base_n)
    emit_strassen(lines, h, der_A21mA11, der_B11pB12, M6, alloc, base_n)
    emit_strassen(lines, h, der_A12mA22, der_B21pB22, M7, alloc, base_n)

    # Phase 3: combine into C blocks
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    for ii in range(h):
        for jj in range(h):
            c = C11(ii, jj)
            lines.append(f"add {c},{M1(ii, jj)},{M4(ii, jj)}")
            lines.append(f"sub {c},{c},{M5(ii, jj)}")
            lines.append(f"add {c},{c},{M7(ii, jj)}")

            c = C12(ii, jj)
            lines.append(f"add {c},{M3(ii, jj)},{M5(ii, jj)}")

            c = C21(ii, jj)
            lines.append(f"add {c},{M2(ii, jj)},{M4(ii, jj)}")

            c = C22(ii, jj)
            lines.append(f"sub {c},{M1(ii, jj)},{M2(ii, jj)}")
            lines.append(f"add {c},{c},{M3(ii, jj)}")
            lines.append(f"add {c},{c},{M6(ii, jj)}")


# ---------------------------------------------------------------------------
# Build the full 16×16 IR for a given base_n.
# ---------------------------------------------------------------------------

def build_ir(base_n: int) -> str:
    inputs = ([A_at(i, j) for i in range(N) for j in range(N)] +
              [B_at(i, j) for i in range(N) for j in range(N)])
    outputs = [C_at(i, j) for i in range(N) for j in range(N)]

    lines = [",".join(map(str, inputs))]
    alloc = AddrAlloc(WORK_BASE)
    emit_strassen(lines, N, A_at, B_at, C_at, alloc, base_n)
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    results = {}

    for base_n in (8, 4, 2, 1):
        depth = {8: 1, 4: 2, 2: 3, 1: 4}[base_n]
        print(f"\n=== base_n={base_n}  depth={depth} ===")
        ir = build_ir(base_n)
        n_ops = len(ir.splitlines()) - 2
        print(f"  raw IR ops: {n_ops}")
        raw_cost = score_16x16(ir)
        print(f"  raw cost (no rename): {raw_cost:,}")

        new_ir, info = rename_optimal(ir)
        new_cost = score_16x16(new_ir)
        print(f"  renamed cost        : {new_cost:,}")
        print(f"  cells used (post-rename): {len(info['reads'])}")

        results[base_n] = {
            "depth": depth,
            "ops": n_ops,
            "raw_cost": raw_cost,
            "renamed_cost": new_cost,
            "cells": len(info["reads"]),
        }

        if new_cost < PREV_RECORD:
            out_path = os.path.join(HERE, "records", f"record_{new_cost}_lane0.ir")
            with open(out_path, "w") as f:
                f.write(new_ir + "\n")
            print(f"  NEW RECORD: {new_cost} (saved to {out_path})")
            with open(os.path.join(HERE, "events.jsonl"), "a") as f:
                f.write(json.dumps({
                    "ts": int(time.time()),
                    "type": "new_record",
                    "cost": new_cost,
                    "prev": PREV_RECORD,
                    "lane": 0,
                    "file": os.path.basename(__file__),
                }) + "\n")
        else:
            print(f"  no new record (best={PREV_RECORD})")

    print("\n=== summary ===")
    print(f"{'base':>5} {'depth':>5} {'ops':>8} {'cells':>6} {'raw':>10} {'renamed':>10}")
    for base_n, r in results.items():
        print(f"{base_n:>5} {r['depth']:>5} {r['ops']:>8} {r['cells']:>6} "
              f"{r['raw_cost']:>10,} {r['renamed_cost']:>10,}")

    # Token-log marker (count not measured here; manager can fill in)
    with open(os.path.join(HERE, "token_log.jsonl"), "a") as f:
        f.write(json.dumps({
            "ts": int(time.time()),
            "lane": 0,
            "exp": os.path.basename(__file__),
            "tokens": 0,
        }) + "\n")
