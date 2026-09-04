"""Track B joint generator: order-aware interval coloring placement.

The validated decode says the record's structure is: a schedule order
fixes value lifetimes; placement is interval coloring where cost
depends on the address. This generator implements exactly that:

1. Schedule: output order permutation + k-interleaving policy fixes
   the event sequence and therefore every value's [birth, death].
2. Placement: greedy interval coloring - each new value takes the
   cheapest address whose occupant intervals do not overlap it.
   Input cells participate: when an input's last consumer has run,
   its address joins the colorable pool.
3. Exit staging into dead cells.

Action space (the MCTS knobs): output order, per-output k order,
which inputs start high, staging pool cells. This file implements the
generator + exact scorer; search drivers sweep the knobs.
"""
from __future__ import annotations

import itertools
import math
import random

K = 4


def rc(a: int) -> int:
    return math.isqrt(a - 1) + 1


def generate(out_order, k_orders, far_keys):
    """out_order: permutation of (i,j). k_orders: per-output k order.
    far_keys: inputs placed at high addresses (staged by cost, not copy).
    Returns (ops, inputs, outputs) in competition format."""
    near = [a for a in range(1, 26) if a not in (22, 23, 24, 25)]
    far = list(range(26, 38))
    keys = [("a", i, k) for i in range(K) for k in range(K)] + \
           [("b", k, j) for k in range(K) for j in range(K)]
    addr = {}
    ni = fi = 0
    far_x = list(far)
    for key in keys:
        if key in far_keys or ni >= len(near):
            if fi >= len(far_x):
                far_x.append(38 + fi - len(far))
            addr[key] = far_x[fi]; fi += 1
        else:
            addr[key] = near[ni]; ni += 1

    # pass 1: emit the event sequence symbolically, computing lifetimes
    events = []          # (kind, i, j, k)
    for (i, j) in out_order:
        for k in k_orders[(i, j)]:
            events.append(("mul", i, j, k))
            events.append(("add", i, j, k))
    birth, death = {}, {}
    acc_sym = {}
    t = 0
    for kind, i, j, k in events:
        if kind == "mul":
            sym = ("p", i, j, k)
            birth[sym] = t
            # input last-use tracking
        else:
            if (i, j, 0) in acc_sym and k == 0:
                pass
        t += 1
    # simpler: two-pass with symbolic values
    # value symbols: products p(i,j,k); accumulators a(i,j,t) t=0..3
    seq = []
    for (i, j) in out_order:
        kseq = k_orders[(i, j)]
        for n, k in enumerate(kseq):
            seq.append(("mul", i, j, k))
            if n == 0:
                seq.append(("seed", i, j, k))   # product becomes acc
            else:
                seq.append(("add", i, j, k))
    # lifetimes: product born at its mul, dies at its add (or lives on as acc)
    # accumulator for (i,j): born at seed/each add, dies at next add or exit
    val_birth, val_death = {}, {}
    # per-output chain: the product whose add just ran holds the acc
    # until the NEXT add of the same output reads it; the last product
    # holds it to the exit copy
    next_add = {}   # event idx of add_k(n+1) keyed by (i,j,k_n)
    for (i, j) in out_order:
        kseq = k_orders[(i, j)]
        add_idx = {}
        for idx, ev in enumerate(seq):
            kind, ii, jj, kk = ev
            if kind in ("add", "seed") and (ii, jj) == (i, j):
                add_idx[kk] = idx
        for n, k in enumerate(kseq):
            if n + 1 < len(kseq):
                next_add[(i, j, k)] = add_idx[kseq[n + 1]]
            else:
                next_add[(i, j, k)] = 10_000 + (i * K + j)
    for idx, ev in enumerate(seq):
        kind, i, j, k = ev
        if kind == "mul":
            v = ("p", i, j, k)
            val_birth[v] = idx
        elif kind in ("seed", "add"):
            v = ("p", i, j, k)
            # death = when the NEXT add (or exit) has consumed this cell
            val_death[v] = next_add[(i, j, k)] + 1
    # products that live to exit: their death = program end
    # input cell intervals: [0, last consumer event index]
    input_death = {}
    for idx, ev in enumerate(seq):
        kind, i, j, k = ev
        if kind in ("mul",):
            input_death[("a", i, k)] = idx
            input_death[("b", k, j)] = idx

    # placement: greedy interval coloring, cheapest non-overlapping addr
    occupied = {}  # addr -> list of (birth, death)
    def can_take(a, b, d):
        for (ob, od) in occupied.get(a, []):
            if not (d <= ob or b >= od):
                return False
        return True
    def place(b, d):
        a = 1
        while True:
            if can_take(a, b, d):
                occupied.setdefault(a, []).append((b, d))
                return a
            a += 1
    val_addr = {}
    # inputs first (they own their fixed cells for their lifetime)
    for key in keys:
        occupied.setdefault(addr[key], []).append((0, input_death.get(key, len(seq))))
    # place values in birth order
    to_place = sorted([v for v in val_birth], key=lambda v: val_birth[v])
    for v in to_place:
        val_addr[v] = place(val_birth[v], val_death[v])

    # pass 2: emit ops with real addresses
    ops = []
    acc = {}
    for idx, ev in enumerate(seq):
        kind, i, j, k = ev
        if kind == "mul":
            ops.append(("mul", val_addr[("p", i, j, k)],
                        [addr[("a", i, k)], addr[("b", k, j)]]))
        elif kind == "seed":
            acc[(i, j)] = val_addr[("p", i, j, k)]
        elif kind == "add":
            ops.append(("add", val_addr[("p", i, j, k)],
                        [acc[(i, j)], val_addr[("p", i, j, k)]]))
            acc[(i, j)] = val_addr[("p", i, j, k)]
    # exit staging into cheapest dead cells
    acc_cells = set(acc.values())
    dead = sorted(
        (set(list(addr.values()) + [22, 23, 24, 25]) - acc_cells),
        key=lambda a: (rc(a), a))
    outs_order = sorted(acc.keys(), key=lambda ij: ij[0] * K + ij[1])
    staged = {ij: dead[n] for n, ij in enumerate(outs_order)}
    for ij in outs_order:
        ops.append(("copy", staged[ij], [acc[ij]]))
    outputs = [staged[ij] for ij in outs_order]
    return ops, [addr[k] for k in keys], outputs


def score(ops, outputs):
    c = 0
    for _, _, srcs in ops:
        c += sum(rc(s) for s in srcs)
    c += sum(rc(o) for o in outputs)
    return c


def main():
    rng = random.Random(2026_0905)
    keys = [("a", i, k) for i in range(K) for k in range(K)] + \
           [("b", k, j) for k in range(K) for j in range(K)]
    best = None
    for trial in range(3000):
        ij = [(i, j) for i in range(K) for j in range(K)]
        out_order = ij[:]
        if trial % 3 == 1:
            rng.shuffle(out_order)
        elif trial % 3 == 2:  # row-major with row reversal mixing
            out_order = sorted(ij, key=lambda x: (x[0] % 2, x[0], x[1]))
        k_orders = {}
        for o in ij:
            kk = list(range(K))
            if trial % 2 == 1:
                rng.shuffle(kk)
            k_orders[o] = kk
        far = set(rng.sample(keys, rng.choice([0, 4, 8, 12])))
        ops, inputs, outputs = generate(out_order, k_orders, far)
        c = score(ops, outputs)
        if best is None or c < best[0]:
            best = (c, trial)
            print(f"trial {trial}: {c}")
    print("best:", best[0])


if __name__ == "__main__":
    main()
