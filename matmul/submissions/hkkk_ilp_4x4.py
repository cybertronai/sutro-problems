"""Generator for the 4x4 matmul record ``hkkk_ilp_4x4.ir`` (cost 683).

Approach: build the computation as an SSA-style op trace (values with
birth/last-read times), then solve an exact ILP for address assignment
(weighted interval coloring: each value gets one address; co-live values
must differ; objective = sum over values of nreads * ceil(sqrt(addr))).

The op order is "HKKK": row 0 j-outer reading A from its home cells,
rows 1-3 k-outer with just-in-time staging. See hkkk_ilp_4x4.md.

Requires numpy and scipy (scipy.optimize.milp / HiGHS).

Run: python3 hkkk_ilp_4x4.py   (writes hkkk_ilp_4x4.ir next to this file)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import matmul  # noqa: E402

EXPECTED_SCORE = 683
EXPECTED_SHA256 = "562af9a9b848ca325499a63525b5e2d614339632365ebb712c651202a627ab3c"
IR_PATH = Path(__file__).with_name("hkkk_ilp_4x4.ir")


def generate_hkkk_ilp_4x4() -> str:
    """Return the exact leaderboard IR without requiring the ILP dependencies."""
    return IR_PATH.read_text(encoding="utf-8")


def cell_cost(a: int) -> int:
    return math.isqrt(a - 1) + 1


# ---------------------------------------------------------------------------
# Trace construction
# ---------------------------------------------------------------------------
# A trace is a list of ops: (opcode, out_value, [in_values]).
# Values are strings. Inputs are values "A_i_k" / "B_k_j" born at time -1
# (gap 0). Outputs are values "O_i_j", with a virtual exit read at time T.

def build_trace(stage_a: bool = True,
                row_order=(0, 1, 2, 3),
                col_order=(0, 1, 2, 3),
                k_order=(0, 1, 2, 3),
                stage_rows_early: bool = False):
    """Inner-product order with optional per-row staging of A into sA cells.

    stage_a=True: copy A[i][k] -> sA_i_k (4 staged values per row), then the
    16 muls of row i read the staged values.
    """
    ops = []
    if stage_rows_early and stage_a:
        for i in row_order:
            for k in k_order:
                ops.append(("copy", f"s_{i}_{k}", [f"A_{i}_{k}"]))
    for i in row_order:
        if stage_a and not stage_rows_early:
            for k in k_order:
                ops.append(("copy", f"s_{i}_{k}", [f"A_{i}_{k}"]))
        for j in col_order:
            terms = []
            for t, k in enumerate(k_order):
                a_op = f"s_{i}_{k}" if stage_a else f"A_{i}_{k}"
                p = f"p_{i}_{j}_{k}"
                ops.append(("mul", p, [a_op, f"B_{k}_{j}"]))
                terms.append(p)
            # sum chain: s1 = p0+p1, s2 = s1+p2, out = s2+p3
            acc = terms[0]
            for t in range(1, 4):
                nxt = f"O_{i}_{j}" if t == 3 else f"s_{i}_{j}_{t}"
                ops.append(("add", nxt, [acc, terms[t]]))
                acc = nxt
    return ops


def build_trace_colmajor(col_order=(0, 1, 2, 3),
                         row_order=(0, 1, 2, 3),
                         k_order=(0, 1, 2, 3)):
    """Symmetric variant: stage B columns, read A from homes."""
    ops = []
    for j in col_order:
        for k in k_order:
            ops.append(("copy", f"t_{k}_{j}", [f"B_{k}_{j}"]))
        for i in row_order:
            terms = []
            for k in k_order:
                p = f"p_{i}_{j}_{k}"
                ops.append(("mul", p, [f"A_{i}_{k}", f"t_{k}_{j}"]))
                terms.append(p)
            acc = terms[0]
            for t in range(1, 4):
                nxt = f"O_{i}_{j}" if t == 3 else f"s_{i}_{j}_{t}"
                ops.append(("add", nxt, [acc, terms[t]]))
                acc = nxt
    return ops


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------

def compute_liveness(ops):
    """Return values, birth, last_read (exit reads at time T for O_*)."""
    birth = {}
    last_read = {}
    inputs = []
    for t, (opc, out, ins) in enumerate(ops):
        for v in ins:
            if v not in birth and not v.startswith(("A_", "B_")):
                raise ValueError(f"read of unborn value {v} at op {t}")
            last_read[v] = t
        if out in birth:
            raise ValueError(f"value {out} written twice")
        birth[out] = t
    T = len(ops)
    for i in range(4):
        for j in range(4):
            last_read[f"O_{i}_{j}"] = T  # exit read
    # inputs: born at -1 (live from gap 0)
    for v in list(last_read):
        if v.startswith(("A_", "B_")):
            birth[v] = -1
            inputs.append(v)
    values = list(birth)
    return values, birth, last_read, T, inputs


# ---------------------------------------------------------------------------
# Exact address assignment via ILP
# ---------------------------------------------------------------------------

def solve_assignment(ops, max_addr=None, time_limit=120.0, verbose=False):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    values, birth, last_read, T, inputs = compute_liveness(ops)
    nV = len(values)
    # Searching nV addresses is complete: an assignment uses at most one
    # address per value. If it used a label above nV, rank-compressing its
    # used labels to 1..k preserves all conflicts and cannot raise read cost.
    if max_addr is None:
        max_addr = nV
    vidx = {v: i for i, v in enumerate(values)}
    nreads = {}
    for v in values:
        nreads[v] = 0
    for opc, out, ins in ops:
        for v in ins:
            nreads[v] += 1
    for v in values:
        if v.startswith("O_"):
            nreads[v] += 1  # exit read

    addrs = list(range(1, max_addr + 1))
    nA = len(addrs)
    costs = np.array([cell_cost(a) for a in addrs], dtype=float)

    # variable x[v, a] -> index v * nA + ai
    nX = nV * nA
    c = np.zeros(nX)
    for vi, v in enumerate(values):
        c[vi * nA:(vi + 1) * nA] = nreads[v] * costs

    # constraints:
    #  (1) each value: sum_a x[v,a] == 1
    #  (2) each gap g, each addr a: sum_{v live in g} x[v,a] <= 1
    # value v live in gap g (g in 0..T) iff birth(v) < g <= last_read(v)
    rows = []
    lb = []
    ub = []
    # Use sparse assembly
    # estimate nnz: assignment rows nA each + gap rows: live counts
    A_mat = lil_matrix((0, nX), dtype=float)

    def add_row(ind_dict, lo, hi):
        r = lil_matrix((1, nX), dtype=float)
        for k, val in ind_dict.items():
            r[0, k] = val
        rows.append(r)
        lb.append(lo)
        ub.append(hi)

    for vi in range(nV):
        add_row({vi * nA + ai: 1.0 for ai in range(nA)}, 1.0, 1.0)

    # gap liveness
    for g in range(0, T + 1):
        live = [vi for vi, v in enumerate(values)
                if birth[v] < g <= last_read[v]]
        if not live:
            continue
        for ai in range(nA):
            add_row({vi * nA + ai: 1.0 for vi in live}, 0.0, 1.0)

    A_all = lil_matrix((len(rows), nX), dtype=float)
    for r_i, r in enumerate(rows):
        A_all[r_i] = r
    A_all = A_all.tocsr()
    constraints = LinearConstraint(A_all, np.array(lb), np.array(ub))

    integrality = np.ones(nX)
    res = milp(c=c, constraints=constraints,
               bounds=Bounds(0, 1), integrality=integrality,
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    # A time-limited MILP may return a feasible incumbent in ``res.x`` while
    # status is non-optimal.  This generator is used as an exact certificate,
    # so never emit such an incumbent as though it proved the optimum.
    mip_gap = getattr(res, "mip_gap", None)
    if (not res.success or res.status != 0 or res.x is None or
            (mip_gap is not None and mip_gap > 1e-9)):
        raise RuntimeError(
            f"ILP did not prove optimality: status={res.status}, "
            f"success={res.success}, mip_gap={mip_gap}, message={res.message}")
    if verbose:
        print(f"  ILP status={res.status} obj={res.fun:.1f} "
              f"nodes={getattr(res, 'mip_node_count', '?')}")
    x = res.x.reshape(nV, nA)
    fractionality = float(np.max(np.abs(x - np.rint(x))))
    if fractionality > 1e-7:
        raise RuntimeError(
            f"optimal solution is not integral: max fractionality "
            f"{fractionality:.3g}")
    assign = {}
    for vi, v in enumerate(values):
        ai = int(np.argmax(x[vi]))
        if x[vi, ai] <= 0.5:
            raise RuntimeError(f"value {v} has no integral address assignment")
        assign[v] = addrs[ai]
    return assign, res.fun


# ---------------------------------------------------------------------------
# IR emission
# ---------------------------------------------------------------------------

def emit_ir(ops, assign):
    inp = [assign[f"A_{i}_{k}"] for i in range(4) for k in range(4)]
    inp += [assign[f"B_{k}_{j}"] for k in range(4) for j in range(4)]
    out = [assign[f"O_{i}_{j}"] for i in range(4) for j in range(4)]
    lines = [",".join(map(str, inp))]
    for opc, o, ins in ops:
        lines.append(f"{opc} {assign[o]},{','.join(str(assign[v]) for v in ins)}")
    lines.append(",".join(map(str, out)))
    return "\n".join(lines)


def build_trace_best(defer_output: bool = True):
    """Winning structure (cost 683): row 0 computed j-outer reading A from
    its (cheap, early-freed) home cells; rows 1-3 computed k-outer with
    just-in-time staging of A[i][k] (copy immediately before its k-step, so
    the staged value lives at addr 1 and dies after 4 reads).  One independent
    row-2 output add is delayed into the start of row 3 to lower peak address
    pressure."""
    ops = []
    # row 0: j-outer, home reads
    for j in range(4):
        acc = None
        for t, k in enumerate((0, 1, 2, 3)):
            p = f"p_0_{j}_{k}"
            ops.append(("mul", p, [f"A_0_{k}", f"B_{k}_{j}"]))
            if t == 0:
                acc = p
            else:
                nxt = f"O_0_{j}" if t == 3 else f"h_0_{j}_{t}"
                ops.append(("add", nxt, [acc, p]))
                acc = nxt
    # rows 1-3: k-outer, just-in-time staged A
    for i in (1, 2, 3):
        for ki, k in enumerate((0, 1, 2, 3)):
            ops.append(("copy", f"s_{i}_{k}", [f"A_{i}_{k}"]))
            for j in range(4):
                p = f"p_{i}_{j}_{k}"
                ops.append(("mul", p, [f"s_{i}_{k}", f"B_{k}_{j}"]))
                if ki > 0:
                    prev = (f"p_{i}_{j}_{k-1}" if ki == 1
                            else f"h_{i}_{j}_{k-1}")
                    nxt = f"O_{i}_{j}" if ki == 3 else f"h_{i}_{j}_{k}"
                    ops.append(("add", nxt, [prev, p]))
    if defer_output:
        # Delay O[2,1]'s final add until just after row 3's first multiply.
        # The arithmetic DAG is unchanged, while the exact allocation optimum
        # across the complete address search drops from 686 to 683. ``main``
        # solves both
        # schedules so this improvement is part of the reproducible artifact.
        delayed_index = next(
            i for i, (_op, out, _ins) in enumerate(ops) if out == "O_2_1"
        )
        delayed = ops.pop(delayed_index)
        anchor_index = next(
            i for i, (_op, out, _ins) in enumerate(ops) if out == "p_3_0_0"
        )
        ops.insert(anchor_index + 1, delayed)
    return ops


def main():
    base_ops = build_trace_best(defer_output=False)
    _base_assign, base_obj = solve_assignment(base_ops, verbose=True)
    ops = build_trace_best()
    assign, obj = solve_assignment(ops, verbose=True)
    if abs(base_obj - 686) > 1e-6 or abs(obj - 683) > 1e-6:
        raise RuntimeError(
            f"unexpected certified objectives: base={base_obj}, deferred={obj}")
    ir = emit_ir(ops, assign)
    cost = matmul.score_4x4(ir)
    print(f"best (HKKK: home-row0 + k-outer-jit rows): "
          f"ilp_obj={obj:.0f}  verified_cost={cost}  ops={len(ops)}")
    if abs(obj - cost) >= 1e-6:
        raise RuntimeError(f"ILP/scorer cost mismatch: {obj} != {cost}")
    IR_PATH.write_text(ir + "\n", encoding="utf-8")
    print(f"saved -> {IR_PATH}")


if __name__ == "__main__":
    main()
