"""Semantic SSA and verified artifacts for the 16x16 matmul campaign."""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_IR = HERE.parent / "best_64431.ir"
sys.path.insert(0, str(REPO))


@dataclass(frozen=True)
class Op:
    dest: int
    code: str
    src: tuple[int, ...]


@dataclass
class Program:
    inputs: tuple[int, ...]
    ops: list[Op]
    outputs: tuple[int, ...]
    assignment: dict[int, int]

    def clone(self):
        return Program(self.inputs, list(self.ops), self.outputs, dict(self.assignment))


def cost(addr):
    return math.isqrt(int(addr) - 1) + 1


def parse_ir(text: str) -> Program:
    lines = [s.strip() for s in text.replace(";", "\n").splitlines() if s.strip()]
    addrs = list(map(int, lines[0].split(",")))
    if len(set(addrs)) != len(addrs) or any(a < 1 for a in addrs):
        raise ValueError("Input placement must be positive and distinct")
    current = {a: i for i, a in enumerate(addrs)}
    assignment = dict(enumerate(addrs))
    ops = []
    for line in lines[1:-1]:
        code, args = line.split(None, 1)
        aa = list(map(int, args.split(",")))
        if code == "copy":
            assert len(aa) == 2
            source = aa[1:]
        else:
            assert code in ("mul", "add", "sub") and len(aa) in (2, 3)
            source = aa[1:] if len(aa) == 3 else aa
        src = tuple(current[a] for a in source)
        dest = len(assignment)
        ops.append(Op(dest, code, src))
        assignment[dest] = aa[0]
        current[aa[0]] = dest
    return Program(tuple(range(len(addrs))), ops,
                   tuple(current[int(a)] for a in lines[-1].split(",")), assignment)


def load(path=DEFAULT_IR):
    return parse_ir(Path(path).read_text())


def lifetimes(p: Program):
    """Dict value -> [definition event, last read event, number of reads].

    Half-open intervals permit a destination to reuse an operand on its last
    read. Every input is present at entry; every output survives until exit.
    """
    values = {v: [0, 0, 0] for v in p.inputs}
    for t, op in enumerate(p.ops, 1):
        for v in op.src:
            if v not in values:
                raise ValueError(f"Value {v} read before definition at {t}")
            values[v][1] = t
            values[v][2] += 1
        if op.dest in values:
            raise ValueError(f"SSA destination {op.dest} defined twice")
        values[op.dest] = [t, t, 0]
    for v in p.outputs:
        values[v][1] = len(p.ops) + 1
        values[v][2] += 1
    return values


def assignment_score(p, assignment=None):
    a = p.assignment if assignment is None else assignment
    return sum(n * cost(a[v]) for v, (_, _, n) in lifetimes(p).items() if n)


def check_assignment(p, assignment=None):
    a = p.assignment if assignment is None else assignment
    lives = lifetimes(p)
    by_addr = defaultdict(list)
    for v, (start, end, reads) in lives.items():
        if a[v] < 1:
            raise ValueError("Nonpositive address")
        if reads:
            by_addr[a[v]].append((start, end, v))
    if len({a[v] for v in p.inputs}) != len(p.inputs):
        raise ValueError("Aliased inputs")
    for addr, intervals in by_addr.items():
        prev_end = -1
        for start, end, v in sorted(intervals):
            if start < prev_end:
                raise ValueError(f"Address {addr} overlaps at value {v}: {start} < {prev_end}")
            prev_end = end
    # An unread destination still writes memory. Require its address to be
    # disjoint from read-bearing cells; emission supplies fresh dead slots.
    used = set(by_addr)
    for v, (_, _, reads) in lives.items():
        if not reads and a[v] in used:
            raise ValueError(f"Dead write {v} overwrites a used address")
    return True


def emit(p, assignment=None):
    a = dict(p.assignment if assignment is None else assignment)
    fresh = max(a.values(), default=0) + 1
    for v, (_, _, n) in lifetimes(p).items():
        if not n:
            a[v] = fresh
            fresh += 1
    check_assignment(p, a)
    lines = [",".join(str(a[v]) for v in p.inputs)]
    lines += [op.code + " " + ",".join(str(a[v]) for v in (op.dest, *op.src)) for op in p.ops]
    lines.append(",".join(str(a[v]) for v in p.outputs))
    return "\n".join(lines) + "\n"


def color_tiers(p, tiers):
    """Optimal interval coloring within fixed cost tiers; raises if infeasible."""
    groups = defaultdict(list)
    lives = lifetimes(p)
    for v, (start, end, reads) in lives.items():
        if reads:
            groups[int(tiers[v])].append((start, end, v))
    assignment = {}
    for tier, intervals in groups.items():
        free = list(range((tier - 1) ** 2 + 1, tier ** 2 + 1))
        active = []
        for start, end, v in sorted(intervals):
            while active and active[0][0] <= start:
                _, addr = heapq.heappop(active)
                heapq.heappush(free, addr)
            if not free:
                raise ValueError(f"Tier {tier} exceeds capacity at {start}")
            addr = heapq.heappop(free)
            assignment[v] = addr
            heapq.heappush(active, (end, addr))
    fresh = max(assignment.values(), default=0) + 1
    for v in lives:
        if v not in assignment:
            assignment[v] = fresh
            fresh += 1
    check_assignment(p, assignment)
    return assignment


def verify_ir(ir):
    from matmul import score_16x16
    from matmul.submissions.best_66178 import _prove
    score = score_16x16(ir)
    counts, costs = _prove(ir)
    assert score == sum(costs.values())
    return {"score": score, "sha256": hashlib.sha256(ir.encode()).hexdigest(),
            "operations": dict(counts), "read_costs": dict(costs)}


def save(p, name, metadata=None):
    ir = emit(p)
    verified = verify_ir(ir)
    path = HERE / "artifacts" / f"{name}_{verified['score']}.ir"
    path.parent.mkdir(exist_ok=True)
    path.write_text(ir)
    path.with_suffix(".json").write_text(json.dumps({**(metadata or {}), **verified}, indent=2) + "\n")
    return path, verified


if __name__ == "__main__":
    p = load()
    ir = emit(p)
    assert assignment_score(p) == 64431
    assert color_tiers(p, {v: cost(a) for v, a in p.assignment.items()})
    print(verify_ir(ir))
