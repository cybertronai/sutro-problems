"""Energy-efficient matrix multiplication scorer + baselines.

Scores IR programs that compute ``C = A @ B`` under the
[simplified Dally model](https://github.com/cybertronai/simplified-dally-model)
using the
[v0 instruction set](https://github.com/cybertronai/simplified-dally-model/tree/main/instruction-sets/v0)
(``add``, ``sub``, ``mul``, ``copy``).

**Cost model (v0).** Processor at the origin, memory laid out as a
2D upper half-plane indexed by **positive integers**; the cell at
linear index ``addr`` sits at Manhattan distance ``⌈√addr⌉`` from the
core. Each operand read pays that distance; writes and arithmetic are
free; inputs are placed for free at caller-specified addresses; every
output address pays one standard read at exit.

Three-address-code IR (one instruction per line; ``;`` is also a
line separator so that single-line strings work):

    1,2                   ← input placement: A@1, B@2
    mul 3,1,2             ← mem[3] = mem[1] * mem[2]; reads ⌈√1⌉ + ⌈√2⌉
    3                     ← exit: read mem[3]; cost ⌈√3⌉

Supported ops (all four come straight from v0):

* ``add dest, src1, src2``  — ``mem[dest] = mem[src1] + mem[src2]``
* ``sub dest, src1, src2``  — ``mem[dest] = mem[src1] - mem[src2]``
* ``mul dest, src1, src2``  — ``mem[dest] = mem[src1] * mem[src2]``
* ``copy dest, src``        — ``mem[dest] = mem[src]``  (1 read; needed
  for scratchpad-style tiling, where you copy a far-away cell into a
  cheap scratch slot once and then re-read it many times)

Two-operand short form for the binary ops: ``add dest, src`` is wire
sugar for ``add dest, dest, src`` (in-place accumulate); the ops it
expands into are still v0-conformant. Addresses must be positive
integers; ``addr ≤ 0`` raises.
"""
from __future__ import annotations

import math
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def _cost(addr: int) -> int:
    """``⌈√addr⌉`` for a positive integer ``addr``; raises otherwise.

    The simplified Dally model only addresses cells with linear index
    ``≥ 1``; ``addr = 0`` is the (off-grid) ALU position and is not a
    valid memory cell. Rejecting non-positive addresses here means a
    typo'd or otherwise malformed IR can't slip through with a free
    read.
    """
    if not isinstance(addr, int) or addr < 1:
        raise ValueError(
            f"addresses must be positive integers; got {addr!r}")
    return math.isqrt(addr - 1) + 1


def _check_addrs(addrs, where):
    for a in addrs:
        if not isinstance(a, int) or a < 1:
            raise ValueError(
                f"{where}: addresses must be positive integers; got {a!r}")


# ---------------------------------------------------------------------------
# Parser + simulator
# ---------------------------------------------------------------------------

_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
}


def _parse(ir: str):
    text = ir.replace(";", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("IR needs at least an input line and an output line")
    input_addrs = [int(x) for x in lines[0].split(",")]
    output_addrs = [int(x) for x in lines[-1].split(",")]
    _check_addrs(input_addrs,  "input line")
    _check_addrs(output_addrs, "output line")
    ops = []
    for ln in lines[1:-1]:
        head, _, rest = ln.partition(" ")
        if not rest:
            raise ValueError(f"malformed instruction: {ln!r}")
        operands = [int(x) for x in rest.split(",")]
        _check_addrs(operands, f"`{head}` operands")
        ops.append((head, operands))
    return input_addrs, ops, output_addrs


def _simulate(ir: str, inputs: List[int]) -> Tuple[List[int], int]:
    input_addrs, ops, output_addrs = _parse(ir)
    if len(input_addrs) != len(inputs):
        raise ValueError(
            f"IR declares {len(input_addrs)} inputs; {len(inputs)} provided")
    if len(set(input_addrs)) != len(input_addrs):
        raise ValueError("input addresses must be distinct")
    mem = {a: v for a, v in zip(input_addrs, inputs)}
    cost = 0
    for op, oprs in ops:
        if op == "copy":
            if len(oprs) != 2:
                raise ValueError(f"copy needs 2 operands: copy {oprs}")
            dest, src = oprs
            if src not in mem:
                raise ValueError(
                    f"copy {dest},{src} reads uninitialized addr {src}")
            cost += _cost(src)
            mem[dest] = mem[src]
            continue
        if op not in _BINARY:
            raise ValueError(f"unknown op: {op!r}  (v0 supports add/sub/mul/copy)")
        if len(oprs) == 3:
            dest, s1, s2 = oprs
        elif len(oprs) == 2:
            dest, s2 = oprs
            s1 = dest
        else:
            raise ValueError(f"{op} needs 2 or 3 operands; got {oprs}")
        for src in (s1, s2):
            if src not in mem:
                raise ValueError(
                    f"{op} {','.join(map(str,oprs))} reads "
                    f"uninitialized addr {src}")
        cost += _cost(s1) + _cost(s2)
        mem[dest] = _BINARY[op](mem[s1], mem[s2])
    outputs = []
    for a in output_addrs:
        if a not in mem:
            raise ValueError(f"output addr {a} never written")
        cost += _cost(a)
        outputs.append(mem[a])
    return outputs, cost


# ---------------------------------------------------------------------------
# Test matrices + scorers
# ---------------------------------------------------------------------------

def _matmul_test(n: int):
    """Deterministic ``A``, ``B``, expected ``C = A @ B``.

    Inputs convention (used by all scorers and baseline generators):
    A flattened row-major first (``n²`` values), then B flattened
    row-major (``n²`` values), so ``2 n²`` inputs total. Outputs:
    C flattened row-major (``n²`` values).

    The two formulas below produce distinct, **non-symmetric** test
    data on purpose — earlier versions used ``B = A.T`` which made
    ``C = A·B = A·A.T`` symmetric, allowing IRs that confused the
    ``i, j`` indices to pass coincidentally. With the current data,
    ``C[i,j] != C[j,i]`` for ``n ≥ 2``, and the 1×1 case has
    ``A = [[1]], B = [[3]], C = [[3]]`` so an identity-IR returning
    ``A[0][0]`` no longer passes ``score_1x1``.
    """
    A = [[i * n + j + 1 for j in range(n)] for i in range(n)]
    B = [[i + 2 * j + 3 for j in range(n)] for i in range(n)]
    C = [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)]
         for i in range(n)]
    inputs = ([A[i][j] for i in range(n) for j in range(n)] +
              [B[i][j] for i in range(n) for j in range(n)])
    expected = [C[i][j] for i in range(n) for j in range(n)]
    return inputs, expected


def _score_n(ir: str, n: int) -> int:
    inputs, expected = _matmul_test(n)
    actual, cost = _simulate(ir, inputs)
    if actual != expected:
        raise ValueError(
            f"correctness failed (n={n}):\n  got      {actual}\n"
            f"  expected {expected}")
    return cost


def score_1x1(ir: str) -> int: return _score_n(ir, 1)
def score_4x4(ir: str) -> int: return _score_n(ir, 4)
def score_16x16(ir: str) -> int: return _score_n(ir, 16)


# ---------------------------------------------------------------------------
# Baseline generators — naive triple loop
# ---------------------------------------------------------------------------

def _baseline(n: int) -> str:
    """Naive triple loop: ``C[i,j] = Σ_k A[i,k]·B[k,j]``.

    Layout (worst case — bulk arrays placed contiguously after the
    scratchpad):
      A at addrs ``1 .. n²``       (row-major)
      B at addrs ``n²+1 .. 2n²``   (row-major)
      C at addrs ``2n²+1 .. 3n²``  (row-major; output)
      tmp scratch at addr ``3n²+1``
    """
    A_at = lambda i, j: 1 + i * n + j
    B_at = lambda i, j: 1 + n * n + i * n + j
    C_at = lambda i, j: 1 + 2 * n * n + i * n + j
    tmp = 3 * n * n + 1

    inputs = ([A_at(i, j) for i in range(n) for j in range(n)] +
              [B_at(i, j) for i in range(n) for j in range(n)])
    outputs = [C_at(i, j) for i in range(n) for j in range(n)]

    lines = [",".join(map(str, inputs))]
    for i in range(n):
        for j in range(n):
            # First product initializes C[i,j] (no const-zero needed).
            lines.append(f"mul {C_at(i,j)},{A_at(i,0)},{B_at(0,j)}")
            for k in range(1, n):
                lines.append(f"mul {tmp},{A_at(i,k)},{B_at(k,j)}")
                lines.append(f"add {C_at(i,j)},{tmp}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def generate_baseline_4x4() -> str: return _baseline(4)
def generate_baseline_16x16() -> str: return _baseline(16)


# ---------------------------------------------------------------------------
# Tiled 16×16 — scratchpad-cached 4×4 tiles
# ---------------------------------------------------------------------------

def generate_tiled_16x16() -> str:
    """Tiled matmul with 4×4 scratchpad-cached A/B tiles + a 4×4 sC
    accumulator. ``copy`` instructions move each A/B tile into the
    cheapest 48 cells of memory once per ``(bi, bj, bk)`` block, so
    the inner ``mul`` reads hit short-distance addresses. The
    accumulated sC tile is then ``copy``-ed out to its final position
    in the C bulk.

    Layout:
      sA at 1..16   (4×4 cached A-tile)
      sB at 17..32  (4×4 cached B-tile)
      sC at 33..48  (4×4 accumulator for the current C-tile)
      tmp at 49
      A bulk: 50..305
      B bulk: 306..561
      C bulk: 562..817 (output)
    """
    n, T = 16, 4
    sA = lambda ii, kk: 1 + ii * T + kk
    sB = lambda kk, jj: 1 + T * T + kk * T + jj
    sC = lambda ii, jj: 1 + 2 * T * T + ii * T + jj
    tmp = 3 * T * T + 1

    A_base = tmp + 1
    B_base = A_base + n * n
    C_base = B_base + n * n
    A_at = lambda i, j: A_base + i * n + j
    B_at = lambda i, j: B_base + i * n + j
    C_at = lambda i, j: C_base + i * n + j

    inputs = ([A_at(i, j) for i in range(n) for j in range(n)] +
              [B_at(i, j) for i in range(n) for j in range(n)])
    outputs = [C_at(i, j) for i in range(n) for j in range(n)]

    lines = [",".join(map(str, inputs))]
    nb = n // T
    for bi in range(nb):
        for bj in range(nb):
            for bk in range(nb):
                # Load A-tile A[bi*T:.., bk*T:..] -> sA
                for ii in range(T):
                    for kk in range(T):
                        lines.append(
                            f"copy {sA(ii,kk)},{A_at(bi*T+ii, bk*T+kk)}")
                # Load B-tile B[bk*T:.., bj*T:..] -> sB
                for kk in range(T):
                    for jj in range(T):
                        lines.append(
                            f"copy {sB(kk,jj)},{B_at(bk*T+kk, bj*T+jj)}")
                # Inner T³ contraction: sC[ii,jj] += sA[ii,kk] * sB[kk,jj]
                for ii in range(T):
                    for jj in range(T):
                        for kk in range(T):
                            lines.append(
                                f"mul {tmp},{sA(ii,kk)},{sB(kk,jj)}")
                            if bk == 0 and kk == 0:
                                # First product initializes sC[ii,jj].
                                lines.append(f"copy {sC(ii,jj)},{tmp}")
                            else:
                                lines.append(f"add {sC(ii,jj)},{tmp}")
            # End of (bi, bj) block: write sC out to bulk C.
            for ii in range(T):
                for jj in range(T):
                    lines.append(
                        f"copy {C_at(bi*T+ii, bj*T+jj)},{sC(ii,jj)}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


__all__ = [
    "score_1x1", "score_4x4", "score_16x16",
    "generate_baseline_4x4", "generate_baseline_16x16",
    "generate_tiled_16x16",
]


# ---------------------------------------------------------------------------
# Reproducer for the record-history IR files (run `python matmul.py`).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    ir_dir = os.path.join(here, "submissions")
    os.makedirs(ir_dir, exist_ok=True)
    artifacts = [
        ("baseline_4x4.ir",   generate_baseline_4x4(),   score_4x4),
        ("baseline_16x16.ir", generate_baseline_16x16(), score_16x16),
        ("tiled_16x16.ir",    generate_tiled_16x16(),    score_16x16),
    ]
    for name, ir, scorer in artifacts:
        cost = scorer(ir)
        path = os.path.join(ir_dir, name)
        with open(path, "w") as f:
            f.write(ir)
            f.write("\n")
        n_ops = len(ir.splitlines()) - 2
        print(f"  {name:<22} cost={cost:>10,}  ops={n_ops:>6,}  -> {path}")
