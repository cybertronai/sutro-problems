"""16x16 matmul: liveness-ordered outputs, repacked inputs, scratch tail.

This is the 68,392 follow-up to ``colmajor_fused_16x16``.  It keeps the
same 4x8 super-block arithmetic schedule and fused final copy-out, then
changes only address/liveness choices:

* use super-block order 00,01,10,20,30,11,21,31;
* prefer dead B homes, then dead A homes, then fresh spill homes;
* pack input cells that also serve as outputs at the front of their input
  regions, because they get one extra exit read;
* in the final block, compute local column 7 last and leave its four outputs
  directly in sA0, sA1, TMP, and SB.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

N = 16
TIO = 4
TJO = 8
TII = 4
N_IBO = N // TIO
N_JBO = N // TJO
N_JBI = TJO

SB = 1
TMP = 2
SUPERBLOCK_ORDER = (
    (0, 0),
    (0, 1),
    (1, 0),
    (2, 0),
    (3, 0),
    (1, 1),
    (2, 1),
    (3, 1),
)
SCRATCH_JB_IN = 7
EXPECTED_COST = 68_392

Block = tuple[int, int]
Output = tuple[int, int]
Home = tuple[str, int, int] | tuple[str, int]


def sA(ii: int) -> int:
    return 3 + ii


def sC(jb_in: int, ii: int) -> int:
    return 7 + jb_in * TII + ii


def block_outputs(block: Block) -> list[Output]:
    bi_o, bj_o = block
    return [
        (bi_o * TIO + ii, bj_o * TJO + jb_in)
        for jb_in in range(N_JBI)
        for ii in range(TII)
    ]


def final_tail_homes() -> dict[tuple[int, int], int]:
    """Map the final local column's four outputs into cheap scratch cells."""
    return {
        (SCRATCH_JB_IN, 0): sA(0),
        (SCRATCH_JB_IN, 1): sA(1),
        (SCRATCH_JB_IN, 2): TMP,
        (SCRATCH_JB_IN, 3): SB,
    }


@dataclass(frozen=True)
class Assignment:
    output_homes: dict[Output, Home]
    spill_count: int


@dataclass(frozen=True)
class Addressing:
    a_addr: dict[tuple[int, int], int]
    b_addr: dict[tuple[int, int], int]
    spill_addr: dict[int, int]

    def home_addr(self, home: Home) -> int:
        tag = home[0]
        if tag == "A":
            return self.a_addr[(home[1], home[2])]  # type: ignore[index]
        if tag == "B":
            return self.b_addr[(home[1], home[2])]  # type: ignore[index]
        if tag == "SPILL":
            return self.spill_addr[home[1]]  # type: ignore[index]
        if tag == "SC":
            return sC(home[1], home[2])  # type: ignore[index]
        if tag == "SCRATCH":
            return home[1]  # type: ignore[index]
        raise ValueError(f"unknown home: {home!r}")


def _input_cells() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    a_cells = [(i, k) for i in range(N) for k in range(N)]
    b_cells = [(k, j) for k in range(N) for j in range(N)]
    return a_cells, b_cells


def assign_output_homes() -> Assignment:
    """Assign output homes using cells proven dead by super-block liveness."""
    order = SUPERBLOCK_ORDER
    expected_blocks = {(bi, bj) for bj in range(N_JBO) for bi in range(N_IBO)}
    if set(order) != expected_blocks:
        raise ValueError("super-block order must cover each block exactly once")

    final_block = order[-1]
    row_remaining = Counter(bi for bi, _bj in order)
    col_remaining = Counter(bj for _bi, bj in order)
    dead_a: list[Home] = []
    dead_b: list[Home] = []
    output_homes: dict[Output, Home] = {}
    spill_count = 0

    for block_index, block in enumerate(order):
        bi_o, bj_o = block
        row_remaining[bi_o] -= 1
        col_remaining[bj_o] -= 1

        if row_remaining[bi_o] == 0:
            for ii in range(TIO):
                i = bi_o * TIO + ii
                for k in range(N):
                    dead_a.append(("A", i, k))
        if col_remaining[bj_o] == 0:
            for k in range(N):
                for jb_in in range(N_JBI):
                    j = bj_o * TJO + jb_in
                    dead_b.append(("B", k, j))

        if block == final_block and block_index == len(order) - 1:
            scratch_homes = final_tail_homes()
            for i, j in block_outputs(block):
                local_ii = i - bi_o * TIO
                local_jb = j - bj_o * TJO
                scratch_addr = scratch_homes.get((local_jb, local_ii))
                if scratch_addr is None:
                    output_homes[(i, j)] = ("SC", local_jb, local_ii)
                else:
                    output_homes[(i, j)] = ("SCRATCH", scratch_addr)
            continue

        # B cells are read four times and A cells twice, so if a cell also
        # carries an output exit read, using dead B homes first is cheaper.
        available = sorted(dead_b) + sorted(dead_a)
        for output in block_outputs(block):
            if available:
                home = available.pop(0)
                if home[0] == "A":
                    dead_a.remove(home)
                elif home[0] == "B":
                    dead_b.remove(home)
                else:
                    raise AssertionError(f"unexpected dead home: {home!r}")
            else:
                home = ("SPILL", spill_count)
                spill_count += 1
            output_homes[output] = home

    return Assignment(output_homes=output_homes, spill_count=spill_count)


def build_addressing(assignment: Assignment) -> Addressing:
    """Pack output-bearing input cells first, then ordinary inputs."""
    a_cells, b_cells = _input_cells()
    next_addr = 39

    b_outputs = sorted(
        (home[1], home[2])  # type: ignore[index]
        for home in assignment.output_homes.values()
        if home[0] == "B"
    )
    b_output_set = set(b_outputs)
    b_order = b_outputs + [cell for cell in b_cells if cell not in b_output_set]

    a_outputs = sorted(
        (home[1], home[2])  # type: ignore[index]
        for home in assignment.output_homes.values()
        if home[0] == "A"
    )
    a_output_set = set(a_outputs)
    a_order = a_outputs + [cell for cell in a_cells if cell not in a_output_set]

    b_addr = {cell: next_addr + idx for idx, cell in enumerate(b_order)}
    next_addr += len(b_order)
    a_addr = {cell: next_addr + idx for idx, cell in enumerate(a_order)}
    next_addr += len(a_order)
    spill_addr = {idx: next_addr + idx for idx in range(assignment.spill_count)}
    return Addressing(a_addr=a_addr, b_addr=b_addr, spill_addr=spill_addr)


def generate_output_repacked_tail_16x16() -> str:
    assignment = assign_output_homes()
    addressing = build_addressing(assignment)

    A = addressing.a_addr.__getitem__
    B = addressing.b_addr.__getitem__
    out_addr = {
        cell: addressing.home_addr(home)
        for cell, home in assignment.output_homes.items()
    }

    inputs = ([A((i, k)) for i in range(N) for k in range(N)] +
              [B((k, j)) for k in range(N) for j in range(N)])
    outputs = [out_addr[(i, j)] for i in range(N) for j in range(N)]
    lines = [",".join(map(str, inputs))]

    final_block = SUPERBLOCK_ORDER[-1]
    scratch_direct = final_tail_homes()

    for block in SUPERBLOCK_ORDER:
        bi_o, bj_o = block
        is_last_super = block == final_block
        jb_order = list(range(N_JBI))
        if is_last_super:
            jb_order.remove(SCRATCH_JB_IN)
            jb_order.append(SCRATCH_JB_IN)

        for k in range(N):
            for ii in range(TII):
                lines.append(f"copy {sA(ii)},{A((bi_o * TIO + ii, k))}")
            for jb_in in jb_order:
                j = bj_o * TJO + jb_in
                lines.append(f"copy {SB},{B((k, j))}")
                for ii in range(TII):
                    i = bi_o * TIO + ii
                    out = out_addr[(i, j)]
                    if k == 0:
                        lines.append(f"mul {sC(jb_in, ii)},{sA(ii)},{SB}")
                    elif (
                        k == N - 1
                        and is_last_super
                        and scratch_direct.get((jb_in, ii)) == out
                    ):
                        if out == sA(ii):
                            lines.append(f"mul {out},{sA(ii)},{SB}")
                            lines.append(f"add {out},{sC(jb_in, ii)},{out}")
                        elif out == TMP:
                            lines.append(f"mul {TMP},{sA(ii)},{SB}")
                            lines.append(f"add {TMP},{sC(jb_in, ii)},{TMP}")
                        elif out == SB:
                            lines.append(f"mul {SB},{sA(ii)},{SB}")
                            lines.append(f"add {SB},{sC(jb_in, ii)},{SB}")
                        else:
                            raise AssertionError(f"unsupported scratch home {out}")
                    elif k == N - 1 and not is_last_super:
                        if ii < TII - 1:
                            lines.append(f"mul {TMP},{sA(ii)},{SB}")
                            lines.append(f"add {out},{sC(jb_in, ii)},{TMP}")
                        else:
                            lines.append(f"mul {SB},{sA(ii)},{SB}")
                            lines.append(f"add {out},{sC(jb_in, ii)},{SB}")
                    else:
                        if ii < TII - 1:
                            lines.append(f"mul {TMP},{sA(ii)},{SB}")
                            lines.append(f"add {sC(jb_in, ii)},{TMP}")
                        else:
                            lines.append(f"mul {SB},{sA(ii)},{SB}")
                            lines.append(f"add {sC(jb_in, ii)},{SB}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def _parse_ir(ir: str) -> tuple[list[int], list[tuple[str, list[int]]], list[int]]:
    lines = [line.strip() for line in ir.splitlines() if line.strip()]
    inputs = [int(part) for part in lines[0].split(",")]
    outputs = [int(part) for part in lines[-1].split(",")]
    ops: list[tuple[str, list[int]]] = []
    for line in lines[1:-1]:
        op, _sep, rest = line.partition(" ")
        ops.append((op, [int(part) for part in rest.split(",")]))
    return inputs, ops, outputs


def _assert_submission_invariants(ir: str) -> None:
    """Catch malformed address use and unsafe input/output aliasing."""
    inputs, ops, outputs = _parse_ir(ir)
    assert len(inputs) == 2 * N * N
    assert len(set(inputs)) == len(inputs)
    assert len(outputs) == N * N
    assert len(set(outputs)) == len(outputs)
    assert all(addr > 0 for addr in inputs + outputs)

    input_addrs = set(inputs)
    first_write: dict[int, int] = {}
    output_writes: set[int] = set()

    for op_index, (op, operands) in enumerate(ops, start=1):
        assert all(addr > 0 for addr in operands)
        if op == "copy":
            dest, src = operands
            reads = [src]
        elif op in {"add", "sub", "mul"} and len(operands) == 3:
            dest, src1, src2 = operands
            reads = [src1, src2]
        elif op in {"add", "sub", "mul"} and len(operands) == 2:
            dest, src2 = operands
            reads = [dest, src2]
        else:
            raise AssertionError(f"bad op at {op_index}: {op} {operands}")

        for src in reads:
            if src in input_addrs and src in first_write:
                raise AssertionError(
                    f"input addr {src} read at op {op_index} after "
                    f"overwrite at op {first_write[src]}"
                )

        first_write.setdefault(dest, op_index)
        output_writes.add(dest)

    missing_outputs = [addr for addr in outputs if addr not in output_writes]
    assert not missing_outputs, missing_outputs
    assert len(input_addrs & set(outputs)) == 160


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from matmul import score_16x16  # noqa: E402

    ir = generate_output_repacked_tail_16x16()
    _assert_submission_invariants(ir)
    cost = score_16x16(ir)
    assert cost == EXPECTED_COST, cost

    out_path = Path(__file__).with_suffix(".ir")
    out_path.write_text(ir + "\n")
    print(f"{out_path.name}  cost={cost:,}")
