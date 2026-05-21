"""Greedy deterministic value-split search for 16x16 matmul.

This generalizes ``search_preevict_deterministic.py`` from one input split to
several splits.  At each iteration it:

1. builds SSA-like values for the current trace,
2. enumerates legal suffix splits after an already-scheduled read,
3. DP-chain-colors each candidate, and
4. keeps the best improving split.

The split is deterministic and semantics-preserving:

    copy fresh, old

is inserted immediately after a read of ``old`` only when that instruction did
not overwrite ``old``.  Later reads of that same SSA value are redirected to
``fresh``.  This models throwing away the old placement after a hot read while
shuffling the future suffix forward for later reuse.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matmul import score_16x16  # noqa: E402
from matmul.submissions import search_value_coloring as coloring  # noqa: E402

N = 16
DEFAULT_IR = Path(__file__).with_name(
    "output_repacked_tail_deferred_value_colored_live_b_tiny_a_endpoint_16x16.ir"
)


@dataclass(frozen=True)
class ReadSite:
    op_index: int
    operand_pos: int


@dataclass(frozen=True)
class Split:
    value_id: int
    kind: str
    cut_after_read: int
    total_reads: int
    insert_after_op: int
    first_read_op: int
    last_read_op: int

    def describe(self) -> str:
        return (
            f"{self.kind}{self.value_id} after read "
            f"{self.cut_after_read}/{self.total_reads}; "
            f"copy after op {self.insert_after_op}; "
            f"span {self.first_read_op}..{self.last_read_op}"
        )


def parse_ir_text(ir: str) -> tuple[list[int], list[tuple[str, list[int]]], list[int]]:
    lines = [line.strip() for line in ir.splitlines() if line.strip()]
    inputs = [int(part) for part in lines[0].split(",")]
    outputs = [int(part) for part in lines[-1].split(",")]
    ops: list[tuple[str, list[int]]] = []
    for line in lines[1:-1]:
        op, _sep, rest = line.partition(" ")
        ops.append((op, [int(part) for part in rest.split(",")]))
    return inputs, ops, outputs


def emit_raw_ir(
    inputs: list[int],
    ops: list[tuple[str, list[int]]],
    outputs: list[int],
) -> str:
    lines = [",".join(map(str, inputs))]
    for op, operands in ops:
        lines.append(f"{op} {','.join(map(str, operands))}")
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def value_kind(value_id: int) -> str:
    if value_id < N * N:
        return "A"
    if value_id < 2 * N * N:
        return "B"
    return "derived"


def read_sites(
    ops: list[tuple[str, list[int]]],
    op_values: list[list[int]],
    value_count: int,
) -> list[list[ReadSite]]:
    sites: list[list[ReadSite]] = [[] for _ in range(value_count)]
    for op_index, (op, operands) in enumerate(ops):
        for read_number, operand_pos in enumerate(
            coloring.read_positions(op, operands),
            start=1,
        ):
            value_id = op_values[op_index][read_number]
            sites[value_id].append(ReadSite(op_index, operand_pos))
    return sites


def dp_color(
    inputs: list[int],
    ops: list[tuple[str, list[int]]],
    outputs: list[int],
) -> tuple[str, int, int]:
    values, op_values, input_values, output_values = coloring.to_values(
        inputs, ops, outputs
    )
    assignment = coloring.allocate_dp_chains(values)
    ir = coloring.emit_ir(
        inputs,
        ops,
        values,
        op_values,
        input_values,
        output_values,
        assignment,
    )
    score = coloring.current_score(*parse_ir_text(ir)[1:])
    return ir, score, max(assignment.values(), default=0)


def max_addr(inputs: list[int], ops: list[tuple[str, list[int]]]) -> int:
    return max(max(inputs), max(addr for _op, operands in ops for addr in operands))


def split_ops(
    inputs: list[int],
    ops: list[tuple[str, list[int]]],
    sites: list[list[ReadSite]],
    value_id: int,
    cut_after_read: int,
) -> tuple[list[tuple[str, list[int]]], Split] | None:
    value_sites = sites[value_id]
    if not 0 < cut_after_read < len(value_sites):
        return None

    insert_site = value_sites[cut_after_read - 1]
    source_operands = ops[insert_site.op_index][1]
    source_addr = source_operands[insert_site.operand_pos]
    if source_operands[0] == source_addr:
        return None

    fresh_addr = max_addr(inputs, ops) + 1
    redirect = {
        (site.op_index, site.operand_pos)
        for site in value_sites[cut_after_read:]
    }

    new_ops: list[tuple[str, list[int]]] = []
    for op_index, (op, operands) in enumerate(ops):
        new_operands = list(operands)
        for operand_pos in coloring.read_positions(op, operands):
            if (op_index, operand_pos) in redirect:
                new_operands[operand_pos] = fresh_addr
        new_ops.append((op, new_operands))
        if op_index == insert_site.op_index:
            new_ops.append(("copy", [fresh_addr, source_addr]))

    split = Split(
        value_id=value_id,
        kind=value_kind(value_id),
        cut_after_read=cut_after_read,
        total_reads=len(value_sites),
        insert_after_op=insert_site.op_index + 1,
        first_read_op=value_sites[0].op_index + 1,
        last_read_op=value_sites[-1].op_index + 1,
    )
    return new_ops, split


def best_single_split(
    inputs: list[int],
    ops: list[tuple[str, list[int]]],
    outputs: list[int],
    kinds: set[str],
    current_score: int,
) -> tuple[int, int, str, list[tuple[str, list[int]]], Split] | None:
    values, op_values, _input_values, _output_values = coloring.to_values(
        inputs, ops, outputs
    )
    sites = read_sites(ops, op_values, len(values))
    searched = 0
    best: tuple[int, int, str, list[tuple[str, list[int]]], Split] | None = None

    for value_id, value_sites in enumerate(sites):
        kind = value_kind(value_id)
        if kind not in kinds or len(value_sites) < 2:
            continue
        for cut_after_read in range(1, len(value_sites)):
            transformed = split_ops(inputs, ops, sites, value_id, cut_after_read)
            if transformed is None:
                continue
            searched += 1
            candidate_ops, split = transformed
            colored_ir, score, addrs = dp_color(inputs, candidate_ops, outputs)
            if score < current_score and (
                best is None or score < best[0]
            ):
                best = (score, addrs, colored_ir, candidate_ops, split)

    print(f"  searched_legal_splits={searched:,}", flush=True)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ir", nargs="?", type=Path, default=DEFAULT_IR)
    parser.add_argument("--kinds", nargs="+", choices=("A", "B", "derived"), default=("A", "B"))
    parser.add_argument("--max-iters", type=int, default=4)
    parser.add_argument("--write-best", type=Path)
    parser.add_argument("--write-raw", type=Path)
    args = parser.parse_args()

    inputs, ops, outputs = coloring.parse_ir(args.ir)
    current_ir, current_score, current_addrs = dp_color(inputs, ops, outputs)
    raw_ops = list(ops)
    history: list[Split] = []
    print(f"initial score={current_score:,} addrs={current_addrs}")

    for iteration in range(1, args.max_iters + 1):
        print(f"iteration={iteration}", flush=True)
        best = best_single_split(
            inputs,
            raw_ops,
            outputs,
            set(args.kinds),
            current_score,
        )
        if best is None:
            print("  no improving split")
            break
        score, addrs, colored_ir, candidate_ops, split = best
        raw_ops = candidate_ops
        current_ir = colored_ir
        current_score = score
        current_addrs = addrs
        history.append(split)
        verified = score_16x16(current_ir)
        if verified != current_score:
            raise AssertionError((verified, current_score, split))
        print(
            f"  accepted score={current_score:,} addrs={current_addrs} "
            f"{split.describe()}",
            flush=True,
        )

    print(f"final score={current_score:,} addrs={current_addrs} splits={len(history)}")
    for index, split in enumerate(history, start=1):
        print(f"  {index}. {split.describe()}")
    if args.write_best:
        args.write_best.write_text(current_ir + "\n")
        print(f"wrote={args.write_best}")
    if args.write_raw:
        args.write_raw.write_text(emit_raw_ir(inputs, raw_ops, outputs) + "\n")
        print(f"wrote_raw={args.write_raw}")


if __name__ == "__main__":
    main()
