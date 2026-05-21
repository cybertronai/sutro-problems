"""General copy-removal hill climb for colored matmul raw IRs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matmul import score_16x16  # noqa: E402
from matmul.submissions import search_preevict_greedy as greedy  # noqa: E402
from matmul.submissions import search_value_coloring as coloring  # noqa: E402


def remove_self_copies(
    ops: list[tuple[str, list[int]]],
) -> tuple[list[tuple[str, list[int]]], int]:
    kept = [
        (op, list(operands))
        for op, operands in ops
        if not (op == "copy" and operands[0] == operands[1])
    ]
    return kept, len(ops) - len(kept)


def try_remove_copy(
    ops: list[tuple[str, list[int]]],
    index: int,
) -> tuple[list[tuple[str, list[int]]], int] | None:
    op, operands = ops[index]
    if op != "copy":
        return None
    dest, src = operands
    if dest == src:
        new_ops = [(op, list(operands)) for op, operands in ops]
        del new_ops[index]
        return new_ops, 0

    new_ops: list[tuple[str, list[int]]] = []
    redirected = 0
    active = True
    for op_index, (next_op, next_operands) in enumerate(ops):
        if op_index == index:
            continue
        rewritten = list(next_operands)
        if op_index > index and active:
            if rewritten[0] == src:
                return None
            for pos in coloring.read_positions(next_op, rewritten):
                if rewritten[pos] == dest:
                    rewritten[pos] = src
                    redirected += 1
            if rewritten[0] == dest:
                active = False
        new_ops.append((next_op, rewritten))
    if redirected == 0:
        return None
    return new_ops, redirected


def copy_count(ops: list[tuple[str, list[int]]]) -> int:
    return sum(1 for op, _operands in ops if op == "copy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ir", type=Path)
    parser.add_argument("--top", type=int, default=2000)
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--write-best", type=Path)
    parser.add_argument("--write-raw", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--checkpoint-each-accept", action="store_true")
    parser.add_argument(
        "--order",
        choices=("forward", "reverse", "dest_desc", "src_desc"),
        default="forward",
    )
    args = parser.parse_args()

    inputs, raw_ops, outputs = coloring.parse_ir(args.ir)
    raw_ops, removed_self = remove_self_copies(raw_ops)
    current_ir, current_score, current_addrs = greedy.dp_color(inputs, raw_ops, outputs)
    if score_16x16(current_ir) != current_score:
        raise AssertionError(current_score)
    print(
        f"initial score={current_score:,} addrs={current_addrs} "
        f"copies={copy_count(raw_ops)} removed_self={removed_self}",
        flush=True,
    )

    history: list[dict[str, int]] = []
    for iteration in range(1, args.max_iters + 1):
        copy_indexes = [
            index
            for index, (op, operands) in enumerate(raw_ops)
            if op == "copy" and operands[0] != operands[1]
        ]
        if args.order == "reverse":
            copy_indexes = list(reversed(copy_indexes))
        elif args.order == "dest_desc":
            copy_indexes = sorted(
                copy_indexes,
                key=lambda index: (raw_ops[index][1][0], index),
                reverse=True,
            )
        elif args.order == "src_desc":
            copy_indexes = sorted(
                copy_indexes,
                key=lambda index: (raw_ops[index][1][1], index),
                reverse=True,
            )
        copy_indexes = copy_indexes[: args.top]
        print(
            f"iteration={iteration} evaluating_copies={len(copy_indexes)} "
            f"order={args.order}",
            flush=True,
        )
        best: tuple[int, int, str, list[tuple[str, list[int]]], int, int] | None = None
        for index in copy_indexes:
            transformed = try_remove_copy(raw_ops, index)
            if transformed is None:
                continue
            candidate_ops, redirected = transformed
            colored_ir, score, addrs = greedy.dp_color(inputs, candidate_ops, outputs)
            if score < current_score and (best is None or score < best[0]):
                best = (score, addrs, colored_ir, candidate_ops, index, redirected)
                print(
                    f"  candidate score={score:,} copy_op={index + 1} "
                    f"redirected={redirected}",
                    flush=True,
                )
        if best is None:
            print("  no improving copy removal")
            break
        score, addrs, current_ir, raw_ops, index, redirected = best
        if score_16x16(current_ir) != score:
            raise AssertionError(score)
        current_score = score
        current_addrs = addrs
        history.append({"copy_op": index + 1, "redirected": redirected, "score": score})
        print(
            f"  accepted score={score:,} addrs={addrs} "
            f"copy_op={index + 1} redirected={redirected}",
            flush=True,
        )
        if args.checkpoint_each_accept:
            if args.write_best:
                args.write_best.write_text(current_ir + "\n")
                print(f"  checkpoint={args.write_best}", flush=True)
            if args.write_raw:
                args.write_raw.write_text(greedy.emit_raw_ir(inputs, raw_ops, outputs) + "\n")
                print(f"  checkpoint_raw={args.write_raw}", flush=True)

    print(
        f"final score={current_score:,} addrs={current_addrs} "
        f"copies={copy_count(raw_ops)} moves={len(history)}"
    )
    if args.write_best:
        args.write_best.write_text(current_ir + "\n")
        print(f"wrote={args.write_best}")
    if args.write_raw:
        args.write_raw.write_text(greedy.emit_raw_ir(inputs, raw_ops, outputs) + "\n")
        print(f"wrote_raw={args.write_raw}")
    if args.journal:
        args.journal.parent.mkdir(parents=True, exist_ok=True)
        args.journal.write_text(
            json.dumps(
                {
                    "name": "copy_elim_general",
                    "start_ir": str(args.ir),
                    "score": current_score,
                    "addrs": current_addrs,
                    "removed_self": removed_self,
                    "moves": history,
                    "best_artifact": str(args.write_best) if args.write_best else None,
                    "raw_artifact": str(args.write_raw) if args.write_raw else None,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"journal={args.journal}")


if __name__ == "__main__":
    main()
