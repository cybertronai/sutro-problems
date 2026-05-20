"""Pressure-ranked deterministic pre-eviction search for 16x16 matmul.

This is the interactive hill-climb version of ``search_preevict_greedy``.  It
does not scan every legal split.  Instead it ranks split points by a simple
future-pressure heuristic:

    future_reads * lifetime_suffix_length

and DP-colors only the top candidates per iteration.  The goal is to test the
"keep far, push close near use, then throw away" idea quickly enough to run
several simulations.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matmul import score_16x16  # noqa: E402
from matmul.submissions import search_preevict_greedy as greedy  # noqa: E402
from matmul.submissions import search_value_coloring as coloring  # noqa: E402

DEFAULT_IR = greedy.DEFAULT_IR


@dataclass(frozen=True)
class RankedSplit:
    pressure: int
    value_id: int
    cut_after_read: int
    future_reads: int
    suffix_span: int
    kind: str


def ranked_splits(
    inputs: list[int],
    ops: list[tuple[str, list[int]]],
    outputs: list[int],
    kinds: set[str],
    rank: str,
) -> list[RankedSplit]:
    values, op_values, _input_values, _output_values = coloring.to_values(
        inputs, ops, outputs
    )
    sites = greedy.read_sites(ops, op_values, len(values))
    ranked: list[RankedSplit] = []
    for value_id, value_sites in enumerate(sites):
        kind = greedy.value_kind(value_id)
        if kind not in kinds or len(value_sites) < 2:
            continue
        for cut_after_read in range(1, len(value_sites)):
            first_future = value_sites[cut_after_read].op_index + 1
            last_future = value_sites[-1].op_index + 1
            future_reads = len(value_sites) - cut_after_read
            suffix_span = last_future - first_future
            pressure = future_reads * max(1, suffix_span)
            ranked.append(
                RankedSplit(
                    pressure=pressure,
                    value_id=value_id,
                    cut_after_read=cut_after_read,
                    future_reads=future_reads,
                    suffix_span=suffix_span,
                    kind=kind,
                )
            )
    if rank == "reads":
        key = lambda item: (
            -item.future_reads,
            -item.pressure,
            -item.suffix_span,
            item.kind,
            item.value_id,
            item.cut_after_read,
        )
    elif rank == "span":
        key = lambda item: (
            -item.suffix_span,
            -item.future_reads,
            -item.pressure,
            item.kind,
            item.value_id,
            item.cut_after_read,
        )
    else:
        key = lambda item: (
            -item.pressure,
            -item.future_reads,
            -item.suffix_span,
            item.kind,
            item.value_id,
            item.cut_after_read,
        )
    return sorted(ranked, key=key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ir", nargs="?", type=Path, default=DEFAULT_IR)
    parser.add_argument("--kinds", nargs="+", choices=("A", "B", "derived"), default=("A", "B"))
    parser.add_argument("--top", type=int, default=160)
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument("--rank", choices=("pressure", "reads", "span"), default="pressure")
    parser.add_argument("--write-best", type=Path)
    parser.add_argument("--write-raw", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument(
        "--checkpoint-each-accept",
        action="store_true",
        help="Update --write-best/--write-raw after every accepted split.",
    )
    args = parser.parse_args()

    inputs, raw_ops, outputs = coloring.parse_ir(args.ir)
    current_ir, current_score, current_addrs = greedy.dp_color(inputs, raw_ops, outputs)
    history: list[greedy.Split] = []
    print(f"initial score={current_score:,} addrs={current_addrs}")

    for iteration in range(1, args.max_iters + 1):
        candidates = ranked_splits(inputs, raw_ops, outputs, set(args.kinds), args.rank)[: args.top]
        print(
            f"iteration={iteration} evaluating_top={len(candidates)} "
            f"rank={args.rank}",
            flush=True,
        )
        best: tuple[int, int, str, list[tuple[str, list[int]]], greedy.Split] | None = None
        values, op_values, _input_values, _output_values = coloring.to_values(
            inputs, raw_ops, outputs
        )
        sites = greedy.read_sites(raw_ops, op_values, len(values))
        for candidate in candidates:
            transformed = greedy.split_ops(
                inputs,
                raw_ops,
                sites,
                candidate.value_id,
                candidate.cut_after_read,
            )
            if transformed is None:
                continue
            candidate_ops, split = transformed
            colored_ir, score, addrs = greedy.dp_color(inputs, candidate_ops, outputs)
            if score < current_score and (best is None or score < best[0]):
                best = (score, addrs, colored_ir, candidate_ops, split)
                print(
                    f"  candidate score={score:,} pressure={candidate.pressure:,} "
                    f"{split.describe()}",
                    flush=True,
                )
        if best is None:
            print("  no improving pressure-ranked split")
            break
        score, addrs, colored_ir, raw_ops, split = best
        verified = score_16x16(colored_ir)
        if verified != score:
            raise AssertionError((verified, score, split))
        current_ir = colored_ir
        current_score = score
        current_addrs = addrs
        history.append(split)
        print(f"  accepted score={score:,} addrs={addrs} {split.describe()}", flush=True)
        if args.checkpoint_each_accept:
            if args.write_best:
                args.write_best.write_text(current_ir + "\n")
                print(f"  checkpoint={args.write_best}", flush=True)
            if args.write_raw:
                args.write_raw.write_text(greedy.emit_raw_ir(inputs, raw_ops, outputs) + "\n")
                print(f"  checkpoint_raw={args.write_raw}", flush=True)

    print(f"final score={current_score:,} addrs={current_addrs} splits={len(history)}")
    for index, split in enumerate(history, start=1):
        print(f"  {index}. {split.describe()}")
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
                    "name": "pressure_preevict",
                    "start_ir": str(args.ir),
                    "score": current_score,
                    "addrs": current_addrs,
                    "rank": args.rank,
                    "kinds": list(args.kinds),
                    "top": args.top,
                    "max_iters": args.max_iters,
                    "splits": [split.describe() for split in history],
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
