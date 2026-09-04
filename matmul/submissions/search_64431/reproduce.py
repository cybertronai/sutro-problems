"""Rebuild the 64,431 submission from a self-contained schedule generator."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO)]

import core
from pair_alloc import improve
from matmul.submissions.search_value_coloring import Value, allocate_dp_chains
import numpy
import ortools


COLUMN_WIDTHS = (6, 10)
ROW_WIDTHS_BY_PANEL = ((4, 4, 4, 4), (5, 5, 6))
CHUNKS_BY_PANEL = (1, 2)
K_OFFSETS_BY_PANEL = ((3, 11, 0, 0), (0, 0, 4))
REVERSE_BY_PANEL = ((False, False, False, False), (True, False, True))
CAPTURE_THRESHOLD = 8
MOVES = ((7902, 7905, 2), (7893, 7991, 1), (1823, 1825, 1),
         (226, 222, 4), (1281, 1287, 1))

EXPECTED = {
    "captured_generator": (1177063, "6dca745d2e7fff3e23794dc28c6481fde140bb522922ab34c06d9e2c3cfb089a"),
    "reordered_generator": (1177063, "3c86365a522ca3909e8b1a892ef38e9edc21f3c54eb873f074854ff7e638be13"),
    "dp_initial": (64458, "0393e6775e7b73ea8f117dd541247a07f5e8de146359b2ac422cf87d9155f184"),
    "pair2_seed0_no_ties": (64431, "66929dab27c72e9714bf8a1ae77f1942b4201fe81d0d7a255a8547108eabadc3"),
    "pair5_seed11_ties": (64431, "9d94114a87fecd30168fbcf63931bbc98a50778984a11fe0c3b16940218bcf11"),
}
EXPECTED_HISTORIES = {
    "pair2_seed0_no_ties": [{"round": 0, "tiers": [2, 3], "gain": 27, "score": 64431}],
    "pair5_seed11_ties": [],
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def groups(widths):
    answer, start = [], 0
    for width in widths:
        answer.append(list(range(start, start + width)))
        start += width
    assert start == 16
    return answer


def generate_base():
    """Emit the literal seven-tile geometry before captures or reordering."""
    ops, outputs = [], {}
    next_value = 512

    def emit(code, *sources):
        nonlocal next_value
        dest = next_value
        next_value += 1
        ops.append(core.Op(dest, code, sources))
        return dest

    panels = zip(groups(COLUMN_WIDTHS), ROW_WIDTHS_BY_PANEL, CHUNKS_BY_PANEL,
                 K_OFFSETS_BY_PANEL, REVERSE_BY_PANEL)
    for columns, row_widths, chunk, offsets, reversals in panels:
        row_groups = groups(row_widths)
        assert len(row_groups) == len(offsets) == len(reversals)
        for rows, offset, reverse in zip(row_groups, offsets, reversals):
            contraction = [(k + offset) % 16 for k in range(16)]
            assert sorted(contraction) == list(range(16))
            accumulators = {}
            for chunk_id, start_k in enumerate(range(0, 16, chunk)):
                staged_a = {}
                traversal = list(reversed(columns)) if (chunk_id % 2) ^ reverse else columns
                for j in traversal:
                    staged_b = {}
                    for i in rows:
                        local = accumulators.get((i, j))
                        for k in contraction[start_k:start_k + chunk]:
                            if k not in staged_b:
                                staged_b[k] = emit("copy", 256 + k * 16 + j)
                            if (i, k) not in staged_a:
                                staged_a[i, k] = emit("copy", i * 16 + k)
                            product = emit("mul", staged_a[i, k], staged_b[k])
                            local = product if local is None else emit("add", local, product)
                        accumulators[i, j] = local
            assert not set(outputs).intersection(accumulators)
            outputs.update(accumulators)
    assert set(outputs) == {(i, j) for i in range(16) for j in range(16)}
    return core.Program(tuple(range(512)), ops,
                        tuple(outputs[i, j] for i in range(16) for j in range(16)),
                        {v: v + 1 for v in range(next_value)})


def capture_b(program):
    """Retain first staged B[k,j] for columns 8..15."""
    counts = Counter(op.src[0] for op in program.ops if op.code == "copy")
    result = program.clone()
    result.ops = []
    saved = {}
    next_value = max(program.assignment) + 1
    for op in program.ops:
        origin = op.src[0]
        selected = (op.code == "copy" and 256 <= origin < 512
                    and counts[origin] > 1 and (origin - 256) % 16 >= CAPTURE_THRESHOLD)
        if not selected:
            result.ops.append(op)
        elif origin in saved:
            result.ops.append(replace(op, src=(saved[origin],)))
        else:
            result.ops.append(op)
            saved[origin] = next_value
            result.ops.append(core.Op(next_value, "copy", (op.dest,)))
            result.assignment[next_value] = next_value + 1
            next_value += 1
    assert len(saved) == 128
    return result


def reorder(program):
    """Apply the five dependency-safe operation-list edits."""
    frozen = {op.dest: op for op in program.ops}
    for source, destination, length in MOVES:
        block = program.ops[source:source + length]
        del program.ops[source:source + length]
        program.ops[destination:destination] = block
        seen = set(program.inputs)
        for op in program.ops:
            assert all(value in seen for value in op.src)
            assert op.dest not in seen and op == frozen[op.dest]
            seen.add(op.dest)
        assert len(program.ops) == len(frozen)
        assert all(value in seen for value in program.outputs)
    return program


def initial_allocation(program):
    lives = core.lifetimes(program)
    assignment = allocate_dp_chains(
        [Value(value, start, end, reads)
         for value, (start, end, reads) in lives.items()]
    )
    fresh = max(assignment.values()) + 1
    for value in lives:
        if value not in assignment:
            assignment[value] = fresh
            fresh += 1
    result = program.clone()
    result.assignment = assignment
    core.check_assignment(result)
    return result


def reproduce(output=None, manifest=None):
    started = time.monotonic()
    stages = []

    def checkpoint(name, program, history=None):
        ir = core.emit(program)
        proof = core.verify_ir(ir)
        expected_score, expected_hash = EXPECTED[name]
        assert (proof["score"], proof["sha256"]) == (expected_score, expected_hash), (name, proof)
        record = {"name": name, **proof}
        if history is not None:
            assert history == EXPECTED_HISTORIES[name]
            record["accepted_history"] = history
        stages.append(record)
        print(json.dumps({"stage": name, "score": expected_score,
                          "sha256": expected_hash}), flush=True)
        return ir

    program = capture_b(generate_base())
    checkpoint("captured_generator", program)
    program = reorder(program)
    checkpoint("reordered_generator", program)
    program = initial_allocation(program)
    program = core.parse_ir(checkpoint("dp_initial", program))
    for name, rounds, seed, ties in (
        ("pair2_seed0_no_ties", 2, 0, False),
        ("pair5_seed11_ties", 5, 11, True),
    ):
        program, history = improve(program, rounds=rounds, seed=seed,
                                   max_seconds=float("inf"), allow_ties=ties,
                                   checkpoint=None)
        checkpoint(name, program, history)

    ir = core.emit(program)
    submission = HERE.parent / "best_64431.ir"
    assert ir.encode() == submission.read_bytes(), "Replay differs from submission"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(ir)
    result = {
        "score": 64431,
        "sha256": EXPECTED["pair5_seed11_ties"][1],
        "fresh_process_replay": True,
        "byte_identical_to_submission": True,
        "elapsed_seconds": time.monotonic() - started,
        "configuration": {
            "column_widths": COLUMN_WIDTHS,
            "row_widths_by_panel": ROW_WIDTHS_BY_PANEL,
            "chunks_by_panel": CHUNKS_BY_PANEL,
            "k_offsets_by_panel": K_OFFSETS_BY_PANEL,
            "reverse_by_panel": REVERSE_BY_PANEL,
            "capture_threshold": CAPTURE_THRESHOLD,
            "moves": MOVES,
        },
        "versions": {"python": platform.python_version(), "numpy": numpy.__version__,
                     "ortools": ortools.__version__},
        "helpers": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), HERE / "core.py", HERE / "pair_alloc.py")
        },
        "stages": stages,
    }
    if manifest is not None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(result, indent=2) + "\n")
    print("Self-contained replay is byte-identical to best_64431.ir.", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    reproduce(args.output, args.manifest)
