#!/usr/bin/env python3
"""Generate, execute, and score the MNIST-small 1NN v4 instruction stream.

This is a submission-owned interpreter, not an official reference evaluator.
The scoring rules are pinned to simplified-dally-model commit
26abcca402de647381d31286d42dfbb7a001763d. Arithmetic interpretation and tape
serialization are explicit submission conventions; see scoring-notes.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import platform
from pathlib import Path
import re
import struct
import sys
import time

import numpy as np

SPEC_COMMIT = "26abcca402de647381d31286d42dfbb7a001763d"
N_TRAIN, N_TEST, FEATURES = 600, 600, 9
TEMP, DIST, BEST_DIST, BEST_LABEL, COND = 10, 11, 12, 13, 14
TRAIN_BASE = 15
LABEL_BASE = TRAIN_BASE + N_TRAIN * FEATURES
WORDS = LABEL_BASE + N_TRAIN - 1
EXPECTED_INPUT_WORDS = N_TRAIN * FEATURES + N_TRAIN + N_TEST * FEATURES
EXPECTED_OUTPUT_WORDS = N_TEST
F32 = struct.Struct("<f")
U32 = struct.Struct("<I")


def bits_to_float(word):
    return F32.unpack(U32.pack(int(word)))[0]


def float_to_bits(value):
    return U32.unpack(F32.pack(value))[0]


def placement(words=WORDS):
    """Address 1 onward fills half-diamond shells, then ascending x."""
    coords = [None]
    h = 1
    while len(coords) <= words:
        for x in range(-(h - 1), h):
            if len(coords) > words:
                break
            coords.append((x, h - abs(x)))
        h += 1
    return coords


def program(n_train=N_TRAIN, n_test=N_TEST, features=FEATURES):
    """Yield actual v4 instructions. Loops here only generate straight-line IR.

    Tuple forms are (opcode, destination, sources...), except send's sole
    operand is its source and set's final argument is an immediate word.
    """
    label_base = TRAIN_BASE + n_train * features
    for address in range(TRAIN_BASE, label_base + n_train):
        yield ("recv", address)
    for _ in range(n_test):
        for pixel in range(features):
            yield ("recv", pixel + 1)
        for row in range(n_train):
            yield ("set", DIST, 0)
            for pixel in range(features):
                yield ("sub", TEMP, pixel + 1, TRAIN_BASE + row * features + pixel)
                yield ("mul", TEMP, TEMP, TEMP)
                yield ("add", DIST, DIST, TEMP)
            if row == 0:
                yield ("copy", BEST_DIST, DIST)
                yield ("copy", BEST_LABEL, label_base)
            else:
                yield ("cmp", COND, DIST, BEST_DIST)
                yield ("select", BEST_DIST, COND, DIST, BEST_DIST)
                yield ("select", BEST_LABEL, COND, label_base + row, BEST_LABEL)
        yield ("send", BEST_LABEL)


def instruction_text(inst):
    op, *args = inst
    if op == "send":
        return f"send %{args[0]}\n"
    dest, *sources = args
    if op == "recv":
        return f"%{dest} = recv\n"
    if op == "set":
        return f"%{dest} = set {sources[0]}\n"
    operands = ", ".join(f"%{source}" for source in sources)
    if op == "cmp":
        operands += ", lt"
    return f"%{dest} = {op} {operands}\n"


def parse_program(path):
    """Read the exported straight-line v4 subset without implicit operations."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="ascii") as handle:
        for number, line in enumerate(handle, 1):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            if match := re.fullmatch(r"send %(\d+)", text):
                yield ("send", int(match[1]))
                continue
            match = re.fullmatch(r"%(\d+) = (recv|set|sub|mul|add|copy|cmp|select)(?: (.*))?", text)
            if not match:
                raise ValueError(f"Unsupported v4 text at line {number}: {text}")
            dest, op, arguments = int(match[1]), match[2], match[3]
            if op == "recv":
                if arguments is not None:
                    raise ValueError(f"recv arguments at line {number}")
                yield (op, dest)
                continue
            if op == "set":
                if arguments is None or not re.fullmatch(r"\d+", arguments):
                    raise ValueError(f"Invalid set literal at line {number}")
                yield (op, dest, int(arguments))
                continue
            arguments = (arguments or "").split(", ")
            if op == "cmp":
                if arguments[-1] != "lt":
                    raise ValueError(f"Only cmp lt is supported at line {number}")
                arguments.pop()
            arity = {"sub": 2, "mul": 2, "add": 2, "copy": 1, "cmp": 2, "select": 3}[op]
            if len(arguments) != arity or any(not re.fullmatch(r"%\d+", item) for item in arguments):
                raise ValueError(f"Invalid operand list at line {number}")
            yield (op, dest, *(int(item[1:]) for item in arguments))


class Machine:
    """Raw-word scratch, source-before-destination execution, exact costs.

    Time accumulates in 0.2 ps integer ticks, avoiding floating-point cost
    accumulation. Float arithmetic rounds each instruction to binary32.
    Python binary64 exactly represents a binary32 product and is sufficient
    for these bounded nonnegative distance sums before binary32 rounding.
    """

    def __init__(self, coordinates, tape, arithmetic="fp32"):
        if arithmetic not in ("fp32", "u32"):
            raise ValueError("Unsupported arithmetic convention")
        self.arithmetic = arithmetic
        self.coordinates = coordinates
        assert len(set(coordinates[1:])) == len(coordinates) - 1
        assert all(-16000 <= x <= 16000 and 1 <= y <= 16000
                   for x, y in coordinates[1:])
        hops = [0] + [abs(x) + y for x, y in coordinates[1:]]
        self.access_energy = [max(50, 2 * h) for h in hops]
        self.read_ticks = [max(250, 4 * h) for h in hops]
        self.write_ticks = [max(250, 2 * h) for h in hops]
        self.memory = [None] * len(coordinates)
        self.input = [int(word) for word in tape]
        assert all(0 <= word <= 0xffffffff for word in self.input)
        self.input_position = 0
        self.output = []
        self.energy_fj = 0
        self.time_ticks = 0
        self.instructions = Counter()
        self.read_counts = [0] * len(coordinates)
        self.write_counts = [0] * len(coordinates)
        self.initialized_words = 0
        self.peak_initialized_words = 0

    def _valid(self, address):
        if not isinstance(address, int) or not 1 <= address < len(self.memory):
            raise ValueError(f"Unallocated scratch address {address}")

    def step(self, inst):
        op, d, *args = inst
        self._valid(d)
        self.instructions[op] += 1
        if op == "recv":
            if self.input_position == len(self.input):
                raise ValueError("Read past end of input tape")
            value = self.input[self.input_position]
            self.input_position += 1
        elif op == "send":
            if self.memory[d] is None:
                raise ValueError("Uninitialized output source")
            self.output.append(self.memory[d])
            return
        else:
            sources = [] if op == "set" else args
            values = []
            for address in sources:
                self._valid(address)
                value = self.memory[address]
                if value is None:
                    raise ValueError(f"Uninitialized source %{address}")
                values.append(value)
                self.energy_fj += self.access_energy[address]
                self.time_ticks += self.read_ticks[address]
                self.read_counts[address] += 1
            self.energy_fj += self.access_energy[d]
            self.time_ticks += self.write_ticks[d]
            self.write_counts[d] += 1
            if op == "set":
                value = args[0]
            elif op == "copy":
                value = values[0]
            elif op == "select":
                value = values[1] if values[0] else values[2]
            elif op in ("add", "sub", "mul", "cmp"):
                a, b = values
                if self.arithmetic == "fp32":
                    a, b = bits_to_float(a), bits_to_float(b)
                if op == "cmp":
                    value = int(a < b)
                else:
                    result = a + b if op == "add" else a - b if op == "sub" else a * b
                    value = float_to_bits(result) if self.arithmetic == "fp32" else result & 0xffffffff
            else:
                raise ValueError(f"Unsupported opcode {op}")
        if not 0 <= value <= 0xffffffff:
            raise ValueError("Result is not a 32-bit word")
        if self.memory[d] is None:
            self.initialized_words += 1
            self.peak_initialized_words = max(self.peak_initialized_words, self.initialized_words)
        self.memory[d] = value

    def run(self, instructions):
        for inst in instructions:
            self.step(inst)
        if self.input_position != len(self.input):
            raise ValueError("Input tape was not consumed completely")
        return np.asarray(self.output, dtype=np.uint32)


def input_tape(dataset):
    # Only these three NPZ entries are accessed. No test-label dependency.
    with np.load(dataset, allow_pickle=False) as data:
        train = np.ascontiguousarray(data["train_images"], dtype="<f4")
        labels = data["train_labels"]
        test = np.ascontiguousarray(data["test_images"], dtype="<f4")
    assert train.shape == (600, 1, 3, 3) and test.shape == (600, 1, 3, 3)
    assert labels.shape == (600,) and np.issubdtype(labels.dtype, np.integer)
    assert np.all((labels >= 0) & (labels <= 9))
    assert np.isfinite(train).all() and np.isfinite(test).all()
    assert np.all((train >= 0) & (train <= 1))
    assert np.all((test >= 0) & (test <= 1))
    return np.concatenate((train.reshape(-1).view("<u4"),
                           labels.astype("<u4"), test.reshape(-1).view("<u4")))


def independent_predictions(tape):
    """Independent vectorized FP32 distance oracle; includes no test labels."""
    train = tape[:5400].view("<f4").reshape(600, 9)
    labels = tape[5400:6000]
    test = tape[6000:].view("<f4").reshape(600, 9)
    distances = np.zeros((600, 600), dtype=np.float32)
    for pixel in range(9):
        diff = test[:, pixel, None] - train[None, :, pixel]
        distances = np.add(distances, np.multiply(diff, diff))
    return labels[np.argmin(distances, axis=1)]


def expected_counts():
    return {"recv": 11400, "send": 600, "set": 360000,
            "sub": 3240000, "mul": 3240000, "add": 3240000,
            "copy": 1200, "cmp": 359400, "select": 718800}


def closed_form_cost(coordinates):
    """Independent count formula: each training word read once per query.

    Every non-training source/destination has a 50-unit access floor. The
    first row costs set+27 arithmetic+2 copy = 86 accesses/query; each
    subsequent row costs set+27 arithmetic+cmp+2 select = 93 accesses.
    """
    accesses = N_TEST * (86 + (N_TRAIN - 1) * 93)
    persistent_reads = N_TEST * N_TRAIN * (FEATURES + 1)
    hot_accesses = accesses - persistent_reads
    hops = [abs(x) + y for x, y in coordinates[TRAIN_BASE:]]
    energy = hot_accesses * 50 + N_TEST * sum(max(50, 2 * h) for h in hops)
    ticks = hot_accesses * 250 + N_TEST * sum(max(250, 4 * h) for h in hops)
    return {"energy_fj": energy, "time_ticks": ticks,
            "charged_accesses": accesses, "persistent_reads": persistent_reads,
            "hot_accesses": hot_accesses}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def self_test():
    # Published cost example (integer arithmetic mode), including source/dest alias.
    example = Machine([None, (3, 4), (0, 1), (0, 100)], [0x123456ab], "u32")
    output = example.run([("recv", 1), ("set", 2, 1), ("add", 1, 1, 2),
                          ("copy", 3, 1), ("send", 3)])
    assert output.tolist() == [0x123456ac]
    assert (example.energy_fj, example.time_ticks) == (450, 1500)
    # Far recv/send do not inherit floors or any implicit final-output charge.
    passthrough = Machine([None, (16000, 16000)], [0xffffffff])
    assert passthrough.run([("recv", 1), ("send", 1)]).tolist() == [0xffffffff]
    assert passthrough.energy_fj == passthrough.time_ticks == 0
    # Reject read-before-write even for a non-selected select operand.
    for inst in [("send", 1), ("copy", 1, 2), ("select", 1, 1, 1, 2)]:
        machine = Machine([None, (0, 1), (0, 2)], [])
        if inst[0] == "select":
            machine.step(("set", 1, 1))
        try:
            machine.step(inst)
        except ValueError:
            pass
        else:
            raise AssertionError("Uninitialized read accepted")
    # Independent 1D toy data: equidistant rows choose the first training label.
    toy = np.array([0.0, 2.0], dtype="<f4").view("<u4").tolist()
    toy += [7, 3]
    toy += np.array([1.0, 2.0], dtype="<f4").view("<u4").tolist()
    machine = Machine(placement(18), toy)
    assert machine.run(program(n_train=2, n_test=2, features=1)).tolist() == [7, 3]
    assert machine.instructions["recv"] == 6 and machine.instructions["send"] == 2
    print("Self-tests passed: published example, free tape I/O, uninitialized reads, ties.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("model-results"))
    parser.add_argument("--emit-ir", type=Path,
                        help="Optional fully expanded standard v4 text (.gz supported)")
    parser.add_argument("--replay-ir", type=Path,
                        help="Execute saved expanded v4 text instead of regenerating it")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.data is None:
        if args.self_test:
            return
        parser.error("--data is required unless only running --self-test")
    args.output.mkdir(parents=True, exist_ok=True)
    tape = input_tape(args.data)
    assert len(tape) == EXPECTED_INPUT_WORDS
    coordinates = placement()
    # Declared scoring clock starts before machine construction and includes
    # all instruction generation, validation, FP32 execution, and accounting.
    start = time.perf_counter()
    machine = Machine(coordinates, tape)
    predictions = machine.run(parse_program(args.replay_ir) if args.replay_ir else program())
    time_to_score = time.perf_counter() - start
    assert len(predictions) == EXPECTED_OUTPUT_WORDS
    assert machine.instructions == Counter(expected_counts())
    assert machine.peak_initialized_words == WORDS
    formula = closed_form_cost(coordinates)
    assert machine.energy_fj == formula["energy_fj"]
    assert machine.time_ticks == formula["time_ticks"]
    assert sum(machine.read_counts) + sum(machine.write_counts) == formula["charged_accesses"]
    assert all(count == N_TEST for count in machine.read_counts[TRAIN_BASE:])
    oracle = independent_predictions(tape)
    assert np.array_equal(predictions, oracle), "Interpreter disagrees with independent oracle"
    np.save(args.output / "model-predictions.npy", predictions.astype(np.int64))
    tape.astype("<u4").tofile(args.output / "input-tape.u32le")
    predictions.astype("<u4").tofile(args.output / "output-tape.u32le")
    with open(args.output / "placement.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["address", "x_um", "y_um", "manhattan_hops", "charged_reads", "charged_writes"])
        for address, (x, y) in enumerate(coordinates[1:], 1):
            writer.writerow([address, x, y, abs(x) + y,
                             machine.read_counts[address], machine.write_counts[address]])
    x, y = zip(*coordinates[1:])
    result = {
        "model": "single-core-with-tape", "instruction_set": "v4",
        "model_spec_commit": SPEC_COMMIT,
        "scorer_kind": "submission-owned full instruction interpreter",
        "arithmetic": "FP32 round-to-nearest-ties-to-even per add/sub/mul; no FMA; cmp is FP32 lt; raw-word select/copy",
        "time_ps": machine.time_ticks / 5,
        "time_seconds": machine.time_ticks / 5 * 1e-12,
        "energy_fj": machine.energy_fj,
        "energy_joules": machine.energy_fj * 1e-15,
        "area_um2_occupied_cells": WORDS,
        "area_mm2_occupied_cells": WORDS * 1e-6,
        "bounding_rectangle_um2": (max(x) - min(x) + 1) * (max(y) - min(y) + 1),
        "peak_allocated_scratch_words": WORDS,
        "peak_initialized_scratch_words": machine.peak_initialized_words,
        "peak_allocated_scratch_bytes": WORDS * 4,
        "max_manhattan_hops": max(abs(a) + b for a, b in coordinates[1:]),
        "time_to_score_seconds": time_to_score,
        "time_to_score_scope": "perf_counter around Machine construction plus full streamed instruction generation (or text parsing if replayed), uninitialized-read checks, FP32 execution and integer accounting; excludes input loading, placement generation, independent verification, file output and optional IR emission",
        "instruction_source": args.replay_ir.name if args.replay_ir else "Python generator",
        "instructions": dict(machine.instructions),
        "total_instructions": sum(machine.instructions.values()),
        "charged_reads": sum(machine.read_counts),
        "charged_writes": sum(machine.write_counts),
        "input_tape_words": machine.input_position,
        "output_tape_words": len(machine.output),
        "input_tape_sha256": hashlib.sha256(tape.astype("<u4").tobytes()).hexdigest(),
        "output_tape_sha256": hashlib.sha256(predictions.astype("<u4").tobytes()).hexdigest(),
        "prediction_npy_sha256": sha256_file(args.output / "model-predictions.npy"),
        "source_sha256": sha256_file(Path(__file__)),
        "independent_fp32_predictions_match": True,
        "independent_closed_form_scores_match": True,
        "closed_form_counts": formula,
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "processor": platform.processor(), "python": sys.version,
                 "numpy": np.__version__},
        "limitations": [
            "v4 does not specify numeric types/rounding; FP32 interpretation is a submission convention.",
            "Workload does not specify canonical word tape serialization; this submission declares one.",
            "Area is occupied scratch cells at 1 um pitch; no processor, tape, instruction-store, or routing area is specified or included.",
            "Instruction and scoring execution time is host interpreter time, not modeled end-to-end physical time.",
            "Only the v4 opcode subset used by this submission is implemented; scorer is not an official reference evaluator."
        ],
    }
    if args.emit_ir:
        args.emit_ir.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if args.emit_ir.suffix == ".gz" else open
        digest = hashlib.sha256()
        with opener(args.emit_ir, "wt", encoding="ascii", newline="\n") as handle:
            for inst in program():
                text = instruction_text(inst)
                digest.update(text.encode("ascii"))
                handle.write(text)
        result["expanded_ir_uncompressed_sha256"] = digest.hexdigest()
        result["expanded_ir_file"] = args.emit_ir.name
        result["expanded_ir_file_bytes"] = args.emit_ir.stat().st_size
    with open(args.output / "model-score.json", "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
