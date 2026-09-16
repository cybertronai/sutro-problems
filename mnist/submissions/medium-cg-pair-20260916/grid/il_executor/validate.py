#!/usr/bin/env python3
"""Generate independent interpreter scratch snapshots and compare C++ output."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def execute_with_memory(document, tape_in):
    """The pair_ir.execute primitive dispatch with its final memory exposed."""
    from affine import Program, expand
    program = Program(document)
    mem = np.zeros(program.words + 1, dtype=np.uint32)
    fmem, tape, out = mem.view(np.float32), iter(tape_in), []
    with np.errstate(all="ignore"):
        for op in expand(document):
            code = op[0]
            if code == "set": mem[op[1]] = np.uint32(op[2])
            elif code == "recv": mem[op[1]] = np.uint32(next(tape))
            elif code == "send": out.append(int(mem[op[1]]))
            elif code == "copy": mem[op[1]] = mem[op[2]]
            elif code == "add": fmem[op[1]] = fmem[op[2]] + fmem[op[3]]
            elif code == "sub": fmem[op[1]] = fmem[op[2]] - fmem[op[3]]
            elif code == "mul": fmem[op[1]] = fmem[op[2]] * fmem[op[3]]
            elif code == "div": fmem[op[1]] = fmem[op[2]] / fmem[op[3]]
            elif code == "cmp": mem[op[1]] = np.uint32(1 if fmem[op[2]] < fmem[op[3]] else 0)
            elif code == "cmp_eq": mem[op[1]] = np.uint32(1 if mem[op[2]] == mem[op[3]] else 0)
            elif code == "select": mem[op[1]] = mem[op[3]] if mem[op[2]] != 0 else mem[op[4]]
            else: raise ValueError(code)
    return {"memory": mem, "output": np.asarray(out, dtype=np.uint32)}


def fixture():
    from affine import make_program, ref, ins
    r = lambda i: ref("r", i)
    doc = make_program([("r", 4)], [
        ins("recv", r(0)), ins("recv", r(1)), ins("set", r(2), 0),
        ins("cmp", r(3), r(1), r(2), predicate="eq"),
        ins("select", r(1), r(3), r(0), r(2)),
        ins("add", r(0), r(0), r(0)), ins("sub", r(0), r(0), r(1)),
        ins("mul", r(0), r(0), r(0)), ins("div", r(0), r(0), r(0)),
        ins("cmp", r(3), r(1), r(0)), ins("copy", r(1), r(3)), ins("send", r(1)),
    ], {"fixture": "12 instructions; all supported opcodes; aliased operands and signed-zero raw equality"})
    return doc, np.array([1.5, -0.0], dtype=np.float32).view(np.uint32)


def edge_fixture():
    from affine import make_program, ref, ins, loop
    r = lambda offset=0, **kw: ref("r", offset, **kw)
    inputs = np.array([0x00000001, 0x00800000, 0x3f800000, 0x33800000,
                       0x80000000, 0x00000000, 0x7fc12345, 0x7f800000], dtype=np.uint32)
    body = [loop("i", 8, [ins("recv", r(i=1))]), loop("zero", 8, [ins("set", r(8, zero=1), 0)])]
    body += [ins("add", r(8), r(0), r(0)), ins("mul", r(9), r(1), r(1)),
             ins("add", r(10), r(2), r(3)), ins("cmp", r(11), r(4), r(5), predicate="eq"),
             ins("cmp", r(12), r(6), r(2)), ins("select", r(13), r(6), r(6), r(2)),
             ins("div", r(14), r(5), r(5)), ins("sub", r(15), r(7), r(7)),
             loop("send", 8, [ins("send", r(8, send=1))])]
    return make_program([("r", 16)], body, {"fixture": "subnormals, ties-to-even, signed zero, NaN raw select, numeric NaN cmp"}), inputs


def save_reference(directory, document, tape):
    from affine import Program
    from compiler import document_digest
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "program.json").write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))
    np.save(directory / "tape.npy", tape, allow_pickle=False)
    start = time.perf_counter()
    result = execute_with_memory(document, tape)
    elapsed = time.perf_counter() - start
    for key, value in result.items():
        np.save(directory / (key + ".reference.npy"), value, allow_pickle=False)
    program = Program(document)
    summary = {"program_sha256": document_digest(document), "elapsed_seconds": elapsed,
               "instructions": int(sum(program.instructions.values())), "scratch_words_including_zero": len(result["memory"]),
               "memory_sha256": hashlib.sha256(result["memory"].tobytes()).hexdigest(), "output": result["output"].tolist()}
    (directory / "reference.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"fixture": str(directory), **summary}), flush=True)


def compare(directory):
    from compiler import compile_shared, run_shared, document_digest
    document = json.loads((directory / "program.json").read_text())
    library = compile_shared(document, directory / "build")
    result = run_shared(library, np.load(directory / "tape.npy"), document_digest(document))
    checks = {}
    for key in ("memory", "output"):
        expected = np.load(directory / (key + ".reference.npy"))
        actual = result[key]
        wrong = np.flatnonzero(actual != expected)
        checks[key] = {"words": len(actual), "differing_words": len(wrong), "first_mismatches": wrong[:20].tolist(),
                       "sha256": hashlib.sha256(actual.tobytes()).hexdigest()}
        np.save(directory / (key + ".compiled.npy"), actual, allow_pickle=False)
    report = {"program_sha256": result["program_sha256"], "elapsed_seconds": result["elapsed_seconds"], "checks": checks,
              "pass": all(check["differing_words"] == 0 for check in checks.values())}
    (directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"fixture": str(directory), **report}), flush=True)
    if not report["pass"]:
        raise AssertionError("Compiled program differs from Python interpreter")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["reference", "compare"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--affine-dir", type=Path, required=True)
    parser.add_argument("--grid-dir", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.affine_dir))
    if args.mode == "reference":
        save_reference(args.output / "tiny", *fixture())
        save_reference(args.output / "edges", *edge_fixture())
        if args.grid_dir:
            sys.path.insert(0, str(args.grid_dir))
            import spatial_program
            rng = np.random.Generator(np.random.PCG64(84631))
            train = rng.random((16, 81), dtype=np.float32)
            query = rng.random((2, 81), dtype=np.float32)
            # Explicit range endpoints and outside-range values test clamp.
            train[0, :4] = np.array([-0.125, 0, 1, 1.125], dtype=np.float32)
            labels = np.arange(16, dtype=np.uint32) % 10
            tape = np.concatenate([train.reshape(-1).view(np.uint32), labels, query.reshape(-1).view(np.uint32)])
            save_reference(args.output / "grid", spatial_program.build(16, 2, 64, 2), tape)
    else:
        for directory in sorted(args.output.iterdir()):
            if directory.is_dir() and (directory / "memory.reference.npy").exists():
                compare(directory)


if __name__ == "__main__":
    main()
