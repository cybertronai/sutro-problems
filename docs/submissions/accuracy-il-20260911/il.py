#!/usr/bin/env python3
"""A restricted, inspectable loop representation of straight-line v4 programs.

This is a research prototype, not an official language or scorer. Loop syntax
compresses program generation only: every leaf is a priced v4 instruction.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

FORMAT = "sutro-affine-v4/0.1"
SPEC_COMMIT = "26abcca402de647381d31286d42dfbb7a001763d"
ARITIES = {"set": 0, "recv": 0, "send": 1, "copy": 1,
           "add": 2, "sub": 2, "mul": 2, "cmp": 2, "select": 3}
MAX_COUNT = (1 << 62) - 1


def ref(region, offset=0, **coefficients):
    return {"region": region, "offset": offset, "coefficients": coefficients}


def ins(opcode, dst, *src, predicate=None):
    if opcode == "send":
        if src:
            raise ValueError("send takes exactly one source")
        return {"op": opcode, "src": [dst]}
    result = {"op": opcode, "dst": dst}
    if opcode == "set":
        if len(src) != 1:
            raise ValueError("set takes one immediate")
        result["imm"] = src[0]
    else:
        result["src"] = list(src)
    if predicate is not None:
        if opcode != "cmp":
            raise ValueError("Only cmp has a predicate")
        result["predicate"] = predicate
    return result


def loop(var, count, body, start=0):
    return {"loop": var, "start": start, "count": count, "body": body}


def make_program(regions, body, metadata=None):
    return {"format": FORMAT, "model_spec_commit": SPEC_COMMIT,
            "arithmetic": "fp32-rne; cmp=lt; select=raw-word",
            "placement": "half-diamond-ascending-x",
            "regions": [{"name": name, "words": words} for name, words in regions],
            "body": body, "metadata": metadata or {}}


def _integer(value, description, minimum=None, maximum=None):
    if type(value) is not int:
        raise ValueError(f"{description} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{description} is below {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{description} exceeds {maximum}")
    return value


def placement(words):
    result = [None]
    h = 1
    while len(result) <= words:
        for x in range(-(h - 1), h):
            if len(result) > words:
                break
            result.append((x, h - abs(x)))
        h += 1
    return result


class Program:
    """Validate a finite affine program and expose its static leaf scopes."""

    def __init__(self, document):
        self.document = document
        if document.get("format") != FORMAT:
            raise ValueError("Unrecognized IL format")
        if document.get("model_spec_commit") != SPEC_COMMIT:
            raise ValueError("Unrecognized model revision")
        if document.get("placement") != "half-diamond-ascending-x":
            raise ValueError("Unsupported placement")
        if document.get("arithmetic") != "fp32-rne; cmp=lt; select=raw-word":
            raise ValueError("Unsupported arithmetic convention")
        self.regions = {}
        self.words = 0
        for region in document["regions"]:
            if set(region) != {"name", "words"}:
                raise ValueError("Unexpected region fields")
            name = region["name"]
            if not isinstance(name, str) or not name or name in self.regions:
                raise ValueError("Region names must be unique nonempty strings")
            words = _integer(region["words"], "Region length", 1, 1_000_000)
            self.regions[name] = (self.words + 1, words)
            self.words += words
        if not self.words or self.words > 1_000_000:
            raise ValueError("Prototype supports 1..1,000,000 scratch words")
        self.coordinates = placement(self.words)
        self.leaves = []
        self.instructions = Counter()
        self.node_count = 0
        self._validate_body(document["body"], {})
        if sum(self.instructions.values()) > MAX_COUNT // 4:
            raise ValueError("Dynamic instruction count exceeds exact int64 limit")
        self.initialization_visits = 0
        self._validate_initialization()

    def _operand(self, operand, scope):
        if not isinstance(operand, dict) or set(operand) != {"region", "offset", "coefficients"}:
            raise ValueError("Operand must be a region and an affine integer offset")
        if operand["region"] not in self.regions:
            raise ValueError("Unknown operand region")
        offset = _integer(operand["offset"], "Operand offset")
        coefficients = operand["coefficients"]
        if not isinstance(coefficients, dict):
            raise ValueError("Affine coefficients must be an object")
        lower = upper = offset
        for var, coefficient in coefficients.items():
            if var not in scope:
                raise ValueError(f"Unbound loop variable {var}")
            coefficient = _integer(coefficient, "Affine coefficient")
            start, count = scope[var]
            # Empty loop bodies still undergo syntactic checks, but never access.
            a, b = coefficient * start, coefficient * (start + max(1, count) - 1)
            lower += min(a, b)
            upper += max(a, b)
        _, words = self.regions[operand["region"]]
        if all(count for _, count in scope.values()) and not (0 <= lower <= upper < words):
            raise ValueError(f"Operand escapes region {operand['region']}: [{lower}, {upper}]")

    def _validate_body(self, body, scope):
        if not isinstance(body, list):
            raise ValueError("Body must be a list")
        for node in body:
            self.node_count += 1
            if "loop" in node:
                if set(node) != {"loop", "start", "count", "body"}:
                    raise ValueError("Unexpected loop fields")
                var = node["loop"]
                if not isinstance(var, str) or not var or var in scope:
                    raise ValueError("Loop variables must be nonempty and cannot shadow")
                start = _integer(node["start"], "Loop start")
                count = _integer(node["count"], "Loop count", 0, MAX_COUNT)
                self._validate_body(node["body"], {**scope, var: (start, count)})
                continue
            opcode = node.get("op")
            if opcode not in ARITIES:
                raise ValueError(f"Unsupported opcode {opcode}")
            expected = {"op", "src"} if opcode == "send" else {"op", "dst", "imm" if opcode == "set" else "src"}
            if opcode == "cmp" and "predicate" in node:
                expected.add("predicate")
                if node["predicate"] not in ("eq", "ne", "lt", "le", "gt", "ge"):
                    raise ValueError("Unsupported cmp predicate")
            if set(node) != expected:
                raise ValueError(f"Unexpected fields for {opcode}")
            if opcode != "send":
                self._operand(node["dst"], scope)
            if opcode == "set":
                _integer(node["imm"], "32-bit immediate", 0, 0xffffffff)
            else:
                if not isinstance(node["src"], list) or len(node["src"]) != ARITIES[opcode]:
                    raise ValueError(f"Wrong source count for {opcode}")
                for source in node["src"]:
                    self._operand(source, scope)
            multiplicity = math.prod(count for _, count in scope.values())
            self.instructions[opcode] += multiplicity
            self.leaves.append((node, tuple(scope.items()), multiplicity))

    def address(self, operand, environment):
        base, _ = self.regions[operand["region"]]
        return base + operand["offset"] + sum(c * environment[v] for v, c in operand["coefficients"].items())

    def _validate_initialization(self):
        """Sound definite-write proof. Writes never invalidate a scratch word.

        Walk real ordering until every allocated word has been written, after
        which static address validation suffices. A loop whose variable occurs
        in no descendant address needs just its first iteration: later passes
        use the identical ordered address stream with a superset of initialized
        words. Other loops are enumerated only for this proof; a work budget
        causes explicit rejection, never assumed validity.
        """
        initialized = np.zeros(self.words + 1, dtype=np.bool_)
        initialized_count = 0
        variable_cache = {}
        def variables(body):
            key = id(body)
            if key not in variable_cache:
                result = set()
                for node in body:
                    if "loop" in node:
                        result.update(variables(node["body"]))
                    else:
                        operands = node.get("src", []) + ([node["dst"]] if "dst" in node else [])
                        for operand in operands:
                            result.update(v for v, c in operand["coefficients"].items() if c)
                variable_cache[key] = result
            return variable_cache[key]
        def visit(body, environment):
            nonlocal initialized_count
            for node in body:
                if initialized_count == self.words:
                    return
                self.initialization_visits += 1
                if self.initialization_visits > 2_000_000:
                    raise ValueError("Initialization proof budget exceeded; add explicit initialization or a stronger verifier")
                if "loop" in node:
                    count = node["count"]
                    if node["loop"] not in variables(node["body"]):
                        count = min(1, count)
                    for value in range(node["start"], node["start"] + count):
                        visit(node["body"], {**environment, node["loop"]: value})
                        if initialized_count == self.words:
                            return
                else:
                    for operand in node.get("src", []):
                        address = self.address(operand, environment)
                        if not initialized[address]:
                            raise ValueError(f"Uninitialized read from {operand['region']} (address {address})")
                    if "dst" in node:
                        address = self.address(node["dst"], environment)
                        if not initialized[address]:
                            initialized[address] = True
                            initialized_count += 1
        visit(self.document["body"], {})
        self.initialized_words = initialized_count

    def histogram(self, operand, scope):
        """Count each affine address with sliding-window discrete convolution.

        For each referenced loop, convolve its arithmetic-progression address
        histogram. Unreferenced loop dimensions only multiply counts. Runtime
        depends on represented address span, not total dynamic instructions.
        """
        base, _ = self.regions[operand["region"]]
        minimum = base + operand["offset"]
        histogram = np.ones(1, dtype=np.int64)
        multiplier = 1
        for variable, (start, count) in scope:
            if not count:
                return minimum, np.zeros(0, dtype=np.int64)
            coefficient = operand["coefficients"].get(variable, 0)
            if not coefficient:
                multiplier *= count
                continue
            minimum += coefficient * start + min(0, coefficient * (count - 1))
            stride = abs(coefficient)
            length = len(histogram) + stride * (count - 1)
            output = np.zeros(length, dtype=np.int64)
            output[:len(histogram)] = histogram
            for residue in range(min(stride, length)):
                cumulative = np.cumsum(output[residue::stride], dtype=np.int64)
                if len(cumulative) > count:
                    cumulative[count:] -= cumulative[:-count].copy()
                output[residue::stride] = cumulative
            histogram = output
        if multiplier != 1:
            histogram *= multiplier
        return minimum, histogram

    def score(self):
        reads = np.zeros(self.words + 1, dtype=np.int64)
        writes = np.zeros(self.words + 1, dtype=np.int64)
        cache = {}
        def charge(counts, operand, scope):
            key = (operand["region"], operand["offset"], tuple(sorted(operand["coefficients"].items())), scope)
            if key not in cache:
                cache[key] = self.histogram(operand, scope)
            first, histogram = cache[key]
            counts[first:first + len(histogram)] += histogram
        for node, scope, multiplicity in self.leaves:
            if not multiplicity or node["op"] in ("recv", "send"):
                continue
            for source in node.get("src", []):
                charge(reads, source, scope)
            charge(writes, node["dst"], scope)
        total_instructions = sum(self.instructions.values())
        # Each instruction has at most three reads and one write. Conservative
        # guard keeps all vector arithmetic exact signed int64 throughout.
        if total_instructions > MAX_COUNT // 4:
            raise ValueError("Aggregate accesses exceed exact int64 prototype bound")
        energy_fj = 0
        ticks = 0
        for address, (x, y) in enumerate(self.coordinates[1:], 1):
            hops = abs(x) + y
            r, w = int(reads[address]), int(writes[address])
            energy_fj += (r + w) * max(50, 2 * hops)
            ticks += r * max(250, 4 * hops) + w * max(250, 2 * hops)
        digest = hashlib.sha256(json.dumps(self.document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {"format": FORMAT, "model_spec_commit": SPEC_COMMIT,
                "program_sha256": digest,
                "scorer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "score_kind": "static exact access aggregation; no numerical execution",
                "instructions": dict(self.instructions), "total_instructions": total_instructions,
                "charged_reads": int(reads.sum()), "charged_writes": int(writes.sum()),
                "charged_accesses": int(reads.sum()) + int(writes.sum()),
                "time_ticks_0_2_ps": ticks, "time_ps": ticks / 5, "energy_fj": energy_fj,
                "area_um2_occupied_cells": self.words,
                "peak_initialized_scratch_words": self.initialized_words,
                "input_tape_words": self.instructions.get("recv", 0),
                "output_tape_words": self.instructions.get("send", 0),
                "static_leaf_nodes": len(self.leaves), "static_nodes": self.node_count,
                "initialization_proof_visits": self.initialization_visits,
                "histograms_computed": len(cache)}, reads, writes


def score(document, include_counts=False):
    start = time.perf_counter()
    program = Program(document)
    result, reads, writes = program.score()
    result["time_to_score_seconds"] = time.perf_counter() - start
    result["time_to_score_scope"] = "schema/bounds/definite-initialization validation, placement generation, exact static access histograms, integer cost sums, and canonical program hashing; excludes JSON loading, numerical execution/accuracy verification, file I/O"
    return (result, reads, writes) if include_counts else result


def expand(document):
    program = Program(document)
    def emit(body, environment):
        for node in body:
            if "loop" in node:
                for value in range(node["start"], node["start"] + node["count"]):
                    yield from emit(node["body"], {**environment, node["loop"]: value})
            elif node["op"] == "send":
                yield ("send", program.address(node["src"][0], environment))
            else:
                arguments = [node["imm"]] if node["op"] == "set" else [program.address(source, environment) for source in node.get("src", [])]
                opcode = node["op"]
                if opcode == "cmp" and node.get("predicate", "lt") != "lt":
                    opcode += "_" + node["predicate"]
                yield (opcode, program.address(node["dst"], environment), *arguments)
    yield from emit(document["body"], {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expand", type=Path, help="Optionally emit the expanded primitive stream, one JSON tuple per line")
    args = parser.parse_args()
    document = json.loads(args.program.read_text())
    result = score(document)
    if args.expand:
        with args.expand.open("w") as handle:
            for instruction in expand(document):
                handle.write(json.dumps(instruction) + "\n")
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
