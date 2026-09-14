#!/usr/bin/env python3
"""A restricted, inspectable loop representation of straight-line v4 programs.

This is a research prototype, not an official language or scorer. Loop syntax
compresses program generation only: every leaf is a priced v4 instruction.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

FORMAT = "sutro-serial-spatial-affine-v4/1"
SPEC_COMMIT = "01a0bd5e0d2564825b0f53dd766f763c82dbc7c0"
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
            "placement": "nearest-tile-nearest-legal-cell; reserve-bottom-stage",
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


class Program:
    """Validate a finite affine program and expose its static leaf scopes."""

    def __init__(self, document):
        self.document = document
        if document.get("format") != FORMAT:
            raise ValueError("Unrecognized IL format")
        if document.get("model_spec_commit") != SPEC_COMMIT:
            raise ValueError("Unrecognized model revision")
        if document.get("placement") != "nearest-tile-nearest-legal-cell; reserve-bottom-stage":
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
            words = _integer(region["words"], "Region length", 1, 384_000_000 - 250)
            self.regions[name] = (self.words + 1, words)
            self.words += words
        if not self.words or self.words > 384_000_000 - 250:
            raise ValueError("Program plus 250 stages exceeds spatial scratch capacity")
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
                if self.initialization_visits > 2 * self.words + 100_000:
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


def expand(document):
    """Expand the finite affine loops to declared primitive instructions."""
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
