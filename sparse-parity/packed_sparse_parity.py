"""Packed-column solver for the MASK32 sparse-parity benchmark.

The existing full-recovery generator stores the 18 x 33 augmented matrix one
bit per cell and walks 32-bit candidates.  This module exploits the v3 ISA's
8-bit cells in both phases:

* each augmented column is three non-negative 6-bit row masks;
* Gaussian elimination updates all 18 rows with three bytewise XORs;
* the affine-space walk stores only the three packed pivot-row masks;
* coefficient vectors are visited in a bounded-weight Gray order;
* target weights zero through three use dedicated bit predicates; and
* SSA liveness allocation reuses cells separately in the elimination and walk
  phases before an exact fixed-trace frequency layout.

The generated program remains ordinary v3 IR and is scored by
``mask_sparse_parity.evaluate_mask`` without evaluator changes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq
from math import comb, isqrt
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import mask_sparse_parity as mp


MASK32 = mp.MASK32
OP_CAP = mp.OP_CAP

_BINARY_OPS = {"add", "sub", "mul", "div", "and", "or", "xor"}
_UNARY_OPS = {"copy", "not", "abs"}
_CMP_PREDICATES = {"eq", "ne", "lt", "le", "gt", "ge"}


def bounded_weight_gray_states(dimension: int, max_weight: int) -> List[int]:
    """Return BRGC masks whose Hamming weight is at most ``max_weight``.

    For MASK32 at ``dimension=14, max_weight=5`` this visits all 3,473
    coefficient masks and has total Hamming transition cost 4,759.  The latter
    is optimal for any walk starting at zero that visits this vertex set; see
    ``doc/packed_scan_lower_bound.md``.
    """
    if dimension < 0:
        raise ValueError("dimension must be non-negative")
    if not 0 <= max_weight <= dimension:
        raise ValueError("max_weight must be in [0, dimension]")
    return [
        gray
        for i in range(1 << dimension)
        if (gray := i ^ (i >> 1)).bit_count() <= max_weight
    ]


def transition_cost(states: Sequence[int]) -> int:
    """Total number of coefficient-bit flips along ``states``."""
    return sum((a ^ b).bit_count() for a, b in zip(states, states[1:]))


def bounded_weight_transition_lower_bound(
    dimension: int, max_weight: int
) -> int:
    """Parity lower bound for a zero-start walk visiting every allowed mask.

    Expanding a transition of Hamming distance d into d hypercube edges gives
    a unit-edge walk.  Starting at an even vertex, a C-edge walk has at most
    floor(C/2)+1 even positions and ceil(C/2) odd positions.  It therefore
    needs at least max(2*(E-1), 2*O-1) edges to visit E even and O odd masks.

    The bound is exact for the benchmark's (dimension=14, max_weight=5) set.
    """
    if dimension < 0:
        raise ValueError("dimension must be non-negative")
    if not 0 <= max_weight <= dimension:
        raise ValueError("max_weight must be in [0, dimension]")
    even = sum(
        comb(dimension, weight)
        for weight in range(0, max_weight + 1, 2)
    )
    odd = sum(
        comb(dimension, weight)
        for weight in range(1, max_weight + 1, 2)
    )
    return max(2 * (even - 1), 2 * odd - 1)


def _parse_instruction(line: str) -> Tuple[str, List[str]]:
    op, separator, rest = line.partition(" ")
    if not separator:
        raise ValueError(f"malformed instruction: {line!r}")
    return op, [part.strip() for part in rest.split(",")]


def _source_addresses(op: str, args: Sequence[str]) -> List[int]:
    """Return every operand read, including implicit in-place reads."""
    if op == "set":
        return []
    if op == "cmp":
        if len(args) != 4 or args[3] not in _CMP_PREDICATES:
            raise ValueError(f"malformed cmp operands: {args!r}")
        return [int(args[1]), int(args[2])]
    if op == "select":
        if len(args) != 4:
            raise ValueError(f"malformed select operands: {args!r}")
        return [int(args[1]), int(args[2]), int(args[3])]
    if op in _UNARY_OPS:
        if len(args) != 2:
            raise ValueError(f"malformed unary operands: {args!r}")
        return [int(args[1])]
    if op in _BINARY_OPS:
        if len(args) == 3:
            return [int(args[1]), int(args[2])]
        if len(args) == 2:
            # ``xor d,s`` means ``d = d xor s``: d is a read as well as a write.
            return [int(args[0]), int(args[1])]
    raise ValueError(f"unknown or malformed instruction: {op} {args!r}")


def _remap_segment(lines: Iterable[str], mapping: Dict[int, int]) -> List[str]:
    result: List[str] = []
    for line in lines:
        op, args = _parse_instruction(line)
        destination = mapping[int(args[0])]
        if op == "set":
            result.append(f"set {destination},{args[1]}")
            continue
        suffix = ""
        sources = args[1:]
        if op == "cmp":
            suffix = f",{args[-1]}"
            sources = args[1:-1]
        remapped = ",".join(str(mapping[int(source)]) for source in sources)
        result.append(f"{op} {destination},{remapped}{suffix}")
    return result


def _parallel_copy(moves: Dict[int, int], scratch: int) -> List[str]:
    """Emit cycle-safe parallel assignment ``destination <- source``."""
    pending = {
        destination: source
        for destination, source in moves.items()
        if destination != source
    }
    result: List[str] = []
    while pending:
        sources = set(pending.values())
        ready = sorted(
            destination for destination in pending if destination not in sources
        )
        if ready:
            for destination in ready:
                result.append(f"copy {destination},{pending.pop(destination)}")
            continue

        # A remaining component is a cycle.  Save one old value, rotate, then
        # restore the saved value into the final destination.
        start = min(pending)
        result.append(f"copy {scratch},{start}")
        destination = start
        while True:
            source = pending[destination]
            if source == start:
                result.append(f"copy {destination},{scratch}")
                del pending[destination]
                break
            result.append(f"copy {destination},{source}")
            del pending[destination]
            destination = source
    return result


@dataclass
class _SSAValue:
    identifier: int
    start: int
    end: int
    reads: int = 0
    input_index: int | None = None
    defining_instruction: int | None = None


@dataclass
class _SSAInstruction:
    op: str
    destination: int
    sources: List[int]
    extra: str | None = None
    original_index: int = 0


def _to_ssa(ir: str):
    """Parse IR into SSA values while preserving instruction order.

    Reads happen before writes.  Giving them distinct odd/even timestamps lets
    a register be recycled for a destination written by the instruction that
    performs the source's final read.
    """
    lines = [line.strip() for line in ir.splitlines() if line.strip()]
    input_addresses = [int(text) for text in lines[0].split(",")]
    output_addresses = [int(text) for text in lines[-1].split(",")]

    values: List[_SSAValue] = []
    current: Dict[int, int] = {}
    for input_index, address in enumerate(input_addresses):
        identifier = len(values)
        values.append(
            _SSAValue(
                identifier,
                start=0,
                end=0,
                input_index=input_index,
            )
        )
        current[address] = identifier

    instructions: List[_SSAInstruction] = []
    for original_index, line in enumerate(lines[1:-1]):
        op, args = _parse_instruction(line)
        read_time = 2 * original_index + 1
        write_time = read_time + 1
        source_identifiers: List[int] = []
        for address in _source_addresses(op, args):
            try:
                identifier = current[address]
            except KeyError as exc:
                raise ValueError(
                    f"uninitialized address {address} at instruction "
                    f"{original_index}: {line}"
                ) from exc
            source_identifiers.append(identifier)
            values[identifier].end = max(values[identifier].end, read_time)
            values[identifier].reads += 1

        destination_address = int(args[0])
        destination_identifier = len(values)
        values.append(
            _SSAValue(
                destination_identifier,
                start=write_time,
                end=write_time,
                defining_instruction=len(instructions),
            )
        )
        current[destination_address] = destination_identifier
        extra = args[1] if op == "set" else (args[3] if op == "cmp" else None)
        instructions.append(
            _SSAInstruction(
                op,
                destination_identifier,
                source_identifiers,
                extra,
                original_index,
            )
        )

    output_time = 2 * len(instructions) + 1
    output_identifiers: List[int] = []
    for address in output_addresses:
        try:
            identifier = current[address]
        except KeyError as exc:
            raise ValueError(f"output address {address} is uninitialized") from exc
        output_identifiers.append(identifier)
        values[identifier].end = max(values[identifier].end, output_time)
        values[identifier].reads += 1

    return (
        input_addresses,
        output_addresses,
        values,
        instructions,
        output_identifiers,
    )


def _live_ssa_instructions(
    instructions: Sequence[_SSAInstruction], output_identifiers: Sequence[int]
) -> Set[int]:
    """Backward dead-code elimination for the straight-line SSA graph."""
    defining = {
        instruction.destination: index
        for index, instruction in enumerate(instructions)
    }
    needed_values = set(output_identifiers)
    needed_instructions: Set[int] = set()
    stack = list(output_identifiers)
    while stack:
        identifier = stack.pop()
        instruction_index = defining.get(identifier)
        if instruction_index is None or instruction_index in needed_instructions:
            continue
        needed_instructions.add(instruction_index)
        for source in instructions[instruction_index].sources:
            if source not in needed_values:
                needed_values.add(source)
                stack.append(source)
    return needed_instructions


def _allocate_phase(
    values: Sequence[_SSAValue],
    instructions: Sequence[_SSAInstruction],
    instruction_indices: Sequence[int],
    *,
    live_ins: Set[int],
    live_outs: Set[int],
    output_identifiers: Sequence[int],
    input_count: int,
    include_inputs: bool,
    output_phase: bool,
):
    """Linear-scan allocation, biased toward short read-dense intervals.

    The interval ordering is an energy heuristic: at one program point, assign
    scarce low reusable slots first to values with low lifetime/read ratio.
    The number of slots remains the exact interval-graph chromatic number for
    this fixed instruction order because any free slot is valid.
    """
    include = set(live_ins)
    if include_inputs:
        include.update(range(input_count))
    for instruction_index in instruction_indices:
        instruction = instructions[instruction_index]
        include.add(instruction.destination)
        include.update(instruction.sources)

    local_position = {
        instruction_index: position
        for position, instruction_index in enumerate(instruction_indices)
    }
    starts: Dict[int, int] = {}
    ends: Dict[int, int] = {}
    for identifier in include:
        if identifier in live_ins or (
            include_inputs and identifier < input_count
        ):
            starts[identifier] = 0
            ends[identifier] = 0
        else:
            defining = values[identifier].defining_instruction
            if defining is None:
                raise ValueError(f"SSA value {identifier} has no local definition")
            position = local_position[defining]
            starts[identifier] = 2 * position + 2
            ends[identifier] = starts[identifier]

    reads: Counter[int] = Counter()
    for position, instruction_index in enumerate(instruction_indices):
        read_time = 2 * position + 1
        for source in instructions[instruction_index].sources:
            ends[source] = max(ends[source], read_time)
            reads[source] += 1

    final_time = 2 * len(instruction_indices) + 1
    for identifier in live_outs:
        ends[identifier] = max(ends.get(identifier, 0), final_time)
        reads[identifier] += 1
    if output_phase:
        for identifier in output_identifiers:
            ends[identifier] = max(ends.get(identifier, starts[identifier]), final_time)
            reads[identifier] += 1

    by_start: Dict[int, List[int]] = defaultdict(list)
    for identifier in include:
        by_start[starts[identifier]].append(identifier)

    active: List[Tuple[int, int]] = []
    free_slots: Set[int] = set()
    slot_for: Dict[int, int] = {}
    slot_reads: Counter[int] = Counter()
    next_slot = 1

    for start_time in sorted(by_start):
        while active and active[0][0] < start_time:
            _end, slot = heapq.heappop(active)
            free_slots.add(slot)

        ordered = sorted(
            by_start[start_time],
            key=lambda identifier: (
                (ends[identifier] - starts[identifier])
                / (reads[identifier] + 1),
                identifier,
            ),
        )
        for identifier in ordered:
            if free_slots:
                slot = min(free_slots)
                free_slots.remove(slot)
            else:
                slot = next_slot
                next_slot += 1
            slot_for[identifier] = slot
            slot_reads[slot] += reads[identifier]
            heapq.heappush(active, (ends[identifier], slot))

    return slot_for, slot_reads, next_slot - 1


def _bucketed_phase_addresses(
    prefix_slots: int,
    walk_slots: int,
    prefix_reads: Counter[int],
    walk_reads: Counter[int],
    prefix_slot_for: Dict[int, int],
    walk_slot_for: Dict[int, int],
    live_across: Set[int],
):
    """Give each phase its frequency-optimal cost buckets and align live cells."""
    prefix_order = sorted(
        range(1, prefix_slots + 1),
        key=lambda slot: (-prefix_reads[slot], slot),
    )
    walk_order = sorted(
        range(1, walk_slots + 1),
        key=lambda slot: (-walk_reads[slot], slot),
    )

    def cost_bucket(position: int) -> int:
        return isqrt(position - 1) + 1

    prefix_groups: Dict[int, List[int]] = defaultdict(list)
    walk_groups: Dict[int, List[int]] = defaultdict(list)
    for position, slot in enumerate(prefix_order, start=1):
        prefix_groups[cost_bucket(position)].append(slot)
    for position, slot in enumerate(walk_order, start=1):
        walk_groups[cost_bucket(position)].append(slot)

    prefix_map: Dict[int, int] = {}
    walk_map: Dict[int, int] = {}
    for bucket in sorted(set(prefix_groups) | set(walk_groups)):
        prefix_bucket_slots = prefix_groups.get(bucket, [])
        walk_bucket_slots = walk_groups.get(bucket, [])
        first_address = (bucket - 1) ** 2 + 1
        prefix_addresses = list(
            range(first_address, first_address + len(prefix_bucket_slots))
        )
        walk_addresses = list(
            range(first_address, first_address + len(walk_bucket_slots))
        )
        prefix_set = set(prefix_bucket_slots)
        walk_set = set(walk_bucket_slots)
        common = sorted(
            identifier
            for identifier in live_across
            if prefix_slot_for[identifier] in prefix_set
            and walk_slot_for[identifier] in walk_set
        )

        used_prefix: Set[int] = set()
        used_walk: Set[int] = set()
        for identifier, address in zip(common, prefix_addresses):
            prefix_map[prefix_slot_for[identifier]] = address
            walk_map[walk_slot_for[identifier]] = address
            used_prefix.add(address)
            used_walk.add(address)

        remaining_prefix_slots = [
            slot for slot in prefix_bucket_slots if slot not in prefix_map
        ]
        remaining_walk_slots = [
            slot for slot in walk_bucket_slots if slot not in walk_map
        ]
        remaining_prefix_addresses = [
            address for address in prefix_addresses if address not in used_prefix
        ]
        remaining_walk_addresses = [
            address for address in walk_addresses if address not in used_walk
        ]
        prefix_map.update(zip(remaining_prefix_slots, remaining_prefix_addresses))
        walk_map.update(zip(remaining_walk_slots, remaining_walk_addresses))

    return prefix_map, walk_map


def _emit_ssa_instruction(
    instruction: _SSAInstruction,
    slot_for: Dict[int, int],
    address_for_slot: Dict[int, int],
) -> str:
    destination = address_for_slot[slot_for[instruction.destination]]
    sources = [address_for_slot[slot_for[source]] for source in instruction.sources]
    if instruction.op == "set":
        return f"set {destination},{instruction.extra}"
    if instruction.op == "cmp":
        return f"cmp {destination},{sources[0]},{sources[1]},{instruction.extra}"
    if instruction.op == "select":
        return (
            f"select {destination},{sources[0]},{sources[1]},{sources[2]}"
        )
    if instruction.op in _UNARY_OPS:
        return f"{instruction.op} {destination},{sources[0]}"
    # Emit the explicit binary form even when the source used an implicit
    # in-place operand.  The SSA graph has already made both reads explicit.
    return f"{instruction.op} {destination},{sources[0]},{sources[1]}"


def _global_frequency_layout(ir: str) -> str:
    """Exact optimal static address assignment for the emitted fixed trace."""
    lines = [line.strip() for line in ir.splitlines() if line.strip()]
    input_addresses = [int(text) for text in lines[0].split(",")]
    output_addresses = [int(text) for text in lines[-1].split(",")]
    reads: Counter[int] = Counter()
    universe = set(input_addresses) | set(output_addresses)
    for line in lines[1:-1]:
        op, args = _parse_instruction(line)
        universe.add(int(args[0]))
        for source in _source_addresses(op, args):
            universe.add(source)
            reads[source] += 1
    for address in output_addresses:
        reads[address] += 1

    order = sorted(universe, key=lambda address: (-reads[address], address))
    mapping = {old: new for new, old in enumerate(order, start=1)}
    result = [
        ",".join(str(mapping[address]) for address in input_addresses)
    ]
    result.extend(_remap_segment(lines[1:-1], mapping))
    result.append(
        ",".join(str(mapping[address]) for address in output_addresses)
    )
    return "\n".join(result)


def _optimize_two_phase_ir(ir: str, walk_start_destination: int) -> str:
    """DCE + exact-liveness register allocation across RREF and walk phases."""
    body = [line.strip() for line in ir.splitlines()[1:-1] if line.strip()]
    split_original_index = next(
        index
        for index, line in enumerate(body)
        if (parsed := _parse_instruction(line))[0] == "set"
        and int(parsed[1][0]) == walk_start_destination
    )

    (
        input_addresses,
        _output_addresses,
        values,
        instructions,
        output_identifiers,
    ) = _to_ssa(ir)
    needed = _live_ssa_instructions(instructions, output_identifiers)
    prefix_indices = [
        index
        for index, instruction in enumerate(instructions)
        if instruction.original_index < split_original_index and index in needed
    ]
    walk_indices = [
        index
        for index, instruction in enumerate(instructions)
        if instruction.original_index >= split_original_index and index in needed
    ]

    walk_sources: Set[int] = set(output_identifiers)
    for index in walk_indices:
        walk_sources.update(instructions[index].sources)
    split_time = 2 * split_original_index
    live_across = {
        identifier
        for identifier in walk_sources
        if values[identifier].start <= split_time
    }

    prefix_slot_for, prefix_reads, prefix_slots = _allocate_phase(
        values,
        instructions,
        prefix_indices,
        live_ins=set(),
        live_outs=live_across,
        output_identifiers=output_identifiers,
        input_count=len(input_addresses),
        include_inputs=True,
        output_phase=False,
    )
    walk_slot_for, walk_reads, walk_slots = _allocate_phase(
        values,
        instructions,
        walk_indices,
        live_ins=live_across,
        live_outs=set(),
        output_identifiers=output_identifiers,
        input_count=len(input_addresses),
        include_inputs=False,
        output_phase=True,
    )
    prefix_address_for_slot, walk_address_for_slot = _bucketed_phase_addresses(
        prefix_slots,
        walk_slots,
        prefix_reads,
        walk_reads,
        prefix_slot_for,
        walk_slot_for,
        live_across,
    )

    result = [
        ",".join(
            str(prefix_address_for_slot[prefix_slot_for[identifier]])
            for identifier in range(len(input_addresses))
        )
    ]
    result.extend(
        _emit_ssa_instruction(
            instructions[index], prefix_slot_for, prefix_address_for_slot
        )
        for index in prefix_indices
    )

    moves = {
        walk_address_for_slot[walk_slot_for[identifier]]:
        prefix_address_for_slot[prefix_slot_for[identifier]]
        for identifier in live_across
    }
    occupied = set(moves) | set(moves.values())
    scratch = next(
        (
            address
            for address in range(1, max(prefix_slots, walk_slots) + 1)
            if address not in occupied
        ),
        max(prefix_slots, walk_slots) + 1,
    )
    result.extend(_parallel_copy(moves, scratch))
    result.extend(
        _emit_ssa_instruction(
            instructions[index], walk_slot_for, walk_address_for_slot
        )
        for index in walk_indices
    )
    result.append(
        ",".join(
            str(walk_address_for_slot[walk_slot_for[identifier]])
            for identifier in output_identifiers
        )
    )

    # The phase allocator minimizes live storage and puts each phase in the
    # right cost buckets.  This final rearrangement-inequality pass makes the
    # whole emitted trace exactly frequency-optimal after bridge insertion.
    return _global_frequency_layout("\n".join(result))


def generate_packed_scan(
    weight_cap: int = 5,
    *,
    spec=MASK32,
    op_cap: int = OP_CAP,
    optimize_layout: bool = True,
) -> str:
    """Generate the packed-column bounded-weight affine scan.

    ``weight_cap`` is the largest number of free variables set in a visited
    coefficient mask.  For MASK32, caps 1, 2, 3 and 5 are useful for the 20%,
    40%, 60/80% and 100% leaderboard bands respectively.

    The implementation intentionally targets MASK32.  Its three 6-bit chunks,
    two-byte coefficient checkpoint and exact zero/one predicates rely on
    ``n=32, m=18, k=5``.
    """
    if tuple(spec) != tuple(MASK32):
        raise ValueError("generate_packed_scan currently supports MASK32 only")

    n, k, m = spec.n_bits, spec.k_secret, spec.m_train
    free_dimension = n - m
    if not 0 <= weight_cap <= k:
        raise ValueError(f"weight_cap must be in [0, {k}]")

    chunks = 3
    chunk_bits = 6
    next_address = 1

    def allocate(size: int) -> int:
        nonlocal next_address
        base = next_address
        next_address += size
        return base

    # Walk-hot cells are allocated first.  ``current_low`` is the phase marker
    # passed to _stage_two_phase_layout.
    pivot_state_base = allocate(chunks)
    best_low = allocate(1)
    best_high = allocate(1)
    current_low = allocate(1)
    current_high = allocate(1)
    popcount_base = allocate(chunks)
    weight_sum = allocate(1)
    candidate_ok = allocate(1)
    popcount_tmp = allocate(1)
    basis_base = allocate(free_dimension * chunks)

    # Packed RREF and metadata.
    matrix_base = allocate((n + 1) * chunks)
    used_base = allocate(chunks)
    eligible_base = allocate(chunks)
    lowbit_base = allocate(chunks)
    selected_pivot_base = allocate(chunks)
    eliminate_base = allocate(chunks)
    zero = allocate(1)
    one = allocate(1)
    mask6 = allocate(1)
    powers_base = allocate(chunk_bits - 1)
    rank_constants_base = allocate(free_dimension)
    free_count = allocate(1)
    rank_base = allocate(n)
    negative = allocate(1)
    any_value = allocate(1)
    found = allocate(1)
    pivot_bit = allocate(1)
    gate = allocate(1)
    temporary0 = allocate(1)
    temporary1 = allocate(1)
    temporary2 = allocate(1)
    temporary = allocate(1)
    base_solution_base = allocate(chunks)
    best_coeff_base = allocate(free_dimension)
    best_pivot_base = allocate(chunks)
    rank_bit_base = allocate(5)
    output_base = allocate(n)
    valid = allocate(1)
    bitmask = allocate(1)
    x_base = allocate(n * m)
    y_base = allocate(m)

    pivot_state = lambda q: pivot_state_base + q
    popcount = lambda q: popcount_base + q
    basis = lambda j, q: basis_base + j * chunks + q
    matrix = lambda c, q: matrix_base + c * chunks + q
    used = lambda q: used_base + q
    eligible = lambda q: eligible_base + q
    lowbit = lambda q: lowbit_base + q
    selected_pivot = lambda q: selected_pivot_base + q
    eliminate = lambda q: eliminate_base + q
    power = lambda bit: powers_base + bit - 1
    rank_constant = lambda j: rank_constants_base + j
    target_constant = lambda coefficient_weight: rank_constant(
        k - coefficient_weight
    )
    rank = lambda c: rank_base + c
    base_solution = lambda q: base_solution_base + q
    best_coeff = lambda j: best_coeff_base + j
    best_pivot = lambda q: best_pivot_base + q
    rank_bit = lambda bit: rank_bit_base + bit
    output = lambda c: output_base + c
    x_input = lambda row, column: x_base + row * n + column
    y_input = lambda row: y_base + row

    inputs = [x_input(row, column) for row in range(m) for column in range(n)]
    inputs.extend(y_input(row) for row in range(m))
    lines = [",".join(map(str, inputs))]
    emit = lines.append

    # Start by consuming one input into each packed cell.  At entry all 594
    # input cells are simultaneously live; freeing 99 of them before defining
    # constants lets the register allocator stay at that unavoidable floor.
    for column in range(n + 1):
        for chunk in range(chunks):
            row = chunk * chunk_bits
            source = y_input(row) if column == n else x_input(row, column)
            emit(f"copy {matrix(column, chunk)},{source}")

    # Bootstrap the five power constants from input slots as they become
    # dead.  The first weight-2 term is formed as source+source; each following
    # bit-1 term frees one slot before the next persistent constant is set.
    entries = [
        (column, chunk)
        for column in range(n + 1)
        for chunk in range(chunks)
    ]
    column, chunk = entries[0]
    row = chunk * chunk_bits + 1
    source = y_input(row) if column == n else x_input(row, column)
    emit(f"add {temporary},{source},{source}")
    emit(f"add {matrix(column, chunk)},{temporary}")
    emit(f"set {power(1)},2")

    for bit, (column, chunk) in zip(range(2, chunk_bits), entries[1:]):
        row = chunk * chunk_bits + 1
        source = y_input(row) if column == n else x_input(row, column)
        emit(f"mul {temporary},{source},{power(1)}")
        emit(f"add {matrix(column, chunk)},{temporary}")
        emit(f"set {power(bit)},{1 << bit}")

    for column, chunk in entries[chunk_bits - 1:]:
        row = chunk * chunk_bits + 1
        source = y_input(row) if column == n else x_input(row, column)
        emit(f"mul {temporary},{source},{power(1)}")
        emit(f"add {matrix(column, chunk)},{temporary}")

    for column, chunk in entries:
        for bit in range(2, chunk_bits):
            row = chunk * chunk_bits + bit
            source = y_input(row) if column == n else x_input(row, column)
            emit(f"mul {temporary},{source},{power(bit)}")
            emit(f"add {matrix(column, chunk)},{temporary}")

    emit(f"set {zero},0")
    emit(f"set {one},1")
    emit(f"set {mask6},63")
    for j in range(free_dimension):
        emit(f"set {rank_constant(j)},{j}")

    for chunk in range(chunks):
        emit(f"set {used(chunk)},0")
    emit(f"set {free_count},0")

    # Full RREF.  A selected pivot is a one-hot 18-row mask; E is the set of
    # non-pivot rows whose current-column bit must be eliminated.  For each
    # augmented column, one pivot-bit test gates three bytewise XORs.
    for column in range(n):
        for chunk in range(chunks):
            emit(f"xor {temporary},{used(chunk)},{mask6}")
            emit(f"and {eligible(chunk)},{matrix(column, chunk)},{temporary}")
            emit(f"sub {negative},{zero},{eligible(chunk)}")
            emit(f"and {lowbit(chunk)},{eligible(chunk)},{negative}")

        emit(f"copy {selected_pivot(0)},{lowbit(0)}")
        emit(f"select {selected_pivot(1)},{eligible(0)},{zero},{lowbit(1)}")
        emit(f"or {any_value},{eligible(0)},{eligible(1)}")
        emit(f"select {selected_pivot(2)},{any_value},{zero},{lowbit(2)}")
        emit(f"or {found},{any_value},{eligible(2)}")
        emit(f"select {temporary},{found},{zero},{one}")
        emit(f"select {rank(column)},{found},{mask6},{free_count}")
        emit(f"add {free_count},{temporary}")
        emit(f"or {gate},{selected_pivot(0)},{selected_pivot(1)}")
        emit(f"or {gate},{selected_pivot(2)}")

        for chunk in range(chunks):
            emit(f"or {used(chunk)},{selected_pivot(chunk)}")
            emit(
                f"xor {eliminate(chunk)},{matrix(column, chunk)},"
                f"{selected_pivot(chunk)}"
            )

        # Earlier pivot columns are already unit columns.  The newly selected
        # pivot row is unused, hence those columns contain zero at that row and
        # would be unchanged; update only the current/future columns and y.
        for augmented_column in range(column, n + 1):
            if augmented_column == column:
                # A successful pivot turns the current column into its one-hot
                # pivot mask.  A free column must be preserved as basis data.
                for chunk in range(chunks):
                    emit(f"select {matrix(column, chunk)},{found},"
                         f"{selected_pivot(chunk)},{matrix(column, chunk)}")
                continue
            emit(f"select {temporary0},{eligible(1)},"
                 f"{matrix(augmented_column, 1)},"
                 f"{matrix(augmented_column, 2)}")
            emit(f"select {temporary0},{eligible(0)},"
                 f"{matrix(augmented_column, 0)},{temporary0}")
            emit(f"and {pivot_bit},{temporary0},{gate}")
            emit(f"cmp {pivot_bit},{pivot_bit},{zero},ne")
            for chunk in range(chunks):
                emit(f"mul {temporary},{eliminate(chunk)},{pivot_bit}")
                emit(f"xor {matrix(augmented_column, chunk)},"
                     f"{matrix(augmented_column, chunk)},{temporary}")

    for chunk in range(chunks):
        # Consistency guarantees that rows never selected as pivots are zero
        # in y after elimination, so no used-row mask is required here.
        emit(f"copy {base_solution(chunk)},{matrix(n, chunk)}")
        emit(f"copy {pivot_state(chunk)},{base_solution(chunk)}")

    # Gather the first 14 free columns into packed pivot-space basis vectors.
    # Compare a column's tagged rank once, then use that condition for all
    # three chunks.  This removes the 448-cell rank-match table entirely.
    for j in range(free_dimension):
        for chunk in range(chunks):
            emit(f"set {basis(j, chunk)},0")
        for column in range(n):
            emit(f"cmp {temporary},{rank(column)},"
                 f"{rank_constant(j)},eq")
            for chunk in range(chunks):
                emit(f"select {basis(j, chunk)},{temporary},"
                     f"{matrix(column, chunk)},{basis(j, chunk)}")

    emit(f"set {best_low},0")
    emit(f"set {best_high},0")
    states = bounded_weight_gray_states(free_dimension, weight_cap)

    def emit_packed_popcount(cells: Sequence[int]) -> None:
        """Emit popcount of three 6-bit chunks into ``weight_sum``.

        Per bit position, parity + 2*majority equals the number of occupied
        chunks.  This reduces three six-bit popcounts to two.
        """
        emit(f"xor {temporary0},{cells[0]},{cells[1]}")
        emit(f"xor {temporary1},{temporary0},{cells[2]}")
        emit(f"and {temporary2},{cells[0]},{cells[1]}")
        emit(f"and {temporary},{temporary0},{cells[2]}")
        emit(f"or {temporary2},{temporary}")

        for destination, value in (
            (popcount(0), temporary1),
            (popcount(1), temporary2),
        ):
            # For 0 <= value < 64:
            # popcount(value) = value - sum_{b=1}^5 floor(value / 2^b).
            emit(f"div {popcount_tmp},{value},{power(1)}")
            emit(f"sub {destination},{value},{popcount_tmp}")
            for bit in range(2, chunk_bits):
                emit(f"div {popcount_tmp},{value},{power(bit)}")
                emit(f"sub {destination},{popcount_tmp}")
        emit(f"add {weight_sum},{popcount(1)},{popcount(1)}")
        emit(f"add {weight_sum},{popcount(0)}")

    def capture(state: int) -> None:
        coefficient_weight = state.bit_count()
        target = k - coefficient_weight
        low_byte = state & 0xFF
        high_byte = state >> 8
        # The checkpoint is initialized to state zero.  By uniqueness there
        # is at most one accepted state, so a zero byte never needs writing.
        # Keep one explicit first-state set as the phase-boundary marker.
        if state == 0:
            emit(f"set {current_low},0")
        if low_byte:
            emit(f"set {current_low},{low_byte}")
        if high_byte:
            emit(f"set {current_high},{high_byte}")
        cells = [pivot_state(chunk) for chunk in range(chunks)]

        if target == 0:
            emit(f"or {any_value},{cells[0]},{cells[1]}")
            emit(f"or {any_value},{cells[2]}")
            emit(f"cmp {candidate_ok},{any_value},{zero},eq")
        elif target == 1:
            # The aligned OR must be a nonzero power of two, and the majority
            # bitset must be zero so that two chunks cannot occupy that bit.
            emit(f"or {any_value},{cells[0]},{cells[1]}")
            emit(f"or {any_value},{cells[2]}")
            emit(f"sub {temporary2},{any_value},{one}")
            emit(f"and {temporary2},{any_value}")
            emit(f"xor {temporary0},{cells[0]},{cells[1]}")
            emit(f"and {temporary1},{cells[0]},{cells[1]}")
            emit(f"and {temporary0},{cells[2]}")
            emit(f"or {temporary1},{temporary0}")
            # Both the power-of-two error and the cross-chunk majority must
            # be zero.  Combining them before one comparison saves a boolean
            # comparison/AND pair.
            emit(f"or {temporary2},{temporary1}")
            emit(f"cmp {candidate_ok},{temporary2},{zero},eq")
            # Reject zero; a nonzero one-hot value is a valid select condition.
            emit(f"mul {candidate_ok},{candidate_ok},{any_value}")
        elif target == 2:
            # Let p be the per-bit parity across chunks and m the majority.
            # Total weight two means either popcount(p)=2,m=0 or p=0 and
            # popcount(m)=1.  The tests below keep a nonzero one-hot value as
            # the select condition, avoiding booleanization where possible.
            emit(f"xor {temporary0},{cells[0]},{cells[1]}")
            emit(f"xor {temporary1},{temporary0},{cells[2]}")
            emit(f"and {temporary2},{cells[0]},{cells[1]}")
            emit(f"and {temporary},{temporary0},{cells[2]}")
            emit(f"or {temporary2},{temporary}")

            # parity has exactly two bits and majority is zero.  For
            # y=p&(p-1), y is a nonzero power of two; combine the remaining
            # power-of-two error with majority before one comparison.
            emit(f"sub {popcount_tmp},{temporary1},{one}")
            emit(f"and {popcount(0)},{temporary1},{popcount_tmp}")
            emit(f"sub {popcount_tmp},{popcount(0)},{one}")
            emit(f"and {popcount(1)},{popcount(0)},{popcount_tmp}")
            emit(f"or {popcount(1)},{temporary2}")
            emit(f"cmp {candidate_ok},{popcount(1)},{zero},eq")
            emit(f"mul {candidate_ok},{candidate_ok},{popcount(0)}")

            # majority has one bit and parity is zero.
            emit(f"sub {popcount_tmp},{temporary2},{one}")
            emit(f"and {popcount(0)},{temporary2},{popcount_tmp}")
            emit(f"or {popcount(0)},{temporary1}")
            emit(f"cmp {popcount(0)},{popcount(0)},{zero},eq")
            emit(f"mul {popcount(0)},{popcount(0)},{temporary2}")
            emit(f"or {candidate_ok},{popcount(0)}")
        elif target == 3:
            # With parity p and majority m, total weight three is either
            # popcount(p)=3,m=0 or popcount(p)=1,popcount(m)=1.
            emit(f"xor {temporary0},{cells[0]},{cells[1]}")
            emit(f"xor {temporary1},{temporary0},{cells[2]}")
            emit(f"and {temporary2},{cells[0]},{cells[1]}")
            emit(f"and {temporary},{temporary0},{cells[2]}")
            emit(f"or {temporary2},{temporary}")

            # p has exactly three bits and majority is zero.  After clearing
            # two low bits, z must be a nonzero power of two; combine the last
            # clear-bit error with majority before one comparison.
            emit(f"sub {popcount_tmp},{temporary1},{one}")
            emit(f"and {popcount(0)},{temporary1},{popcount_tmp}")
            emit(f"sub {popcount_tmp},{popcount(0)},{one}")
            emit(f"and {popcount(1)},{popcount(0)},{popcount_tmp}")
            emit(f"sub {popcount_tmp},{popcount(1)},{one}")
            emit(f"and {weight_sum},{popcount(1)},{popcount_tmp}")
            emit(f"or {weight_sum},{temporary2}")
            emit(f"cmp {candidate_ok},{weight_sum},{zero},eq")
            emit(f"mul {candidate_ok},{candidate_ok},{popcount(1)}")

            # p and m each have exactly one bit.
            emit(f"sub {popcount_tmp},{temporary1},{one}")
            emit(f"and {popcount(0)},{temporary1},{popcount_tmp}")
            emit(f"cmp {popcount(0)},{popcount(0)},{zero},eq")
            emit(f"cmp {found},{temporary1},{zero},ne")
            emit(f"and {popcount(0)},{found}")
            emit(f"sub {popcount_tmp},{temporary2},{one}")
            emit(f"and {popcount(1)},{temporary2},{popcount_tmp}")
            emit(f"cmp {popcount(1)},{popcount(1)},{zero},eq")
            emit(f"mul {popcount(1)},{popcount(1)},{temporary2}")
            emit(f"mul {popcount(1)},{popcount(1)},{popcount(0)}")
            emit(f"or {candidate_ok},{popcount(1)}")
        else:
            emit_packed_popcount(cells)
            emit(
                f"cmp {candidate_ok},{weight_sum},"
                f"{target_constant(coefficient_weight)},eq"
            )

        if low_byte:
            emit(f"select {best_low},{candidate_ok},{current_low},{best_low}")
        if high_byte:
            emit(f"select {best_high},{candidate_ok},{current_high},{best_high}")

    capture(states[0])
    previous = states[0]
    for state in states[1:]:
        changed = previous ^ state
        for j in range(free_dimension):
            if (changed >> j) & 1:
                for chunk in range(chunks):
                    emit(f"xor {pivot_state(chunk)},{basis(j, chunk)}")
        capture(state)
        previous = state

    # Decode the captured two-byte coefficient mask.
    for j in range(free_dimension):
        source = best_low if j < 8 else best_high
        if (j & 7) == 0:
            emit(f"and {best_coeff(j)},{source},{one}")
        else:
            emit(f"set {bitmask},{1 << (j & 7)}")
            emit(f"and {best_coeff(j)},{source},{bitmask}")
            emit(f"cmp {best_coeff(j)},{best_coeff(j)},{zero},ne")

    for chunk in range(chunks):
        emit(f"copy {best_pivot(chunk)},{base_solution(chunk)}")
        for j in range(free_dimension):
            emit(f"mul {temporary},{basis(j, chunk)},{best_coeff(j)}")
            emit(f"xor {best_pivot(chunk)},{temporary}")

    # If no state was captured, the checkpoint is still zero.  Rechecking the
    # final total weight distinguishes that case from a legitimate state zero.
    emit_packed_popcount([best_pivot(chunk) for chunk in range(chunks)])
    for j in range(free_dimension):
        emit(f"add {weight_sum},{best_coeff(j)}")
    emit(f"cmp {valid},{weight_sum},{target_constant(0)},eq")

    for column in range(n):
        emit(f"and {temporary0},{best_pivot(0)},{matrix(column, 0)}")
        emit(f"and {temporary1},{best_pivot(1)},{matrix(column, 1)}")
        emit(f"and {temporary2},{best_pivot(2)},{matrix(column, 2)}")
        emit(f"or {temporary},{temporary0},{temporary1}")
        emit(f"or {temporary},{temporary2}")
        emit(f"cmp {output(column)},{temporary},{zero},ne")

        # Select one bit from the two-byte coefficient checkpoint.  A
        # three-level mux chooses the bit mask, rank bit 3 chooses the byte,
        # and rank bit 4 distinguishes the pivot sentinel 63 from ranks 0..13.
        rank_masks = [one, power(1), power(2), power(3), power(4)]
        for bit, bit_cell in enumerate(rank_masks):
            emit(f"and {rank_bit(bit)},{rank(column)},{bit_cell}")
        # 2^r factors as 2^(b0) * 4^(b1) * 16^(b2), reducing the
        # eight-way bit-mask lookup to three selects and two multiplies.
        emit(f"select {temporary0},{rank_bit(0)},{power(1)},{one}")
        emit(f"select {temporary1},{rank_bit(1)},{power(2)},{one}")
        emit(f"mul {temporary0},{temporary1}")
        emit(f"select {temporary1},{rank_bit(2)},{power(4)},{one}")
        emit(f"mul {temporary0},{temporary1}")
        emit(f"select {temporary},{rank_bit(3)},{best_high},{best_low}")
        emit(f"and {temporary},{temporary0}")
        emit(f"cmp {temporary},{temporary},{zero},ne")
        emit(f"select {output(column)},{rank_bit(4)},"
             f"{output(column)},{temporary}")
        emit(f"and {output(column)},{valid}")

    lines.append(",".join(str(output(column)) for column in range(n)))
    raw = "\n".join(lines)
    if len(lines) > op_cap:
        raise ValueError(
            f"packed scan IR has {len(lines):,} lines, over the {op_cap:,} cap"
        )

    ir = (
        _optimize_two_phase_ir(raw, walk_start_destination=current_low)
        if optimize_layout
        else raw
    )
    if len(ir.splitlines()) > op_cap:
        raise ValueError(
            f"optimized packed scan IR has {len(ir.splitlines()):,} lines, "
            f"over the {op_cap:,} cap"
        )
    return ir


__all__ = [
    "MASK32",
    "OP_CAP",
    "bounded_weight_gray_states",
    "bounded_weight_transition_lower_bound",
    "transition_cost",
    "generate_packed_scan",
]
