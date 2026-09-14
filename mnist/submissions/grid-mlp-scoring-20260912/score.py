"""Exactly price a fully serialized spatial-computer v4 MLP schedule.

This is a conservative executable schedule, not a parallel-performance model.
Every issued access finishes before any processor issues another access.
The affine instruction representation compresses fixed loops only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from affine import Program, SPEC_COMMIT, expand
from model_ir import build_mlp

HERE = Path(__file__).resolve().parent
COMPUTE = (125, 0)
PITCH = 128
TILE_WORDS = 12_288
PORTS = 250
MAX_WORDS = 384_000_000


def legal_cells():
    """Return (u,v) cells ordered by (distance from core, u, v)."""
    # The processor is at local coordinates (u=64,v=63), since y starts at 1.
    return sorted(((u, v) for u in range(128) for v in range(128)
                   if not (32 <= u <= 95 and 32 <= v <= 95)),
                  key=lambda c: (abs(c[0] - 64) + abs(c[1] - 63), *c))


CELLS = legal_cells()
STAGE_CELL = CELLS[0]
STAGE_DISTANCE = abs(STAGE_CELL[0] - 64) + abs(STAGE_CELL[1] - 63)


def coordinate(tile, cell):
    return (-16_000 + 128 * tile[0] + cell[0], 128 * tile[1] + 1 + cell[1])


def placement(words):
    """Fixed injective mapping for program addresses 1..words and 250 stages.

    Tiles are ordered by (Manhattan mesh distance from COMPUTE, row, column).
    Within each tile use CELLS order, skipping STAGE_CELL in every bottom tile.
    Stages have distinct names stage:0..249 and occupy those reserved cells.
    """
    if not 0 <= words <= MAX_WORDS - PORTS:
        raise ValueError('Program plus stages exceeds spatial scratch capacity')
    owners = np.empty((words + 1, 2), dtype=np.int16)
    local = np.empty((words + 1, 2), dtype=np.int16)
    owners[0] = local[0] = -1
    tiles = sorted(((i, j) for i in range(250) for j in range(125)),
                   key=lambda t: (abs(t[0] - COMPUTE[0]) + t[1], t[1], t[0]))
    first = 1
    occupancy = {}
    for tile in tiles:
        if first > words:
            break
        cells = CELLS[1:] if tile[1] == 0 else CELLS
        count = min(len(cells), words - first + 1)
        owners[first:first + count] = tile
        local[first:first + count] = cells[:count]
        occupancy[tile] = count
        first += count
    assert first == words + 1
    for column in range(PORTS):
        occupancy[(column, 0)] = occupancy.get((column, 0), 0) + 1
    assert max(occupancy.values()) <= TILE_WORDS
    distance = np.abs(local[:, 0].astype(np.int64) - 64) + np.abs(local[:, 1].astype(np.int64) - 63)
    links = np.abs(owners[:, 0].astype(np.int64) - COMPUTE[0]) + owners[:, 1]
    return owners, local, distance, links, occupancy


def access_cost(issuer, owner, cell, kind):
    """One blocking scratch access, including all request/data wire words."""
    distance = abs(cell[0] - 64) + abs(cell[1] - 63)
    if cell not in CELLS_SET:
        raise ValueError('Reserved or out-of-bounds scratch cell')
    if not (0 <= owner[0] < 250 and 0 <= owner[1] < 125):
        raise ValueError('Out-of-bounds owning tile')
    links = abs(issuer[0] - owner[0]) + abs(issuer[1] - owner[1])
    if kind not in ('read', 'write'):
        raise ValueError('Unknown scratch access')
    energy = 256 * links + max(50, 2 * distance)
    cycles = 1 if links == 0 else (2 * links + 1 if kind == 'read' else links + 2)
    return energy, cycles


CELLS_SET = frozenset(CELLS)


def tape_cost(tile, cell):
    distance = abs(cell[0] - 64) + abs(cell[1] - 63)
    return 128 * tile[1] + 64 + max(50, 2 * distance), tile[1] + 2


def histogram_counts(program):
    """Exact address-use counts, including actual tape copy source/destinations."""
    counts = {name: np.zeros(program.words + 1, dtype=np.int64)
              for name in ('reads', 'writes', 'input_destinations', 'output_sources')}
    cache = {}
    def charge(name, operand, scope):
        key = (operand['region'], operand['offset'], tuple(sorted(operand['coefficients'].items())), scope)
        if key not in cache:
            cache[key] = program.histogram(operand, scope)
        first, histogram = cache[key]
        counts[name][first:first + len(histogram)] += histogram
    for node, scope, multiplicity in program.leaves:
        if not multiplicity:
            continue
        if node['op'] == 'recv':
            charge('input_destinations', node['dst'], scope)
        elif node['op'] == 'send':
            charge('output_sources', node['src'][0], scope)
        else:
            for source in node.get('src', []):
                charge('reads', source, scope)
            charge('writes', node['dst'], scope)
    return counts, len(cache)


def exact_dot(counts, costs):
    # Avoid an overflowing int64 dot product, including for larger programs.
    return sum(int(count) * int(cost) for count, cost in zip(counts, costs) if count)


def score(document):
    start = time.perf_counter()
    program = Program(document)
    owners, local, distances, links, occupancy = placement(program.words)
    counts, histograms = histogram_counts(program)
    access_energy = 256 * links + np.maximum(50, 2 * distances)
    read_cycles = 2 * links + 1
    write_cycles = np.where(links == 0, 1, links + 2)
    components = {}
    for name, kind in (('reads', 'read'), ('writes', 'write'),
                       ('input_destinations', 'write'), ('output_sources', 'read')):
        components[name] = {
            'scratch_accesses': int(counts[name].sum()),
            'energy_fj': exact_dot(counts[name], access_energy),
            'cycles': exact_dot(counts[name], read_cycles if kind == 'read' else write_cycles),
        }
    n_input = program.instructions.get('recv', 0)
    n_output = program.instructions.get('send', 0)
    port_counts = {'input': [], 'output': []}
    for name, total, kind in (('input', n_input, 'read'), ('output', n_output, 'write')):
        energy = cycles = 0
        for port in range(PORTS):
            count = total // PORTS + (port < total % PORTS)
            port_counts[name].append(count)
            stage_energy, stage_cycles = access_cost(COMPUTE, (port, 0), STAGE_CELL, kind)
            tape_energy, tape_cycles = tape_cost((port, 0), STAGE_CELL)
            energy += count * (stage_energy + tape_energy)
            cycles += count * (stage_cycles + tape_cycles)
        components[name + '_stage_and_tape'] = {
            'stage_scratch_accesses': total,
            'tape_instructions': total,
            'energy_fj': energy,
            'cycles': cycles,
        }
    energy = sum(c['energy_fj'] for c in components.values())
    cycles = sum(c['cycles'] for c in components.values())
    serialized = json.dumps(document, sort_keys=True, separators=(',', ':')).encode()
    result = {
        'model': 'spatial-computer pitch128 / instruction-set v4',
        'model_spec_commit': SPEC_COMMIT,
        'model_url': f'https://github.com/cybertronai/simplified-dally-model/blob/{SPEC_COMMIT}/models/spatial-computer/README.md',
        'score_kind': 'exact static counts of a globally serialized legal schedule',
        'numeric_execution_checked_by_scorer': False,
        'program_sha256': hashlib.sha256(serialized).hexdigest(),
        'configuration': document['metadata'],
        'energy_fj': energy,
        'energy_mj': energy / 1e12,
        'word_node_hops': energy,
        'cycles': cycles,
        'time_ms': cycles / 1e6,
        'time_ns': cycles,
        'components': components,
        'logical_instructions': dict(program.instructions),
        'executed_instructions': {**dict(program.instructions), 'copy': program.instructions.get('copy', 0) + n_input + n_output},
        'total_executed_instructions': sum(program.instructions.values()) + n_input + n_output,
        'input_tape_words': n_input,
        'output_tape_words': n_output,
        'per_port_words': port_counts,
        'program_scratch_words': program.words,
        'stage_scratch_words': PORTS,
        'peak_allocated_scratch_words': program.words + PORTS,
        'peak_allocated_scratch_bytes': 4 * (program.words + PORTS),
        'peak_initialized_program_words': program.initialized_words,
        'memory_tiles': len(occupancy),
        'max_tile_scratch_words': max(occupancy.values()),
        'compute_processor': list(COMPUTE),
        'instruction_issuing_processors': len({COMPUTE} | {(i, 0) for i, (a, b) in enumerate(zip(port_counts['input'], port_counts['output'])) if a or b}),
        'max_simultaneous_outstanding_accesses': 1,
        'max_simultaneous_instructions': 1,
        'stage_local_coordinate': list(STAGE_CELL),
        'placement_sha256': hashlib.sha256(owners.astype('<i2').tobytes() + local.astype('<i2').tobytes()).hexdigest(),
        'histograms_computed': histograms,
        'initialization_proof_visits': program.initialization_visits,
        'schedule': 'Expand original affine loops in lexicographic iteration order. Complete every access before starting the next. Each normal instruction runs on P(125,0), reading listed sources then writing destination. Lower kth recv to P(k mod250,0):recv stage; P(125,0):copy original_dst,stage. Lower qth send to P(125,0):copy stage,original_src; P(q mod250,0):send stage. Route remote requests horizontally then vertically; responses retrace. Global serialization defines access/transfer cycle starts by prefix sums of blocking latencies; no contention or waiting queues occur.',
        'time_to_score_seconds': time.perf_counter() - start,
        'time_to_score_scope': 'Program/schema/address/initialization validation, physical placement, affine access histograms, exact integer costs, port counts and program/placement hashing; excludes program generation, JSON/file I/O and numerical accuracy execution',
    }
    return result


def scheduled_instructions(document):
    """Concrete per-processor primitive stream, suitable for tiny trace expansion.

    stage cells are named tuples, program cells are integer addresses. The
    emitted order is also the global dependency order; finish an instruction
    before starting its successor. This fully determines the cycle schedule.
    """
    input_index = output_index = 0
    for instruction in expand(document):
        op, *operands = instruction
        if op == 'recv':
            port = input_index % PORTS
            input_index += 1
            yield (port, 0), ('recv', ('stage', port))
            yield COMPUTE, ('copy', operands[0], ('stage', port))
        elif op == 'send':
            port = output_index % PORTS
            output_index += 1
            yield COMPUTE, ('copy', ('stage', port), operands[0])
            yield (port, 0), ('send', ('stage', port))
        else:
            yield COMPUTE, instruction


def scheduled_accesses(document):
    """Expand the exact cycle schedule lazily, one scratch/tape access at a time.

    A remote read's path carries address at start+k, local service at start+L,
    and response along the reverse path at start+L+1+k. A remote write carries
    address/data at start+k and start+k+1, then local service at start+L+1.
    Bottom recv uses its port link at start and scratch at start+1; send reverses
    that order. Each yielded completion is the next access's start boundary.
    """
    program = Program(document)
    owners, cells, _, _, _ = placement(program.words)
    tick = 0
    def location(address):
        if isinstance(address, tuple):
            return (address[1], 0), STAGE_CELL
        return tuple(map(int, owners[address])), tuple(map(int, cells[address]))
    for instruction_index, (issuer, instruction) in enumerate(scheduled_instructions(document)):
        op, *arguments = instruction
        if op in ('recv', 'send'):
            accesses = [(op, arguments[0])]
        elif op == 'set':
            accesses = [('write', arguments[0])]
        else:
            accesses = [('read', source) for source in arguments[1:]] + [('write', arguments[0])]
        for kind, address in accesses:
            owner, cell = location(address)
            if kind in ('recv', 'send'):
                assert issuer == owner
                energy, cycles = tape_cost(issuer, cell)
            else:
                energy, cycles = access_cost(issuer, owner, cell, kind)
            yield {'instruction': instruction_index, 'opcode': op, 'issuer': issuer,
                   'kind': kind, 'address': address, 'owner': owner,
                   'coordinate': coordinate(owner, cell), 'start_cycle': tick,
                   'end_cycle': tick + cycles, 'energy_fj': energy}
            tick += cycles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--features', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--n-train', type=int, required=True)
    parser.add_argument('--n-test', type=int, required=True)
    parser.add_argument('--learning-rate', type=float, required=True)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    document = build_mlp(width=args.width, features=args.features, epochs=args.epochs,
                         batch=args.batch, n_train=args.n_train, n_test=args.n_test,
                         learning_rate=args.learning_rate, seed=args.seed)
    result = score(document)
    result['software'] = {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()}
    result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (HERE / 'affine.py', HERE / 'model_ir.py', Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=True)
    program_file = args.output / 'program.spatial.json'
    program_file.write_text(json.dumps(document, indent=2) + '\n')
    result['program_file_sha256'] = hashlib.sha256(program_file.read_bytes()).hexdigest()
    (args.output / 'grid-score.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('energy_mj','time_ms','energy_fj','cycles','peak_allocated_scratch_bytes','time_to_score_seconds')}, indent=2))


if __name__ == '__main__':
    main()
