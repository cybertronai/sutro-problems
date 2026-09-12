"""Independent tiny expanded execution checks for the compressed grid scorer."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
import unittest

import numpy as np

from affine import Program, ins, loop, make_program, ref
from model_ir import build_mlp, initial_parameters
from score import COMPUTE, PORTS, STAGE_CELL, access_cost, coordinate, placement, score, scheduled_accesses, scheduled_instructions, tape_cost


def route(start, end):
    """Independent horizontal-then-vertical nearest-neighbor route."""
    x, y = start
    result = []
    while x != end[0]:
        nx = x + (1 if x < end[0] else -1)
        result.append(((x, y), (nx, y)))
        x = nx
    while y != end[1]:
        ny = y + (1 if y < end[1] else -1)
        result.append(((x, y), (x, ny)))
        y = ny
    return result


def reference_run(document, input_words, execute=True):
    """Expand all primitives; charge wire events and local scratch independently.

    Numeric words are uint32. This reference has no histogram aggregation and
    does not call score.access_cost/tape_cost or use their closed-form costs.
    Link occupancy, scratch service and per-port tape cursors are validated.
    """
    program = Program(document)
    owners, cells, _, _, _ = placement(program.words)
    memory = {}
    tapes = [list(input_words[i::PORTS]) for i in range(PORTS)]
    cursors = [0] * PORTS
    outputs = [[] for _ in range(PORTS)]
    tick = energy = 0
    events = Counter()
    instructions = Counter()
    float_word = lambda value: np.asarray(value, dtype=np.uint32).view(np.float32)[()]
    raw = lambda value: int(np.asarray(value, dtype=np.float32).view(np.uint32)[()])

    def location(address):
        if isinstance(address, tuple):
            return (address[1], 0), STAGE_CELL
        return tuple(map(int, owners[address])), tuple(map(int, cells[address]))

    def local_access(owner, cell, start):
        nonlocal energy
        key = ('scratch', owner, start)
        events[key] += 1
        if events[key] != 1:
            raise AssertionError('Scratch service contention')
        core = coordinate(owner, (64, 63))
        point = coordinate(owner, cell)
        distance = abs(core[0] - point[0]) + abs(core[1] - point[1])
        energy += max(50, distance * 2)

    def link(edge, start, hops=128):
        nonlocal energy
        key = ('link', edge, start)
        events[key] += 1
        if events[key] != 1:
            raise AssertionError('Directed-link contention')
        energy += hops

    def scratch(issuer, address, kind):
        nonlocal tick
        owner, cell = location(address)
        path = route(issuer, owner)
        if not path:
            local_access(owner, cell, tick)
            tick += 1
        elif kind == 'read':
            for offset, edge in enumerate(path):
                link(edge, tick + offset)
            local_access(owner, cell, tick + len(path))
            for offset, (a, b) in enumerate(reversed(path)):
                link((b, a), tick + len(path) + 1 + offset)
            tick += len(path) * 2 + 1
        else:
            # Address and data words follow at consecutive boundaries. At an
            # intermediate router the first word dequeues as the next arrives.
            for offset, edge in enumerate(path):
                link(edge, tick + offset)
                link(edge, tick + offset + 1)
            local_access(owner, cell, tick + len(path) + 1)
            tick += len(path) + 2

    for issuer, instruction in scheduled_instructions(document):
        op, *args = instruction
        instructions['cmp' if op.startswith('cmp_') else op] += 1
        if op == 'recv':
            dst = args[0]
            owner, cell = location(dst)
            assert owner == issuer and issuer[1] == 0
            port = issuer[0]
            if cursors[port] >= len(tapes[port]):
                raise AssertionError('Read beyond input tape')
            value = int(tapes[port][cursors[port]])
            cursors[port] += 1
            link((('port', port), issuer), tick, 64)
            local_access(owner, cell, tick + 1)
            tick += 2
            memory[dst] = value
        elif op == 'send':
            src = args[0]
            owner, cell = location(src)
            assert owner == issuer and issuer[1] == 0
            if src not in memory:
                raise AssertionError('Read uninitialized output')
            local_access(owner, cell, tick)
            link((issuer, ('port', issuer[0])), tick + 1, 64)
            tick += 2
            outputs[issuer[0]].append(memory[src])
        else:
            dst = args[0]
            values = []
            if op == 'set':
                value = args[1]
            else:
                for src in args[1:]:
                    if src not in memory:
                        raise AssertionError('Read uninitialized operand')
                    scratch(issuer, src, 'read')
                    values.append(memory[src])
                if not execute:
                    value = values[0]
                elif op == 'copy':
                    value = values[0]
                elif op == 'select':
                    value = values[1] if values[0] else values[2]
                elif op.startswith('cmp'):
                    predicate = op[4:] if op.startswith('cmp_') else 'lt'
                    a, b = map(float_word, values)
                    value = int({'eq': a == b, 'lt': a < b}[predicate])
                else:
                    a, b = map(float_word, values)
                    value = raw({'add': np.add, 'sub': np.subtract, 'mul': np.multiply}[op](a, b, dtype=np.float32))
            scratch(issuer, dst, 'write')
            memory[dst] = value
    assert cursors == [len(tape) for tape in tapes]
    n_output = sum(map(len, outputs))
    ordered_output = [outputs[q % PORTS][q // PORTS] for q in range(n_output)]
    return {'energy_fj': energy, 'cycles': tick, 'memory': memory,
            'output': ordered_output, 'instructions': dict(instructions),
            'wire_and_scratch_events': len(events)}


def ordered_mm(a, b):
    result = np.zeros((len(a), b.shape[1]), dtype=np.float32)
    for k in range(a.shape[1]):
        result = result + a[:, k, None] * b[None, k, :]
    return result


def ordered_sum(a):
    result = np.zeros(a.shape[1], dtype=np.float32)
    for row in a:
        result = result + row
    return result


def independent_learner(train, labels, query, width, batch, epochs, rate, seed):
    x = train * np.float32(4) - np.float32(.5)
    q = query * np.float32(4) - np.float32(.5)
    target = (labels[:, None] == np.arange(10)).astype(np.float32)
    parameters = initial_parameters(x.shape[1], width, seed)
    def forward(values, parameters):
        w1, b1, w2, b2 = parameters
        h = np.maximum(ordered_mm(values, w1) + b1, np.float32(0))
        return h, ordered_mm(h, w2) + b2
    step = np.float32(rate / batch)
    for _ in range(epochs):
        for start in range(0, len(x), batch):
            xb, yb = x[start:start + batch], target[start:start + batch]
            w1, b1, w2, b2 = parameters
            h, scores = forward(xb, parameters)
            d2 = scores - yb
            d1 = np.where(h > 0, ordered_mm(d2, w2.T), np.float32(0))
            parameters = [w1 - step * ordered_mm(xb.T, d1),
                          b1 - step * ordered_sum(d1),
                          w2 - step * ordered_mm(h.T, d2),
                          b2 - step * ordered_sum(d2)]
    _, scores = forward(q, parameters)
    return scores.argmax(axis=1), parameters


class SpatialTests(unittest.TestCase):
    def test_spec_worked_examples(self):
        cell = (32, 31)
        self.assertEqual(access_cost((0, 0), (0, 0), cell, 'write'), (128, 1))
        self.assertEqual(access_cost((0, 1), (0, 0), cell, 'read'), (384, 3))
        self.assertEqual(access_cost((0, 0), (0, 1), cell, 'write'), (384, 3))
        self.assertEqual(access_cost((0, 0), (2, 1), cell, 'read'), (896, 7))
        self.assertEqual(access_cost((0, 0), (2, 1), cell, 'write'), (896, 5))
        steps = [tape_cost((0,0), cell),
                 access_cost((0,1), (0,0), cell, 'read'),
                 access_cost((0,1), (0,1), cell, 'write'),
                 tape_cost((0,1), cell)]
        self.assertEqual(sum(s[0] for s in steps), 1024)
        self.assertEqual(sum(s[1] for s in steps), 9)

    def test_placement_capacity_and_reservation(self):
        owners, cells, _, _, occupancy = placement(40_000)
        positions = {coordinate(tuple(tile), tuple(cell)) for tile, cell in zip(owners[1:], cells[1:])}
        stages = {coordinate((i, 0), STAGE_CELL) for i in range(PORTS)}
        self.assertEqual(len(positions), 40_000)
        self.assertFalse(positions & stages)
        self.assertLessEqual(max(occupancy.values()), 12_288)
        self.assertTrue(all(-16_000 <= x <= 15_999 and 1 <= y <= 16_000 for x,y in positions))
        self.assertTrue(all(not (32 <= u <= 95 and 32 <= v <= 95) for u,v in cells[1:]))
        self.assertEqual(tuple(owners[12_287]), COMPUTE)
        self.assertNotEqual(tuple(owners[12_288]), COMPUTE)

    def test_tape_wrap_and_remote_memory_match_expanded_events(self):
        document = make_program([('x', 12_301)], [
            loop('i', 12_301, [ins('set', ref('x', i=1), 0)]),
            loop('k', 253, [ins('recv', ref('x', 12_000, k=1)), ins('send', ref('x', 12_000, k=1))]),
            ins('copy', ref('x', 12_300), ref('x', 12_252)),
            ins('send', ref('x', 12_300)),
        ])
        words = np.arange(253, dtype=np.uint32)
        actual = reference_run(document, words)
        aggregate = score(document)
        self.assertEqual(actual['output'], list(range(253)) + [252])
        for key in ('energy_fj', 'cycles'):
            self.assertEqual(actual[key], aggregate[key])
        self.assertEqual(actual['instructions'], aggregate['executed_instructions'])
        accesses = list(scheduled_accesses(document))
        self.assertEqual(accesses[0]['start_cycle'], 0)
        self.assertTrue(all(a['end_cycle'] == b['start_cycle'] for a,b in zip(accesses, accesses[1:])))
        self.assertEqual(accesses[-1]['end_cycle'], actual['cycles'])
        self.assertEqual(sum(a['energy_fj'] for a in accesses), actual['energy_fj'])

    def test_tiny_mlp_matches_numerics_and_expanded_physics(self):
        dimensions = dict(features=3, width=4, n_train=4, n_test=3, batch=2,
                          epochs=2, learning_rate=.1, seed=101)
        document = build_mlp(**dimensions)
        rng = np.random.default_rng(309)
        train = rng.random((4,3), dtype=np.float32)
        query = rng.random((3,3), dtype=np.float32)
        labels = np.array([1,4,9,2], dtype=np.uint32)
        words = np.concatenate([train.ravel().view(np.uint32), labels, query.ravel().view(np.uint32)])
        actual = reference_run(document, words)
        aggregate = score(document)
        expected_predictions, parameters = independent_learner(train, labels, query, 4, 2, 2, .1, 101)
        self.assertEqual(actual['output'], expected_predictions.tolist())
        program = Program(document)
        for region, expected in zip(('w1','b1','w2','b2'), parameters):
            first, words = program.regions[region]
            actual_words = np.array([actual['memory'][address] for address in range(first, first+words)], dtype=np.uint32)
            np.testing.assert_array_equal(actual_words, expected.ravel().view(np.uint32))
        for key in ('energy_fj', 'cycles'):
            self.assertEqual(actual[key], aggregate[key])
        self.assertEqual(actual['instructions'], aggregate['executed_instructions'])

    def test_undefined_reads_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Uninitialized'):
            score(make_program([('x',1)], [ins('send', ref('x'))]))


if __name__ == '__main__':
    begin = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SpatialTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {'tests_run': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'successful': result.wasSuccessful(),
              'runtime_seconds': time.perf_counter() - begin,
              'checks': ['specification access examples', 'legal injective placement and tile capacity',
                         '253 input/254 output port wrap and remote-memory event expansion',
                         'complete tiny MLP: bitwise parameters, predictions, and expanded wire/scratch energy/time',
                         'uninitialized source rejection']}
    Path(__file__).with_name('test-results.json').write_text(json.dumps(record, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
