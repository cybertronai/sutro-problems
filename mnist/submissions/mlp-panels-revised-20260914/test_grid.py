"""Bounded exhaustive spatial-event and ordered-FP32 checks of panel programs."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import time
import unittest

import numpy as np

import grid_score as grid


def expanded_run(document, input_words):
    """Expand primitives and individual wire words, without static cost formulas.

    Only one blocking access's event set is retained. All its events must finish
    before the next access starts, so discarded history cannot hide contention.
    """
    program = grid.affine.Program(document)
    owners, cells, _, _, _ = grid.spatial.placement(program.words)
    memory = {}
    counts = {name: np.zeros(program.words + 1, np.int64) for name in
              ('reads', 'writes', 'input_destinations', 'output_sources')}
    instructions = Counter()
    cursors = [0] * 250
    output = []
    tick = energy = event_count = max_events = 0
    input_index = output_index = 0

    def access(issuer, address, kind):
        nonlocal tick, energy, event_count, max_events
        if isinstance(address, tuple):
            owner, cell = (address[1], 0), grid.spatial.STAGE_CELL
        else:
            owner = tuple(map(int, owners[address]))
            cell = tuple(map(int, cells[address]))
        events = set()
        start = tick

        def event(resource, cycle, cost):
            nonlocal energy, event_count
            assert cycle >= start
            key = (resource, cycle)
            assert key not in events, ('contention', key)
            events.add(key)
            energy += cost
            event_count += 1

        def local(cycle):
            distance = abs(cell[0] - 64) + abs(cell[1] - 63)
            event(('scratch', owner), cycle, max(50, 2 * distance))

        if kind in ('recv', 'send'):
            assert issuer == owner and owner[1] == 0
            if kind == 'recv':
                event(('wire', ('port', owner[0]), owner), tick, 64)
                local(tick + 1)
            else:
                local(tick)
                event(('wire', owner, ('port', owner[0])), tick + 1, 64)
            tick += 2
        else:
            path = []
            x, y = issuer
            while x != owner[0]:
                nx = x + (1 if x < owner[0] else -1)
                path.append(((x, y), (nx, y)))
                x = nx
            while y != owner[1]:
                ny = y + (1 if y < owner[1] else -1)
                path.append(((x, y), (x, ny)))
                y = ny
            if not path:
                local(tick)
                tick += 1
            elif kind == 'read':
                for offset, (a, b) in enumerate(path):
                    event(('wire', a, b), tick + offset, 128)
                local(tick + len(path))
                for offset, (a, b) in enumerate(reversed(path)):
                    event(('wire', b, a), tick + len(path) + 1 + offset, 128)
                tick += 2 * len(path) + 1
            else:
                assert kind == 'write'
                for offset, (a, b) in enumerate(path):
                    event(('wire', a, b), tick + offset, 128)
                    event(('wire', a, b), tick + offset + 1, 128)
                local(tick + len(path) + 1)
                tick += len(path) + 2
        assert events and max(cycle for _, cycle in events) < tick
        max_events = max(max_events, len(events))

    # Independently lower tape operations rather than reuse scheduled_instructions.
    for instruction in grid.affine.expand(document):
        op, *args = instruction
        instructions['cmp' if op.startswith('cmp_') else op] += 1
        if op == 'recv':
            dst = args[0]
            port = input_index % 250
            index = port + cursors[port] * 250
            assert index == input_index and index < len(input_words)
            stage = ('stage', port)
            access((port, 0), stage, 'recv')
            access((125, 0), stage, 'read')
            access((125, 0), dst, 'write')
            memory[dst] = int(input_words[index])
            counts['input_destinations'][dst] += 1
            instructions['copy'] += 1
            cursors[port] += 1
            input_index += 1
        elif op == 'send':
            src = args[0]
            assert src in memory
            port = output_index % 250
            stage = ('stage', port)
            access((125, 0), src, 'read')
            access((125, 0), stage, 'write')
            access((port, 0), stage, 'send')
            counts['output_sources'][src] += 1
            instructions['copy'] += 1
            output.append(memory[src])
            output_index += 1
        else:
            dst = args[0]
            if op == 'set':
                value = args[1]
            else:
                values = []
                for src in args[1:]:
                    assert src in memory, ('uninitialized', src)
                    access((125, 0), src, 'read')
                    counts['reads'][src] += 1
                    values.append(memory[src])
                if op == 'copy':
                    value = values[0]
                elif op == 'select':
                    value = values[1] if values[0] else values[2]
                else:
                    a, b = np.asarray(values, np.uint32).view(np.float32)
                    if op.startswith('cmp'):
                        value = int({'cmp': np.less, 'cmp_eq': np.equal,
                                     'cmp_ne': np.not_equal, 'cmp_le': np.less_equal,
                                     'cmp_gt': np.greater, 'cmp_ge': np.greater_equal}[op](a, b))
                    else:
                        value = int({'add': np.add, 'sub': np.subtract, 'mul': np.multiply}[op](
                            a, b, dtype=np.float32).view(np.uint32))
            access((125, 0), dst, 'write')
            counts['writes'][dst] += 1
            memory[dst] = value
    assert input_index == len(input_words)
    return {'memory': memory, 'output': output, 'counts': counts,
            'instructions': dict(instructions), 'energy_fj': energy, 'cycles': tick,
            'wire_and_scratch_events': event_count, 'max_retained_events': max_events}


RECORDS = []


class PanelGridTests(unittest.TestCase):
    def test_bounded_runner_against_original_event_expansion(self):
        original = grid.load_module('_validated_spatial_tests', grid.GRID / 'test_score.py')
        a = grid.affine
        document = a.make_program([('x', 12301)], [
            a.loop('i', 12301, [a.ins('set', a.ref('x', i=1), 0)]),
            a.loop('k', 253, [a.ins('recv', a.ref('x', 12000, k=1)),
                              a.ins('send', a.ref('x', 12000, k=1))]),
            a.ins('copy', a.ref('x', 12300), a.ref('x', 12252)),
            a.ins('send', a.ref('x', 12300))])
        words = np.arange(253, dtype=np.uint32)
        actual = expanded_run(document, words)
        expected = original.reference_run(document, words)
        for key in ('energy_fj', 'cycles', 'output', 'instructions', 'wire_and_scratch_events'):
            self.assertEqual(actual[key], expected[key])
        for address in range(1, 12302):
            self.assertEqual(actual['memory'][address], expected['memory'][address])
        self.assertLessEqual(actual['max_retained_events'], 251)

    def test_header_only(self):
        panels = grid.load_module('panels', grid.HERE / 'panels.py')
        config = json.loads((grid.HERE / 'selected.json').read_text())
        original = panels.build(config, 1, 25, 2)
        before = copy.deepcopy(original)
        converted = grid.spatial_document(original)
        self.assertEqual(original, before)
        for key in ('body', 'regions', 'metadata'):
            self.assertEqual(converted[key], original[key])
        grid.affine.Program(converted)

    def test_full_tiny_panel_programs(self):
        panels = grid.load_module('panels', grid.HERE / 'panels.py')
        reference = grid.load_module('reference', grid.HERE / 'reference.py')
        selected = json.loads((grid.HERE / 'selected.json').read_text())
        tail = copy.deepcopy(selected)
        stages = ('both', 'left_panel', 'right_panel', 'left', 'right', 'none', 'both')
        for name, stage in zip(panels.PRODUCTS, stages):
            tail['products'][name] = {'m': 4, 'n': '6+10', 'stage': stage, 'order': 'column'}
        for config_name, config in (('selected', selected), ('asymmetric_tails', tail)):
            self.assertEqual(config['inference_batch'], 25)
            document = grid.spatial_document(panels.build(config, 1, 25, 2))
            self.assertEqual(document['metadata']['batch_size'], 25)
            aggregate = grid.spatial.score(document)
            program = grid.affine.Program(document)
            histograms, _ = grid.spatial.histogram_counts(program)
            for mutated in (False, True):
                with self.subTest(config=config_name, mutated=mutated):
                    rng = np.random.default_rng(781 if mutated else 309)
                    data = {'train_images': rng.random((25, 9), dtype=np.float32),
                            'test_images': rng.random((2, 9), dtype=np.float32),
                            'train_labels': (np.arange(25, dtype=np.uint32) + (3 if mutated else 0)) % 10}
                    words = np.concatenate([data['train_images'].ravel().view(np.uint32),
                                            data['train_labels'], data['test_images'].ravel().view(np.uint32)])
                    actual = expanded_run(document, words)
                    # None product configurations force the independent ascending-K CPU path.
                    baseline = panels.default_config()
                    baseline['inference_batch'] = 25
                    expected = reference.cpu(data, baseline, 1)
                    parameters = []
                    for name in ('w1', 'b1', 'w2', 'b2'):
                        first, size = program.regions[name]
                        parameters.extend(actual['memory'][address] for address in range(first, first + size))
                    np.testing.assert_array_equal(np.asarray(parameters, np.uint32), expected['params'].view(np.uint32))
                    first, _ = program.regions['d2']
                    scores = np.asarray([actual['memory'][address] for address in range(first, first + 20)], np.uint32)
                    np.testing.assert_array_equal(scores, expected['scores'].ravel().view(np.uint32))
                    np.testing.assert_array_equal(np.asarray(actual['output'], np.uint32), expected['predictions'].astype(np.uint32))
                    for key in ('energy_fj', 'cycles'):
                        self.assertEqual(actual[key], aggregate[key])
                    self.assertEqual(actual['instructions'], aggregate['executed_instructions'])
                    for name in histograms:
                        np.testing.assert_array_equal(actual['counts'][name], histograms[name])
                    RECORDS.append({'config': config_name, 'mutated': mutated, 'train': 25, 'query': 2,
                                    'epochs': 1, 'all_parameter_score_prediction_bits_equal': True,
                                    'all_address_counts_and_expanded_costs_equal': True,
                                    'parameter_sha256': hashlib.sha256(np.asarray(parameters, '<u4').tobytes()).hexdigest(),
                                    **{key: actual[key] for key in ('energy_fj', 'cycles',
                                       'wire_and_scratch_events', 'max_retained_events')}})


if __name__ == '__main__':
    begin = time.perf_counter()
    # Run the original validated scorer suite without its artifact-writing main.
    original = grid.load_module('_validated_spatial_tests', grid.GRID / 'test_score.py')
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(original.SpatialTests),
                               unittest.defaultTestLoader.loadTestsFromTestCase(PanelGridTests)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'successful': result.wasSuccessful(), 'runtime_seconds': time.perf_counter() - begin,
              'panel_checks': RECORDS, 'source_sha256': grid.source_hashes()}
    (grid.HERE / 'grid-test-results.json').write_text(json.dumps(record, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
