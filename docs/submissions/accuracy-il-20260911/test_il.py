#!/usr/bin/env python3
"""Independent count and expansion checks for the compact affine IL scorer."""
from collections import Counter
import copy
import csv
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import unittest

import numpy as np

from il import Program, expand, ins, loop, make_program, ref, score
from nn_il import nearest_neighbor

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent.parent / 'submissions' / '1nn-v4-20260911'
spec = importlib.util.spec_from_file_location('baseline_score_v4', BASELINE / 'score_v4.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class CompactILTests(unittest.TestCase):
    def test_full_baseline_exact_per_address(self):
        result, reads, writes = score(nearest_neighbor(), include_counts=True)
        previous = json.loads((BASELINE / 'model-score.json').read_text())
        self.assertEqual(result['instructions'], baseline.expected_counts())
        for name in ('total_instructions', 'charged_reads', 'charged_writes', 'time_ps', 'energy_fj',
                     'area_um2_occupied_cells', 'peak_initialized_scratch_words', 'input_tape_words', 'output_tape_words'):
            self.assertEqual(result[name], previous[name], name)
        rows = list(csv.DictReader((BASELINE / 'placement.csv').open()))
        np.testing.assert_array_equal(reads[1:], [int(row['charged_reads']) for row in rows])
        np.testing.assert_array_equal(writes[1:], [int(row['charged_writes']) for row in rows])
        self.assertEqual(result['time_ticks_0_2_ps'], baseline.closed_form_cost(baseline.placement())['time_ticks'])

    def test_expansion_and_numeric_execution(self):
        for train, test, features in ((2, 2, 1), (3, 4, 2), (4, 3, 9)):
            document = nearest_neighbor(train, test, features)
            emitted = list(expand(document))
            self.assertEqual(emitted, list(baseline.program(train, test, features)))
            rng = np.random.default_rng(31)
            train_values = rng.random(train * features).astype('<f4').view('<u4').tolist()
            labels = list(range(train))
            queries = rng.random(test * features).astype('<f4').view('<u4').tolist()
            tape = train_values + labels + queries
            machine = baseline.Machine(baseline.placement(14 + train * (features + 1)), tape)
            machine.run(iter(emitted))
            result, reads, writes = score(document, include_counts=True)
            self.assertEqual(result['energy_fj'], machine.energy_fj)
            self.assertEqual(result['time_ticks_0_2_ps'], machine.time_ticks)
            np.testing.assert_array_equal(reads, machine.read_counts)
            np.testing.assert_array_equal(writes, machine.write_counts)

    def test_affine_histograms_negative_strides_and_collisions(self):
        rng = np.random.default_rng(99)
        for _ in range(50):
            counts = rng.integers(1, 6, 3).tolist()
            strides = rng.integers(-4, 5, 3).tolist()
            starts = rng.integers(-3, 4, 3).tolist()
            minimum = sum(min(c*s, c*(s+n-1)) for c,s,n in zip(strides, starts, counts))
            source = ref('memory', -minimum, **dict(zip(('i','j','k'), strides)))
            body = [ins('mul', ref('memory', 100), source, source),
                    ins('select', ref('memory', 100), ref('memory', 100), source, source)]
            for v,n,s in reversed(list(zip(('i','j','k'), counts, starts))):
                body = [loop(v,n,body,start=s)]
            document = make_program([('memory',101)], [loop('init',101,[ins('set',ref('memory',init=1),0)])]+body)
            result, reads, writes = score(document, include_counts=True)
            machine = baseline.Machine(baseline.placement(101), [])
            machine.run(expand(document))
            np.testing.assert_array_equal(reads, machine.read_counts)
            np.testing.assert_array_equal(writes, machine.write_counts)
            self.assertEqual(result['energy_fj'], machine.energy_fj)
            self.assertEqual(result['time_ticks_0_2_ps'], machine.time_ticks)

    def test_far_access_floors_and_free_tape(self):
        document = make_program([('memory',10000)], [ins('recv',ref('memory',9999)),
            ins('set',ref('memory'),1),ins('add',ref('memory'),ref('memory'),ref('memory')),
            ins('copy',ref('memory',9999),ref('memory')), ins('send',ref('memory',9999))])
        machine = baseline.Machine(baseline.placement(10000), [5], 'u32')
        machine.run(expand(document))
        result = score(document)
        self.assertEqual(result['energy_fj'],machine.energy_fj)
        self.assertEqual(result['time_ticks_0_2_ps'],machine.time_ticks)
        self.assertEqual(result['charged_reads'],3)
        self.assertEqual(result['charged_writes'],3)

    def test_reject_uninitialized_unchosen_select_source(self):
        document = make_program([('memory',2)], [ins('set',ref('memory'),1),
            ins('select',ref('memory'),ref('memory'),ref('memory'),ref('memory',1))])
        with self.assertRaisesRegex(ValueError,'Uninitialized'):
            score(document)

    def test_alias_source_is_read_before_write(self):
        document = make_program([('memory',1)], [ins('add',ref('memory'),ref('memory'),ref('memory'))])
        with self.assertRaisesRegex(ValueError,'Uninitialized'):
            score(document)

    def test_initialization_loop_skip_is_sound(self):
        valid = make_program([('memory',3)], [ins('set',ref('memory'),1),
            loop('iteration',10**9,[ins('copy',ref('memory',1),ref('memory'))])])
        result = score(valid)
        self.assertLess(result['initialization_proof_visits'],10)
        # Address-dependent loop cannot be skipped: the third read is unwritten.
        invalid = make_program([('memory',4)], [ins('set',ref('memory'),1),
            loop('i',3,[ins('copy',ref('memory',3),ref('memory',i=1))])])
        with self.assertRaisesRegex(ValueError,'Uninitialized'):
            score(invalid)

    def test_zero_trip_loop_has_no_effect(self):
        document = make_program([('memory',1)], [loop('i',0,[ins('copy',ref('memory',999),ref('memory',999))])])
        result = score(document)
        self.assertEqual(result['total_instructions'],0)
        self.assertEqual(result['charged_accesses'],0)

    def test_validation_rejects_bad_schema_bounds_and_overflow(self):
        cases = [make_program([('memory',2)],[loop('i',3,[ins('set',ref('memory',i=1),0)])]),
                 make_program([('memory',1)],[ins('set',ref('memory',i=1),0)]),
                 make_program([('memory',1)],[ins('set',ref('memory'),-1)]),
                 make_program([('memory',1)],[loop('i',10**19,[ins('set',ref('memory'),0)])]),
                 make_program([('memory',1)],[loop('i',2,[loop('i',2,[ins('set',ref('memory'),0)])])]),
                 make_program([('memory',1)],[{'op':'algorithm','certificate':{'cost':0}}])]
        for document in cases:
            with self.assertRaises(ValueError):
                score(document)

    def test_compact_mlp_counts_against_expansion(self):
        from mlp_il import build_mlp
        document = build_mlp(width=3, epochs=2, learning_rate=.05, n_train=4, n_test=3, batch=2)
        result, reads, writes = score(document, include_counts=True)
        program = Program(document)
        expected_reads = np.zeros(program.words + 1, dtype=np.int64)
        expected_writes = np.zeros(program.words + 1, dtype=np.int64)
        opcodes = Counter()
        for opcode, destination, *sources in expand(document):
            opcodes["cmp" if opcode.startswith("cmp_") else opcode] += 1
            if opcode in ("recv", "send"):
                continue
            if opcode != "set":
                for address in sources:
                    expected_reads[address] += 1
            expected_writes[destination] += 1
        self.assertEqual(dict(opcodes), result["instructions"])
        np.testing.assert_array_equal(reads, expected_reads)
        np.testing.assert_array_equal(writes, expected_writes)

    def test_cmp_predicates_have_equal_access_costs(self):
        values = []
        for predicate in ('eq','ne','lt','le','gt','ge'):
            document = make_program([('memory',1)],[ins('set',ref('memory'),0),
                       ins('cmp',ref('memory'),ref('memory'),ref('memory'),predicate=predicate)])
            values.append(score(document)['energy_fj'])
        self.assertEqual(len(set(values)),1)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CompactILTests)
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    output = {'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
              'passed':result.wasSuccessful(),'duration_seconds':time.perf_counter()-start,
              'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
              'checks':['exact full-baseline opcode totals and all 6,014 per-address read/write multiplicities',
                        'small-program expanded instructions and numeric execution against previous v4 interpreter',
                        '50 negative-stride/collision/alias affine programs independently enumerated',
                        'distance floors, asymmetric read/write times, free recv/send',
                        'unselected select sources initialized and charged',
                        'source-before-destination aliasing',
                        'sound invariant-loop initialization skipping',
                        'zero-trip loops','malformed programs, bounds, overflow, unbound/shadowed indices rejected',
                        'comparison predicate access pricing',
                        'compact MLP all per-address counts independently expanded on a two-batch two-epoch toy']}
    (HERE/'il-validation.json').write_text(json.dumps(output,indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
