"""Independent expanded semantics and baseline regressions for the generic IL."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest
import numpy as np

from model_ir import build_mlp, EXPERIMENT
from il import Program, expand, score

spec = importlib.util.spec_from_file_location('small_semantic_verifier', EXPERIMENT / 'validate_mlp_il.py')
small = importlib.util.module_from_spec(spec)
spec.loader.exec_module(small)
HERE = Path(__file__).resolve().parent


def ordered_mm(a, b):
    result = np.zeros((a.shape[0], b.shape[1]), np.float32)
    for k in range(a.shape[1]):
        result = result + a[:, k, None] * b[None, k, :]
    return result


def ordered_rows(a):
    result = np.zeros(a.shape[1], np.float32)
    for row in a:
        result = result + row
    return result


def reference(train, labels, queries, width, epochs, rate, batch, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    d = train.shape[1]
    w1 = rng.uniform(-1 / np.sqrt(d), 1 / np.sqrt(d), (d, width)).astype(np.float32)
    b1 = np.zeros(width, np.float32)
    w2 = rng.uniform(-1 / np.sqrt(width), 1 / np.sqrt(width), (width, 10)).astype(np.float32)
    b2 = np.zeros(10, np.float32)
    x = train * np.float32(4) - np.float32(.5)
    q = queries * np.float32(4) - np.float32(.5)
    targets = (labels[:, None] == np.arange(10)).astype(np.float32)
    step = np.float32(rate / batch)
    for _ in range(epochs):
        for first in range(0, len(x), batch):
            xb = x[first:first + batch]
            z = ordered_mm(xb, w1) + b1
            h = np.where(z > 0, z, np.float32(0))
            d2 = ordered_mm(h, w2) + b2 - targets[first:first + batch]
            d1 = np.where(z > 0, ordered_mm(d2, w2.T), np.float32(0))
            g1, gb1 = ordered_mm(xb.T, d1), ordered_rows(d1)
            g2, gb2 = ordered_mm(h.T, d2), ordered_rows(d2)
            w1 = w1 - step * g1
            b1 = b1 - step * gb1
            w2 = w2 - step * g2
            b2 = b2 - step * gb2
    h = np.maximum(ordered_mm(q, w1) + b1, np.float32(0))
    return [w1, b1, w2, b2], (ordered_mm(h, w2) + b2).argmax(axis=1)


class GeneralizedILTests(unittest.TestCase):
    def test_exact_legacy_small_regression(self):
        document = build_mlp(32, 300, .2, n_train=600, n_test=600, features=9, stream_queries=False)
        result = score(document)
        previous = json.loads((EXPERIMENT / 'h32-e300-lr0.2.score.json').read_text())
        for key in ('program_sha256','instructions','total_instructions','charged_reads','charged_writes',
                    'time_ticks_0_2_ps','energy_fj','area_um2_occupied_cells','peak_initialized_scratch_words'):
            self.assertEqual(result[key], previous[key], key)

    def test_expanded_semantics_and_accesses(self):
        rng = np.random.default_rng(71)
        for features, stream in ((1, True), (9, False), (9, True), (81, True)):
            with self.subTest(features=features, streamed=stream):
                train = rng.uniform(0,1,(4,features)).astype(np.float32)
                labels = np.array([0,3,5,8], dtype=np.uint32)
                queries = rng.uniform(0,1,(3,features)).astype(np.float32)
                document = build_mlp(3,2,.05,n_train=4,n_test=3,batch=2,features=features,stream_queries=stream)
                parsed = Program(document)
                tape = np.concatenate([train.ravel().view(np.uint32),labels,queries.ravel().view(np.uint32)])
                machine = small.SemanticMachine(parsed.coordinates, tape)
                observed = machine.run(expand(document))
                params, expected = reference(train,labels,queries,3,2,.05,2,101)
                np.testing.assert_array_equal(observed,expected)
                for name, value in zip(('w1','b1','w2','b2'), params):
                    start, words = parsed.regions[name]
                    actual = np.array(machine.memory[start:start+words], dtype=np.uint32)
                    np.testing.assert_array_equal(actual, value.ravel().view(np.uint32))
                result, reads, writes = score(document,include_counts=True)
                self.assertEqual(result['energy_fj'],machine.energy_fj)
                self.assertEqual(result['time_ticks_0_2_ps'],machine.time_ticks)
                self.assertEqual(result['instructions'],dict(machine.instructions))
                np.testing.assert_array_equal(reads,machine.read_counts)
                np.testing.assert_array_equal(writes,machine.write_counts)

    def test_medium_streamed_allocation_stays_within_unchanged_guard(self):
        for width in (128,256,512):
            document = build_mlp(width,50,.05)
            regions = {region['name']:region['words'] for region in document['regions']}
            self.assertEqual(regions['q'],81)
            self.assertEqual(regions['x'],6000*81)
            self.assertLess(sum(regions.values()),1000000)
            self.assertEqual(document['regions'][2]['name'],'q')

    def test_bad_dimensions_rejected(self):
        for options in ({'batch':0},{'n_train':6010},{'features':0},{'learning_rate':float('nan')}):
            args={'width':32,'epochs':1,'learning_rate':.05,**options}
            with self.assertRaises(ValueError):
                build_mlp(**args)


if __name__ == '__main__':
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GeneralizedILTests))
    record={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'wall_seconds':time.perf_counter()-started,'python':sys.version,'numpy':np.__version__,
            'source_sha256':{str(path.relative_to(HERE.parents[2])):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__),HERE/'model_ir.py',EXPERIMENT/'il.py',EXPERIMENT/'validate_mlp_il.py',HERE.parent/'1nn-v4-20260911/score_v4.py')},
            'scope':'Legacy small canonical program/hash/exact cost identity; independent 1/9/81-feature expanded numerical semantics and every address count; streamed medium allocations; invalid shapes rejected'}
    (HERE/'model-ir-validation.json').write_text(json.dumps(record,indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
