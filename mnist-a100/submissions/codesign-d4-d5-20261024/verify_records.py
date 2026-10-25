"""Reconstruct all initial author canonical results using the unchanged scorer."""
from pathlib import Path
import collections
import hashlib
import json
import statistics
import sys

here = Path(__file__).resolve().parent
root = here.parents[1]
sys.path.insert(0, str(root))
import mnist

manifest = json.loads((here / 'manifest.json').read_text())
assert hashlib.sha256((root / 'mnist.py').read_bytes()).hexdigest() == manifest['scorer_sha256']
assert hashlib.sha256((root / 'run_modal.py').read_bytes()).hexdigest() == manifest['runner_sha256']
for config in manifest['candidates']:
    source = here / config['source']
    data = source.read_bytes()
    assert len(data) == config['bytes'] <= 20480
    assert hashlib.sha256(data).hexdigest() == config['sha256']
    mnist.check_source(data, config['function'], source.name)
    folder = here / config['records']
    paths = sorted(folder.glob('run-*.json'))
    assert [p.name for p in paths] == ['run-1.json', 'run-2.json', 'run-3.json']
    times = []
    correct = total = 0
    for path in paths:
        saved = json.loads(path.read_text())
        assert saved['returncode'] == 0
        record = saved['record']
        assert record['source_sha256'] == config['sha256']
        assert record['function'] == config['function']
        assert Path(record['file']).name == source.name
        assert record['sandboxed'] is True and 'A100' in record['device']
        band = mnist.DIFFICULTY[config['difficulty']][0]
        assert record['band_bp'] == band
        calls = [mnist.Call(**row) for row in record['calls']]
        assert collections.Counter(c.dataset for c in calls) == {'mnist': 11, record['holdout']: 4}
        assert all(c.total == 10000 for c in calls)
        problems, ranked, _ = mnist.judge(calls, band)
        assert problems == record['problems'] == []
        assert abs(ranked - record['ranked_ms']) < 1e-9
        times.append(ranked)
        correct += sum(c.correct for c in calls if c.dataset == 'mnist')
        total += sum(c.total for c in calls if c.dataset == 'mnist')
    print(json.dumps(dict(difficulty=config['difficulty'], passed='3/3', median_ms=statistics.median(times), pooled_error_pct=100*(1-correct/total))))
