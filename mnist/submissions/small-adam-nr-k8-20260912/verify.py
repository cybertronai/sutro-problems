"""Re-check frozen manifests, inputs, predictions and accuracy."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
E = HERE / 'evidence/accuracy'


def repo():
    env = os.environ.get('SUTRO_REPO')
    if env:
        return Path(env)
    for base in (HERE, *HERE.parents):
        if (base / 'mnist/code/data.py').exists():
            return base
        if (base / 'vendor/sutro-problems/mnist/code/data.py').exists():
            return base / 'vendor/sutro-problems'
    raise SystemExit('sutro-problems checkout not found; set SUTRO_REPO')


sys.path.insert(0, str(repo()))
from mnist.code import data as ds

def raw():
    env = os.environ.get('SUTRO_RAW')
    if env:
        return Path(env)
    for base in (HERE, *HERE.parents):
        c = base / 'data/certification/draw-00/raw'
        if c.exists():
            return c
    raise SystemExit('raw MNIST dir not found; set SUTRO_RAW')

draws = json.loads((E / 'draw_manifest.json').read_text())
preds = json.loads((E / 'prediction_manifest.json').read_text())
acc = json.loads((E / 'accuracy.json').read_text())
assert [d['dataset_seed'] for d in draws['draws']] == [d['dataset_seed'] for d in preds['draws']]
labels = ds.read_idx(raw() / 'train-labels-idx1-ubyte.gz', 60000, False)
total = 0
for d, record in zip(draws['draws'], preds['draws'], strict=True):
    order = np.random.Generator(np.random.PCG64(d['dataset_seed'])).permutation(60000)
    train, test = order[:1000], order[1000:2000]
    assert ds.array_hash(train) == d['train_indices_sha256']
    assert ds.array_hash(test) == d['test_indices_sha256']
    path = E / record['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['prediction_sha256']
    pred = np.load(path, allow_pickle=False)
    assert pred.shape == (1000,) and ((pred >= 0) & (pred < 10)).all()
    assert ds.array_hash(pred) == record['array_sha256']
    total += int((pred == labels[test]).sum())
assert total == acc['correct'] == sum(r['correct'] for r in acc['draws'])
assert acc['target_met'] is True
print('PASS: draws, indices, input hashes, prediction hashes and label-derived accuracy consistent')
