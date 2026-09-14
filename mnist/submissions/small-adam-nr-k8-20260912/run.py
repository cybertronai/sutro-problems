"""Prepare, freeze and score the NR-K8 Adam MNIST-small submission.

Two-phase evaluation: predictions are written and hashed before any
evaluation-label slice is opened; scoring verifies every hash first.
"""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'evidence/accuracy'
sys.path.insert(0, str(HERE))
from reference import train_predict, ordered_mm, ordered_rows, f32

SEEDS = list(range(20261201, 20261212))
CONFIG = {'width': 32, 'epochs': 100, 'nr': 8, 'batch': 25, 'lr': 0.2}

from mnist.code import data as ds  # repository data module
def _raw():
    import os
    env = os.environ.get('SUTRO_RAW')
    if env:
        return Path(env)
    for base in (HERE, *HERE.parents):
        candidate = base / 'data/certification/draw-00/raw'
        if candidate.exists():
            return candidate
    raise SystemExit('raw MNIST dir not found; set SUTRO_RAW')

RAW = _raw()


def _load(seed):
    order = np.random.Generator(np.random.PCG64(seed)).permutation(60000)
    train, test = order[:1000].astype(np.int64), order[1000:2000].astype(np.int64)
    pixels = ds.read_idx(RAW / 'train-images-idx3-ubyte.gz', 60000, True)
    labels = ds.read_idx(RAW / 'train-labels-idx1-ubyte.gz', 60000, False)
    x = ds.area_resize(pixels[train].astype(f32) / f32(255), 3).reshape(1000, 9) * f32(4) - f32(.5)
    q = ds.area_resize(pixels[test].astype(f32) / f32(255), 3).reshape(1000, 9) * f32(4) - f32(.5)
    target = (labels[train][:, None] == np.arange(10)).astype(f32)
    return train, test, x, q, target, pixels, labels


def prepare():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if (EVIDENCE / 'draw_manifest.json').exists():
        raise FileExistsError('draw manifest exists; never overwrite')
    draws = []
    for index, seed in enumerate(SEEDS):
        train, test, *_ = _load(seed)
        draws.append({'draw': index, 'dataset_seed': seed,
                      'train_indices_sha256': ds.array_hash(train), 'test_indices_sha256': ds.array_hash(test),
                      'train_indices': train.tolist(), 'test_indices': test.tolist()})
        print('prepared draw', index, flush=True)
    (EVIDENCE / 'draw_manifest.json').write_text(json.dumps({'seeds': SEEDS, 'draws': draws}, indent=2) + '\n')


def freeze():
    assert not (EVIDENCE / 'prediction_manifest.json').exists(), 'do not overwrite frozen predictions'
    predictions = EVIDENCE / 'predictions'
    predictions.mkdir(parents=True, exist_ok=True)
    records = []
    for index, seed in enumerate(SEEDS):
        train, test, x, q, target, pixels, labels = _load(seed)
        params, scores, pred = train_predict(x, q, target, CONFIG, ordered_mm, ordered_rows)
        path = predictions / f'draw-{index:02d}.npy'
        np.save(path, pred)
        records.append({'draw': index, 'dataset_seed': seed, 'path': str(path.relative_to(EVIDENCE)),
                        'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'array_sha256': ds.array_hash(pred),
                        'parameter_sha256': ds.array_hash(params), 'scores_sha256': ds.array_hash(scores)})
        print('frozen draw', index, flush=True)
    (EVIDENCE / 'prediction_manifest.json').write_text(json.dumps(
        {'frozen_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'config': CONFIG,
         'seeds': SEEDS, 'draws': records, 'test_labels_opened': False}, indent=2) + '\n')


def score():
    manifest = json.loads((EVIDENCE / 'prediction_manifest.json').read_text())
    predictions = []
    for record in manifest['draws']:
        path = EVIDENCE / record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['prediction_sha256']
        pred = np.load(path, allow_pickle=False)
        assert ds.array_hash(pred) == record['array_sha256']
        predictions.append(pred)
    labels = ds.read_idx(RAW / 'train-labels-idx1-ubyte.gz', 60000, False)
    counts = []
    for record, pred in zip(manifest['draws'], predictions, strict=True):
        order = np.random.Generator(np.random.PCG64(record['dataset_seed'])).permutation(60000)
        correct = int((pred == labels[order[1000:2000]]).sum())
        counts.append({'draw': record['draw'], 'dataset_seed': record['dataset_seed'], 'correct': correct, 'total': 1000})
    acc = np.array([c['correct'] / 1000 for c in counts])
    total = sum(c['correct'] for c in counts)
    result = {'draws': counts, 'correct': total, 'total': 11000, 'mean_accuracy': float(acc.mean()),
              'sample_sd_pp': float(acc.std(ddof=1) * 100), 'target_correct': 7370, 'target_met': total >= 7370}
    (EVIDENCE / 'accuracy.json').write_text(json.dumps(result, indent=2) + '\n')
    (EVIDENCE / 'evaluation_freeze.json').write_text(json.dumps(
        {'evaluated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'prediction_manifest_sha256': hashlib.sha256((EVIDENCE / 'prediction_manifest.json').read_bytes()).hexdigest()},
        indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    {'prepare': prepare, 'freeze': freeze, 'score': score}[sys.argv[1]]()
