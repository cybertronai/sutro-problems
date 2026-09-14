"""Honest two-phase evidence for the NR-K8 Adam w32/e100 entry.

freeze: train 11 official draws, save+hash predictions and write draw manifests.
        The MNIST label file is parsed by the loader, but only train-index slices
        are used; no evaluation-label slice is read or used for training.
score:  verify the freeze manifest and every prediction hash FIRST, then open
        evaluation-label slices and count.
"""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import sweep
from adam_k8_final import ordered_mm, ordered_rows, train_predict, f32

SEEDS = list(range(20261201, 20261212))
CONFIG = {'width': 32, 'epochs': 100, 'nr': 8, 'batch': 25, 'lr': 0.2}
EVIDENCE = HERE / 'evidence-adam-v2'
sys.path.insert(0, str(ROOT / 'vendor/sutro-problems'))
from mnist.code import data as ds

RAW = ROOT / 'data/certification/draw-00/raw'


def build(seed):
    order = np.random.Generator(np.random.PCG64(seed)).permutation(60000)
    train, test = order[:1000].astype(np.int64), order[1000:2000].astype(np.int64)
    pixels = ds.read_idx(RAW / 'train-images-idx3-ubyte.gz', 60000, True)
    labels = ds.read_idx(RAW / 'train-labels-idx1-ubyte.gz', 60000, False)
    x = ds.area_resize(pixels[train].astype(f32) / f32(255), 3).reshape(1000, 9) * f32(4) - f32(.5)
    q = ds.area_resize(pixels[test].astype(f32) / f32(255), 3).reshape(1000, 9) * f32(4) - f32(.5)
    target = (labels[train][:, None] == np.arange(10)).astype(f32)
    return train, test, x, q, target, pixels[train], pixels[test], labels[train]


def freeze():
    EVIDENCE.mkdir(exist_ok=True)
    assert not (EVIDENCE / 'predictions_frozen.json').exists(), 'do not overwrite'
    predictions = EVIDENCE / 'predictions'
    predictions.mkdir(exist_ok=True)
    records, manifest = [], []
    for index, seed in enumerate(SEEDS):
        train, test, x, q, target, train_px, test_px, train_lab = build(seed)
        params, scores, pred = train_predict(x, q, target, CONFIG, ordered_mm, ordered_rows)
        path = predictions / f'draw-{index:02d}.npy'
        np.save(path, pred)
        records.append({'draw': index, 'seed': seed, 'file': str(path.relative_to(EVIDENCE)),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'array_sha256': ds.array_hash(pred),
                        'parameter_sha256': ds.array_hash(params), 'scores_sha256': ds.array_hash(scores)})
        manifest.append({'draw': index, 'dataset_seed': seed,
                         'train_indices_sha256': ds.array_hash(train), 'test_indices_sha256': ds.array_hash(test),
                         'input_sha256': {'train_images': ds.array_hash(train_px), 'train_labels': ds.array_hash(train_lab),
                                          'test_images': ds.array_hash(test_px)},
                         'train_indices': train.tolist(), 'test_indices': test.tolist()})
        print('frozen', index, seed, flush=True)
    frozen = {'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'config': CONFIG, 'seeds': SEEDS, 'records': records, 'test_labels_opened': False}
    (EVIDENCE / 'predictions_frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    (EVIDENCE / 'draw_manifest.json').write_text(json.dumps({'seeds': SEEDS, 'draws': manifest}, indent=2) + '\n')
    print('freeze complete; no test label array was read')


def score():
    frozen = json.loads((EVIDENCE / 'predictions_frozen.json').read_text())
    verified = []
    for record in frozen['records']:
        path = EVIDENCE / record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        pred = np.load(path, allow_pickle=False)
        assert ds.array_hash(pred) == record['array_sha256']
        verified.append(pred)
    labels = ds.read_idx(RAW / 'train-labels-idx1-ubyte.gz', 60000, False)
    counts = []
    for record, pred in zip(frozen['records'], verified, strict=True):
        order = np.random.Generator(np.random.PCG64(record['seed'])).permutation(60000)
        test = order[1000:2000]
        counts.append({'draw': record['draw'], 'seed': record['seed'], 'correct': int((pred == labels[test]).sum()), 'total': 1000})
    acc = np.array([c['correct'] / 1000 for c in counts])
    total = sum(c['correct'] for c in counts)
    result = {'draws': counts, 'correct': total, 'total': 11000, 'mean_accuracy': float(acc.mean()),
              'sample_sd_pp': float(acc.std(ddof=1) * 100), 'target_correct': 7370,
              'target_met': total >= 7370, 'grid_energy_mj': 0.185228620462, 'grid_time_ms': 1798.760521,
              'prediction_manifest_sha256': hashlib.sha256((EVIDENCE / 'predictions_frozen.json').read_bytes()).hexdigest()}
    (EVIDENCE / 'accuracy.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    {'freeze': freeze, 'score': score}[sys.argv[1]]()
