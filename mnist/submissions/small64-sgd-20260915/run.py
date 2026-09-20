"""Prepare, freeze and score the 9-64-10 SGD MNIST-small submission.

Two-phase evaluation: predictions are written and hashed before any
evaluation-label slice is opened; scoring verifies every hash first.
Configuration was predeclared on disjoint pilot seeds (see pilot.py and
evidence/pilot_results.json) before the official draws were evaluated.
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
import reference
from reference import train_predict, ordered_mm, ordered_rows, f32

SEEDS = list(range(20261201, 20261212))
CONFIG = dict(reference.CONFIG)
TARGET_CORRECT = 7370


def _repo():
    import os
    env = os.environ.get('SUTRO_REPO')
    if env:
        return Path(env)
    for base in (HERE, *HERE.parents):
        if (base / 'mnist/code/data.py').exists():
            return base
        if (base / 'vendor/sutro-problems/mnist/code/data.py').exists():
            return base / 'vendor/sutro-problems'
    raise SystemExit('sutro-problems checkout not found; set SUTRO_REPO')


sys.path.insert(0, str(_repo()))
from mnist.code import data as ds  # repository data module
def _raw():
    import os
    env = os.environ.get('SUTRO_RAW')
    if env:
        return Path(env)
    candidates = []
    for base in (HERE, *HERE.parents):
        candidates.append(base / 'data/certification/draw-00/raw')
    candidates.append(_repo() / 'matmul/mnist_cache')
    for candidate in candidates:
        if (candidate / 'train-images-idx3-ubyte.gz').exists():
            return candidate
    raise SystemExit('raw MNIST dir not found; set SUTRO_RAW')

RAW = _raw()


def resize_recorded(images, size=3):
    """Reproduce the source BLAS's ordered FP32 multiply-adds for the 3x3 area resize.

    Different BLAS builds round the repository `area_resize` matrix products
    differently, so a float32 matmul does not reproduce the archived input
    hashes on every machine (macOS ARM vs Linux x86). This keeps the
    repository's box-area weights and increasing reduction order, taking a
    float64 product and sum before each FP32 accumulation; it reproduces all
    33 archived input hashes of the official draws bit-exactly and is free of
    BLAS dispatch. Same construction as the merged medium PCA-QDA and
    MNIST-small QDA entries' `resize_recorded`, at size 3.
    """
    weights = ds.area_weights(28, size).astype(np.float64)
    images = np.asarray(images, dtype=f32)
    horizontal = np.zeros((len(images), size, 28), dtype=f32)
    for k in range(28):
        horizontal = (horizontal.astype(np.float64)
                      + weights[None, :, k, None] * images[:, None, k, :].astype(np.float64)).astype(f32)
    result = np.zeros((len(images), size, size), dtype=f32)
    for k in range(28):
        result = (result.astype(np.float64)
                  + horizontal[:, :, k, None].astype(np.float64) * weights[None, None, :, k]).astype(f32)
    return result


def _load(seed):
    order = np.random.Generator(np.random.PCG64(seed)).permutation(60000)
    train, test = order[:1000].astype(np.int64), order[1000:2000].astype(np.int64)
    pixels = ds.read_idx(RAW / 'train-images-idx3-ubyte.gz', 60000, True)
    labels = ds.read_idx(RAW / 'train-labels-idx1-ubyte.gz', 60000, False)
    x = resize_recorded(pixels[train].astype(f32) / f32(255)).reshape(1000, 9) * f32(4) - f32(.5)
    q = resize_recorded(pixels[test].astype(f32) / f32(255)).reshape(1000, 9) * f32(4) - f32(.5)
    target = (labels[train][:, None] == np.arange(10)).astype(f32)
    return train, test, x, q, target, pixels, labels


def _arrays(pixels, labels, train, test):
    return {'train_images': resize_recorded(pixels[train].astype(f32) / f32(255)).reshape(1000, 1, 3, 3),
            'train_labels': labels[train].astype(np.int64),
            'test_images': resize_recorded(pixels[test].astype(f32) / f32(255)).reshape(1000, 1, 3, 3)}


def prepare():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    private = HERE / 'private'
    private.mkdir(exist_ok=True)
    if (EVIDENCE / 'draw_manifest.json').exists():
        raise FileExistsError('draw manifest exists; never overwrite')
    draws = []
    for index, seed in enumerate(SEEDS):
        train, test, *_ , pixels, labels = _load(seed)
        arrays = _arrays(pixels, labels, train, test)
        np.savez(private / f'draw-{index:02d}.npz', **arrays)
        draws.append({'draw': index, 'dataset_seed': seed,
                      'train_indices_sha256': ds.array_hash(train), 'test_indices_sha256': ds.array_hash(test),
                      'train_indices': train.tolist(), 'test_indices': test.tolist(),
                      'input_sha256': {name: ds.array_hash(value) for name, value in arrays.items()}})
        print('prepared draw', index, flush=True)
    (EVIDENCE / 'draw_manifest.json').write_text(json.dumps(
        {'seeds': SEEDS, 'draws': draws,
         'raw_gz_sha256': {name: ds.file_hash(RAW / name) for name in
                           ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')}}, indent=2) + '\n')


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
              'sample_sd_pp': float(acc.std(ddof=1) * 100), 'target_correct': TARGET_CORRECT,
              'target_met': total >= TARGET_CORRECT}
    (EVIDENCE / 'accuracy.json').write_text(json.dumps(result, indent=2) + '\n')
    (EVIDENCE / 'evaluation_freeze.json').write_text(json.dumps(
        {'evaluated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'prediction_manifest_sha256': hashlib.sha256((EVIDENCE / 'prediction_manifest.json').read_bytes()).hexdigest()},
        indent=2) + '\n')
    print(json.dumps(result, indent=2))


def gpu_payload():
    """Write generated/gpu-payload.json (base64 arrays) for the Modal client."""
    import base64
    generated = HERE / 'generated'
    generated.mkdir(exist_ok=True)
    manifest = json.loads((EVIDENCE / 'draw_manifest.json').read_text())
    draw = manifest['draws'][0]
    arrays = np.load(HERE / 'private' / 'draw-00.npz', allow_pickle=False)
    payload = {}
    for name in ('train_images', 'train_labels', 'test_images'):
        array = np.ascontiguousarray(arrays[name])
        assert ds.array_hash(array) == draw['input_sha256'][name], name
        payload[name] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                         'bytes_b64': base64.b64encode(array.tobytes()).decode('ascii')}
    (generated / 'gpu-payload.json').write_text(json.dumps(payload) + '\n')
    print('wrote generated/gpu-payload.json')


if __name__ == '__main__':
    {'prepare': prepare, 'freeze': freeze, 'score': score,
     'gpu-payload': gpu_payload}[sys.argv[1]]()
