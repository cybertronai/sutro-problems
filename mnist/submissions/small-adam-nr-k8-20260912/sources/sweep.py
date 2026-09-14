"""CPU-only Pareto sweep for MNIST-small (1000/1000, 3x3), official learner family.

Selection uses pilot seeds 20261001-20261005 only; the official evaluation seeds
20261201-20261211 are generated but never inspected here.
"""
import itertools
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCORER = ROOT / 'vendor/sutro-problems/mnist/submissions/grid-mlp-scoring-20260912/score.py'
RAW = ROOT / 'data/certification/draw-00/raw'
PILOT_SEEDS = [20261001, 20261002, 20261003, 20261004, 20261005]
f32 = np.float32
_CACHE = {}


def area_resize(images, size):
    def weights(input_size, output_size):
        left = np.arange(output_size, dtype=np.int64)[:, None] * input_size
        right = left + input_size
        pixel_left = np.arange(input_size, dtype=np.int64)[None, :] * output_size
        pixel_right = pixel_left + output_size
        overlap = np.maximum(0, np.minimum(right, pixel_right) - np.maximum(left, pixel_left))
        return (overlap / input_size).astype(np.float32)
    return np.matmul(np.matmul(weights(images.shape[1], size), images.astype(f32, copy=False)),
                     weights(images.shape[2], size).T).astype(f32)


def load_raw():
    if 'raw' not in _CACHE:
        import gzip
        with gzip.open(RAW / 'train-images-idx3-ubyte.gz', 'rb') as f:
            f.read(16)
            images = np.frombuffer(f.read(), np.uint8).reshape(-1, 28, 28)
        with gzip.open(RAW / 'train-labels-idx1-ubyte.gz', 'rb') as f:
            f.read(8)
            labels = np.frombuffer(f.read(), np.uint8)
        _CACHE['raw'] = (images, labels)
    return _CACHE['raw']


def draw(seed):
    key = ('draw', seed)
    if key not in _CACHE:
        images, labels = load_raw()
        order = np.random.Generator(np.random.PCG64(seed)).permutation(60000)
        train, test = order[:1000], order[1000:2000]
        x = transform(area_resize(images[train] / f32(255), 3))
        q = transform(area_resize(images[test] / f32(255), 3))
        target = (labels[train][:, None] == np.arange(10)).astype(f32)
        _CACHE[key] = (x, q, target, labels[test].copy())
    return _CACHE[key]


def transform(images):
    return images.reshape(len(images), 9) * f32(4) - f32(.5)


def mm(a, b):
    return np.einsum('ik,kj->ij', a, np.ascontiguousarray(b), optimize=False, dtype=np.float32)


def rows(a):
    return np.einsum('ij->j', a, optimize=False, dtype=np.float32)


def parameters(width, seed=101):
    rng = np.random.Generator(np.random.PCG64(seed))
    return [rng.uniform(-1/3, 1/3, (9, width)).astype(f32), np.zeros(width, f32),
            rng.uniform(-1/np.sqrt(width), 1/np.sqrt(width), (width, 10)).astype(f32), np.zeros(10, f32)]


def train_eval(config, seed):
    width, epochs, batch, lr = config
    x, q, target, truth = draw(seed)
    w1, b1, w2, b2 = parameters(width)
    step = f32(lr / batch)
    for _ in range(epochs):
        for start in range(0, 1000, batch):
            xb, tb = x[start:start+batch], target[start:start+batch]
            z = mm(xb, w1) + b1
            h = np.where(z > f32(0), z, f32(0))
            d2 = mm(h, w2) + b2 - tb
            d1 = np.where(z > f32(0), mm(d2, w2.T), f32(0))
            w1 = w1 - step * mm(xb.T, d1)
            b1 = b1 - step * rows(d1)
            w2 = w2 - step * mm(h.T, d2)
            b2 = b2 - step * rows(d2)
    scores = mm(np.where(mm(q, w1) + b1 > f32(0), mm(q, w1) + b1, f32(0)), w2) + b2
    return float((scores.argmax(1) == truth).mean())


def run_config(config):
    accs = [train_eval(config, seed) for seed in PILOT_SEEDS]
    width, epochs, batch, lr = config
    out = Path('/tmp/sweep-grid') / f'w{width}-e{epochs}-b{batch}-lr{lr}'
    subprocess.run([sys.executable, str(SCORER), '--features', '9', '--width', str(width), '--epochs', str(epochs),
                    '--batch', str(batch), '--n-train', '1000', '--n-test', '1000', '--learning-rate', str(lr),
                    '--seed', '101', '--output', str(out)], cwd=str(SCORER.parent), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    grid = json.loads((out / 'grid-score.json').read_text())
    return {'config': {'width': width, 'epochs': epochs, 'batch': batch, 'lr': lr},
            'pilot_mean': float(np.mean(accs)), 'pilot_sd': float(np.std(accs, ddof=1)),
            'pilot_accs': [round(a, 4) for a in accs],
            'energy_mj': grid['energy_mj'], 'time_ms': grid['time_ms']}


if __name__ == '__main__':
    widths = [32, 40, 48, 64, 80, 96, 128]
    epochs_list = [100, 150, 200, 300, 400]
    batches = [25, 50]
    lrs = [0.2, 0.4]
    configs = list(itertools.product(widths, epochs_list, batches, lrs))
    print(len(configs), 'configs x', len(PILOT_SEEDS), 'pilots', flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run_config, configs))
    results.sort(key=lambda r: -r['pilot_mean'])
    (HERE / 'sweep.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f"{'width':>5}{'ep':>5}{'b':>4}{'lr':>5}  {'pilot_mean':>10}{'sd':>7}{'mJ':>8}{'ms':>8}")
    for r in results[:25]:
        c = r['config']
        print(f"{c['width']:>5}{c['epochs']:>5}{c['batch']:>4}{c['lr']:>5}  {r['pilot_mean']*100:>10.2f}{r['pilot_sd']*100:>7.2f}{r['energy_mj']:>8.3f}{r['time_ms']:>8.0f}")
