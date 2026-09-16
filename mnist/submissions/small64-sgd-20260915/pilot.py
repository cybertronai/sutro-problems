"""Pilot hyperparameter sweep used to predeclare the frozen configuration.

Pilot seeds 20261120-20261130 are disjoint from the official draw seeds
20261201-20261211. Config selection used ONLY pilot draws; the official
draws were frozen afterwards with the two-phase protocol in run.py.

Learner semantics match the reviewed SGD scorer path: 9-H-10 ReLU MLP,
squared error, batch 25, constant lr, step = lr/batch, gradients from
pre-update weights, ascending-index FP32 reductions, fresh seed-101
initialization per draw, 4*x-0.5 input transform. Downsampling uses
run.resize_recorded (ordered accumulation, float64 product/sum intermediates,
float32 cast after each step), matching run.py and verify.py. The sweep uses
einsum-based matmuls with contiguous row-major operands, which preserve
the ascending-K accumulation order (equivalence re-checked on the frozen
configuration by verify.py against the ordered loops in reference.py).
"""
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist.code import data as ds  # noqa: E402
import run  # noqa: E402  (deterministic resize_recorded downsampling)

RAW = Path(__file__).resolve().parents[3] / 'matmul' / 'mnist_cache'
f32 = np.float32
PILOT_SEEDS = list(range(20261120, 20261131))

CONFIGS = []
for width in (32, 48, 64):
    for epochs in (300, 500):
        for lr in (0.1, 0.2, 0.3, 0.4):
            CONFIGS.append({'width': width, 'epochs': epochs, 'lr': lr})


def fast_mm(a, b):
    return np.einsum('ik,kj->ij', a, np.ascontiguousarray(b),
                     optimize=False, dtype=np.float32)


def fast_rows(a):
    return np.einsum('ij->j', a, optimize=False, dtype=np.float32)


def parameters(width, seed=101):
    rng = np.random.Generator(np.random.PCG64(seed))
    return [rng.uniform(-1 / 3, 1 / 3, (9, width)).astype(f32),
            np.zeros(width, dtype=f32),
            rng.uniform(-1 / math.sqrt(width), 1 / math.sqrt(width),
                        (width, 10)).astype(f32),
            np.zeros(10, dtype=f32)]


_CACHE = {}


def load(seed):
    if seed not in _CACHE:
        order = np.random.Generator(np.random.PCG64(seed)).permutation(60000)
        train, test = order[:1000].astype(np.int64), order[1000:2000].astype(np.int64)
        pixels = ds.read_idx(RAW / 'train-images-idx3-ubyte.gz', 60000, True)
        labels = ds.read_idx(RAW / 'train-labels-idx1-ubyte.gz', 60000, False)
        x = run.resize_recorded(pixels[train].astype(f32) / f32(255)).reshape(1000, 9) * f32(4) - f32(.5)
        q = run.resize_recorded(pixels[test].astype(f32) / f32(255)).reshape(1000, 9) * f32(4) - f32(.5)
        target = (labels[train][:, None] == np.arange(10)).astype(f32)
        _CACHE[seed] = (x, q, target, labels[test])
    return _CACHE[seed]


def train_predict(x, q, target, width, epochs, lr):
    w1, b1, w2, b2 = parameters(width)
    step, zero = f32(lr / 25), f32(0)
    for _ in range(epochs):
        for start in range(0, 1000, 25):
            xb, tb = x[start:start + 25], target[start:start + 25]
            z = fast_mm(xb, w1) + b1
            h = np.where(z > zero, z, zero)
            d2 = fast_mm(h, w2) + b2 - tb
            d1 = np.where(z > zero, fast_mm(d2, w2.T), zero)
            g1, gb1 = fast_mm(xb.T, d1), fast_rows(d1)
            g2, gb2 = fast_mm(h.T, d2), fast_rows(d2)
            w1 = w1 - step * g1
            b1 = b1 - step * gb1
            w2 = w2 - step * g2
            b2 = b2 - step * gb2
    zh = fast_mm(q, w1) + b1
    scores = fast_mm(np.where(zh > zero, zh, zero), w2) + b2
    return scores.argmax(1).astype(np.int64)


def run_one(job):
    cfg, seed = job
    x, q, target, test_labels = load(seed)
    t0 = time.time()
    pred = train_predict(x, q, target, cfg['width'], cfg['epochs'], cfg['lr'])
    correct = int((pred == test_labels).sum())
    return cfg['width'], cfg['epochs'], cfg['lr'], seed, correct, time.time() - t0


def main():
    jobs = [(cfg, seed) for cfg in CONFIGS for seed in PILOT_SEEDS]
    results = []
    with Pool(24) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs)):
            results.append(r)
            if (i + 1) % 55 == 0:
                print(f'{i + 1}/{len(jobs)} done', flush=True)
    summary = {}
    for width, epochs, lr, seed, correct, secs in results:
        key = f'w{width}-e{epochs}-lr{lr}'
        summary.setdefault(key, {'width': width, 'epochs': epochs, 'lr': lr, 'correct': []})
        summary[key]['correct'].append(correct)
    out = []
    for key, s in summary.items():
        acc = np.array(s['correct']) / 1000
        out.append({'config': key, 'width': s['width'], 'epochs': s['epochs'], 'lr': s['lr'],
                    'total': int(sum(s['correct'])), 'mean_pp': float(acc.mean() * 100),
                    'sd_pp': float(acc.std(ddof=1) * 100),
                    'per_draw_correct': s['correct']})
    out.sort(key=lambda r: -r['total'])
    evidence = Path(__file__).resolve().parent / 'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'pilot_results.json').write_text(json.dumps(
        {'pilot_seeds': PILOT_SEEDS, 'official_seeds_excluded': list(range(20261201, 20261212)),
         'batch': 25, 'learner_seed': 101, 'swept_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'results': out}, indent=2) + '\n')
    for r in out:
        print(f"{r['config']:>20}: {r['total']}/11000 = {r['mean_pp']:.2f}% +/- {r['sd_pp']:.2f}")


if __name__ == '__main__':
    main()
