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
import reference
from reference import train_predict, ordered_mm, ordered_rows, f32

SEEDS = list(range(20261201, 20261212))
CONFIG = {'width': 32, 'epochs': 100, 'nr': 8, 'batch': 25, 'lr': 0.2}

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


def gpu_payload():
    """Regenerate generated/adam11-{payload,expected}.npz for the GPU runner."""
    generated = HERE / 'generated'
    generated.mkdir(exist_ok=True)
    order = np.random.Generator(np.random.PCG64(SEEDS[0])).permutation(60000)
    train0, test0 = order[:1000].astype(np.int64), order[1000:2000].astype(np.int64)
    pixels = ds.read_idx(RAW / 'train-images-idx3-ubyte.gz', 60000, True)
    labels = ds.read_idx(RAW / 'train-labels-idx1-ubyte.gz', 60000, False)
    raw = lambda idx: ds.area_resize(pixels[idx].astype(f32) / f32(255), 3).reshape(1000, 9)
    x0, q0 = raw(train0), raw(test0)
    xt0, qt0 = x0 * f32(4) - f32(.5), q0 * f32(4) - f32(.5)
    target0 = (labels[train0][:, None] == np.arange(10)).astype(f32)
    params, scores, pred = train_predict(xt0, qt0, target0, CONFIG, ordered_mm, ordered_rows)
    ym = (labels[train0] + 1) % 10
    params_m, scores_m, pred_m = train_predict(xt0, qt0, (ym[:, None] == np.arange(10)).astype(f32), CONFIG, ordered_mm, ordered_rows)
    qm = np.random.default_rng(20260911).uniform(0, 1, (1000, 9)).astype(f32)
    w1, b1 = params[:288].reshape(9, 32), params[288:320]
    w2, b2 = params[320:640].reshape(32, 10), params[640:]
    zh = ordered_mm(qm * f32(4) - f32(.5), w1) + b1
    scores_q = ordered_mm(np.where(zh > f32(0), zh, f32(0)), w2) + b2
    # one-minibatch reference
    w1b, b1b, w2b, b2b = reference.parameters(CONFIG['width'])
    step, eps = f32(CONFIG['lr'] / CONFIG['batch']), f32(1e-8)
    xb, tb = xt0[:CONFIG['batch']], target0[:CONFIG['batch']]
    z = ordered_mm(xb, w1b) + b1b
    h = np.where(z > f32(0), z, f32(0))
    d2 = ordered_mm(h, w2b) + b2b - tb
    d1 = np.where(z > f32(0), ordered_mm(d2, w2b.T), f32(0))
    g1, gb1 = ordered_mm(xb.T, d1), ordered_rows(d1)
    g2, gb2 = ordered_mm(h.T, d2), ordered_rows(d2)
    c1s, c2s = f32(1) - f32(.9), f32(1) - f32(.999)
    def adam(g, p0v):
        m = f32(.1) * g
        v = f32(.001) * g * g
        mh, vh = m / c1s, v / c2s
        y = vh + f32(1e-6)
        for _ in range(CONFIG['nr']):
            y = f32(.5) * (y + vh / y)
        return p0v - step * mh / (y + eps)
    step1 = np.concatenate([adam(g1, w1b).ravel(), adam(gb1, b1b).ravel(),
                            adam(g2, w2b).ravel(), adam(gb2, b2b).ravel()]).astype(f32)
    steps = CONFIG['epochs'] * 40
    c1 = np.array([f32(1) - f32(.9) ** t for t in range(1, steps + 1)], f32)
    c2 = np.array([f32(1) - f32(.999) ** t for t in range(1, steps + 1)], f32)
    payload = {'train_images': x0, 'test_images': q0, 'train_labels': labels[train0].astype(np.int32),
               'mutated_queries': qm,
               'initial': np.concatenate([a.ravel() for a in reference.parameters(CONFIG['width'])]).astype(f32),
               'c1': c1, 'c2': c2}
    expected = {'canonical_params': params, 'canonical_scores': scores, 'canonical_predictions': pred,
                'changed_labels_params': params_m, 'changed_labels_scores': scores_m, 'changed_labels_predictions': pred_m,
                'changed_queries_params': params, 'changed_queries_scores': scores_q,
                'changed_queries_predictions': scores_q.argmax(1).astype(np.int64), 'step1_params': step1}
    for index, seed in enumerate(SEEDS):
        o = np.random.Generator(np.random.PCG64(seed)).permutation(60000)
        tr, te = o[:1000].astype(np.int64), o[1000:2000].astype(np.int64)
        payload[f'x{index}'] = raw(tr)
        payload[f'q{index}'] = raw(te)
        payload[f'y{index}'] = labels[tr].astype(np.int32)
        expected[f'pred{index}'] = np.load(EVIDENCE / 'predictions' / f'draw-{index:02d}.npy')
    np.savez(generated / 'adam11-payload.npz', **payload)
    np.savez(generated / 'adam11-expected.npz', **expected)
    print('wrote generated/adam11-payload.npz and generated/adam11-expected.npz')


if __name__ == '__main__':
    {'prepare': prepare, 'freeze': freeze, 'score': score, 'gpu-payload': gpu_payload}[sys.argv[1]]()
