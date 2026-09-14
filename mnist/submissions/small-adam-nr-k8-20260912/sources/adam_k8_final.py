"""Predeclared selection + frozen official evaluation for the NR-K8 Adam rule.

Selection: among the fixed candidate list, require pilot mean >= baseline
(0.6708) AND grid energy < 0.2 mJ (user target); choose the highest pilot mean.
Then train the winner with the identical ordered rule on the 11 official draws,
freeze predictions before opening labels, and score.
"""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import sweep

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOTS = list(range(20261006, 20261016))
OFFICIAL = list(range(20261201, 20261212))
CANDIDATES = [
    {'width': 32, 'epochs': 75, 'nr': 8, 'batch': 25, 'lr': 0.2, 'energy_mj': 0.136044949888},
    {'width': 32, 'epochs': 100, 'nr': 8, 'batch': 25, 'lr': 0.2, 'energy_mj': 0.185228620462},
    {'width': 48, 'epochs': 75, 'nr': 8, 'batch': 25, 'lr': 0.2, 'energy_mj': 0.209327139884},
]
BASELINE = 0.6708181818181819
f32 = np.float32


def ordered_mm(a, b):
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    for k in range(a.shape[1]):
        out = out + a[:, k, None] * b[None, k, :]
    return out


def ordered_rows(a):
    out = np.zeros(a.shape[1], dtype=np.float32)
    for row in a:
        out = out + row
    return out


def sqrt_nr(x, iterations):
    y = x + f32(1e-6)
    for _ in range(iterations):
        y = f32(.5) * (y + x / y)
    return y


def train_predict(x, q, target, cfg, mm, rows):
    w, e, b, n = cfg['width'], cfg['epochs'], cfg['batch'], cfg['nr']
    w1, b1, w2, b2 = sweep.parameters(w)
    step, eps = f32(cfg['lr'] / b), f32(1e-8)
    m = {k: np.zeros_like(v) for k, v in (('w1', w1), ('b1', b1), ('w2', w2), ('b2', b2))}
    v = {k: np.zeros_like(x) for k, x in m.items()}
    t = 0
    for _ in range(e):
        for start in range(0, len(x), b):
            t += 1
            xb, tb = x[start:start+b], target[start:start+b]
            z = mm(xb, w1) + b1
            h = np.where(z > f32(0), z, f32(0))
            d2 = mm(h, w2) + b2 - tb
            d1 = np.where(z > f32(0), mm(d2, w2.T), f32(0))
            grads = {'w1': mm(xb.T, d1), 'b1': rows(d1), 'w2': mm(h.T, d2), 'b2': rows(d2)}
            params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
            for key in grads:
                g = grads[key]
                m[key] = f32(.9) * m[key] + f32(.1) * g
                v[key] = f32(.999) * v[key] + f32(.001) * g * g
                mhat = m[key] / (1 - f32(.9) ** t)
                vhat = v[key] / (1 - f32(.999) ** t)
                params[key] = params[key] - step * mhat / (sqrt_nr(vhat, n) + eps)
            w1, b1, w2, b2 = params['w1'], params['b1'], params['w2'], params['b2']
    zh = mm(q, w1) + b1
    scores = mm(np.where(zh > f32(0), zh, f32(0)), w2) + b2
    params = np.concatenate([a.ravel() for a in (w1, b1, w2, b2)])
    return params, scores, scores.argmax(1).astype(np.int64)


def pilot(cfg):
    accs = []
    for seed in PILOTS:
        x, q, target, truth = sweep.draw(seed)
        _, _, pred = train_predict(x, q, target, cfg, sweep.mm, sweep.rows)
        accs.append(float((pred == truth).mean()))
    return accs


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=8) as pool:
        pilot_results = list(pool.map(pilot, CANDIDATES))
    for cfg, accs in zip(CANDIDATES, pilot_results):
        cfg['pilot_mean'] = float(np.mean(accs))
        cfg['pilot_sd'] = float(np.std(accs, ddof=1))
        print(f"candidate w{cfg['width']} e{cfg['epochs']} K{cfg['nr']}: "
              f"{cfg['pilot_mean']*100:.2f}% +/- {cfg['pilot_sd']*100:.2f} mJ={cfg['energy_mj']:.4f}")
    eligible = [c for c in CANDIDATES if c['pilot_mean'] >= BASELINE and c['energy_mj'] < 0.2]
    assert eligible, 'no candidate meets predeclared rule'
    winner = max(eligible, key=lambda c: c['pilot_mean'])
    print('winner:', json.dumps(winner))

    raw_labels = ROOT / 'data/certification/draw-00/raw/train-labels-idx1-ubyte.gz'
    import sys
    sys.path.insert(0, str(ROOT / 'vendor/sutro-problems'))
    from mnist.code import data as ds
    labels = ds.read_idx(raw_labels, 60000, False)
    pixels = ds.read_idx(ROOT / 'data/certification/draw-00/raw/train-images-idx3-ubyte.gz', 60000, True)
    evidence = HERE / 'evidence-adam'
    evidence.mkdir(exist_ok=True)
    assert not (evidence / 'accuracy.json').exists(), 'already evaluated'
    plan = {'created_utc': __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime()),
            'rule': 'NR-K8 Adam, ordered FP32; selection on pilots 20261006-15, official draws opened once',
            'candidates': CANDIDATES, 'winner': winner, 'baseline_mean': BASELINE, 'test_labels_opened': False}
    (evidence / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    predictions = evidence / 'predictions'
    predictions.mkdir(exist_ok=True)
    records, draws = [], []
    for index, seed in enumerate(OFFICIAL):
        order = np.random.Generator(np.random.PCG64(seed)).permutation(60000)
        train, test = order[:1000], order[1000:2000]
        x = ds.area_resize(pixels[train].astype(f32) / f32(255), 3).reshape(1000, 9)
        q = ds.area_resize(pixels[test].astype(f32) / f32(255), 3).reshape(1000, 9)
        x = x * f32(4) - f32(.5)
        q = q * f32(4) - f32(.5)
        target = (labels[train][:, None] == np.arange(10)).astype(f32)
        params, scores, pred = train_predict(x, q, target, winner, ordered_mm, ordered_rows)
        path = predictions / f'draw-{index:02d}.npy'
        np.save(path, pred)
        correct = int((pred == labels[test]).sum())
        records.append({'draw': index, 'seed': seed, 'file': str(path.relative_to(evidence)),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'array_sha256': ds.array_hash(pred), 'correct': correct,
                        'parameter_sha256': ds.array_hash(params), 'scores_sha256': ds.array_hash(scores)})
        draws.append(test)
        print(f'draw {index:02d} {seed}: {correct}/1000', flush=True)
    frozen = {'plan_sha256': hashlib.sha256((evidence / 'plan.json').read_bytes()).hexdigest(),
              'winner': winner, 'records': records, 'test_labels_opened': False}
    (evidence / 'predictions_frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    accs = np.array([r['correct'] / 1000 for r in records])
    total = sum(r['correct'] for r in records)
    result = {'draws': [{'draw': r['draw'], 'seed': r['seed'], 'correct': r['correct'], 'total': 1000} for r in records],
              'correct': total, 'total': 11000, 'mean_accuracy': float(accs.mean()),
              'sample_sd_pp': float(accs.std(ddof=1) * 100),
              'baseline_correct': 7379, 'baseline_mean_accuracy': BASELINE,
              'target_accuracy': 0.67, 'target_met': (total / 11000) >= 0.67,
              'grid_energy_mj': winner['energy_mj'], 'grid_time_ms': 1798.760521,
              'prediction_manifest_sha256': hashlib.sha256((evidence / 'predictions_frozen.json').read_bytes()).hexdigest()}
    (evidence / 'accuracy.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
