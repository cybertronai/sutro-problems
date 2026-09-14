"""Payload + ordered expected outputs for the NR-K8 Adam w32/e100 winner (draw 0)."""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import sweep
from adam_k8_final import ordered_mm, ordered_rows, train_predict, f32

RAW = ROOT / 'data/certification/draw-00/raw'
SEED = 20261201
CONFIG = {'width': 32, 'epochs': 100, 'nr': 8, 'batch': 25, 'lr': 0.2}
sys.path.insert(0, str(ROOT / 'vendor/sutro-problems'))
from mnist.code import data as ds


def main():
    pixels = ds.read_idx(RAW / 'train-images-idx3-ubyte.gz', 60000, True)
    labels = ds.read_idx(RAW / 'train-labels-idx1-ubyte.gz', 60000, False)
    order = np.random.Generator(np.random.PCG64(SEED)).permutation(60000)
    train, test = order[:1000], order[1000:2000]
    x = ds.area_resize(pixels[train].astype(f32) / f32(255), 3).reshape(1000, 9)
    q = ds.area_resize(pixels[test].astype(f32) / f32(255), 3).reshape(1000, 9)
    y = labels[train].astype(np.int32)
    ym = ((y + 1) % 10).astype(np.int32)
    qm = np.random.default_rng(20260911).uniform(0, 1, (1000, 9)).astype(f32)
    xt, qt, qmt = x * f32(4) - f32(.5), q * f32(4) - f32(.5), qm * f32(4) - f32(.5)
    initial = np.concatenate([a.ravel() for a in sweep.parameters(CONFIG['width'])]).astype(f32)
    steps = CONFIG['epochs'] * 40
    c1 = np.array([f32(1) - f32(.9) ** t for t in range(1, steps + 1)], f32)
    c2 = np.array([f32(1) - f32(.999) ** t for t in range(1, steps + 1)], f32)

    expected = {}
    params, scores, pred = train_predict(xt, qt, (y[:, None] == np.arange(10)).astype(f32), CONFIG, ordered_mm, ordered_rows)
    expected.update({'canonical_params': params, 'canonical_scores': scores, 'canonical_predictions': pred})
    print('canonical trained')
    params_m, scores_m, pred_m = train_predict(xt, qt, (ym[:, None] == np.arange(10)).astype(f32), CONFIG, ordered_mm, ordered_rows)
    expected.update({'changed_labels_params': params_m, 'changed_labels_scores': scores_m, 'changed_labels_predictions': pred_m})
    print('changed_labels trained')
    w1 = params[:288].reshape(9, 32); b1 = params[288:320]
    w2 = params[320:640].reshape(32, 10); b2 = params[640:]
    zh = ordered_mm(qmt, w1) + b1
    scores_q = ordered_mm(np.where(zh > f32(0), zh, f32(0)), w2) + b2
    expected.update({'changed_queries_params': params, 'changed_queries_scores': scores_q,
                     'changed_queries_predictions': scores_q.argmax(1).astype(np.int64)})

    # one-minibatch reference for the GPU debug gate
    def one_step(x, q, target, cfg, mm, rows):
        w1, b1, w2, b2 = sweep.parameters(cfg['width'])
        w, e, b, n = cfg['width'], 1, cfg['batch'], cfg['nr']
        step, eps = f32(cfg['lr'] / b), f32(1e-8)
        steps = 1
        c1s = f32(1) - f32(.9) ** steps
        c2s = f32(1) - f32(.999) ** steps
        m = {k: np.zeros_like(v) for k, v in (('w1', w1), ('b1', b1), ('w2', w2), ('b2', b2))}
        v = {k: np.zeros_like(x) for k, x in m.items()}
        xb, tb = x[:b], target[:b]
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
            mhat = m[key] / c1s
            vhat = v[key] / c2s
            y = vhat + f32(1e-6)
            for _ in range(n):
                y = f32(.5) * (y + vhat / y)
            params[key] = params[key] - step * mhat / (y + eps)
        return np.concatenate([params[k].ravel() for k in ('w1', 'b1', 'w2', 'b2')]).astype(f32)

    expected['step1_params'] = one_step(xt, qt, (y[:, None] == np.arange(10)).astype(f32), CONFIG, ordered_mm, ordered_rows)

    generated = HERE / 'generated'
    generated.mkdir(exist_ok=True)
    payload = {'train_images': x, 'test_images': q, 'train_labels': y, 'mutated_queries': qm,
               'initial': initial, 'c1': c1, 'c2': c2}
    np.savez(generated / 'adam-payload.npz', **payload)
    np.savez(generated / 'adam-expected.npz', **expected)
    print('wrote adam payload/expected', {k: v.shape for k, v in payload.items()})


if __name__ == '__main__':
    main()
