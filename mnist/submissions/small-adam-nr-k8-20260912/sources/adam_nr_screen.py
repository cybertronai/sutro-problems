"""Screen Adam with the exact Newton-Raphson-sqrt rule the scorer prices (K=12).

Matches scorer-adam's op sequence: q = x/y; q = y + q; y = 0.5*q; y0 = x + 1e-6.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import sweep

HERE = Path(__file__).resolve().parent
SEEDS = list(range(20261006, 20261011))
CHECKPOINTS = [50, 75, 100, 125, 150]
WIDTHS = [32, 48]
f32 = np.float32


def sqrt_nr(x):
    y = x + f32(1e-6)
    for _ in range(12):
        y = f32(.5) * (y + x / y)
    return y


def run(case):
    width, seed, batch = case
    x, q, target, truth = sweep.draw(seed)
    w1, b1, w2, b2 = sweep.parameters(width)
    step = f32(0.2 / batch)
    eps = f32(1e-8)
    m = {k: np.zeros_like(v) for k, v in (('w1', w1), ('b1', b1), ('w2', w2), ('b2', b2))}
    v = {k: np.zeros_like(x) for k, x in m.items()}
    accs = {}
    t = 0
    for epoch in range(1, max(CHECKPOINTS) + 1):
        for start in range(0, 1000, batch):
            t += 1
            xb, tb = x[start:start+batch], target[start:start+batch]
            z = sweep.mm(xb, w1) + b1
            h = np.where(z > f32(0), z, f32(0))
            d2 = sweep.mm(h, w2) + b2 - tb
            d1 = np.where(z > f32(0), sweep.mm(d2, w2.T), f32(0))
            grads = {'w1': sweep.mm(xb.T, d1), 'b1': sweep.rows(d1),
                     'w2': sweep.mm(h.T, d2), 'b2': sweep.rows(d2)}
            params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
            for key in grads:
                g = grads[key]
                m[key] = f32(.9) * m[key] + f32(.1) * g
                v[key] = f32(.999) * v[key] + f32(.001) * g * g
                mhat = m[key] / (1 - f32(.9) ** t)
                vhat = v[key] / (1 - f32(.999) ** t)
                params[key] = params[key] - step * mhat / (sqrt_nr(vhat) + eps)
            w1, b1, w2, b2 = params['w1'], params['b1'], params['w2'], params['b2']
        if epoch in CHECKPOINTS:
            zh = sweep.mm(q, w1) + b1
            scores = sweep.mm(np.where(zh > f32(0), zh, f32(0)), w2) + b2
            accs[epoch] = float((scores.argmax(1) == truth).mean())
    return width, batch, seed, accs


if __name__ == '__main__':
    cases = [(w, s, b) for w in WIDTHS for b in (25, 50) for s in SEEDS]
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, cases))
    table = {}
    for width, batch, seed, accs in results:
        for epoch, acc in accs.items():
            table.setdefault((width, batch, epoch), []).append(acc)
    rows = [{'width': k[0], 'batch': k[1], 'epochs': k[2], 'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1))}
            for k, v in sorted(table.items())]
    (HERE / 'adam-nr-screen.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f"{'w':>4}{'b':>4}  " + ''.join(f"{e:>8}" for e in CHECKPOINTS))
    for width in WIDTHS:
        for batch in (25, 50):
            cells = ''
            for epoch in CHECKPOINTS:
                row = next(r for r in rows if r['width'] == width and r['batch'] == batch and r['epochs'] == epoch)
                cells += f"{row['mean']*100:>8.2f}"
            print(f"{width:>4}{batch:>4}  {cells}")
