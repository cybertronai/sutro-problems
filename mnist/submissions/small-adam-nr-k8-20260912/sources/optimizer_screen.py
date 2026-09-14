"""CPU screen: do better training rules reach 67% in fewer epochs than SGD?

Same MLP, same data/draw protocol, same ordered accumulation; only the update
rule changes. Grid costs are NOT computed here (the accepted scorer does not
price these rules yet) - this is an accuracy-vs-epochs screen only.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import sweep

HERE = Path(__file__).resolve().parent
SEEDS = [20261001, 20261002, 20261003]
CHECKPOINTS = [50, 100, 150, 200, 300]
VARIANTS = ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam']
WIDTHS = [32, 48]
f32 = np.float32


def run(case):
    variant, width, seed = case
    x, q, target, truth = sweep.draw(seed)
    w1, b1, w2, b2 = sweep.parameters(width)
    lr, batch, mu, eps = 0.2, 25, f32(.9), f32(1e-8)
    step = f32(lr / batch)
    state = {k: np.zeros_like(v) for k, v in (('w1', w1), ('b1', b1), ('w2', w2), ('b2', b2))}
    accs = {}
    for epoch in range(1, max(CHECKPOINTS) + 1):
        for start in range(0, 1000, batch):
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
                if variant == 'sgd':
                    params[key] = params[key] - step * g
                elif variant == 'momentum':
                    state[key] = mu * state[key] + g
                    params[key] = params[key] - step * state[key]
                elif variant == 'nesterov':
                    state[key] = mu * state[key] + g
                    params[key] = params[key] - step * (g + mu * state[key])
                elif variant == 'rmsprop':
                    state[key] = f32(.9) * state[key] + f32(.1) * g * g
                    params[key] = params[key] - step * g / (np.sqrt(state[key]) + eps)
                else:
                    state[key] = f32(.9) * state[key] + f32(.1) * g
                    state['v' + key] = f32(.999) * state.get('v' + key, np.zeros_like(g)) + f32(.001) * g * g
                    mhat = state[key] / (1 - f32(.9) ** epoch)
                    vhat = state['v' + key] / (1 - f32(.999) ** epoch)
                    params[key] = params[key] - step * mhat / (np.sqrt(vhat) + eps)
            w1, b1, w2, b2 = params['w1'], params['b1'], params['w2'], params['b2']
        if epoch in CHECKPOINTS:
            zh = sweep.mm(q, w1) + b1
            scores = sweep.mm(np.where(zh > f32(0), zh, f32(0)), w2) + b2
            accs[epoch] = float((scores.argmax(1) == truth).mean())
    return variant, width, seed, accs


if __name__ == '__main__':
    cases = [(v, w, s) for v in VARIANTS for w in WIDTHS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, cases))
    table = {}
    for variant, width, seed, accs in results:
        for epoch, acc in accs.items():
            table.setdefault((variant, width, epoch), []).append(acc)
    rows = []
    for (variant, width, epoch), values in sorted(table.items()):
        rows.append({'variant': variant, 'width': width, 'epochs': epoch,
                     'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1))})
    (HERE / 'optimizer-screen.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f"{'variant':<10}{'w':>4}  " + ''.join(f"{e:>9}" for e in CHECKPOINTS))
    for variant in VARIANTS:
        for width in WIDTHS:
            cells = ''
            for epoch in CHECKPOINTS:
                row = next(r for r in rows if r['variant'] == variant and r['width'] == width and r['epochs'] == epoch)
                cells += f"{row['mean']*100:>9.2f}"
            print(f"{variant:<10}{width:>4}  {cells}")
