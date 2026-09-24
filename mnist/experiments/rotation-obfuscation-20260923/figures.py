#!/usr/bin/env python
"""Figure: what each party sees.  Rows = release variants; columns = a few test
digits.  For each digit: true 9x9 pixels | released coordinates reshaped 9x9 |
informed attacker's reconstruction (attacker-picked init, refined) | blind
reconstruction (sparse unmixing sources on the recovered lattice, if present)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import attacks
import common

RES = common.HERE / 'results'
SEED = 1
VARIANTS = ['rot', 'zca1e-3+rot', 'pca60+rot']
DIGITS = [0, 1, 2, 3, 4, 5]


def main(out=RES / 'reconstructions-s1.png'):
    x, y, tr, te, pub = common.pool()[0], common.pool()[1], *common.draw(SEED)
    idx = te[DIGITS]
    rows = []
    for v in VARIANTS:
        rel = common.release(v, x[tr], x[tr], x[te], SEED)
        inf = json.load(open(RES / f'informed-{v}-s{SEED}.json'))
        inits = {k: e for k, e in inf['inits'].items() if not k.startswith('__')}
        best = min(inits, key=lambda n: inits[n]['final_loss'])
        if 'H' in inits[best]['refined']:
            lin = {'H': np.asarray(inits[best]['refined']['H']), 'offset': np.asarray(inits[best]['refined']['offset'])}
        else:
            maps = np.load(RES / f'informed-maps-s{SEED}.npz')
            lin = {'H': maps[f'{v}|{best}|refined|H'].astype(np.float64), 'offset': maps[f'{v}|{best}|refined|offset'].astype(np.float64)}
        x_hat = attacks.apply_linear(lin, rel['z_test'][DIGITS])
        z = rel['z_test'][DIGITS]
        z_img = np.zeros((len(DIGITS), 81)); z_img[:, :z.shape[1]] = z
        rows.append((v, best, x[idx], z_img, x_hat))
    fig, axes = plt.subplots(len(VARIANTS) * 3, len(DIGITS), figsize=(len(DIGITS) * 1.2, len(VARIANTS) * 3 * 1.25))
    for r, (v, best, xt, zi, xh) in enumerate(rows):
        for c in range(len(DIGITS)):
            for k, (img, lab) in enumerate(((xt[c], 'true pixels'), (zi[c], f'released ({v})'), (xh[c], f'informed recon ({best})'))):
                ax = axes[3 * r + k, c]
                ax.imshow(img.reshape(9, 9), cmap='gray_r' if k != 1 else 'coolwarm', vmin=(0 if k != 1 else None), vmax=(1 if k != 1 else None))
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(lab, fontsize=7)
    fig.suptitle('Seed 1 test digits: true pixels, released coordinates, informed-attack reconstruction', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print('wrote', out)


if __name__ == '__main__':
    main()
