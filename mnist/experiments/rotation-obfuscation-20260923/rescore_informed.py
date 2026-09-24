#!/usr/bin/env python
"""Add active-pixel-only basis metrics to results/informed-*.json (scoring-side; rebuilds the
deterministic release to get the secret map)."""
import glob, json
import numpy as np
import common
x, y = common.pool()
for f in sorted(glob.glob(str(common.HERE / 'results' / 'informed-*-s*.json'))):
    if 'separate' in f:
        continue
    rec = json.load(open(f))
    seed, v = rec['seed'], rec['variant']
    tr, te, pub = common.draw(seed)
    active = x[pub].var(0) > 1e-4
    a = common.release(v, x[tr], x[tr], x[te], seed)['secret']['A']
    for name, entry in rec['inits'].items():
        for tag in ('init', 'refined'):
            comp = np.asarray(entry[tag]['H']) @ a
            e = comp ** 2
            rowfrac = np.diag(e) / np.maximum(e.sum(1), 1e-300)
            entry[tag]['active_diag_energy_mean'] = float(rowfrac[active].mean())
            rep = common.basis_recovery_report(comp[active], active)
            entry[tag]['active_n_above_0.9'] = rep['active_n_above_0.9']
            entry[tag]['active_n_above_0.5'] = rep['active_n_above_0.5']
            entry[tag]['active_matched_energy_mean'] = rep['active_matched_energy_mean']
    common.jdump(f, rec)
    print('rescored', f.split('/')[-1])
