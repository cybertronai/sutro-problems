#!/usr/bin/env python
"""Aggregate results/*.json into markdown tables (mean +- sd over seeds)."""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / 'results'
ORDER = ['raw', 'perm', 'rot', 'zca1e-3', 'zca1e-3+rot', 'zca1e-2+rot', 'pca60+rot']


def load(stage):
    out = defaultdict(list)
    for f in sorted(glob.glob(str(RES / f'{stage}-*-s[0-9]*.json'))):
        r = json.load(open(f))
        if 'variant' not in r:
            continue
        out[r['variant']].append(r)
    return {v: out[v] for v in ORDER if v in out} | {v: r for v, r in out.items() if v not in ORDER}


def ms(vals, fmt='%.2f'):
    vals = [v for v in vals if v is not None]
    if not vals:
        return '—'
    if len(vals) == 1:
        return fmt % vals[0]
    return (fmt + ' ± ' + fmt) % (np.mean(vals), np.std(vals, ddof=1))


def table(rows, header):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        lines.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(lines)


def utility():
    d = load('utility')
    rows = []
    for v, recs in d.items():
        rows.append([v, len(recs), recs[0]['dim'], ms([r['logreg_acc'] for r in recs]), ms([r['mlp_acc'] for r in recs]),
                     ms([r.get('cnn_noaug_acc') for r in recs]), ms([r.get('cnn_aug_acc') for r in recs])])
    return table(rows, ['release', 'seeds', 'dim', 'logreg %', 'MLP-512x2 %', 'CNN (no aug) %', 'CNN (+shift aug) %'])


def separate():
    d = load('separate')
    rows = []
    for v, recs in d.items():
        rows.append([v, len(recs)] + [ms([r[f'{m}_mlp_acc'] for r in recs]) for m in ('shared', 'separate', 'pool')]
                    + [ms([r[f'{m}_logreg_acc'] for r in recs]) for m in ('shared', 'separate', 'pool')]
                    + [ms([r['separate_basis_mismatch_rms'] for r in recs], '%.3f')])
    return table(rows, ['release', 'seeds', 'MLP shared', 'MLP separate', 'MLP pool-fit', 'logreg shared', 'logreg separate', 'logreg pool-fit', 'basis mismatch (separate)'])


def blind():
    d = load('blind')
    rows = []
    for v, recs in d.items():
        for name in recs[0]['methods']:
            ms_ = [r['methods'][name] for r in recs if name in r['methods']]
            base = [v, name, len(ms_), ms([m['mean_localisation_r1.5'] for m in ms_]),
                    ms([m.get('basis_active_matched_energy_mean') for m in ms_]),
                    ms([m.get('basis_active_n_above_0.5') for m in ms_], '%.0f') + ' / ' + ms([m.get('basis_active_n_above_0.9') for m in ms_], '%.0f')]
            for feed in ('abs-pcorr', 'abs-corr', 'pos-pcorr', 'raw-corr'):
                qs = [m['layouts'].get(feed, {}) for m in ms_]
                base.append(ms([q.get('edge_precision_at_1.5') for q in qs]) + ' (chance ' + ms([q.get('chance_precision_at_1.5') for q in qs]) + ')')
            rows.append(base)
    t = table(rows, ['release', 'unmixing', 'seeds', 'localisation r<=1.5', 'active matched energy', 'active sources >0.5 / >0.9 (of 64 live pixels; 60 sources for pca60)', 'edge-prec@1.5 |s| pcorr', '|s| corr', 'max(s,0) pcorr', 's corr (control)'])
    d2 = load('blindcnn')
    if d2:
        rows = []
        keys = sorted({k for recs in d2.values() for r in recs for m in r['methods'].values() for k in m})
        for v, recs in d2.items():
            for name in recs[0]['methods']:
                ms_ = [r['methods'][name] for r in recs if name in r['methods']]
                rows.append([v, name, len(ms_)] + [ms([m.get(k) for m in ms_]) for k in keys])
        t += '\n\nEnd-to-end (fresh CNN trained on the recovered lattice; MLP on the sources):\n\n' + table(rows, ['release', 'unmixing', 'seeds'] + keys)
    return t


def informed():
    d = load('informed')
    rows = []
    for v, recs in d.items():
        for init in recs[0]['inits']:
            es = [r['inits'][init] for r in recs if init in r['inits']]
            label = 'oracle: true map (diagnostic, not an attack)' if init == '__oracle__' else init
            rows.append([v, label, len(es),
                         ms([e['init_loss'] for e in es], '%.0f'), ms([e['final_loss'] for e in es], '%.0f'),
                         ms([e['init'].get('active_diag_energy_mean', e['init']['diag_energy_mean']) for e in es]), ms([e['refined'].get('active_diag_energy_mean', e['refined']['diag_energy_mean']) for e in es]),
                         ms([e['refined'].get('active_n_above_0.9') for e in es], '%.0f'),
                         ms([e['init']['rel_rms'] for e in es]), ms([e['refined']['rel_rms'] for e in es])])
    t = table(rows, ['release', 'init', 'seeds', 'moment loss init', 'final', 'diag energy (active px) init', 'refined', 'active px >0.9 (of 64)', 'pixel rel-RMS init', 'refined'])
    d2 = load('smuggle')
    if d2:
        rows = []
        for v, recs in d2.items():
            r0 = recs[0]
            rows.append([v, len(recs), ms([r['smuggled_cnn_acc_on_true_pixels'] for r in recs]), ms([r.get('smuggled_cnn_acc_on_release_no_attack') for r in recs]),
                         ms([r['attacker_pick_smuggled_cnn_acc'] for r in recs]) + ' (' + '/'.join(r['attacker_pick'] for r in recs) + ')',
                         ms([r['inits'].get('rlc', {}).get('refined', {}).get('smuggled_cnn_acc') for r in recs]),
                         ms([r['inits'].get('eig', {}).get('refined', {}).get('smuggled_cnn_acc') for r in recs]),
                         ms([r.get('adapter', {}).get('random', {}).get('test_acc') for r in recs]),
                         ms([r.get('adapter', {}).get('informed', {}).get('test_acc') for r in recs])])
        t += '\n\nSmuggled offline CNN (trained on the disjoint public 40k) applied to recovered pixels, test accuracy %:\n\n' + table(rows, ['release', 'seeds', 'on true pixels (ceiling)', 'on release, no attack', 'attacker-picked recovery', 'rlc recovery', 'eig recovery', 'adapter (random init)', 'adapter (informed init)'])
    return t


def exact():
    d = load('exact')
    rows = []
    for v, recs in d.items():
        rows.append([v, len(recs), ms([r['pool_fit_norm_match_label_acc'] for r in recs]), ms([r['pool_fit_exact_row_hits'] for r in recs]),
                     ms([r.get('pool_fit_map_from_matches_rel_rms_on_train_rows') for r in recs], '%.4f'),
                     ms([r['train_fit_norm_match_label_acc'] for r in recs]), ms([r['train_fit_exact_row_hits'] for r in recs])])
    return table(rows, ['release', 'seeds', 'label acc % (W fit on pool)', 'exact row hits %', 'full map from matches: rel-RMS on other rows', 'label acc % (W fit on secret train rows, naive)', 'exact row hits %'])


if __name__ == '__main__':
    for name, fn in [('Honest-learner utility', utility), ('Separate vs shared whitening', separate),
                     ('Blind attacks', blind), ('Informed attacks', informed), ('Exact-source norm matching', exact)]:
        try:
            t = fn()
        except Exception as err:
            t = f'(no results yet: {type(err).__name__}: {err})'
        print(f'\n## {name}\n\n{t}')
