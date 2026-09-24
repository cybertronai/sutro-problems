#!/usr/bin/env python
"""Run one stage of the rotation-obfuscation study.

CPU stages (attacks, honest dense learners):
  python run.py utility  --seeds 1,2,3            # MLP / logreg on every release (+ CNN on raw pixels: GPU-friendly)
  python run.py separate --seeds 1,2,3            # shared vs separate vs pool-fitted whitening
  python run.py blind    --seeds 1,2,3            # ICA / sparse unmixing + lattice recovery -> blind-*.npz + .json
  python run.py informed --seeds 1,2,3            # distribution-matching pixel recovery -> informed-*.json (maps stored)
  python run.py exact    --seeds 1,2,3            # norm matching against the exact source pool
GPU stages (consume the CPU stages' outputs):
  python run.py blindcnn --seeds 1,2,3            # CNN on the blind attacker's recovered lattice
  python run.py smuggle  --seeds 1,2,3            # offline CNN on public pixels applied to recovered pixels

Every stage writes results/<stage>-<variant>-s<seed>.json and skips files that exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import attacks
import common
import models

RES = common.HERE / 'results'
RES.mkdir(exist_ok=True)
BLIND_FEEDS = ('abs-pcorr', 'abs-corr', 'pos-pcorr', 'raw-corr')
BLIND_K = 64
CNN_FEEDS = ('abs-pcorr', 'pos-pcorr')


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def acc(pred, y):
    return float((np.asarray(pred) == np.asarray(y)).mean() * 100)


def done(path):
    return path.exists() and path.stat().st_size > 0


def load_draw(seed):
    x, y = common.pool()
    tr, te, pub = common.draw(seed)
    return x, y, tr, te, pub


# ------------------------------------------------------------------ stages
def stage_utility(seeds, variants):
    for seed in seeds:
        x, y, tr, te, pub = load_draw(seed)
        for v in variants:
            out = RES / f'utility-{v}-s{seed}.json'
            if done(out):
                continue
            rel = common.release(v, x[tr], x[tr], x[te], seed)
            rec = {'variant': v, 'seed': seed, 'dim': int(rel['z_train'].shape[1])}
            p, s = models.mlp_fit_predict(rel['z_train'], y[tr], rel['z_test'], seed=seed)
            rec['mlp_acc'], rec['mlp_seconds'] = acc(p, y[te]), s
            p, s = models.logreg_fit_predict(rel['z_train'], y[tr], rel['z_test'])
            rec['logreg_acc'], rec['logreg_seconds'] = acc(p, y[te]), s
            if v == 'raw':
                for aug in (True, False):
                    p, s = models.cnn_fit_predict(x[tr].reshape(-1, 9, 9), y[tr], x[te].reshape(-1, 9, 9), seed=seed, augment=aug)
                    rec['cnn_aug_acc' if aug else 'cnn_noaug_acc'] = acc(p, y[te])
                    rec['cnn_seconds'] = s
            log(f'utility {v} s{seed}: ' + ', '.join(f'{k}={val:.2f}' for k, val in rec.items() if k.endswith('_acc')))
            common.jdump(out, rec)


def stage_separate(seeds, variants):
    for seed in seeds:
        x, y, tr, te, pub = load_draw(seed)
        for v in variants:
            if common.VARIANTS[v]['whiten'] is None:
                continue
            out = RES / f'separate-{v}-s{seed}.json'
            if done(out):
                continue
            rec = {'variant': v, 'seed': seed}
            for mode in ('shared', 'separate', 'pool'):
                if mode == 'shared':
                    rel = common.release(v, x[tr], x[tr], x[te], seed)
                elif mode == 'separate':
                    rel = common.release(v, x[tr], x[tr], x[te], seed, fit_test_separately=True, x_fit_test=x[te])
                else:
                    rel = common.release(v, x, x[tr], x[te], seed)
                a, a_te = rel['secret']['A'], rel['secret']['A_test']
                mism = np.linalg.norm(a_te @ np.linalg.pinv(a) - np.eye(a.shape[0])) / np.sqrt(a.shape[0])
                p, s = models.mlp_fit_predict(rel['z_train'], y[tr], rel['z_test'], seed=seed)
                rec[f'{mode}_mlp_acc'] = acc(p, y[te])
                p, s = models.logreg_fit_predict(rel['z_train'], y[tr], rel['z_test'])
                rec[f'{mode}_logreg_acc'] = acc(p, y[te])
                rec[f'{mode}_basis_mismatch_rms'] = float(mism)
            log(f'separate {v} s{seed}: ' + ', '.join(f'{k}={val:.2f}' for k, val in rec.items() if isinstance(val, float)))
            common.jdump(out, rec)


def stage_blind(seeds, variants, funs=('logcosh',)):
    for seed in seeds:
        x, y, tr, te, pub = load_draw(seed)
        for v in variants:
            out = RES / f'blind-{v}-s{seed}.json'
            if done(out):
                continue
            rel = common.release(v, x[tr], x[tr], x[te], seed)
            a = rel['secret']['A']
            z_all = np.concatenate([rel['z_train'], rel['z_test']])
            rec = {'variant': v, 'seed': seed, 'dim': int(a.shape[0]), 'n_unlabelled': int(len(z_all)), 'methods': {}}
            arrays = {}
            # attacker-side: 9x9 digits have a dead border, so an 81-d release carries ~17
            # near-null directions that whitening-based unmixers would inflate into junk
            # sources.  The attacker keeps the top-k principal directions (k = 64).
            if z_all.shape[1] > BLIND_K:
                zc = z_all.astype(np.float64) - z_all.mean(0)
                lam, ev = np.linalg.eigh(np.cov(zc, rowvar=False))
                e_top = ev[:, ::-1][:, :BLIND_K].T                     # k x d
                z_att = (zc @ e_top.T).astype(np.float32)
                rec['attacker_pca_k'] = BLIND_K
                rec['attacker_dropped_variance_fraction'] = float(1 - lam[::-1][:BLIND_K].sum() / lam.sum())
            else:
                e_top = np.eye(z_all.shape[1]); z_att = z_all
            a_att = e_top @ a                                          # k x 81: released-reduced coords vs pixels
            active = x[pub].var(0) > 1e-4
            # reference: localisation of the released coordinates themselves (identity "attack")
            methods = [('ica-' + f, f) for f in funs] + [('sparse-nonneg', None)]
            for name, fun in methods:
                t0 = time.perf_counter()
                if fun:
                    s, b, mean, iters = attacks.fastica(z_att, fun=fun, seed=seed)
                else:
                    s, b, mean = attacks.sparse_nonneg_unmix(z_att, seed=seed); iters = -1
                comp = b @ a_att                               # rows: sources; cols: true pixels
                m = {'seconds_unmix': time.perf_counter() - t0, 'iterations': iters}
                m.update({'basis_' + k: val for k, val in common.basis_recovery_report(comp, active).items() if k != 'peak_pixel'})
                peak = np.argmax(comp ** 2, axis=1)
                m['n_sources_peaking_on_active_pixels'] = int(active[peak].sum())
                cen, local = attacks.source_centroids(comp)
                m['mean_localisation_r1.5'] = float(local.mean())
                m['median_localisation_r1.5'] = float(np.median(local))
                m['layouts'] = {}
                arrays[f'sources_{name}'] = s.astype(np.float32)
                for feed in BLIND_FEEDS:
                    try:
                        layout, secs, det = attacks.lattice_from_sources(s, feed=feed)
                        q = attacks.layout_quality(layout, comp)
                        q['seconds'] = secs
                        q['qap_objective'] = det.get('qap_objective')
                        q['layout'] = layout.tolist()
                        arrays[f'layout_{name}_{feed}'] = layout
                        m['layouts'][feed] = q
                        log(f'  blind {v} s{seed} {name} feed={feed}: edge-prec@1.5={q["edge_precision_at_1.5"]:.2f} '
                            f'(chance {q["chance_precision_at_1.5"]:.2f}) local={m["mean_localisation_r1.5"]:.2f} {secs:.0f}s')
                    except Exception as err:
                        m['layouts'][feed] = {'error': f'{type(err).__name__}: {err}'}
                        log(f'  blind {v} s{seed} {name} feed={feed}: FAILED {err}')
                rec['methods'][name] = m
            np.savez_compressed(RES / f'blind-{v}-s{seed}.npz', **arrays)
            common.jdump(out, rec)


def stage_blindcnn(seeds, variants):
    for seed in seeds:
        x, y, tr, te, pub = load_draw(seed)
        for v in variants:
            out = RES / f'blindcnn-{v}-s{seed}.json'
            src = RES / f'blind-{v}-s{seed}.npz'
            if done(out) or not src.exists():
                continue
            arrays = dict(np.load(src))
            src2 = RES / f'blind2-{v}-s{seed}.npz'
            if src2.exists():
                arrays.update(dict(np.load(src2)))
            rec = {'variant': v, 'seed': seed, 'methods': {}}
            for name in [k[len('sources_'):] for k in arrays if k.startswith('sources_')]:
                s = arrays[f'sources_{name}']
                m = {}
                p, secs = models.mlp_fit_predict(s[:len(tr)], y[tr], s[len(tr):], seed=seed)
                m['mlp_on_sources_acc'] = acc(p, y[te])
                for feed in (('pixel-pcorr', 'abs-pcorr') if name.startswith('atom-') else CNN_FEEDS):
                    key = f'layout_{name}_{feed}'
                    if key not in arrays:
                        continue
                    imgs = attacks.arrange(s, arrays[key])
                    for aug in (True, False):
                        p, secs = models.cnn_fit_predict(imgs[:len(tr)], y[tr], imgs[len(tr):], seed=seed, augment=aug)
                        m[f'{feed}_cnn_{"aug" if aug else "noaug"}_acc'] = acc(p, y[te])
                # control: the same CNN on a random arrangement of the sources
                imgs = attacks.arrange(s, np.random.default_rng(seed).permutation(81))
                p, secs = models.cnn_fit_predict(imgs[:len(tr)], y[tr], imgs[len(tr):], seed=seed, augment=False)
                m['random_layout_cnn_noaug_acc'] = acc(p, y[te])
                rec['methods'][name] = m
                log(f'blindcnn {v} s{seed} {name}: ' + ', '.join(f'{k}={val:.2f}' for k, val in m.items()))
            common.jdump(out, rec)


def stage_blind2(seeds, variants, steps=800):
    """Zero-atom (facet) polish of the blind unmixings from stage_blind, plus the
    eps-tail second-order leak diagnostic."""
    for seed in seeds:
        x, y, tr, te, pub = load_draw(seed)
        active = x[pub].var(0) > 1e-4
        for v in variants:
            out = RES / f'blind2-{v}-s{seed}.json'
            src = RES / f'blind-{v}-s{seed}.npz'
            if done(out) or not src.exists():
                continue
            rel = common.release(v, x[tr], x[tr], x[te], seed)
            a = rel['secret']['A']
            z_all = np.concatenate([rel['z_train'], rel['z_test']])
            arrays = np.load(src)
            rec = {'variant': v, 'seed': seed, 'methods': {}}
            rec['tail_leak'] = attacks.tail_leak_report(z_all, a)
            if z_all.shape[1] > BLIND_K:
                zc = z_all.astype(np.float64) - z_all.mean(0)
                lam, ev = np.linalg.eigh(np.cov(zc, rowvar=False))
                e_top = ev[:, ::-1][:, :BLIND_K].T
                z_att = (zc @ e_top.T).astype(np.float32)
            else:
                e_top = np.eye(z_all.shape[1]); z_att = z_all
            a_att = e_top @ a
            rec['true_pixel_atom_mass_median'] = float(np.median(attacks.atom_mass(x[np.concatenate([tr, te])][:, active])))
            saved = {}
            for name in [k[len('sources_'):] for k in arrays if k.startswith('sources_')]:
                s0 = arrays[f'sources_{name}'].astype(np.float64)
                b0, mu0 = attacks.unmixing_from_sources(z_att, s0)
                t0 = time.perf_counter()
                s1, b1, mu1 = attacks.atom_polish(z_att, b0, steps=steps, seed=seed)
                m = {'seconds': time.perf_counter() - t0, 'atom_mass_median_before': float(np.median(attacks.atom_mass(s0))),
                     'atom_mass_median_after': float(np.median(attacks.atom_mass(s1)))}
                comp = b1 @ a_att
                m.update({'basis_' + k: val for k, val in common.basis_recovery_report(comp, active).items() if k != 'peak_pixel'})
                cen, local = attacks.source_centroids(comp)
                m['mean_localisation_r1.5'] = float(local.mean()); m['median_localisation_r1.5'] = float(np.median(local))
                m['layouts'] = {}
                # polished sources are pixel-like: shift to non-negative and use the plain pixel lattice attack
                feed = s1 - np.quantile(s1, 0.002, axis=0)[None, :]
                feed = np.clip(feed, 0, None); feed = feed / max(feed.std(), 1e-12) * 0.3
                if feed.shape[1] < 81:
                    feed = np.concatenate([feed, np.zeros((len(feed), 81 - feed.shape[1]))], 1)
                for fname, fn in (('pixel-pcorr', lambda: topology_layout(feed)), ('abs-pcorr', lambda: attacks.lattice_from_sources(s1, 'abs-pcorr'))):
                    try:
                        t0 = time.perf_counter()
                        layout = fn()
                        q = attacks.layout_quality(layout, comp); q['seconds'] = time.perf_counter() - t0; q['layout'] = layout.tolist()
                        m['layouts'][fname] = q
                        saved[f'layout_atom-{name}_{fname}'] = layout
                        log(f'  blind2 {v} s{seed} atom-{name} {fname}: edge-prec@1.5={q["edge_precision_at_1.5"]:.2f} (chance {q["chance_precision_at_1.5"]:.2f}) '
                            f'local={m["mean_localisation_r1.5"]:.2f} n>0.9={m["basis_active_n_above_0.9"]}')
                    except Exception as err:
                        m['layouts'][fname] = {'error': f'{type(err).__name__}: {err}'}
                saved[f'sources_atom-{name}'] = s1.astype(np.float32)
                rec['methods']['atom-' + name] = m
            np.savez_compressed(RES / f'blind2-{v}-s{seed}.npz', **saved)
            common.jdump(out, rec)


def topology_layout(feed, max_seconds=120.0):
    layout = attacks.topology.recover_layout(feed.astype(np.float32), orient=False, max_seconds=max_seconds)
    return np.asarray(layout, np.int64)


def stage_informed(seeds, variants):
    for seed in seeds:
        x, y, tr, te, pub = load_draw(seed)
        active = x[pub].var(0) > 1e-4
        for v in variants:
            out = RES / f'informed-{v}-s{seed}.json'
            if done(out):
                continue
            rel = common.release(v, x[tr], x[tr], x[te], seed)
            a = rel['secret']['A']
            rec = {'variant': v, 'seed': seed, 'dim': int(a.shape[0]), 'inits': {}}
            t0 = time.perf_counter()
            res = attacks.informed_attack(rel['z_train'], y[tr], rel['z_test'], x[pub], y[pub], init='all', seed=seed, log=log, oracle_A=a)
            rec['seconds'] = time.perf_counter() - t0
            for name, r in res.items():
                entry = {'init_loss': r['init_loss'], 'final_loss': r['final_loss']}
                for tag in ('init', 'refined'):
                    comp = r[tag]['H'] @ a                       # 81 x 81: recovered pixel i vs true pixel p
                    e = {k: val for k, val in common.basis_recovery_report(comp).items() if k != 'peak_pixel'}
                    e['diag_energy_mean'] = float(np.mean(np.diag(comp) ** 2 / np.maximum((comp ** 2).sum(1), 1e-300)))
                    x_hat = attacks.apply_linear(r[tag], rel['z_test'])
                    e.update(common.pixel_reconstruction_error(x_hat, x[te], active))
                    e['H'] = r[tag]['H'].tolist()
                    e['offset'] = r[tag]['offset'].tolist()
                    entry[tag] = e
                rec['inits'][name] = entry
                log(f'informed {v} s{seed} init={name}: diag-energy {entry["init"]["diag_energy_mean"]:.2f}->{entry["refined"]["diag_energy_mean"]:.2f}, '
                    f'rel-rms {entry["init"]["rel_rms"]:.2f}->{entry["refined"]["rel_rms"]:.2f}, loss {entry["init_loss"]:.3g}->{entry["final_loss"]:.3g}')
            common.jdump(out, rec)


def smuggled_cnn(seed, x, y, pub, epochs=15):
    """CNN trained offline on the public 40k pixel images (the smuggled artifact)."""
    log(f'  training smuggled CNN on {len(pub)} public rows (seed {seed})')
    return models.cnn_fit(x[pub].reshape(-1, 9, 9), y[pub], seed=seed, epochs=epochs, augment=False)


def stage_smuggle(seeds, variants):
    for seed in seeds:
        x, y, tr, te, pub = load_draw(seed)
        model = st = None
        for v in variants:
            out = RES / f'smuggle-{v}-s{seed}.json'
            src = RES / f'informed-{v}-s{seed}.json'
            if done(out) or not src.exists():
                continue
            if model is None:
                model, st = smuggled_cnn(seed, x, y, pub)
                ceiling = acc(models.cnn_predict(model, st, x[te].reshape(-1, 9, 9)), y[te])
            inf = json.load(open(src))
            inf['inits'] = {k: e for k, e in inf['inits'].items() if not k.startswith('__')}
            best_init = min(inf['inits'], key=lambda n: inf['inits'][n]['final_loss'])
            rel = common.release(v, x[tr], x[tr], x[te], seed)
            rec = {'variant': v, 'seed': seed, 'smuggled_cnn_acc_on_true_pixels': ceiling, 'inits': {}}
            # control: smuggled CNN applied to the released coordinates arranged as an image (no attack)
            if rel['z_test'].shape[1] == 81:
                rec['smuggled_cnn_acc_on_release_no_attack'] = acc(models.cnn_predict(model, st, rel['z_test'].reshape(-1, 9, 9)), y[te])
            for name, entry in inf['inits'].items():
                rec['inits'][name] = {}
                for tag in ('init', 'refined'):
                    if 'H' in entry[tag]:
                        lin = {'H': np.asarray(entry[tag]['H']), 'offset': np.asarray(entry[tag]['offset'])}
                    else:
                        maps = np.load(RES / f'informed-maps-s{seed}.npz')
                        lin = {'H': maps[f'{v}|{name}|{tag}|H'].astype(np.float64), 'offset': maps[f'{v}|{name}|{tag}|offset'].astype(np.float64)}
                    x_hat = attacks.apply_linear(lin, rel['z_test'])
                    rec['inits'][name][tag] = {'smuggled_cnn_acc': acc(models.cnn_predict(model, st, x_hat.reshape(-1, 9, 9)), y[te]),
                                               'rel_rms': entry[tag]['rel_rms'], 'final_loss': entry['final_loss']}
            # frozen-CNN + learned linear adapter (no explicit basis recovery)
            z_tr, z_te = rel['z_train'], rel['z_test']
            d = z_tr.shape[1]
            zc = z_tr.astype(np.float64) - z_tr.mean(0)
            lam, ev = np.linalg.eigh(np.cov(zc, rowvar=False))
            white = (ev / np.sqrt(np.clip(lam, 0, None) + 1e-4)[None, :]) @ ev.T          # attacker's own ZCA
            h_blind = 0.25 * common.haar(81, seed + 5)[:, :d] @ white                       # random orthogonal guess, pixel-ish scale
            c_blind = np.full(81, 0.13)
            rec['adapter'] = {}
            for name, h0, c0 in (('random', h_blind, c_blind),
                                 ('informed', np.asarray(inf['inits'][best_init]['refined']['H']), np.asarray(inf['inits'][best_init]['refined']['offset']))):
                t0 = time.perf_counter()
                p, val = models.adapter_fit_predict(model, st, z_tr, y[tr], z_te, h0, c0, seed=seed)
                rec['adapter'][name] = {'test_acc': acc(p, y[te]), 'val_acc': val, 'seconds': time.perf_counter() - t0}
            log(f'smuggle {v} s{seed}: adapter random {rec["adapter"]["random"]["test_acc"]:.2f}, informed {rec["adapter"]["informed"]["test_acc"]:.2f}')
            # the attacker picks the init with the smallest moment loss (no secrets needed)
            best = best_init
            rec['attacker_pick'] = best
            rec['attacker_pick_smuggled_cnn_acc'] = rec['inits'][best]['refined']['smuggled_cnn_acc']
            log(f'smuggle {v} s{seed}: ceiling {ceiling:.2f}, pick={best} -> {rec["attacker_pick_smuggled_cnn_acc"]:.2f}; '
                + ', '.join(f'{n}={e["refined"]["smuggled_cnn_acc"]:.1f}' for n, e in rec['inits'].items()))
            common.jdump(out, rec)


def stage_exact(seeds, variants):
    for seed in seeds:
        x, y, tr, te, pub = load_draw(seed)
        for v in variants:
            out = RES / f'exact-{v}-s{seed}.json'
            if done(out):
                continue
            rec = {'variant': v, 'seed': seed}
            for fit in ('pool', 'train'):
                rel = common.release(v, x if fit == 'pool' else x[tr], x[tr], x[te], seed)
                vv = common.VARIANTS[v]
                if vv['whiten'] is None:
                    w_att, mu_att = np.eye(81), np.zeros(81)
                else:   # attacker refits the organiser's recipe on the full public pool
                    w_att, mu_att, _ = common.whitener(x, vv['whiten'], vv.get('eps', 0.0), vv.get('k'))
                t0 = time.perf_counter()
                lab, pick = attacks.norm_match_labels(rel['z_test'], x, y, w_att, mu_att)
                rec[f'{fit}_fit_norm_match_label_acc'] = acc(lab, y[te])
                rec[f'{fit}_fit_exact_row_hits'] = float((pick == te).mean() * 100)
                rec[f'{fit}_fit_seconds'] = time.perf_counter() - t0
                if fit == 'pool':
                    h_hat, c_hat = attacks.procrustes_from_matched(rel['z_test'], x[pick])
                    active = x.var(0) > 1e-4
                    x_hat_tr = (rel['z_train'].astype(np.float64) @ h_hat.T + c_hat).astype(np.float32)
                    rec['pool_fit_map_from_matches_rel_rms_on_train_rows'] = common.pixel_reconstruction_error(x_hat_tr, x[tr], active)['rel_rms']
            log(f'exact {v} s{seed}: ' + ', '.join(f'{k}={val:.2f}' for k, val in rec.items() if isinstance(val, float)))
            common.jdump(out, rec)


STAGES = {'utility': stage_utility, 'separate': stage_separate, 'blind': stage_blind, 'blind2': stage_blind2, 'blindcnn': stage_blindcnn,
          'informed': stage_informed, 'smuggle': stage_smuggle, 'exact': stage_exact}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=list(STAGES))
    ap.add_argument('--seeds', default='1,2,3')
    ap.add_argument('--variants', default=None)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    seeds = [int(s) for s in args.seeds.split(',')]
    variants = args.variants.split(',') if args.variants else list(common.VARIANTS)
    STAGES[args.stage](seeds, variants)
