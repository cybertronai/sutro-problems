"""(a) Blind attack on the linear release: can an entrant with no source data
recover the pixel lattice and cash in a spatial prior?

Attacker model: sees only what a submission sees for one draw -- the released
train features, the released (secretly relabelled) train labels and the released
test features.  No MNIST, no pixel statistics, nothing precomputed.

Attack: treat z = Q W (x - mu) as a linear mixture of unknown sources and unmix
it blindly (sparse non-negative unmixing, and FastICA as the second opinion),
then arrange the recovered sources on a 9x9 lattice from their |s| partial
correlations (the pmnist study's embedding + QAP search).  If that lattice is
the pixel lattice, the entrant can convolve.

Scoring (uses the secret map, which an attacker never has): every recovered
source is expressed in true-pixel coordinates through B @ A, and a recovered
lattice edge counts as correct if the two sources it joins have true pixel
centroids at distance <= 1 (and <= 1.5 for the looser count).  Chance is the
same statistic over random layouts.

Then the bottom line an organiser cares about: test accuracy of the study's
small CNN on the recovered lattice, against an honest MLP on the released
features, against the same CNN on a random layout.

Run:
  MNIST_POOL_CACHE=/tmp/gpumode-pool \
  /tmp/pmnist-env/bin/python .../redteam/release/blind_attack.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import harness
from harness import attacks, log, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=harness.CASE_SEED)
    ap.add_argument('--release-dims', type=int, default=harness.RELEASE_DIMS)
    ap.add_argument('--unmix-steps', type=int, default=2000)
    ap.add_argument('--qap-seconds', type=float, default=120.0)
    ap.add_argument('--cnn-epochs', type=int, default=12)
    ap.add_argument('--mlp-epochs', type=int, default=30)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default=str(harness.HERE / 'results-blind.json'))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    t_all = time.perf_counter()
    d = harness.Draw(case_seed=args.seed, release_dims=args.release_dims)
    d.check_oracle()
    log(f'draw: {d.z_train.shape} train / {d.z_test.shape} test, '
        f'oracle map residual {d.oracle_rel_rms:.2e}')
    rec = {'attack': 'blind', 'draw': d.summary(), 'unmix_steps': args.unmix_steps,
           'qap_seconds': args.qap_seconds, 'methods': {}}

    z_all = np.concatenate([d.z_train, d.z_test])          # what a submission holds
    n_tr = len(d.z_train)
    a_secret = d.A                                          # k x 81, SCORING ONLY

    # The release is already exactly white and only 60-dimensional, so there is
    # no further attacker-side PCA to do (the study needed one for 81-d releases
    # that carried ~17 near-null border directions).
    recovered = {}
    for name in ('sparse-nonneg', 'ica-logcosh'):
        t0 = time.perf_counter()
        if name == 'sparse-nonneg':
            s, b, _ = attacks.sparse_nonneg_unmix(z_all, seed=args.seed, steps=args.unmix_steps)
            iters = -1
        else:
            s, b, _, iters = attacks.fastica(z_all, fun='logcosh', seed=args.seed)
        comp = b @ a_secret                                 # sources x true pixels (scoring)
        m = {'seconds_unmix': time.perf_counter() - t0, 'iterations': int(iters)}
        cen, local = attacks.source_centroids(comp)
        m['mean_localisation_r1.5'] = float(local.mean())
        m['median_localisation_r1.5'] = float(np.median(local))
        basis = harness.common.basis_recovery_report(comp)
        m['matched_energy_mean'] = basis['matched_energy_mean']
        m['n_sources_above_0.5'] = basis['n_above_0.5']
        m['n_sources_above_0.9'] = basis['n_above_0.9']
        layout, secs, det = attacks.lattice_from_sources(
            s, feed='abs-pcorr', max_seconds=args.qap_seconds)
        q = attacks.layout_quality(layout, comp)
        q['seconds'] = secs
        q['qap_objective'] = det['qap_objective']
        m['lattice'] = q
        m['edge_precision_ratio_to_chance'] = (
            q['edge_precision_at_1.5'] / max(q['chance_precision_at_1.5'], 1e-9))
        rec['methods'][name] = m
        log(f'{name}: edge-prec@1.0 {q["edge_precision_at_1.0"]:.3f} '
            f'@1.5 {q["edge_precision_at_1.5"]:.3f} (chance {q["chance_precision_at_1.5"]:.3f}, '
            f'x{m["edge_precision_ratio_to_chance"]:.1f}), localisation {local.mean():.2f}')
        recovered[name] = (s, layout)

    # A QAP objective is not comparable across unmixers (different similarity
    # matrices), so the attacker has no sound way to pick one: train the CNN on
    # every recovered lattice and report the best, which is generous to them.
    learners = {}
    rng = np.random.default_rng(args.seed)
    rand_layout = rng.permutation(harness.NFEAT)
    for name, (s, layout) in recovered.items():
        imgs = attacks.arrange(s, layout)                   # n x 9 x 9
        t0 = time.perf_counter()
        p, _ = harness.models.cnn_fit_predict(imgs[:n_tr], d.y_train, imgs[n_tr:],
                                              seed=args.seed, epochs=args.cnn_epochs,
                                              augment=False)
        learners[f'cnn_on_recovered_lattice[{name}]'] = {
            'test_acc': acc(p, d.y_test), 'seconds': time.perf_counter() - t0}
        log(f'cnn on {name} lattice: {learners[f"cnn_on_recovered_lattice[{name}]"]["test_acc"]:.2f}%')
    # control: the same sources placed on a random layout
    s = recovered['ica-logcosh'][0]
    imgs_r = attacks.arrange(s, rand_layout)
    t0 = time.perf_counter()
    p, _ = harness.models.cnn_fit_predict(imgs_r[:n_tr], d.y_train, imgs_r[n_tr:], seed=args.seed,
                                          epochs=args.cnn_epochs, augment=False)
    learners['cnn_on_random_layout'] = {'test_acc': acc(p, d.y_test),
                                        'seconds': time.perf_counter() - t0}
    t0 = time.perf_counter()
    p, _ = harness.models.mlp_fit_predict(d.z_train, d.y_train, d.z_test,
                                          seed=args.seed, epochs=args.mlp_epochs)
    learners['mlp_on_released_features'] = {'test_acc': acc(p, d.y_test),
                                            'seconds': time.perf_counter() - t0}
    rec['learners'] = learners
    best_cnn = max((v['test_acc'] for k, v in learners.items()
                    if k.startswith('cnn_on_recovered_lattice')))
    rec['best_cnn_on_recovered_lattice'] = best_cnn
    rec['cnn_gain_over_mlp_points'] = (best_cnn
                                       - learners['mlp_on_released_features']['test_acc'])
    rec['total_seconds'] = time.perf_counter() - t_all
    for k, v in learners.items():
        log(f'{k}: {v["test_acc"]:.2f}%')
    log(f'CNN(recovered lattice) - MLP(released) = {rec["cnn_gain_over_mlp_points"]:+.2f} points')
    harness.jdump(args.out, rec)
    log(f'wrote {args.out} in {rec["total_seconds"]:.0f}s')


if __name__ == '__main__':
    main()
