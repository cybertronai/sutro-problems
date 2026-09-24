"""(d) The Fashion-MNIST hold-out against the informed attack.

A leaderboard run interleaves ``holdout_draws`` calls drawn from Fashion-MNIST
at positions the secret seed picks; they are released through the same linear
map (their own, fitted on their own training rows), timed and ranked like every
other call, and their accuracy must clear ``holdout_min_bp`` = 3000 bp, i.e.
70%.  The submission cannot tell which call it is on.

This script runs the whole informed pipeline of informed_attack.py -- MNIST
public pool, MNIST class statistics, MNIST offline CNN -- against a Fashion
hold-out draw produced by eval.make_draw, and reports what the smuggler scores
against the 70% floor.  The honest MLP on the same released features is the
control: the floor has to be survivable by someone who actually learns.

Run:
  MNIST_POOL_CACHE=/tmp/gpumode-pool \
  /tmp/pmnist-env/bin/python .../redteam/release/holdout_check.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import harness
from harness import attacks, common, models, log, acc

HOLDOUT_MIN_BP = 3000           # bands.json defaults.holdout_min_bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=harness.CASE_SEED)
    ap.add_argument('--release-dims', type=int, default=harness.RELEASE_DIMS)
    ap.add_argument('--refine-steps', type=int, default=600)
    ap.add_argument('--hops', type=int, default=3)
    ap.add_argument('--offline-cnn-epochs', type=int, default=12)
    ap.add_argument('--mlp-epochs', type=int, default=30)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default=str(harness.HERE / 'results-holdout.json'))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    t_all = time.perf_counter()
    # the attacker's world: MNIST.  The draw: Fashion, exactly as run_case builds it.
    mnist = harness.Draw(case_seed=args.seed, release_dims=args.release_dims)
    hold = harness.Draw(dataset='fashion', case_seed=args.seed, release_dims=args.release_dims)
    hold.check_oracle()
    x_pub, y_pub = mnist.pool_x[mnist.public_rows], mnist.pool_y[mnist.public_rows]
    log(f'hold-out draw: fashion, draw seed {hold.seed}, {hold.z_train.shape}')
    rec = {'attack': 'holdout', 'holdout_draw': hold.summary(),
           'mnist_draw_for_attacker_world': mnist.summary(),
           'holdout_min_bp': HOLDOUT_MIN_BP,
           'holdout_floor_percent': (10000 - HOLDOUT_MIN_BP) / 100.0}

    # step 1: the attacker still has to guess the label permutation, now with
    # class statistics from the wrong dataset.
    t0 = time.perf_counter()
    perm_hat, perm_diag = harness.recover_label_permutation(
        hold.z_train, hold.y_train, x_pub, y_pub, k=args.release_dims)
    perm_true = hold.label_permutation()
    rec['label_permutation'] = {
        'recovered': perm_hat.tolist(), 'true_for_scoring': perm_true.tolist(),
        'classes_correct': int((perm_hat == perm_true).sum()),
        'seconds': time.perf_counter() - t0, **perm_diag}
    log(f'label permutation on the hold-out: '
        f'{rec["label_permutation"]["classes_correct"]}/10 classes correct')
    y_pub_rel = perm_hat[y_pub]

    # step 2: the same moment-matching recovery, MNIST statistics vs a Fashion release
    t0 = time.perf_counter()
    res = attacks.informed_attack(hold.z_train, hold.y_train, hold.z_test, x_pub, y_pub_rel,
                                  init='all', seed=args.seed, log=log,
                                  refine_steps=args.refine_steps, hops=args.hops,
                                  k_att=args.release_dims)
    rec['recovery_seconds'] = time.perf_counter() - t0
    active = hold.pool_x.var(0) > 1e-4
    rec['inits'] = {}
    maps = {}
    for name, r in res.items():
        x_hat = attacks.apply_linear(r['refined'], hold.z_test)
        rec['inits'][name] = {
            'final_loss': r['final_loss'],
            'rel_rms_vs_true_fashion_pixels': common.pixel_reconstruction_error(
                x_hat, hold.x_test, active)['rel_rms']}
        maps[name] = r['refined']
    best = min(rec['inits'], key=lambda n: rec['inits'][n]['final_loss'])
    rec['attacker_pick'] = best

    # step 3: the smuggled MNIST CNN, fired at Fashion
    t0 = time.perf_counter()
    model, st = models.cnn_fit(x_pub.reshape(-1, 9, 9), y_pub_rel, seed=args.seed,
                               epochs=args.offline_cnn_epochs, augment=False)
    rec['offline_cnn_train_seconds'] = time.perf_counter() - t0
    rec['offline_cnn_on_true_holdout_pixels'] = acc(
        models.cnn_predict(model, st, hold.x_test.reshape(-1, 9, 9)), hold.y_test)
    rec['offline_cnn_on_recovered'] = {
        name: acc(models.cnn_predict(
            model, st, attacks.apply_linear(m, hold.z_test).reshape(-1, 9, 9)), hold.y_test)
        for name, m in maps.items()}
    rec['smuggler_accuracy'] = rec['offline_cnn_on_recovered'][best]

    # control: an honest learner on the same hold-out release
    t0 = time.perf_counter()
    p, _ = models.mlp_fit_predict(hold.z_train, hold.y_train, hold.z_test, seed=args.seed,
                                  epochs=args.mlp_epochs)
    rec['honest_mlp_on_holdout_release'] = acc(p, hold.y_test)
    rec['honest_mlp_seconds'] = time.perf_counter() - t0

    floor = rec['holdout_floor_percent']
    rec['smuggler_passes_holdout'] = rec['smuggler_accuracy'] >= floor
    rec['honest_mlp_passes_holdout'] = rec['honest_mlp_on_holdout_release'] >= floor
    rec['total_seconds'] = time.perf_counter() - t_all
    log(f'hold-out floor {floor:.0f}%: smuggler {rec["smuggler_accuracy"]:.2f}% '
        f'({"PASS" if rec["smuggler_passes_holdout"] else "FAIL"}), honest MLP '
        f'{rec["honest_mlp_on_holdout_release"]:.2f}% '
        f'({"PASS" if rec["honest_mlp_passes_holdout"] else "FAIL"})')
    harness.jdump(args.out, rec)
    log(f'wrote {args.out} in {rec["total_seconds"]:.0f}s')


if __name__ == '__main__':
    main()
