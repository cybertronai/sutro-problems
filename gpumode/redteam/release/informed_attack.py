"""(b) Informed attack on the linear release: the entrant holds public MNIST.

Attacker model: the 40,000 pool rows the draw did not use, with their true
labels (public MNIST is public), plus what a submission sees -- the released
train features, the released train labels and the released test features.

Attack, in three attacker-side steps (no secrets anywhere):
  1. undo the evaluator's secret label permutation.  Whiten both sides to their
     own top-60 principal subspace; the two spaces then differ by an unknown
     orthogonal map, under which class-mean norms and class-covariance spectra
     are invariant, so pairing released class ids to public class ids is a 10x10
     assignment problem (harness.recover_label_permutation).
  2. recover the map.  attacks.informed_attack fits the orthogonal map between
     the two whitened spaces by matching class-conditional means and second
     moments (plus a non-negativity prior on the reconstructed pixels), from
     several inits; the attacker picks the init with the lowest moment loss,
     which needs no secret.
  3. cash in.  A CNN trained offline on the public 40,000 pixel rows is applied
     to the reconstructed test pixels.

Scored against: the same CNN on the true test pixels (the ceiling) and an honest
MLP on the released features (what the release is supposed to leave you with).

This attack is expected to WORK.  It is the reason the submission size cap and
the Fashion-MNIST hold-out are load-bearing; artifact_size.py and
holdout_check.py measure those two doors.

Run:
  MNIST_POOL_CACHE=/tmp/gpumode-pool \
  /tmp/pmnist-env/bin/python .../redteam/release/informed_attack.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import harness
from harness import attacks, common, models, log, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=harness.CASE_SEED)
    ap.add_argument('--release-dims', type=int, default=harness.RELEASE_DIMS)
    ap.add_argument('--inits', default='all')
    ap.add_argument('--refine-steps', type=int, default=600)
    ap.add_argument('--hops', type=int, default=3)
    ap.add_argument('--offline-cnn-epochs', type=int, default=12)
    ap.add_argument('--mlp-epochs', type=int, default=30)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default=str(harness.HERE / 'results-informed.json'))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    t_all = time.perf_counter()
    d = harness.Draw(case_seed=args.seed, release_dims=args.release_dims)
    d.check_oracle()
    x_pub, y_pub = d.pool_x[d.public_rows], d.pool_y[d.public_rows]
    log(f'draw {d.z_train.shape}, attacker holds {len(x_pub)} public pixel rows')
    rec = {'attack': 'informed', 'draw': d.summary(), 'n_public_rows': int(len(x_pub))}

    # ---- step 1: the secret label permutation
    t0 = time.perf_counter()
    perm_hat, perm_diag = harness.recover_label_permutation(
        d.z_train, d.y_train, x_pub, y_pub, k=args.release_dims)
    perm_true = d.label_permutation()
    rec['label_permutation'] = {
        'recovered': perm_hat.tolist(), 'true_for_scoring': perm_true.tolist(),
        'classes_correct': int((perm_hat == perm_true).sum()),
        'seconds': time.perf_counter() - t0, **perm_diag}
    log(f'label permutation: {rec["label_permutation"]["classes_correct"]}/10 classes correct '
        f'in {rec["label_permutation"]["seconds"]:.1f}s')
    y_pub_rel = perm_hat[y_pub]              # public labels in the draw's label convention

    # ---- step 2: recover the map
    t0 = time.perf_counter()
    res = attacks.informed_attack(d.z_train, d.y_train, d.z_test, x_pub, y_pub_rel,
                                  init=args.inits, seed=args.seed, log=log,
                                  refine_steps=args.refine_steps, hops=args.hops,
                                  k_att=args.release_dims, oracle_A=d.A)
    rec['recovery_seconds'] = time.perf_counter() - t0
    active = x_pub.var(0) > 1e-4             # the 64 live pixels of a 9x9 MNIST crop
    rec['n_active_pixels'] = int(active.sum())
    rec['inits'] = {}
    maps = {}
    for name, r in res.items():
        entry = {'init_loss': r['init_loss'], 'final_loss': r['final_loss']}
        for tag in ('init', 'refined'):
            comp = r[tag]['H'] @ d.A                                   # SCORING
            e = {k: v for k, v in common.basis_recovery_report(comp).items() if k != 'peak_pixel'}
            e['diag_energy_mean'] = float(np.mean(
                np.diag(comp) ** 2 / np.maximum((comp ** 2).sum(1), 1e-300)))
            x_hat = attacks.apply_linear(r[tag], d.z_test)
            e.update(common.pixel_reconstruction_error(x_hat, d.x_test, active))
            entry[tag] = e
        maps[name] = r['refined']
        rec['inits'][name] = entry
        log(f'  {name}: rel-RMS {entry["init"]["rel_rms"]:.3f} -> {entry["refined"]["rel_rms"]:.3f}, '
            f'loss {entry["init_loss"]:.4g} -> {entry["final_loss"]:.4g}')

    picks = {k: v for k, v in rec['inits'].items() if not k.startswith('__')}
    best = min(picks, key=lambda n: picks[n]['final_loss'])
    rec['attacker_pick'] = best
    rec['attacker_pick_rel_rms'] = picks[best]['refined']['rel_rms']
    log(f'attacker picks init={best} (lowest moment loss), pixel rel-RMS '
        f'{rec["attacker_pick_rel_rms"]:.3f}')

    # ---- step 3: the smuggled offline CNN
    t0 = time.perf_counter()
    model, st = models.cnn_fit(x_pub.reshape(-1, 9, 9), y_pub_rel, seed=args.seed,
                               epochs=args.offline_cnn_epochs, augment=False)
    rec['offline_cnn_train_seconds'] = time.perf_counter() - t0
    ceiling = acc(models.cnn_predict(model, st, d.x_test.reshape(-1, 9, 9)), d.y_test)
    rec['offline_cnn_on_true_test_pixels'] = ceiling
    rec['offline_cnn_on_recovered'] = {}
    for name in picks:
        x_hat = attacks.apply_linear(maps[name], d.z_test)
        rec['offline_cnn_on_recovered'][name] = acc(
            models.cnn_predict(model, st, x_hat.reshape(-1, 9, 9)), d.y_test)
    # control: the smuggled CNN fed the released coordinates as if they were pixels
    if d.z_test.shape[1] == harness.NFEAT:
        rec['offline_cnn_on_release_no_attack'] = acc(
            models.cnn_predict(model, st, d.z_test.reshape(-1, 9, 9)), d.y_test)

    t0 = time.perf_counter()
    p, _ = models.mlp_fit_predict(d.z_train, d.y_train, d.z_test, seed=args.seed,
                                  epochs=args.mlp_epochs)
    rec['honest_mlp_on_released_features'] = acc(p, d.y_test)
    rec['honest_mlp_seconds'] = time.perf_counter() - t0
    rec['attack_accuracy'] = rec['offline_cnn_on_recovered'][best]
    rec['gain_over_honest_mlp_points'] = (rec['attack_accuracy']
                                          - rec['honest_mlp_on_released_features'])
    rec['total_seconds'] = time.perf_counter() - t_all
    log(f'offline CNN: true pixels {ceiling:.2f}%, recovered {rec["attack_accuracy"]:.2f}%, '
        f'honest MLP {rec["honest_mlp_on_released_features"]:.2f}% '
        f'({rec["gain_over_honest_mlp_points"]:+.2f} points)')
    harness.jdump(args.out, rec)
    # the maps are what holdout_check / artifact_size would reuse; keep them next to the json
    np.savez_compressed(harness.HERE / 'informed-maps.npz',
                        **{f'{k}|{t}': np.asarray(v[t]) for k, v in maps.items()
                           for t in ('H', 'offset')})
    log(f'wrote {args.out} in {rec["total_seconds"]:.0f}s')


if __name__ == '__main__':
    main()
