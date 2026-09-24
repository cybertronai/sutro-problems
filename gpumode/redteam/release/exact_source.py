"""(e) Exact-source re-identification: the released rows ARE public MNIST rows.

Attacker model: holds the whole 60,000-row MNIST pool with labels, and one
draw's released test features.

Attack: ``z = Q W (x - mu)`` and ``Q`` is orthogonal, so ``||z||`` is the
Mahalanobis norm ``||W (x - mu)||`` of the source image -- a rotation-invariant
fingerprint.  Refit the organiser's published recipe (exact PCA whitening onto
the top 60 principal directions) on the attacker's own rows, compute the norm of
every pool row, and match each released row to the pool row with the nearest
norm.  Its label is then free, up to the draw's secret label permutation, which
harness.recover_label_permutation strips.

Two things stand between this and a win, and both are measured here:

  * the harness fits ``(mu, W)`` on the draw's own secret 10,000 training rows,
    not on anything public, so the attacker's norms are the right quantity
    computed with the wrong matrix.  We report the re-identification rate, and,
    because "the naive match fails" is not the same as "the channel is closed",
    the rank of the true source row in the attacker's norm-sorted candidate
    list -- how many bits a smarter matcher would still have to find;
  * the table.  Norm matching needs the pool's norms and labels inside a
    20,480-byte submission.  We measure the packing, including the entropy floor
    of the label column, which no coder can beat.

Run:
  MNIST_POOL_CACHE=/tmp/gpumode-pool \
  /tmp/pmnist-env/bin/python .../redteam/release/exact_source.py
"""
from __future__ import annotations

import argparse

import math
import time


import numpy as np

import harness
from harness import attacks, common, log, acc
from artifact_size import MAX_SOURCE_BYTES, packaging


def norm_rank_report(z_test, x_pool, test_rows, w_att, mu_att, sample=2000, seed=0):
    """Where does the true source row sit in the attacker's norm-nearest list?"""
    cand = np.linalg.norm((x_pool.astype(np.float64) - mu_att) @ w_att.T, axis=1)
    order = np.argsort(cand)
    rank_of_row = np.empty(len(cand), np.int64)
    rank_of_row[order] = np.arange(len(cand))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(z_test), min(sample, len(z_test)), replace=False)
    r = np.linalg.norm(z_test[idx].astype(np.float64), axis=1)
    pos = np.searchsorted(cand[order], r)
    true_rank = np.abs(rank_of_row[test_rows[idx]] - pos)
    rel_err = np.abs(r - cand[test_rows[idx]]) / np.maximum(r, 1e-12)
    spacing = np.diff(np.sort(cand))
    return {
        'sampled_rows': int(len(idx)),
        'median_rank_of_true_row': float(np.median(true_rank)),
        'mean_rank_of_true_row': float(true_rank.mean()),
        'fraction_true_row_in_top_1': float((true_rank <= 1).mean()),
        'fraction_true_row_in_top_10': float((true_rank <= 10).mean()),
        'fraction_true_row_in_top_100': float((true_rank <= 100).mean()),
        'median_relative_norm_error': float(np.median(rel_err)),
        'median_pool_norm_spacing_relative': float(np.median(spacing) / float(np.median(cand))),
        'bits_still_missing_at_median_rank': float(np.log2(max(np.median(true_rank), 1.0))),
    }


def lookup_table_sizes(pool_y, norms):
    """What a norm -> label table costs inside a submission."""
    n = len(pool_y)
    counts = np.bincount(pool_y, minlength=10) / n
    entropy_bits = float(-(counts[counts > 0] * np.log2(counts[counts > 0])).sum())
    labels_sorted = pool_y[np.argsort(norms)].astype(np.uint8)
    packed = np.packbits(np.unpackbits(labels_sorted[:, None], axis=1, count=4,
                                       bitorder='little').reshape(-1))
    out = {
        'pool_rows': int(n),
        'label_entropy_bits_per_row': entropy_bits,
        'label_entropy_floor_bytes': int(math.ceil(n * entropy_bits / 8)),
        'labels_4bit_raw_bytes': int(packed.nbytes),
        'labels_4bit_packaged': packaging(packed.tobytes(), 'labels 4-bit, norm order'),
        'norms_fp16_packaged': packaging(
            np.asarray(norms, np.float16).tobytes(), 'pool norms fp16'),
        'norm_breakpoints_4096_fp16_packaged': packaging(
            np.quantile(norms, np.linspace(0, 1, 4096)).astype(np.float16).tobytes(),
            '4096 norm quantiles fp16'),
    }
    minimal = (out['labels_4bit_packaged']['bytes_in_submission']
               + out['norm_breakpoints_4096_fp16_packaged']['bytes_in_submission'])
    out['cheapest_working_table_bytes'] = minimal
    out['cheapest_working_table_over_cap_factor'] = minimal / MAX_SOURCE_BYTES
    # the floor nothing can beat: base64 of an entropy-coded label column alone
    floor = math.ceil(out['label_entropy_floor_bytes'] * 4 / 3)
    out['entropy_floor_base64_bytes'] = floor
    out['entropy_floor_over_cap_factor'] = floor / MAX_SOURCE_BYTES
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=harness.CASE_SEED)
    ap.add_argument('--release-dims', type=int, default=harness.RELEASE_DIMS)
    ap.add_argument('--out', default=str(harness.HERE / 'results-exact-source.json'))
    args = ap.parse_args()

    t_all = time.perf_counter()
    d = harness.Draw(case_seed=args.seed, release_dims=args.release_dims)
    d.check_oracle()
    x_pub, y_pub = d.pool_x[d.public_rows], d.pool_y[d.public_rows]
    rec = {'attack': 'exact-source', 'draw': d.summary()}

    perm_hat, _ = harness.recover_label_permutation(
        d.z_train, d.y_train, x_pub, y_pub, k=args.release_dims)
    rec['label_permutation_classes_correct'] = int((perm_hat == d.label_permutation()).sum())

    fits = {
        'whole_pool_60k': (d.pool_x, 'the attacker refits the recipe on all 60k pool rows '
                                     '(includes the draw\'s secret rows, an upper bound)'),
        'public_40k': (x_pub, 'the attacker refits on the 40k rows the draw did not use'),
        'secret_train_10k_ORACLE': (d.x_train, 'SCORING ONLY: the organiser\'s own fit set, '
                                               'the ceiling a perfect guess of mu and W would give'),
    }
    rec['fits'] = {}
    for name, (x_fit, note) in fits.items():
        t0 = time.perf_counter()
        w_att, mu_att, _ = common.whitener(x_fit, 'pca', eps=0.0, k=args.release_dims)
        lab_true, pick = attacks.norm_match_labels(d.z_test, d.pool_x, d.pool_y, w_att, mu_att)
        entry = {
            'note': note,
            'exact_row_hit_percent': float((pick == d.test_rows).mean() * 100),
            'label_accuracy_percent': acc(perm_hat[lab_true], d.y_test),
            'seconds': time.perf_counter() - t0,
        }
        entry.update(norm_rank_report(d.z_test, d.pool_x, d.test_rows, w_att, mu_att,
                                      seed=args.seed))
        rec['fits'][name] = entry
        log(f'{name}: row hits {entry["exact_row_hit_percent"]:.2f}%, labels '
            f'{entry["label_accuracy_percent"]:.2f}%, median rank of true row '
            f'{entry["median_rank_of_true_row"]:.0f}, relative norm error '
            f'{entry["median_relative_norm_error"]:.2e}')

    w_att, mu_att, _ = common.whitener(x_pub, 'pca', eps=0.0, k=args.release_dims)
    norms = np.linalg.norm((d.pool_x.astype(np.float64) - mu_att) @ w_att.T, axis=1)
    rec['lookup_table'] = lookup_table_sizes(d.pool_y, norms)
    log(f'cheapest norm->label table: {rec["lookup_table"]["cheapest_working_table_bytes"]} bytes '
        f'({rec["lookup_table"]["cheapest_working_table_over_cap_factor"]:.1f}x the '
        f'{MAX_SOURCE_BYTES}-byte cap); entropy floor of the label column alone '
        f'{rec["lookup_table"]["entropy_floor_base64_bytes"]} bytes '
        f'({rec["lookup_table"]["entropy_floor_over_cap_factor"]:.1f}x)')

    rec['total_seconds'] = time.perf_counter() - t_all
    harness.jdump(args.out, rec)
    log(f'wrote {args.out} in {rec["total_seconds"]:.0f}s')


if __name__ == '__main__':
    main()
