"""(c) Does the informed attack fit inside a submission?

The informed attack of informed_attack.py works.  What stops it from being a
leaderboard entry is that the attacker has to carry its offline knowledge inside
``submission.py``, which the harness caps at ``max_source_bytes`` = 20,480 bytes
with no single string or bytes literal over ``max_literal_bytes`` = 4,096
(eval.py checks both before the file is imported).

This script measures the artifact, for real, from the real public pool:

  1. the public class statistics ``attacks.informed_attack`` consumes -- public
     mean, the top-k eigenbasis and eigenvalues of the public covariance, the 10
     class means and the 10 class covariances, all in the attacker's top-k
     whitened subspace -- quantised to int8 (per-row scale) and to fp16, raw and
     zlib-compressed, then base64'd and chopped into <=4,096-byte literals;
  2. the offline CNN's weights (int8), for the study's CNN and for shrunken
     variants;
  3. the attack code itself, measured as the source bytes of the functions it
     actually calls.

Then the experiment that decides whether the cap is enough: the only smuggled
artifact that DOES fit is a tiny CNN with no class statistics at all, used
through a learned linear adapter (models.adapter_fit_predict: freeze the CNN,
fit x_hat = z H^T + c on the released labels).  We train that tiny CNN offline
on the public 40,000 pixel rows and run it against the harness release, and
compare with the honest MLP on the released features.

Run:
  MNIST_POOL_CACHE=/tmp/gpumode-pool \
  /tmp/pmnist-env/bin/python .../redteam/release/artifact_size.py
"""
from __future__ import annotations

import argparse
import base64
import inspect
import math
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness
from harness import attacks, common, models, log, acc

MAX_SOURCE_BYTES = 20480
MAX_LITERAL_BYTES = 4096


# --------------------------------------------------------------- packing maths
def quantise(arrays, dtype):
    """Pack a dict of arrays; int8 carries one fp16 scale per row."""
    blob = bytearray()
    for a in arrays.values():
        a = np.asarray(a, np.float64)
        flat = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a[None, :]
        if dtype == 'int8':
            scale = np.maximum(np.abs(flat).max(1), 1e-30) / 127.0
            q = np.clip(np.round(flat / scale[:, None]), -127, 127).astype(np.int8)
            blob += scale.astype(np.float16).tobytes() + q.tobytes()
        elif dtype == 'fp16':
            blob += flat.astype(np.float16).tobytes()
        else:
            raise ValueError(dtype)
    return bytes(blob)


def packaging(blob, label):
    """What it costs to carry ``blob`` inside a submission file."""
    comp = zlib.compress(blob, 9)
    best = min(blob, comp, key=len)
    b64 = base64.b64encode(best)
    literals = math.ceil(len(b64) / MAX_LITERAL_BYTES)
    # each literal costs quotes plus a comma/newline in a tuple of chunks
    packaged = len(b64) + 4 * literals
    return {'name': label, 'raw_bytes': len(blob), 'zlib_bytes': len(comp),
            'base64_bytes': len(b64), 'literals_at_4096': literals,
            'bytes_in_submission': packaged,
            'fraction_of_20480_cap': packaged / MAX_SOURCE_BYTES}


def class_statistics(x_pub, y_pub, k):
    """Exactly the objects attacks.informed_attack derives from the public pool."""
    x_pub = np.asarray(x_pub, np.float64)
    mu = x_pub.mean(0)
    sig = np.cov(x_pub - mu, rowvar=False)
    lam, e = np.linalg.eigh(0.5 * (sig + sig.T))
    lam = np.clip(lam, 0, None)[::-1][:k]
    e = e[:, ::-1][:, :k]
    u = (x_pub - mu) @ (e / np.sqrt(lam + 1e-4)[None, :])
    means, covs = attacks._class_stats(u, y_pub)
    tri = np.triu_indices(k)
    return {
        'mu_pub': mu,                                   # 81
        'eigenbasis': e.T,                              # k x 81
        'eigenvalues': lam,                             # k
        'class_means': means,                           # 10 x k
        'class_covs_upper': np.stack([c[tri] for c in covs]),   # 10 x k(k+1)/2
    }


def param_count(module):
    return sum(p.numel() for p in module.parameters())


class TinyCNN(nn.Module):
    """The largest convolutional net whose int8 weights fit under the cap:
    8/16/16 channels, 2x2 average pool to 4x4, linear head."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1); self.b1 = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1); self.b2 = nn.BatchNorm2d(16)
        self.c3 = nn.Conv2d(16, 16, 3, padding=1); self.b3 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16 * 16, 10)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = F.relu(self.b3(self.c3(x)))
        return self.fc(F.avg_pool2d(x, 2).flatten(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=harness.CASE_SEED)
    ap.add_argument('--release-dims', type=int, default=harness.RELEASE_DIMS)
    ap.add_argument('--tiny-epochs', type=int, default=12)
    ap.add_argument('--adapter-epochs', type=int, default=40)
    ap.add_argument('--mlp-epochs', type=int, default=30)
    ap.add_argument('--min-payload-k', type=int, default=30)
    ap.add_argument('--refine-steps', type=int, default=600)
    ap.add_argument('--skip-adapter', action='store_true')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default=str(harness.HERE / 'results-artifact-size.json'))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    t_all = time.perf_counter()
    d = harness.Draw(case_seed=args.seed, release_dims=args.release_dims)
    d.check_oracle()
    x_pub, y_pub = d.pool_x[d.public_rows], d.pool_y[d.public_rows]
    rec = {'attack': 'artifact-size', 'draw': d.summary(),
           'caps': {'max_source_bytes': MAX_SOURCE_BYTES,
                    'max_literal_bytes': MAX_LITERAL_BYTES}}

    # ---- 1. class statistics
    k = args.release_dims
    stats = class_statistics(x_pub, y_pub, k)
    rec['class_statistics'] = {
        'k': k,
        'element_counts': {n: int(np.asarray(a).size) for n, a in stats.items()},
        'total_elements': int(sum(np.asarray(a).size for a in stats.values())),
        'int8': packaging(quantise(stats, 'int8'), f'class-stats k={k} int8'),
        'fp16': packaging(quantise(stats, 'fp16'), f'class-stats k={k} fp16'),
    }
    log(f'class statistics k={k}: {rec["class_statistics"]["total_elements"]} numbers, int8 '
        f'{rec["class_statistics"]["int8"]["bytes_in_submission"]} bytes in a submission '
        f'({rec["class_statistics"]["int8"]["fraction_of_20480_cap"]:.1f}x the whole cap)')

    # how far the attacker has to shrink k before the statistics alone fit
    sweep = []
    for kk in (10, 15, 20, 25, 30, 40, 50, 60):
        s = class_statistics(x_pub, y_pub, kk)
        p = packaging(quantise(s, 'int8'), f'k={kk}')
        sweep.append({'k': kk, 'bytes_in_submission': p['bytes_in_submission'],
                      'fits_alone': p['bytes_in_submission'] <= MAX_SOURCE_BYTES})
    rec['class_statistics_k_sweep'] = sweep
    log('k sweep (int8, bytes in submission): '
        + ', '.join(f'{s["k"]}:{s["bytes_in_submission"]}' for s in sweep))

    # ---- 2. offline model weights
    nets = {'study_CNN_32_64_64': models.CNN(), 'CNN_16_32_32': models.CNN(16, 32, 32),
            'CNN_8_16_16': models.CNN(8, 16, 16), 'TinyCNN_8_16_16_pool': TinyCNN()}
    rec['offline_models'] = {}
    for name, net in nets.items():
        n = param_count(net)
        # real (trained-scale) weights, one int8 scale per output row of each tensor
        tensors = {f'p{i}': (v.detach().numpy().reshape(v.shape[0], -1)
                             if v.ndim > 1 else v.detach().numpy()[None, :])
                   for i, v in enumerate(net.parameters())}
        p = packaging(quantise(tensors, 'int8'), name)
        p['parameters'] = n
        rec['offline_models'][name] = p
        log(f'{name}: {n} params, int8 {p["bytes_in_submission"]} bytes in a submission')

    # ---- 3. the attack code itself
    srcs = {
        'attacks.informed_attack': inspect.getsource(attacks.informed_attack),
        'attacks._class_stats': inspect.getsource(attacks._class_stats),
        'attacks._procrustes': inspect.getsource(attacks._procrustes),
        'attacks._skew_along': inspect.getsource(attacks._skew_along),
        'attacks.apply_linear': inspect.getsource(attacks.apply_linear),
        'harness.recover_label_permutation': inspect.getsource(harness.recover_label_permutation),
        'harness.invariant_class_features': inspect.getsource(harness.invariant_class_features),
        'models.CNN': inspect.getsource(models.CNN),
    }
    code_bytes = {n: len(s.encode()) for n, s in srcs.items()}
    rec['attack_code_bytes'] = {
        'per_function': code_bytes, 'total': int(sum(code_bytes.values())),
        'note': 'verbatim source of the functions the attack calls, comments and all; '
                'a minifier would cut this substantially, but it is the same order'}
    log(f'attack code: {rec["attack_code_bytes"]["total"]} bytes of source')

    # ---- verdict
    stat_bytes = rec['class_statistics']['int8']['bytes_in_submission']
    cnn_bytes = rec['offline_models']['study_CNN_32_64_64']['bytes_in_submission']
    total = stat_bytes + cnn_bytes
    rec['verdict'] = {
        'informed_attack_payload_bytes': total,
        'over_cap_factor': total / MAX_SOURCE_BYTES,
        'fits': total <= MAX_SOURCE_BYTES,
        'statement': (
            f'the informed attack needs {stat_bytes} bytes of class statistics plus '
            f'{cnn_bytes} bytes of int8 CNN weights, {total / MAX_SOURCE_BYTES:.1f}x the '
            f'{MAX_SOURCE_BYTES}-byte cap before a single line of code; no base64 packaging '
            'changes that, because base64 only inflates.')}
    log(rec['verdict']['statement'])

    # ---- 4. what DOES fit: a tiny CNN with no statistics, through a linear adapter
    if not args.skip_adapter:
        tiny_bytes = rec['offline_models']['TinyCNN_8_16_16_pool']['bytes_in_submission']
        log(f'training the cap-compliant TinyCNN ({tiny_bytes} bytes int8) offline on '
            f'{len(x_pub)} public rows')
        perm_hat, _ = harness.recover_label_permutation(
            d.z_train, d.y_train, x_pub, y_pub, k=args.release_dims)
        y_pub_rel = perm_hat[y_pub]
        t0 = time.perf_counter()
        flat = x_pub.reshape(len(x_pub), -1)
        st = models.Standardizer(flat)
        net = models._train(TinyCNN(), st(flat).reshape(-1, 1, 9, 9), y_pub_rel,
                            args.tiny_epochs, 128, 2e-3, args.seed)
        tiny_train_s = time.perf_counter() - t0
        ceiling = acc(models.predict(net, st(d.x_test).reshape(-1, 1, 9, 9)), d.y_test)
        d_rel = d.z_train.shape[1]
        zc = d.z_train.astype(np.float64) - d.z_train.mean(0)
        lam, ev = np.linalg.eigh(np.cov(zc, rowvar=False))
        white = (ev / np.sqrt(np.clip(lam, 0, None) + 1e-4)[None, :]) @ ev.T
        h0 = 0.25 * common.haar(81, args.seed + 5)[:, :d_rel] @ white
        c0 = np.full(81, 0.13)
        t0 = time.perf_counter()
        p, val = models.adapter_fit_predict(net, st, d.z_train, d.y_train, d.z_test, h0, c0,
                                            epochs=args.adapter_epochs, seed=args.seed)
        trained = {f'p{i}': (v.detach().numpy().reshape(v.shape[0], -1)
                             if v.ndim > 1 else v.detach().numpy()[None, :])
                   for i, v in enumerate(net.parameters())}
        tiny_trained = packaging(quantise(trained, 'int8'), 'TinyCNN trained int8')
        rec['cap_compliant_route'] = {
            'artifact_bytes_in_submission': tiny_trained['bytes_in_submission'],
            'artifact_packaging': tiny_trained,
            'tiny_cnn_parameters': rec['offline_models']['TinyCNN_8_16_16_pool']['parameters'],
            'tiny_cnn_train_seconds': tiny_train_s,
            'tiny_cnn_on_true_test_pixels': ceiling,
            'adapter_random_init_test_acc': acc(p, d.y_test),
            'adapter_val_acc': val,
            'adapter_seconds': time.perf_counter() - t0,
        }
        t0 = time.perf_counter()
        pm, _ = models.mlp_fit_predict(d.z_train, d.y_train, d.z_test, seed=args.seed,
                                       epochs=args.mlp_epochs)
        rec['cap_compliant_route']['honest_mlp_on_released_features'] = acc(pm, d.y_test)
        rec['cap_compliant_route']['honest_mlp_seconds'] = time.perf_counter() - t0
        # ---- 5. the cheapest payload that could plausibly fit: small-k class
        # statistics (so the map CAN be recovered) plus the same tiny CNN.
        kk = args.min_payload_k
        s_small = class_statistics(x_pub, y_pub, kk)
        stat_small = packaging(quantise(s_small, 'int8'), f'class-stats k={kk} int8')
        t0 = time.perf_counter()
        res = attacks.informed_attack(d.z_train, d.y_train, d.z_test, x_pub, y_pub_rel,
                                      init='all', seed=args.seed, log=log,
                                      refine_steps=args.refine_steps, hops=1, k_att=kk)
        pick = min(res, key=lambda n: res[n]['final_loss'])
        x_hat = attacks.apply_linear(res[pick]['refined'], d.z_test)
        active = x_pub.var(0) > 1e-4
        small_acc = acc(models.predict(net, st(x_hat).reshape(-1, 1, 9, 9)), d.y_test)
        payload = (stat_small['bytes_in_submission'] + tiny_trained['bytes_in_submission'])
        rec['minimal_payload_attack'] = {
            'k_att': kk,
            'class_statistics_bytes': stat_small['bytes_in_submission'],
            'tiny_cnn_bytes': tiny_trained['bytes_in_submission'],
            'payload_bytes': payload,
            'bytes_left_for_code': MAX_SOURCE_BYTES - payload,
            'verbatim_attack_code_bytes': rec['attack_code_bytes']['total'],
            'fits_with_verbatim_code': (payload + rec['attack_code_bytes']['total']
                                        <= MAX_SOURCE_BYTES),
            'attacker_pick': pick,
            'final_loss': res[pick]['final_loss'],
            'pixel_rel_rms': common.pixel_reconstruction_error(x_hat, d.x_test, active)['rel_rms'],
            'tiny_cnn_on_recovered_test_acc': small_acc,
            'seconds': time.perf_counter() - t0,
        }
        log(f'minimal payload (k={kk} stats {stat_small["bytes_in_submission"]}B + tiny CNN '
            f'{tiny_trained["bytes_in_submission"]}B = {payload}B, '
            f'{MAX_SOURCE_BYTES - payload}B left for code): recovered pixels rel-RMS '
            f'{rec["minimal_payload_attack"]["pixel_rel_rms"]:.3f}, tiny CNN on them '
            f'{small_acc:.2f}%')

        rec['cap_compliant_route']['gain_over_honest_mlp_points'] = (
            rec['cap_compliant_route']['adapter_random_init_test_acc']
            - rec['cap_compliant_route']['honest_mlp_on_released_features'])
        log(f'cap-compliant route: tiny CNN on true pixels {ceiling:.2f}%, through the adapter '
            f'{rec["cap_compliant_route"]["adapter_random_init_test_acc"]:.2f}%, honest MLP '
            f'{rec["cap_compliant_route"]["honest_mlp_on_released_features"]:.2f}%')

    rec['total_seconds'] = time.perf_counter() - t_all
    harness.jdump(args.out, rec)
    log(f'wrote {args.out} in {rec["total_seconds"]:.0f}s')


if __name__ == '__main__':
    main()
