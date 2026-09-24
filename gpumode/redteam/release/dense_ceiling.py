"""Best-known dense learner on the harness's real release: arc-cosine depth-3 kernel ridge.

Evidence for band calibration, not an attack. Uses eval.make_draw(release_dims=60) draws
exactly as the leaderboard builds them (case seed 101, TIMED_STRIDE offsets) and the
pmnist-medium-cutoffs study's classical.fit_predict (CPU). Labels are used only to score.
Run: MNIST_POOL_CACHE=/tmp/gpumode-pool /tmp/penv/bin/python dense_ceiling.py [n_draws]
"""
import json, sys, time
from pathlib import Path
import numpy as np
G = Path('/Users/yaroslavvb/git/sutro-problems/gpumode')
S = Path('/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923')
sys.path.insert(0, str(G)); sys.path.insert(0, str(S))
import mnist_data, eval as harness, classical
n_draws = int(sys.argv[1]) if len(sys.argv) > 1 else 3
pool = mnist_data.load_pool(Path('/tmp/gpumode-pool'), 'mnist', 9)
case_seed = 101
universes = harness.split_universes(len(pool[1]), case_seed, harness.UNIVERSE_SALT)
config = {'family': 'kernel_ridge', 'kernel': 'arccos1', 'depth': 3,
          'lambda_grid': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3], 'cv_subsample': 4000, 'predict_block': 2000, 'kernel_block': 2000}
rows = []
for i in range(n_draws):
    draw_seed = case_seed + harness.TIMED_STRIDE * (i + 1)
    out = {}
    for dims in (60, 0):
        (tx, ty, qx), truth = harness.make_draw(pool, draw_seed, 10000, 10000, universes, release_dims=dims)
        tx2, qx2 = tx.reshape(len(tx), -1), qx.reshape(len(qx), -1)
        t = time.time()
        res = classical.fit_predict(tx2.astype(np.float32), ty.astype(np.uint8), qx2.astype(np.float32),
                                    {**config, 'normalization': 'none'} if dims else config, 11)
        acc = float(np.mean(res['labels'] == truth))
        out[dims] = {'accuracy': acc, 'seconds': round(time.time() - t, 1), 'chosen': res['metrics'].get('chosen_lambda')}
        print(f'draw {i} seed {draw_seed} release_dims={dims}: acc {acc:.4f} ({out[dims]["seconds"]} s)', flush=True)
    rows.append({'draw': i, 'seed': draw_seed, 'release60': out[60], 'pixels': out[0]})
summary = {'entry': 'arc-cosine depth-3 kernel ridge (pmnist-medium-cutoffs classical.py)', 'draws': rows,
           'mean_release60': float(np.mean([r['release60']['accuracy'] for r in rows])),
           'mean_pixels': float(np.mean([r['pixels']['accuracy'] for r in rows]))}
Path(G / 'results/release-dense-ceiling.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'draws'}))
