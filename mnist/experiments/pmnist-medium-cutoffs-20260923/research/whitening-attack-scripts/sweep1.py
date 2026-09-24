import sys, json, time, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923')
sys.path.insert(0,'/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923/research'); import attack_ica_topographic as atk, whiten, study

T = whiten.default_transform()
print('conventions', atk.verify_conventions(T), flush=True)
print('synthetic', atk.synthetic_check(), flush=True)

# null reference: random orthogonal filters
rng = np.random.default_rng(0)
G = rng.standard_normal((500, 81))
print('random-direction top1 energy mean %.4f  window3x3 %.4f' %
      (atk.localisation(G)['mean_top1_energy'], atk.localisation(G)['mean_window3x3_energy']), flush=True)

N = int(sys.argv[1]) if len(sys.argv)>1 else 1000
pool = atk.unlabeled_pool(2026092301, N, T)['pool']
print('N=%d pool rows %d' % (N, pool.shape[0]), flush=True)
rows = []
for fun in ('logcosh','cube'):
    for wmode in (False, 'unit-variance'):
        for k in (81, 64, 50):
            if wmode is False and k != 81:
                continue
            t0 = time.time()
            fit = atk.ica_decompose(pool, T, fun=fun, whiten_mode=wmode, random_state=0,
                                    max_iter=1500, tol=1e-5, n_components=k)
            syn = atk.localisation(fit['synthesis'])
            ana = atk.localisation(atk.variance_weighted(fit['analysis']))
            corr = atk.pixel_correlation(fit['sources'], 2026092301, N)
            cstat = atk.localisation(corr)
            row = dict(fun=fun, whiten=str(wmode), k=k, n_iter=fit['n_iter'],
                       converged=fit['converged'], seconds=round(fit['seconds'],1),
                       syn_top1=round(syn['mean_top1_energy'],3),
                       syn_win=round(syn['mean_window3x3_energy'],3),
                       syn_top1_gt50=syn['n_components_top1_above_0.5'],
                       ana_top1=round(ana['mean_top1_energy'],3),
                       corr_top1=round(cstat['mean_top1_energy'],3),
                       corr_max_mean=round(float(np.abs(corr).max(1).mean()),3),
                       corr_max_med=round(float(np.median(np.abs(corr).max(1))),3),
                       distinct_corr_peak=int(np.unique(np.abs(corr).argmax(1)).size),
                       distinct_syn_peak=syn['distinct_peak_pixels'])
            rows.append(row); print(json.dumps(row), flush=True)
            np.save('/tmp/whiten-attack/sources_%s_%s_%d_N%d.npy' % (fun, wmode, k, N), fit['sources'].astype(np.float32))
            np.save('/tmp/whiten-attack/syn_%s_%s_%d_N%d.npy' % (fun, wmode, k, N), fit['synthesis'])
            np.save('/tmp/whiten-attack/corr_%s_%s_%d_N%d.npy' % (fun, wmode, k, N), corr)
json.dump(rows, open('/tmp/whiten-attack/sweep1_N%d.json'%N,'w'), indent=1)
