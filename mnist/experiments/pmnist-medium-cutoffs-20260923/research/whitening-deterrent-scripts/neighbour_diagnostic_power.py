"""Does whiten.neighbour_correlation_report have any power to detect a BAD transform?"""
import sys, numpy as np
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R)
import study, whiten
U=np.asarray(study.pool_images())
rows=np.asarray(study.draw_order(2026092301)[:20000])
perm=study.feature_permutation()
cases=[('permuted pixels (control)',None),
       ('zca eps=1e-3 + rotation  (the proposal)',dict(epsilon=1e-3,method='zca')),
       ('zca eps=1e-3, NO rotation (NOT a defence: 44% adjacency precision)',
        dict(epsilon=1e-3,method='zca',rotation='none')),
       ('zca eps=0,    NO rotation (NOT a defence)',
        dict(epsilon=0.0,method='zca',rotation='none'))]
for tag,kw in cases:
    if kw is None:
        z=U[rows][:,perm]
    else:
        T=whiten.fit_transform(U,rotation_seed=20260923,**kw); z=whiten.apply(U[rows],T)
    r=whiten.neighbour_correlation_report(z,perm)
    gap=abs(r['mean_abs_corr_neighbour']-r['mean_abs_corr_non_neighbour'])
    passes = gap < 0.1*r['mean_abs_corr_non_neighbour']
    print('%-66s nbr %.4f  non-nbr %.4f  ratio %.3f  -> test_rotation_destroys_neighbour_correlations %s'
          %(tag,r['mean_abs_corr_neighbour'],r['mean_abs_corr_non_neighbour'],
            r['mean_abs_corr_neighbour']/r['mean_abs_corr_non_neighbour'],
            'PASSES' if passes else 'fails'))
