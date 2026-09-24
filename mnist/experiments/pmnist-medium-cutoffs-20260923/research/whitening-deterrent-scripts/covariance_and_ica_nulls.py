import sys, numpy as np, warnings
warnings.filterwarnings('ignore')
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R)
import study, whiten, attack_ica

pool=np.asarray(study.pool_images(),dtype=np.float64)
act=np.where(pool.var(0)>1e-4)[0]
print('active pixels:',act.size)

# --- (1) honest matched-energy null: Hungarian, 64 rows, restricted to active cols
for eps,meth in [(1e-2,'zca'),(0.0,'rotate_only')]:
    T=whiten.fit_transform(pool,epsilon=eps,method=meth,rotation_seed=20260923)
    A=T['A']
    peaks=[];matched=[]
    for s in range(5):
        Rr=whiten.random_rotation(81,1000+s)[:64]      # 64 random orthonormal rows
        comp=Rr@A
        peaks.append(attack_ica.signed_permutation_report(comp)['mean_peak_energy_fraction'])
        matched.append(attack_ica.matching_report(comp,act)['matched_energy_mean'])
    print('%-12s eps=%g  NULL peak-energy %.4f   NULL matched-energy(64 rows, 64 active cols) %.4f'
          %(meth,eps,np.mean(peaks),np.mean(matched)))

# --- (2) ZCA "ceiling" as reported vs restricted to active pixels vs rank-64 exact
for eps in (1e-2,0.0):
    C=np.cov(pool,rowvar=False); v,V=np.linalg.eigh(C)
    zca=(V*(1/np.sqrt(np.clip(v,0,None)+eps))[None,:])@V.T
    print('eps=%g  reported ceiling (81x81 Hungarian) %.4f   restricted-to-active %.4f'
          %(eps, attack_ica.matching_report(zca)['matched_energy_mean'],
            attack_ica.matching_report(zca,act)['matched_energy_mean']))
# rank-64 exact whitening = what a 64-component FastICA is actually constrained to
sel=np.argsort(v)[::-1][:64]
W64=(V[:,sel]*(1/np.sqrt(v[sel]))[None,:])@V[:,sel].T
print('rank-64 exact ZCA ceiling, active cols: %.4f'%attack_ica.matching_report(W64,act)['matched_energy_mean'])

# --- (3) how far is the released covariance from I at the recommended eps
for eps in (1e-2,1e-3,1e-4,0.0):
    T=whiten.fit_transform(pool,epsilon=eps,method='zca',rotation_seed=20260923)
    l=T['eigenvalues']; a=l/(l+eps) if eps>0 else np.ones(81)
    z=whiten.apply(pool,T).astype(np.float64); Cz=np.cov(z,rowvar=False)
    print('eps=%-6g achieved var: min %.2e median %.3f max %.4f ; #<0.99 %d ; max|offdiag Cov| %.3f ; cond(Cov_z) %.3g'
          %(eps,a.min(),np.median(a),a.max(),int((a<0.99).sum()),
            np.abs(Cz-np.diag(np.diag(Cz))).max(), np.linalg.cond(Cz)))
