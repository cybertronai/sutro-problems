import sys, warnings, numpy as np
warnings.filterwarnings('ignore')
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R); sys.path.insert(0,R+'/research')
import attack_ica_topographic as atk, study
from scipy.optimize import linear_sum_assignment
SEED=2026092301
U=np.asarray(study.pool_images(),dtype=np.float64); m=U.mean(0)
C=np.cov((U-m).T); lam,V=np.linalg.eigh(C); lam=np.clip(lam,0,None)
Wz=(V*(1/np.sqrt(lam+1e-3))[None,:])@V.T
def corrmat(Aa,Bb):
    a=Aa-Aa.mean(0); b=Bb-Bb.mean(0)
    return np.nan_to_num((a.T@b)/np.maximum(np.outer(np.linalg.norm(a,axis=0),np.linalg.norm(b,axis=0)),1e-300))
for N in (1000,10000):
    stem='logcosh_unit-variance_81_N%d_s%d'%(N,SEED)
    try: S=np.load('/tmp/whiten-attack/cache/sources_%s.npy'%stem).astype(np.float64)
    except FileNotFoundError: S=np.load('/tmp/whiten-attack/sources_logcosh_unit-variance_81_N%d.npy'%N).astype(np.float64)
    idx=np.concatenate([study.train_indices(SEED,N),study.query_indices(SEED)])
    X=U[idx]-m; Z=X@Wz.T
    cz=np.abs(corrmat(S,Z)); cp=np.abs(corrmat(S,X))
    r,c=linear_sum_assignment(-cz)
    print('N=%d  ICA source vs ZCA-pixel: matched |r| mean %.3f median %.3f  | best-match |r| mean %.3f'
          %(N,cz[r,c].mean(),np.median(cz[r,c]),cz.max(1).mean()),flush=True)
    r2,c2=linear_sum_assignment(-cp)
    print('        ICA source vs RAW pixel: matched |r| mean %.3f median %.3f | best |r| mean %.3f (ZCA ceiling 0.896)'
          %(cp[r2,c2].mean(),np.median(cp[r2,c2]),cp.max(1).mean()),flush=True)
    # kurtosis of ICA sources vs ZCA-pixel signals: which basis is "more independent"?
    def kurt(A):
        a=(A-A.mean(0))/A.std(0); return (a**4).mean(0)-3
    print('        mean excess kurtosis: ICA %.1f  ZCA-pixels %.1f  raw pixels(active) %.1f'
          %(kurt(S).mean(),kurt(Z).mean(),kurt(X[:,X.var(0)>1e-5]).mean()),flush=True)
