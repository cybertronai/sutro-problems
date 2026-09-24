import sys, json, warnings, numpy as np
warnings.filterwarnings('ignore')
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R); sys.path.insert(0,R+'/research')
import attack_ica_topographic as atk, study, whiten
from scipy.optimize import linear_sum_assignment
SEED=2026092301
U = np.asarray(study.pool_images(), dtype=np.float64)
m = U.mean(0); Xc = U - m
C = np.cov(Xc.T); lam, V = np.linalg.eigh(C); lam = np.clip(lam,0,None)
eps = 1e-3
Wz = (V * (1/np.sqrt(lam+eps))[None,:]) @ V.T
Z = Xc @ Wz.T
cz = np.zeros((81,81))
sd_p = Xc.std(0); sd_z = Z.std(0)
cz = (Z.T @ Xc)/len(U)/np.maximum(np.outer(sd_z, sd_p),1e-12)
cz = np.nan_to_num(cz)
mx = np.abs(cz).max(1)
print('ZCA(eps=1e-3) ceiling: corr with best pixel  mean %.3f median %.3f  diag-is-argmax %d/81'
      % (mx.mean(), np.median(mx), int((np.abs(cz).argmax(1)==np.arange(81)).sum())), flush=True)
loc = atk.localisation(Wz)
print('ZCA rows localisation: top1 %.3f  3x3 %.3f' % (loc['mean_top1_energy'], loc['mean_window3x3_energy']), flush=True)
# also eps=0 (exact whitening, dropping null directions)
keep = lam > 1e-6
Wz0 = (V[:,keep] * (1/np.sqrt(lam[keep]))[None,:]) @ V[:,keep].T
Z0 = Xc @ Wz0.T
c0 = np.nan_to_num((Z0.T @ Xc)/len(U)/np.maximum(np.outer(Z0.std(0), sd_p),1e-12))
print('ZCA(eps=0, rank %d) ceiling: corr mean %.3f median %.3f' % (keep.sum(), np.abs(c0).max(1).mean(), np.median(np.abs(c0).max(1))), flush=True)
loc0 = atk.localisation(Wz0)
print('ZCA(eps=0) rows localisation: top1 %.3f  3x3 %.3f' % (loc0['mean_top1_energy'], loc0['mean_window3x3_energy']), flush=True)
# how close is the ICA solution to ZCA?
for N in (1000, 10000):
    stem='logcosh_unit-variance_81_N%d_s%d'%(N,SEED)
    try:
        S=np.load('/tmp/whiten-attack/cache/sources_%s.npy'%stem).astype(np.float64)
    except FileNotFoundError:
        S=np.load('/tmp/whiten-attack/sources_logcosh_unit-variance_81_N%d.npy'%N).astype(np.float64)
    idx=np.concatenate([study.train_indices(SEED,N), study.query_indices(SEED)])
    Xn = U[idx]-m
    F,*_ = np.linalg.lstsq(Xn, S, rcond=None)     # S = Xn @ F  -> analysis filters are F.T
    resid = float(np.abs(S - Xn@F).max()/np.abs(S).max())
    Fa = F.T
    Fa = Fa/np.linalg.norm(Fa,axis=1,keepdims=True)
    Wzn = Wz/np.linalg.norm(Wz,axis=1,keepdims=True)
    cos = np.abs(Fa @ Wzn.T)
    r,c = linear_sum_assignment(-cos)
    la = atk.localisation(F.T)
    print('N=%d ICA vs ZCA: matched |cos| mean %.3f median %.3f min %.3f (lstsq resid %.2e); ICA analysis-row top1 %.3f 3x3 %.3f'
          % (N, cos[r,c].mean(), np.median(cos[r,c]), cos[r,c].min(), resid, la['mean_top1_energy'], la['mean_window3x3_energy']), flush=True)
