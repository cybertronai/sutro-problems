"""Cone/zero-atom attack: find the rotation that makes coordinates zero-inflated.

MNIST pixels are >= 0 and 59% of all pixel values are EXACTLY 0, so in pixel space
the data cloud lies on the faces of a translated simplicial cone with 81 facets.
A linear map preserves that geometry, so the facets -- i.e. the pixel basis -- are
identifiable from the released data *without any independence assumption*.  This is
a strictly stronger prior than the one FastICA uses.  Here it is implemented as:
find the orthogonal Q that makes every coordinate of s = Y Q^T pile up on its own
lower bound (concave penalty above a low quantile).
"""
import sys, time, warnings, numpy as np
warnings.filterwarnings('ignore')
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R); sys.path.insert(0,R+'/research')
import torch; torch.set_num_threads(3)
import study, whiten, attack_ica_topographic as atk
from scipy.optimize import linear_sum_assignment

SEED=2026092301; N=int(sys.argv[1]) if len(sys.argv)>1 else 10000
T=whiten.default_transform()
U=np.asarray(study.pool_images(),dtype=np.float64); m=U.mean(0)
idx=np.concatenate([study.train_indices(SEED,N), study.query_indices(SEED)])
a=whiten.job_arrays(SEED,N,T); Z=np.concatenate([a['train_x'],a['query_x']]).astype(np.float64)
Zc=Z-Z.mean(0); C=np.cov(Zc.T); d,E=np.linalg.eigh(C); d=np.clip(d,1e-12,None)
Y=Zc@((E*(1/np.sqrt(d))[None,:])@E.T).T          # sample-white coordinates
X=U[idx]-m
def corrmat(A,B):
    A=A-A.mean(0); B=B-B.mean(0)
    return np.nan_to_num((A.T@B)/np.maximum(np.outer(np.linalg.norm(A,axis=0),np.linalg.norm(B,axis=0)),1e-300))
def score(S, name, secs=None):
    c=np.abs(corrmat(S,X)); r,cc=linear_sum_assignment(-c)
    print('%-28s matched |r| with raw pixel: mean %.3f median %.3f | best-match mean %.3f%s'
          %(name, c[r,cc].mean(), np.median(c[r,cc]), c.max(1).mean(),
            '' if secs is None else ' (%.0fs)'%secs), flush=True)
    return c
Yt=torch.tensor(Y, dtype=torch.float32)
best=None
for trial, seed in enumerate((0,1,2)):
    g=torch.Generator().manual_seed(seed)
    W=torch.randn(81,81,generator=g)*0.01; W.requires_grad_(True)
    opt=torch.optim.Adam([W], lr=0.02)
    t0=time.time()
    for step in range(400):
        Q=torch.linalg.matrix_exp(W-W.T)
        S=Yt@Q.T
        floor=torch.quantile(S, 0.005, dim=0, keepdim=True)
        u=torch.clamp(S-floor, min=0.0)
        J=torch.sqrt(u+1e-3).mean()
        opt.zero_grad(); J.backward(); opt.step()
        if step%100==0: print('  trial%d step%d J=%.5f'%(trial,step,J.item()), flush=True)
    with torch.no_grad():
        Q=torch.linalg.matrix_exp(W-W.T); S=(Yt@Q.T).numpy().astype(np.float64)
    c=score(S,'nonneg-cone trial%d'%trial, time.time()-t0)
    val=float(np.abs(c).max(1).mean())
    if best is None or val>best[0]: best=(val,S)
# reference points
score(Y,'released (no attack)')
Sica=np.load('/tmp/whiten-attack/cache/sources_logcosh_unit-variance_81_N%d_s%d.npy'%(N,SEED)).astype(np.float64) \
     if N==10000 else np.load('/tmp/whiten-attack/sources_logcosh_unit-variance_81_N1000.npy').astype(np.float64)
score(Sica,'FastICA logcosh')
np.save('/tmp/whiten-attack/nonneg_sources_N%d.npy'%N, best[1].astype(np.float32))
