"""Attacker knows the published protocol (mean, pool covariance, epsilon) but NOT Q.
He recomputes W himself from public MNIST, matches rows by ||W(x-mu)||, then
solves for Q by least squares."""
import sys, time, numpy as np
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R)
import study, whiten, topology
SEED=2026092301; N=10000
U=np.asarray(study.pool_images(),dtype=np.float64)
idx=np.concatenate([study.train_indices(SEED,N), study.query_indices(SEED)])
for tag,kw in [('rotate_only', dict(epsilon=0.0,method='rotate_only')),
               ('zca eps=1e-3',dict(epsilon=1e-3,method='zca')),
               ('zca eps=1e-2',dict(epsilon=1e-2,method='zca'))]:
    T=whiten.fit_transform(U,rotation_seed=20260923,**kw)
    a=whiten.job_arrays(SEED,N,T); Z=np.concatenate([a['train_x'],a['query_x']]).astype(np.float64)
    t0=time.time()
    # attacker rebuilds mu and W from the PUBLIC pool + published epsilon/method
    mu=U.mean(0); C=np.cov((U-mu).T); d,E=np.linalg.eigh(C); d=np.clip(d,0,None)
    W=np.eye(81) if kw['method']=='rotate_only' else (E*(1/np.sqrt(d+kw['epsilon']))[None,:])@E.T
    ref=np.linalg.norm((U-mu)@W.T,axis=1)          # ||W(x-mu)|| for all 60,000 public rows
    nz=np.linalg.norm(Z,axis=1)                    # rotation-invariant, so identical
    o=np.argsort(ref); p=np.clip(np.searchsorted(ref[o],nz),0,len(o)-1)
    best=np.empty(len(nz),dtype=np.int64)
    for i,pp in enumerate(p):
        lo,hi=max(0,pp-3),min(len(o),pp+4); c=o[lo:hi]; best[i]=c[np.argmin(np.abs(ref[c]-nz[i]))]
    acc=float((best==idx).mean())
    Xm=(U[best]-mu)@W.T
    Qhat=np.linalg.lstsq(Xm,Z,rcond=None)[0].T
    xhat=np.clip(Z@np.linalg.pinv(Qhat@W).T+mu,0,1)
    err=float(np.abs(xhat-U[idx]).max())
    lay=topology.recover_layout(np.ascontiguousarray(xhat[:N],dtype=np.float32),max_seconds=60.0)
    ex=100*max(float((rl[lay]==np.arange(81)).mean()) for rl in topology.DIHEDRAL)
    print('%-13s re-identified %6.2f%% of 20,000 rows | max|x_hat - x| %.2e | layout on recovered pixels exact %5.1f%% | %.1fs'
          %(tag,100*acc,err,ex,time.time()-t0), flush=True)
