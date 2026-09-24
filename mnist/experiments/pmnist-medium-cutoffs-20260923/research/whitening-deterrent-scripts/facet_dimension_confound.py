"""Is the 'exact rank-75 whitening resists the facet attack better' result real,
or is it just a bigger search space?  Matched-dimension comparison."""
import sys, time, warnings, numpy as np
warnings.filterwarnings('ignore')
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R)
import torch; torch.set_num_threads(2)
import study, whiten
SEED=2026092301; N=10000; H=0.05; RESTARTS=24; STEPS=400

U=np.asarray(study.pool_images(),dtype=np.float64); mu=U.mean(0)
idx=np.concatenate([study.train_indices(SEED,N), study.query_indices(SEED)])
X=U[idx]; Xc=X-X.mean(0)

def probe(Y,tag):
    dim=Y.shape[1]
    Yt=torch.tensor(Y,dtype=torch.float32)
    g=torch.Generator().manual_seed(0)
    W=torch.randn(RESTARTS,dim,generator=g); W=W/W.norm(dim=1,keepdim=True); W.requires_grad_(True)
    q=torch.zeros(RESTARTS,requires_grad=True); opt=torch.optim.Adam([W,q],lr=0.05)
    t0=time.time()
    for s in range(STEPS):
        Wn=W/W.norm(dim=1,keepdim=True); t=Yt@Wn.T
        loss=-torch.exp(-0.5*((t-q[None,:])/H)**2).mean(0).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): Wn=(W/W.norm(dim=1,keepdim=True)).numpy().astype(np.float64)
    Tp=Y@Wn.T; Tc=Tp-Tp.mean(0)
    cc=np.abs(np.nan_to_num(Tc.T@Xc/len(X)/np.maximum(np.outer(Tp.std(0),X.std(0)),1e-300)))
    r=cc.max(1)
    print('%-42s dim=%2d  median|r| %.3f  max %.3f  #>0.9 %2d  #>0.99 %2d  (%.0fs)'
          %(tag,dim,np.median(r),r.max(),int((r>0.9).sum()),int((r>0.99).sum()),time.time()-t0),flush=True)
    return float(np.median(r))

def sample_white(Z, keep_rule):
    Zc=Z-Z.mean(0); C=np.cov(Zc.T); d,E=np.linalg.eigh(C); d=np.clip(d,1e-12,None)
    keep=keep_rule(d)
    K=(E[:,keep]*(1/np.sqrt(d[keep]))[None,:]).T
    return Zc@K.T

# ---- A: the study's default transform (eps=1e-3), attacker keeps different subspaces
T=whiten.default_transform()             # eps=1e-3, zca, Haar rotation
a=whiten.job_arrays(SEED,N,T); Z=np.concatenate([a['train_x'],a['queryize'] if False else a['query_x']]).astype(np.float64)
probe(sample_white(Z,lambda d: d>0.3),            'eps=1e-3, attacker keeps eig>0.3 (atom.py)')
probe(sample_white(Z,lambda d: np.argsort(np.argsort(-d))<75), 'eps=1e-3, attacker keeps top-75')
probe(sample_white(Z,lambda d: d>-1),             'eps=1e-3, attacker keeps all 81')

# ---- B: the exact rank-75 variant (rank75.py) at matched and reduced dimension
C=np.cov((U-mu).T); lam,V=np.linalg.eigh(C); lam=np.clip(lam,0,None)
sel=np.argsort(lam)[::-1][:75]
A0=(V[:,sel]*(1/np.sqrt(lam[sel]))[None,:]).T
Q=whiten.random_rotation(75,20260923); Z75=(X-mu)@(Q@A0).T
probe(Z75,'eps=0 rank-75 exact (rank75.py), all 75')
# attacker restricted to the SAME pixel-subspace dimension as the eig>0.3 run
k1=int((np.linalg.eigvalsh(np.cov((Z-Z.mean(0)).T))>0.3).sum())
sel2=np.argsort(lam)[::-1][:k1]
A02=(V[:,sel2]*(1/np.sqrt(lam[sel2]))[None,:]).T
Q2=whiten.random_rotation(k1,20260923); Zk=(X-mu)@(Q2@A02).T
probe(Zk,'eps=0 exact rank-%d, matched dim'%k1)

# ---- C: rotate_only (the variant the dense analysis recommends shipping)
Tr=whiten.fit_transform(U,epsilon=0.0,method='rotate_only',rotation_seed=20260923)
ar=whiten.job_arrays(SEED,N,Tr); Zr=np.concatenate([ar['train_x'],ar['query_x']]).astype(np.float64)
lamz=np.linalg.eigvalsh(np.cov((Zr-Zr.mean(0)).T))
probe(sample_white(Zr,lambda d: np.argsort(np.argsort(-d))<k1), 'rotate_only, attacker keeps top-%d'%k1)
probe(sample_white(Zr,lambda d: np.argsort(np.argsort(-d))<75), 'rotate_only, attacker keeps top-75')
