"""Design question: does whitening EXACTLY (no epsilon floor) help the defender?

Builds the sensible eps=0 variant -- drop the null directions, whiten exactly on
the 75-dimensional range, rotate -- and reruns the single-facet probe there.  In
that variant the released covariance is exactly the identity, so the defender has
removed the last second-order leak; the question is what it does to the attacker.
"""
import sys, time, warnings, numpy as np
warnings.filterwarnings('ignore')
Rt='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,Rt); sys.path.insert(0,Rt+'/research')
import torch; torch.set_num_threads(2)
import study, whiten
SEED=2026092301; N=10000; H=0.05; RANK=75
U=np.asarray(study.pool_images(),dtype=np.float64); m=U.mean(0)
idx=np.concatenate([study.train_indices(SEED,N),study.query_indices(SEED)])
C=np.cov((U-m).T); lam,V=np.linalg.eigh(C); lam=np.clip(lam,0,None)
sel=np.argsort(lam)[::-1][:RANK]
A0=(V[:,sel]*(1/np.sqrt(lam[sel]))[None,:]).T          # exact whitening on the range
Q=whiten.random_rotation(RANK, 20260923)
A=Q@A0
X=U[idx]; Y=(X-m)@A.T
print('released rank-%d variant: cov eig min %.4f max %.4f'
      %(RANK,*np.linalg.eigvalsh(np.cov(Y.T))[[0,-1]]), flush=True)
def atom_of(t,h=H):
    t=(t-t.mean())/t.std(); qs=np.quantile(t,np.linspace(0.001,0.999,400))
    return float(np.max([np.exp(-0.5*((t-q)/h)**2).mean() for q in qs]))
rng=np.random.default_rng(0)
print('random-direction atom score mean %.3f'%np.mean([atom_of(Y@rng.standard_normal(RANK)) for _ in range(8)]), flush=True)
Yt=torch.tensor(Y,dtype=torch.float32); best=[]
t0=time.time()
g=torch.Generator().manual_seed(0)
Wm=torch.randn(24,RANK,generator=g); Wm=Wm/Wm.norm(dim=1,keepdim=True); Wm.requires_grad_(True)
q=torch.zeros(24,requires_grad=True); opt=torch.optim.Adam([Wm,q],lr=0.05)
for step in range(400):
    Wn=Wm/Wm.norm(dim=1,keepdim=True); t=Yt@Wn.T
    loss=-torch.exp(-0.5*((t-q[None,:])/H)**2).mean(0).sum()
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad(): Wn=(Wm/Wm.norm(dim=1,keepdim=True)).numpy().astype(np.float64)
Tp=Y@Wn.T; Xc=X-X.mean(0); Tc=Tp-Tp.mean(0)
cc=np.abs(np.nan_to_num(Tc.T@Xc/len(X)/np.maximum(np.outer(Tp.std(0),X.std(0)),1e-300)))
r=cc.max(1)
print('eps=0 rank-75 variant, 24 restarts (%.0fs): |r| with best raw pixel  median %.3f  max %.3f  #>0.99 %d'
      %(time.time()-t0, np.median(r), r.max(), int((r>0.99).sum())), flush=True)
