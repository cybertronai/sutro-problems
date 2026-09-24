import sys, numpy as np, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0,'/tmp/whiten'); from common import *
import study
X=pool(); order=study.draw_order(2026092301); vis=order[:20000]
act=np.where(X.var(0)>1e-4)[0]; d=len(act)
Xa=X[:,act]; mu=Xa.mean(0)
C=np.cov(Xa-mu,rowvar=False); w,V=np.linalg.eigh(C)
Wz=V@np.diag(1/np.sqrt(w+1e-3*w.mean()))@V.T
Q=rand_orth(777,d); M=Wz@Q.T
Y0=((Xa-mu)@M)[vis]
m=Y0.mean(0); Cy=np.cov(Y0-m,rowvar=False); wy,Vy=np.linalg.eigh(Cy)
Wl=Vy@np.diag(1/np.sqrt(np.clip(wy,1e-9,None)))@Vy.T
Y=np.ascontiguousarray(((Y0-m)@Wl)[:8000]); N=len(Y)
Tot=M@Wl
def score(B,tag):
    P=Tot@B; a=np.abs(P/np.linalg.norm(P,axis=0,keepdims=True))
    b=a.max(0); print(f"{tag}: max|coef| med={np.median(b):.3f} axes>0.9={int((b>0.9).sum())}/{d} distinct={len(set(a.argmax(0).tolist()))}/{d}",flush=True)
Btrue=np.linalg.inv(Tot)@np.diag(1/mu)
score(np.eye(d),"identity (whitened axes)"); score(Btrue,"oracle")
print("oracle logdet=%.1f  violation=%.4f"%(np.linalg.slogdet(Btrue)[1], np.maximum(-(Y@Btrue)-1,0).mean()),flush=True)
B=0.15*np.eye(d); lam=1.0; lr=2e-3
mom=np.zeros_like(B)
t=time.time()
for it in range(6000):
    c=Y@B; r=np.maximum(-c-1.0,0.0)
    g=np.linalg.inv(B).T + 2*lam*(Y.T@r)/N
    mom=0.9*mom+g; B=B+lr*mom
    if it%1000==999:
        lam*=3.0
        print(f"  it{it+1} logdet={np.linalg.slogdet(B)[1]:.1f} viol={r.mean():.4f} lam={lam:.0f} {time.time()-t:.0f}s",flush=True)
score(B,"min-volume cone fit (nonneg, no external data)")
np.save('/tmp/whiten/B.npy',B)
