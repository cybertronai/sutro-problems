import sys, numpy as np, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0,'/tmp/whiten'); from common import *
import study
X=pool(); order=study.draw_order(2026092301); vis=order[:20000]
act=np.where(X.var(0)>1e-4)[0]; d=len(act); print("active pixels:",d,flush=True)
Xa=X[:,act]
mu=Xa.mean(0); C=np.cov(Xa-mu,rowvar=False); w,V=np.linalg.eigh(C)
epsr=1e-3*w.mean(); Wz=V@np.diag(1/np.sqrt(w+epsr))@V.T
Q=rand_orth(777,d); M=Wz@Q.T
Y=((Xa-mu)@M)[vis]
m=Y.mean(0); Cy=np.cov(Y-m,rowvar=False); wy,Vy=np.linalg.eigh(Cy)
Wl=Vy@np.diag(1/np.sqrt(np.clip(wy,1e-9,None)))@Vy.T
Yw=np.ascontiguousarray(((Y-m)@Wl)[:8000])      # 8000 rows is plenty and 2.5x faster
Tot=M@Wl
def score(R,tag):
    B=Tot@R; a=np.abs(B/np.linalg.norm(B,axis=0,keepdims=True))
    best=a.max(0); am=a.argmax(0)
    print(f"{tag}: max|coef| med={np.median(best):.3f} axes>0.9={int((best>0.9).sum())}/{d} distinct={len(set(am.tolist()))}/{d}",flush=True)
score(rand_orth(9,d),"random rotation control")
score(np.linalg.inv(Tot),"oracle (true pixel axes)")
def J(R,beta=20.0):
    c=Yw@R; z=-beta*c; mx=z.max(0,keepdims=True)
    return float(((np.log(np.exp(z-mx).mean(0))+mx[0])/beta).sum())
def search(seed,iters=1200,beta=20.0,lr=0.5):
    R=rand_orth(seed,d); t=time.time()
    for it in range(iters):
        c=Yw@R; z=-beta*c; z-=z.max(0,keepdims=True)
        e=np.exp(z); e/=e.sum(0)
        R,_=np.linalg.qr(R+lr*(Yw.T@e))
        if it%400==399:
            lr*=0.5; print(f"   it{it+1} J={J(R):.2f} {time.time()-t:.0f}s",flush=True)
    return R
best=None
for sd in (11,12):
    R=search(sd); v=J(R); print(f" seed{sd} J={v:.2f}",flush=True)
    if best is None or v<best[0]: best=(v,R)
print("J(oracle) =",J(np.linalg.inv(Tot)),flush=True)
score(best[1],"min-enclosing-orthant rotation (no external data)")
