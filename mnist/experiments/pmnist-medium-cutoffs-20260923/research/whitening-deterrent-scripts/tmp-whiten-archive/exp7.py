import sys, numpy as np, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0,'/tmp/whiten'); from common import *
import study
SEED=2026092301
X=pool(); order=study.draw_order(SEED); vis=order[:20000]
act=np.where(X.var(0)>1e-4)[0]; d=len(act); print("active pixels:",d)
Xa=X[:,act]
mu=Xa.mean(0); C=np.cov(Xa-mu,rowvar=False); w,V=np.linalg.eigh(C)
epsr=1e-3*w.mean(); Wz=V@np.diag(1/np.sqrt(w+epsr))@V.T
Q=rand_orth(777,d)
M=Wz@Q.T                      # protocol map (centering + ZCA + random rotation)
Y=((Xa-mu)@M)[vis]
# attacker: sample-whiten (no external data)
m=Y.mean(0); Cy=np.cov(Y-m,rowvar=False); wy,Vy=np.linalg.eigh(Cy)
Wl=Vy@np.diag(1/np.sqrt(np.clip(wy,1e-9,None)))@Vy.T
Yw=(Y-m)@Wl
Tot=M@Wl                      # pixel-space filters = Tot@R

def score(R,tag):
    B=Tot@R; Bn=B/np.linalg.norm(B,axis=0,keepdims=True); a=np.abs(Bn)
    best=a.max(0); am=a.argmax(0)
    print(f"{tag}: max|coef| med={np.median(best):.3f}  axes>0.9={int((best>0.9).sum())}/{d}  distinct pixels={len(set(am.tolist()))}/{d}")
    return best

score(rand_orth(9,d),"random rotation control")
score(np.linalg.pinv(Tot),"oracle (true pixel axes)")

def minorthant(beta=20.0, iters=4000, seed=11, lr=0.3):
    R=rand_orth(seed,d); N=len(Yw)
    for it in range(iters):
        Cm=Yw@R
        z=-beta*Cm; z-=z.max(0,keepdims=True)
        e=np.exp(z); s=e.sum(0)
        # J = sum_k (1/beta) log(mean_i exp(-beta c_ik)) ; dJ/dc_ik = -e_ik/s_k
        G=Yw.T@(-e/s)
        R,_=np.linalg.qr(R-lr*G)
        if it%1500==1499: lr*=0.5
    return R
t=time.time()
best=None
for sd in (11,12):
    R=minorthant(seed=sd)
    J=float(sum((1/20.0)*np.log(np.mean(np.exp(-20*(Yw@R)[:,k]-np.max(-20*(Yw@R)[:,k])))+1e-300)+np.max(-20*(Yw@R)[:,k])/20 for k in range(d)))
    print(f"  seed{sd} min-orthant objective={J:.2f}")
    if best is None or J<best[0]: best=(J,R)
print("nonneg-rotation search %.0fs"%(time.time()-t))
score(best[1],"min-enclosing-orthant rotation (nonnegative ICA, no external data)")
