import sys, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/tmp/whiten'); from common import *
import study
SEED=2026092301
X=pool(); perm=np.asarray(study.feature_permutation())
order=study.draw_order(SEED); vis=order[:20000]
eps=1e-4; mu,W,Winv,ev=zca(X,eps); Q=rand_orth(777)
P=np.eye(81)[:,perm]; M=W@Q.T@P; Minv=np.linalg.inv(M)
Y=((X-mu)@M)[vis]
# attacker re-whitens on its own 20000 rows (no external data used)
m=Y.mean(0); C=np.cov(Y-m,rowvar=False); w,V=np.linalg.eigh(C)
Wl=V@np.diag(1/np.sqrt(np.clip(w,1e-12,None)))@V.T
Yw=(Y-m)@Wl
Tot=M@Wl     # pixel -> attacker-white coords (up to the mean shift)

def pixel_patterns(R):
    """columns of R are attacker axes in white space; return (81,81) pixel-space filters"""
    return (np.linalg.inv(Tot)@R).T   # row k = pattern of axis k in pixel space? use filter view
def score(R,tag):
    B=Tot@R                      # pixel-space *filters*: c_k = (x-mu)@B[:,k]
    Bn=B/np.linalg.norm(B,axis=0,keepdims=True)
    p2=(Bn**2).T
    pr=1/ (p2**2).sum(1)
    best=np.abs(Bn).max(0)
    print(f"{tag}: filter participation-ratio med={np.median(pr):.2f}  max|coef| med={np.median(best):.3f}  "
          f"#axes matching a single pixel(|coef|>0.9)={int((best>0.9).sum())}/81")
    return pr

rng=np.random.default_rng(0)
score(np.eye(81),"random start (identity of white basis)")
best=None
for trial in range(3):
    R=rand_orth(100+trial)
    lr=0.5
    for it in range(3000):
        c=Yw@R
        g=Yw.T@(3*c**2)/len(Yw)            # d/dR mean(sum c^3)
        R,_=np.linalg.qr(R+lr*g)
        if it%1000==999: lr*=0.5
    obj=float((( Yw@R)**3).mean(0).sum())
    print(f"  trial{trial} skewness objective={obj:.2f}")
    if best is None or obj>best[0]: best=(obj,R)
score(best[1],"max-skewness rotation (3rd-order, no external data)")
# how well does each recovered axis line up with a distinct pixel?
B=Tot@best[1]; Bn=np.abs(B/np.linalg.norm(B,axis=0,keepdims=True))
am=Bn.argmax(0); print("distinct pixels claimed:",len(set(am.tolist())),"/81")
np.save('/tmp/whiten/Rskew.npy',best[1])
