import sys, time, numpy as np
sys.path.insert(0,'/tmp/whiten')
from common import *
import topology, study

SEED=2026092301
perm=np.asarray(study.feature_permutation()); inv=np.empty(81,np.int64); inv[perm]=np.arange(81)
X=pool(); order=study.draw_order(SEED); tr=order[:10000]

TRUE=np.zeros((81,81),bool)
for a,b in TRUE_EDGES: TRUE[a,b]=TRUE[b,a]=True

def pair_precision(F, perm_used, sqrt=True):
    """top-144 partial-correlation pairs vs the 144 true lattice edges."""
    A=np.sqrt(np.clip(F,0,None)) if sqrt else F
    v=A.var(0); act=np.where(v>1e-12)[0]
    P=topology.partial_correlation(A[:,act])
    S=np.zeros((81,81)); S[np.ix_(act,act)]=P
    iu=np.triu_indices(81,1)
    # map feature pair -> original pixel pair
    pin=perm_used  # feature j holds original pixel perm_used[j]
    ta=TRUE[pin[iu[0]],pin[iu[1]]]
    s=S[iu]
    top=np.argsort(-s)[:144]
    return ta[top].mean()

def try_layout(F, perm_used, tag):
    try:
        lay,det=topology.recover_layout(F.astype(np.float32),return_details=True)
        f,c=edge_preservation(lay,perm_used); return f"{c:3d}/144 ({f:.3f})"
    except Exception as e: return f"CRASH {type(e).__name__}"

print("chance top-144 precision = 144/3240 =", 144/3240)
raw=X[tr][:,perm]
print(f"A raw permuted        : pairprec(sqrt)={pair_precision(raw,perm):.3f}  pairprec(lin)={pair_precision(raw,perm,False):.3f}  layout={try_layout(raw,perm,'A')}",flush=True)

for eps in (1e-6,1e-4,1e-2):
    mu,W,Winv,ev=zca(X,eps)
    Z=(X-mu)@W
    Q=rand_orth(777); Y=Z@Q.T
    zp=Z[tr][:,perm]
    print(f"B eps={eps:<7g} ZCA only : pairprec(sqrt)={pair_precision(zp,perm):.3f}  pairprec(lin)={pair_precision(zp,perm,False):.3f}  layout={try_layout(zp,perm,'B')}",flush=True)
    yy=Y[tr]
    print(f"C eps={eps:<7g} ZCA+rot  : pairprec(sqrt)={pair_precision(yy,np.arange(81)):.3f}  pairprec(lin)={pair_precision(yy,np.arange(81),False):.3f}  layout={try_layout(yy,np.arange(81),'C')}",flush=True)
