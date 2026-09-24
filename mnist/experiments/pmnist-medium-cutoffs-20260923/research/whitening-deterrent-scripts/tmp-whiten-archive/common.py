import sys, numpy as np
ROOT="/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923"
sys.path.insert(0,ROOT)
import study, topology

GRID=9; NF=81
RC=np.array([(i//GRID,i%GRID) for i in range(NF)],float)
ADJ=(((RC[:,None,:]-RC[None,:,:])**2).sum(-1)<=1.01)
np.fill_diagonal(ADJ,False)
TRUE_EDGES=np.array(np.triu(ADJ,1).nonzero()).T   # (144,2) in ORIGINAL pixel index

def edge_preservation(layout, perm):
    """layout[j]=cell for feature j (feature space = permuted pixels).
    Map original pixel p -> feature perm-index -> cell, count true lattice edges kept."""
    inv=np.empty(NF,np.int64); inv[perm]=np.arange(NF)  # x[:,perm][:,inv]=x  -> feature of pixel p is inv[p]
    cell=layout[inv]           # cell assigned to original pixel p
    keep=ADJ[cell[TRUE_EDGES[:,0]],cell[TRUE_EDGES[:,1]]]
    return float(keep.mean()), int(keep.sum())

def pool():
    X=np.asarray(study.pool_images(),np.float64)
    return X

def zca(Xfit, eps):
    mu=Xfit.mean(0); C=np.cov(Xfit-mu, rowvar=False)
    w,V=np.linalg.eigh(C)
    W=V@np.diag(1.0/np.sqrt(np.clip(w,0,None)+eps))@V.T
    Winv=V@np.diag(np.sqrt(np.clip(w,0,None)+eps))@V.T
    return mu,W,Winv,w

def rand_orth(seed,d=NF):
    g=np.random.Generator(np.random.PCG64(seed)).standard_normal((d,d))
    Q,R=np.linalg.qr(g); return Q*np.sign(np.diag(R))
