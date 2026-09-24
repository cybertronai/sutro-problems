import sys, numpy as np
sys.path.insert(0,'/tmp/whiten'); from common import *
import study, topology
SEED=2026092301
X=pool(); perm=np.asarray(study.feature_permutation())
order=study.draw_order(SEED); tr=order[:10000]; vis=order[:20000]
eps=1e-4
mu,W,Winv,ev=zca(X,eps)
Z=(X-mu)@W
Q=rand_orth(777)
Y=(Z@Q.T)[:,perm]                     # what the learner sees (perm absorbed into rotation)

def canon(D):
    """4th-order cumulant eigenframe of a (n,d) sample; returns coords + eigvals."""
    m=D.mean(0); C=np.cov(D-m,rowvar=False); w,V=np.linalg.eigh(C)
    Wl=V@np.diag(1/np.sqrt(np.clip(w,1e-12,None)))@V.T
    Dw=(D-m)@Wl
    r2=(Dw**2).sum(1)
    M=(Dw*r2[:,None]).T@Dw/len(Dw)
    lam,U=np.linalg.eigh(M); idx=np.argsort(-lam); lam=lam[idx]; U=U[:,idx]
    c=Dw@U
    return c,lam,U,m,Wl

# attacker sees only the 20000 visible rows of Y
cy,lam_y,Uy,my,Wy = canon(Y[vis])
# reference: attacker's own public 9x9 MNIST pool, same 20000-row count for fairness
ref = X[np.random.default_rng(1).permutation(60000)[:20000]]
cz,lam_z,Uz,mz,Wz = canon(ref)
print("4th-order eigenvalue spectrum (ref): top5",np.round(lam_z[:5],2)," min5",np.round(lam_z[-5:],3))
gaps=np.abs(np.diff(lam_z)); print("min relative eigengap:",float((gaps/lam_z[:-1]).min()), " #gaps<1%:",int((gaps/lam_z[:-1]<0.01).sum()))
sk_y=(cy**3).mean(0); sk_z=(cz**3).mean(0)
d=np.sign(sk_y*sk_z); d[d==0]=1
print("|skew| ref: min",np.abs(sk_z).min().round(4)," median",np.median(np.abs(sk_z)).round(3))
# reconstruct pixels
Xhat = (cy*d)@np.linalg.pinv(Wz@Uz) + mz     # inverse of: (x-mz)Wz Uz
Xtrue = X[vis]
cc=np.array([np.corrcoef(Xhat[:,j],Xtrue[:,j])[0,1] for j in range(81)])
print("per-pixel corr(Xhat, Xtrue): median %.3f  min %.3f  #>0.9: %d/81"%(np.median(cc),np.nanmin(cc),int((cc>0.9).sum())))
print("relative reconstruction error:", float(np.linalg.norm(Xhat-Xtrue)/np.linalg.norm(Xtrue-Xtrue.mean(0))))
lay,det=topology.recover_layout(Xhat[:10000].astype(np.float32),return_details=True)
f,c_=edge_preservation(lay,np.arange(81)); print("recover_layout on Xhat: %d/144 (%.3f)"%(c_,f))
np.save('/tmp/whiten/Xhat.npy',Xhat)
