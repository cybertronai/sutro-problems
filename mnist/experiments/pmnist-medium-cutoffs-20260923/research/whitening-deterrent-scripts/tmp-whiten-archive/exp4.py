import sys, numpy as np, time
sys.path.insert(0,'/tmp/whiten'); from common import *
import study, topology
SEED=2026092301
X=pool(); perm=np.asarray(study.feature_permutation())
order=study.draw_order(SEED); vis=order[:20000]
eps=1e-4; mu,W,Winv,ev=zca(X,eps); Z=(X-mu)@W; Q=rand_orth(777)
Y=(Z@Q.T)[:,perm][vis]                         # learner-visible, whitened+rotated+permuted

# ---- ATTACK R: re-identify rows against the PUBLIC pool (external data), then Procrustes
t=time.time()
Zref=Z                                          # attacker recomputes whitening from public pool
ny=np.linalg.norm(Y,axis=1); nz=np.linalg.norm(Zref,axis=1)
o=np.argsort(nz); nzs=nz[o]
j=np.searchsorted(nzs,ny); j=np.clip(j,1,len(nzs)-1)
pick=np.where(np.abs(nzs[j]-ny)<np.abs(nzs[j-1]-ny), j, j-1)
match=o[pick]
acc=(match==vis).mean()
resid=np.abs(nz[match]-ny)
print(f"ATTACK R row re-identification by rotation-invariant norm: {acc*100:.2f}% of 20000 rows correct ({time.time()-t:.1f}s)")
# Procrustes on the matched rows
A=Zref[match]; B=Y
U,s,Vt=np.linalg.svd(A.T@B); Qhat=U@Vt
Xhat=(B@Qhat.T)@Winv+mu
err=np.linalg.norm(Xhat-X[vis])/np.linalg.norm(X[vis])
print(f"ATTACK R Procrustes: ||Xhat-X||/||X|| = {err:.2e};  per-pixel min corr = "
      f"{min(np.corrcoef(Xhat[:,k],X[vis][:,k])[0,1] for k in range(81)):.4f}")
lay=topology.recover_layout(Xhat[:10000].astype(np.float32))
f,c=edge_preservation(lay,np.arange(81)); print(f"ATTACK R -> recover_layout on Xhat: {c}/144 ({f:.3f})")
