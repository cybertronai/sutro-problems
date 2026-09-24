import sys, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/tmp/whiten'); from common import *
import study
from sklearn.decomposition import FastICA
SEED=2026092301
X=pool(); perm=np.asarray(study.feature_permutation())
order=study.draw_order(SEED); vis=order[:20000]
eps=1e-4; mu,W,Winv,ev=zca(X,eps); Z=(X-mu)@W; Q=rand_orth(777)
P=np.eye(81)[:,perm]                     # (x@P)[:,j]=x[:,perm[j]]
M=W@Q.T@P
Y=((X-mu)@M)[vis]
Minv=np.linalg.inv(M)

def loc_stats(patterns, tag):
    """patterns: (k,81) in PIXEL space. report spatial concentration."""
    p2=patterns**2; p2=p2/p2.sum(1,keepdims=True)
    pr=1.0/ (p2**2).sum(1)                                  # participation ratio (1=one pixel,81=flat)
    com=p2@RC
    rad=np.sqrt(((p2[:,:,None]*(RC[None]-com[:,None,:])**2).sum(1)).sum(1))
    print(f"{tag}: participation ratio med={np.median(pr):.1f} (81=delocalised)  rms spatial radius med={np.median(rad):.2f} (uniform 9x9 = 3.67)")
    return com, pr, rad

rng=np.random.default_rng(3)
loc_stats(rand_orth(5)[:,:81].T@np.eye(81), "random orthogonal basis in pixel space")
loc_stats(np.eye(81), "pixel basis (ideal)")

ica=FastICA(n_components=81, whiten='unit-variance', max_iter=2000, tol=1e-4, random_state=0)
S=ica.fit_transform(Y)                    # (20000,81)
A=ica.mixing_                             # (81,81): Y-mu ~ S @ A.T
pix = A.T@Minv                            # (81,81): row k = pixel-space pattern of source k
print("FastICA converged in", ica.n_iter_, "iters")
com,pr,rad=loc_stats(pix, "FastICA components (after whiten+rotation)")

# TICA-style energy correlations -> 2D map, compare to true centres of mass
E=S**2; E=(E-E.mean(0))/E.std(0)
Ce=(E.T@E)/len(E); np.fill_diagonal(Ce,0)
D=np.sqrt(np.clip(Ce.max()-Ce,0,None))
Dc=D**2; J=np.eye(81)-1/81; B=-0.5*J@Dc@J
w2,V2=np.linalg.eigh(B); idx=np.argsort(-w2)[:2]
emb=V2[:,idx]*np.sqrt(np.clip(w2[idx],0,None))
# procrustes correlation between emb and true com (only for reasonably localised comps)
keep=np.where(pr<np.median(pr))[0]
def proc_corr(a,b):
    a=a-a.mean(0); b=b-b.mean(0)
    a/=np.linalg.norm(a); b/=np.linalg.norm(b)
    return float(np.linalg.svd(a.T@b)[1].sum())
print(f"TICA energy-correlation MDS vs true component centres: procrustes corr = {proc_corr(emb[keep],com[keep]):.3f} (1=perfect, ~0.3-0.5 = chance-ish)")
rr=rng.permutation(len(keep))
print(f"   shuffled control = {proc_corr(emb[keep],com[keep][rr]):.3f}")
np.save('/tmp/whiten/pix.npy',pix); np.save('/tmp/whiten/com.npy',com); np.save('/tmp/whiten/emb.npy',emb)
