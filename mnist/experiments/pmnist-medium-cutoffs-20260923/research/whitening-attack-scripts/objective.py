import sys, warnings, numpy as np
warnings.filterwarnings('ignore')
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R); sys.path.insert(0,R+'/research')
import study, whiten
SEED=2026092301; N=10000
U=np.asarray(study.pool_images(),dtype=np.float64); m=U.mean(0)
Cp=np.cov((U-m).T); lam,V=np.linalg.eigh(Cp); lam=np.clip(lam,0,None)
Wz=(V*(1/np.sqrt(lam+1e-3))[None,:])@V.T          # pixel-aligned white basis (ZCA)
T=whiten.default_transform()
idx=np.concatenate([study.train_indices(SEED,N),study.query_indices(SEED)])
X=(U[idx]-m)                                       # raw centred pixels, same rows
Zr=X@T['A'].T                                      # released features
# sample whitening of the released data (exactly FastICA's feasible set)
Zc=Zr-Zr.mean(0); C=np.cov(Zc.T); d,E=np.linalg.eigh(C); d=np.clip(d,1e-12,None)
K=(E*(1/np.sqrt(d))[None,:])@E.T                   # sample whitener
Yw=Zc@K.T                                          # white coordinates; any orthogonal Q of Yw is feasible
def lowdin(M):
    u,s,vt=np.linalg.svd(M); return u@vt
def contrast(Y, kind):
    y=(Y-Y.mean(0))/Y.std(0)
    if kind=='logcosh':
        g=np.log(np.cosh(y)).mean(0); ref=0.37456866      # E log cosh N(0,1)
    elif kind=='cube':
        g=(y**4).mean(0)/4.0; ref=0.75
    return float((((g-ref))**2).sum()), float(np.abs(g-ref).sum())
# ICA solution
Sica=np.load('/tmp/whiten-attack/cache/sources_logcosh_unit-variance_81_N%d_s%d.npy'%(N,SEED)).astype(np.float64)
# ZCA-pixel basis expressed as an orthogonal transform of Yw, then Lowdin-orthogonalised
Zpix=X@Wz.T
B,*_=np.linalg.lstsq(Yw, Zpix-Zpix.mean(0), rcond=None)   # Zpix ~ Yw @ B
Q=lowdin(B)
Zpix_feasible=Yw@Q
print('ZCA-basis reconstruction from white coords: rel resid %.3e ; |B - Q|_max %.3f'
      % (np.abs((Zpix-Zpix.mean(0))-Yw@B).max()/np.abs(Zpix).max(), np.abs(B-Q).max()), flush=True)
for kind in ('logcosh','cube'):
    a=contrast(Sica,kind); b=contrast(Zpix_feasible,kind); c=contrast(Yw,kind)
    print('%-8s contrast sum-of-squares: FastICA %.4g | pixel-aligned(ZCA) %.4g | untouched released %.4g'
          %(kind,a[0],b[0],c[0]), flush=True)
    print('         sum |E g - Eg(N)|: FastICA %.4g | pixel-aligned %.4g | released %.4g'%(a[1],b[1],c[1]), flush=True)
