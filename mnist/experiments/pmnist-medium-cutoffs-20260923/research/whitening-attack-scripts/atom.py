"""Single-facet probe: can ONE pixel direction be found from the released data?

If pixel p is zero in a fraction f_p of the images, then in pixel space the
functional x -> x_p has an atom of mass f_p at its minimum.  Atoms survive any
invertible linear map, so in the released data there exists a direction w whose
projection has an atom of mass f_p.  Random directions have essentially no atom.
This searches for one such direction by gradient ascent on a smoothed atom mass --
a far easier problem than the 81-dimensional rotation, so it bounds how findable
the cone facets are in practice.
"""
import sys, time, warnings, numpy as np
warnings.filterwarnings('ignore')
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R); sys.path.insert(0,R+'/research')
import torch; torch.set_num_threads(4)
import study, whiten
SEED=2026092301; N=10000; H=0.05
T=whiten.default_transform()
U=np.asarray(study.pool_images(),dtype=np.float64); m=U.mean(0)
idx=np.concatenate([study.train_indices(SEED,N),study.query_indices(SEED)])
a=whiten.job_arrays(SEED,N,T); Z=np.concatenate([a['train_x'],a['query_x']]).astype(np.float64)
Zc=Z-Z.mean(0); C=np.cov(Zc.T); d,E=np.linalg.eigh(C); d=np.clip(d,1e-12,None)
keep = d > 0.3          # drop the epsilon-floored (near-dead-pixel) directions
K=(E[:,keep]*(1/np.sqrt(d[keep]))[None,:]).T          # (k,81): Y = Zc @ K.T
Y=Zc@K.T; X=U[idx]
print('kept %d of 81 released-covariance directions (eigenvalue > 0.3)'%int(keep.sum()), flush=True)
print('zero rate per pixel: mean %.2f max %.2f (active pixels: %d)'
      %((X==0).mean(), (X==0).mean(0).max(), int((X.var(0)>1e-5).sum())), flush=True)
# ground truth: what does the atom objective score for TRUE pixel directions?
DIM=Y.shape[1]
def atom_of(t, h=H):
    t=(t-t.mean())/t.std(); qs=np.quantile(t, np.linspace(0.001,0.999,400))
    return float(np.max([np.exp(-0.5*((t-q)/h)**2).mean() for q in qs]))
act=np.where(X.var(0)>1e-5)[0]
true_scores=[atom_of(X[:,p]) for p in act[:12]]
print('atom score of true pixel projections (first 12 active): mean %.3f min %.3f'
      %(np.mean(true_scores), np.min(true_scores)), flush=True)
rng=np.random.default_rng(0)
rand=[atom_of(Y@rng.standard_normal(DIM)) for _ in range(12)]
print('atom score of random directions: mean %.3f max %.3f'%(np.mean(rand), np.max(rand)), flush=True)
Yt=torch.tensor(Y,dtype=torch.float32)
best=[]
t0=time.time()
for trial in range(24):
    g=torch.Generator().manual_seed(trial)
    w=torch.randn(DIM,generator=g); w=w/w.norm(); w.requires_grad_(True)
    q=torch.tensor(0.0,requires_grad=True)
    opt=torch.optim.Adam([w,q],lr=0.05)
    for step in range(300):
        wn=w/w.norm(); t=Yt@wn
        loss=-torch.exp(-0.5*((t-q)/H)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        wn=(w/w.norm()).numpy().astype(np.float64); t=Y@wn
    s=atom_of(t)
    # which pixel is this direction closest to?
    cc=np.abs(np.corrcoef(np.column_stack([t, X]).T)[0,1:])
    cc=np.nan_to_num(cc)
    best.append((s, float(cc.max()), int(cc.argmax())))
best.sort(reverse=True)
print('24 restarts, %.0fs.  top 6 (atom score, |r| with best pixel, pixel):'%(time.time()-t0), flush=True)
for s,c,p in best[:6]: print('   atom %.3f  |r| %.3f  pixel %d'%(s,c,p), flush=True)
print('median over restarts: atom %.3f  |r| %.3f'
      %(np.median([b[0] for b in best]), np.median([b[1] for b in best])), flush=True)
