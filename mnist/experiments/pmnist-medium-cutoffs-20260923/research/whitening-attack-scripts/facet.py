"""Full zero-atom (cone-facet) attack: recover the pixels themselves, then the grid.

Label-free.  Uses only the released train rows + query images of the whitened,
randomly-rotated variant.  Ground truth is touched only to score the result.
"""
import sys, json, time, warnings, numpy as np
warnings.filterwarnings('ignore')
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R); sys.path.insert(0,R+'/research')
import torch; torch.set_num_threads(4)
import study, whiten, topology, attack_ica_topographic as atk
SEED=2026092301; N=int(sys.argv[1]); EIGFLOOR=float(sys.argv[2]); RESTARTS=int(sys.argv[3]); H=0.05
T=whiten.default_transform()
U=np.asarray(study.pool_images(),dtype=np.float64); m=U.mean(0)
idx=np.concatenate([study.train_indices(SEED,N),study.query_indices(SEED)])
a=whiten.job_arrays(SEED,N,T); Z=np.concatenate([a['train_x'],a['query_x']]).astype(np.float64)
Zc=Z-Z.mean(0); C=np.cov(Zc.T); d,E=np.linalg.eigh(C); d=np.clip(d,1e-12,None)
keep=d>EIGFLOOR
K=(E[:,keep]*(1/np.sqrt(d[keep]))[None,:]).T
Y=Zc@K.T; DIM=Y.shape[1]; X=U[idx]
print('N=%d  kept %d/81 directions (released cov eigenvalue > %g)'%(N,DIM,EIGFLOOR), flush=True)
Yt=torch.tensor(Y,dtype=torch.float32)
g=torch.Generator().manual_seed(0)
W=torch.randn(RESTARTS,DIM,generator=g); W=W/W.norm(dim=1,keepdim=True); W.requires_grad_(True)
q=torch.zeros(RESTARTS,requires_grad=True)
opt=torch.optim.Adam([W,q],lr=0.05); t0=time.time()
for step in range(400):
    Wn=W/W.norm(dim=1,keepdim=True)
    t=Yt@Wn.T                                   # (n, R)
    loss=-torch.exp(-0.5*((t-q[None,:])/H)**2).mean(0).sum()
    opt.zero_grad(); loss.backward(); opt.step()
    if step%200==0: print('  step %d  mean atom %.3f (%.0fs)'%(step,-loss.item()/RESTARTS,time.time()-t0), flush=True)
with torch.no_grad():
    Wn=(W/W.norm(dim=1,keepdim=True)).numpy().astype(np.float64); qq=q.numpy().astype(np.float64)
Tp=Y@Wn.T                                        # (n, R) projections
tz=(Tp-Tp.mean(0))/Tp.std(0)
atom=np.array([np.exp(-0.5*((tz[:,i]-((qq[i]-Tp[:,i].mean())/Tp[:,i].std()))/H)**2).mean() for i in range(RESTARTS)])
order=np.argsort(-atom); kept=[]
for i in order:
    if all(abs(np.corrcoef(tz[:,i],tz[:,j])[0,1])<0.9 for j in kept): kept.append(i)
kept=np.array(kept)
print('%d restarts -> %d distinct directions (|r|<0.9 apart), %.0fs'%(RESTARTS,len(kept),time.time()-t0), flush=True)
Xc=X-X.mean(0); Tk=tz[:,kept]
cc=np.abs(np.nan_to_num((Tk-Tk.mean(0)).T@Xc/len(X)/np.maximum(np.outer(Tk.std(0),X.std(0)),1e-300)))
bestpix=cc.argmax(1); bestr=cc.max(1)
good=bestr>0.9
print('recovered directions matching a raw pixel at |r|>0.9: %d  (>0.99: %d) | distinct pixels %d of 75 active'
      %(int(good.sum()), int((bestr>0.99).sum()), int(np.unique(bestpix[good]).size)), flush=True)
print('median |r| over kept directions %.3f'%np.median(bestr), flush=True)
# --- build a pixel-like feature matrix (label-free: sign from skew, shift to min 0)
feat=np.zeros((len(X), 81))
used={}
for j in range(len(kept)):
    t=Tk[:,j]; s=np.sign(((t-t.mean())**3).mean()) or 1.0
    t=t*s; t=t-t.min()
    p=int(bestpix[j])                      # ONLY used to dedupe/report, not to place
    if bestr[j]>0.9 and (p not in used or bestr[j]>used[p][0]): used[p]=(bestr[j], t)
slots=list(used.items())
for col,(p,(r,t)) in enumerate(slots): feat[:,col]=t/max(t.std(),1e-9)
n_used=len(slots)
print('assembled %d recovered pixel channels'%n_used, flush=True)
np.save('/tmp/whiten-attack/facet_feat_N%d.npy'%N, feat.astype(np.float32))
json.dump({'N':N,'dim':DIM,'restarts':RESTARTS,'distinct_directions':int(len(kept)),
           'matched_gt_0.9':int(good.sum()),'matched_gt_0.99':int((bestr>0.99).sum()),
           'distinct_pixels':int(np.unique(bestpix[good]).size),
           'median_abs_r':float(np.median(bestr)),'channels_assembled':n_used,
           'seconds':round(time.time()-t0,1)},
          open('/tmp/whiten-attack/facet_N%d.json'%N,'w'), indent=1)
