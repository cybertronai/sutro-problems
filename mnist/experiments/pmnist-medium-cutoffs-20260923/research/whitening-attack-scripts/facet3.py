"""Fairest version of the facet attack: keep only the high-atom channels.

The smoothed atom mass is a LABEL-FREE quality signal and it is strongly
predictive of whether a direction is a genuine pixel, so an attacker would keep
only the top-scoring channels instead of all of them.
"""
import sys, json, time, warnings, numpy as np
warnings.filterwarnings('ignore')
Rt='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,Rt); sys.path.insert(0,Rt+'/research')
import torch; torch.set_num_threads(3)
import study, whiten, topology, attack_ica_topographic as atk
SEED=2026092301; N=1000; EPOCHS=60; H=0.05; RESTARTS=400; EIGFLOOR=0.3
T=whiten.default_transform()
U=np.asarray(study.pool_images(),dtype=np.float64)
idx=np.concatenate([study.train_indices(SEED,N),study.query_indices(SEED)])
a=whiten.job_arrays(SEED,N,T); Z=np.concatenate([a['train_x'],a['query_x']]).astype(np.float64)
Zc=Z-Z.mean(0); C=np.cov(Zc.T); d,E=np.linalg.eigh(C); d=np.clip(d,1e-12,None)
keep=d>EIGFLOOR; K=(E[:,keep]*(1/np.sqrt(d[keep]))[None,:]).T
Y=Zc@K.T; DIM=Y.shape[1]; X=U[idx]
Yt=torch.tensor(Y,dtype=torch.float32); g=torch.Generator().manual_seed(0)
W=torch.randn(RESTARTS,DIM,generator=g); W=W/W.norm(dim=1,keepdim=True); W.requires_grad_(True)
q=torch.zeros(RESTARTS,requires_grad=True); opt=torch.optim.Adam([W,q],lr=0.05)
for step in range(400):
    Wn=W/W.norm(dim=1,keepdim=True); t=Yt@Wn.T
    loss=-torch.exp(-0.5*((t-q[None,:])/H)**2).mean(0).sum()
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad(): Wn=(W/W.norm(dim=1,keepdim=True)).numpy().astype(np.float64); qq=q.numpy()
Tp=Y@Wn.T; tz=(Tp-Tp.mean(0))/Tp.std(0)
atom=np.array([np.exp(-0.5*((tz[:,i]-(qq[i]-Tp[:,i].mean())/Tp[:,i].std())/H)**2).mean() for i in range(RESTARTS)])
kept=[]
for i in np.argsort(-atom):
    if all(abs(np.corrcoef(tz[:,i],tz[:,j])[0,1])<0.9 for j in kept): kept.append(i)
kept=np.array(kept)
Xc=X-X.mean(0)
out={}
for TOPK in (15,25,40):
    sel=kept[:TOPK]
    feat=np.zeros((len(X),81))
    for col,i in enumerate(sel):
        t=tz[:,i]*(np.sign(((tz[:,i]-tz[:,i].mean())**3).mean()) or 1.0); t=t-t.min()
        feat[:,col]=t/max(t.std(),1e-9)
    F=feat[:,:TOPK]
    cc=np.abs(np.nan_to_num((F-F.mean(0)).T@Xc/len(X)/np.maximum(np.outer(F.std(0),X.std(0)),1e-300)))
    truecell=np.zeros(81,dtype=np.int64); truecell[:TOPK]=cc.argmax(1)
    hits=int((cc.max(1)>0.9).sum())
    try:
        lay=topology.recover_layout(np.ascontiguousarray(feat[:N]), max_seconds=90.0); crashed=False
    except Exception:
        Wsim=np.clip(topology.partial_correlation(np.sqrt(np.clip(feat[:N],0,None))),0.0,None)
        lay,_=atk.layout_from_similarity(Wsim, max_seconds=45.0); crashed=True
    imgs=topology.unpermute(np.ascontiguousarray(feat,dtype=np.float32), lay)
    y_t=np.load(Rt+'/raw/pool_labels.npy')[study.train_indices(SEED,N)]; y_q=atk.dev_query_labels(SEED)
    r=atk.train_cnn(imgs[:N], y_t, imgs[N:], epochs=EPOCHS, member_seed=101)
    e=atk.error_rate(r['predictions'],y_q)
    out['top%d'%TOPK]={'channels':TOPK,'pixels_at_r_gt_0.9':hits,
                       'recover_layout_crashed':crashed,'cnn_error':e}
    print('top%-3d channels: %2d/%d are true pixels (|r|>0.9); cnn err=%.4f'%(TOPK,hits,TOPK,e), flush=True)
json.dump(out, open('/tmp/whiten-attack/facet3_N1000.json','w'), indent=1)
