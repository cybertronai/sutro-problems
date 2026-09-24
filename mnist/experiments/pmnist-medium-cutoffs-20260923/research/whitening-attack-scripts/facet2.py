"""Zero-atom facet attack, end to end and fully label-free in the attack path."""
import sys, json, time, warnings, numpy as np
warnings.filterwarnings('ignore')
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R); sys.path.insert(0,R+'/research')
import torch; torch.set_num_threads(3)
import study, whiten, topology, attack_ica_topographic as atk
SEED=2026092301; N=int(sys.argv[1]); EPOCHS=int(sys.argv[2]); EIGFLOOR=0.3; RESTARTS=400; H=0.05
T=whiten.default_transform()
U=np.asarray(study.pool_images(),dtype=np.float64)
idx=np.concatenate([study.train_indices(SEED,N),study.query_indices(SEED)])
a=whiten.job_arrays(SEED,N,T); Z=np.concatenate([a['train_x'],a['query_x']]).astype(np.float64)
Zc=Z-Z.mean(0); C=np.cov(Zc.T); d,E=np.linalg.eigh(C); d=np.clip(d,1e-12,None)
keep=d>EIGFLOOR; K=(E[:,keep]*(1/np.sqrt(d[keep]))[None,:]).T
Y=Zc@K.T; DIM=Y.shape[1]; X=U[idx]
t0=time.time(); Yt=torch.tensor(Y,dtype=torch.float32)
g=torch.Generator().manual_seed(0)
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
    if len(kept)>=81: break
    if all(abs(np.corrcoef(tz[:,i],tz[:,j])[0,1])<0.9 for j in kept): kept.append(i)
kept=np.array(kept)
# label-free channel construction: orient by skew, shift so the atom sits at 0
feat=np.zeros((len(X),81))
for col,i in enumerate(kept):
    t=tz[:,i]*(np.sign(((tz[:,i]-tz[:,i].mean())**3).mean()) or 1.0)
    feat[:,col]=(t-t.min())/max(t.std(),1e-9)
attack_seconds=time.time()-t0
# ---- evaluation only
Xc=X-X.mean(0); F=feat[:,:len(kept)]
cc=np.abs(np.nan_to_num((F-F.mean(0)).T@Xc/len(X)/np.maximum(np.outer(F.std(0),X.std(0)),1e-300)))
bestr=cc.max(1); truecell=np.full(81,0,dtype=np.int64); truecell[:len(kept)]=cc.argmax(1)
rep={'N':N,'dim':DIM,'restarts':RESTARTS,'channels':int(len(kept)),
     'matched_gt_0.9':int((bestr>0.9).sum()),'matched_gt_0.99':int((bestr>0.99).sum()),
     'distinct_pixels_gt_0.9':int(np.unique(cc.argmax(1)[bestr>0.9]).size),
     'median_abs_r':float(np.median(bestr)),'attack_seconds':round(attack_seconds,1)}
print(json.dumps(rep), flush=True)
# ---- label-free layout recovery on the reconstructed pixel-like channels
t1=time.time()
try:
    lay=topology.recover_layout(np.ascontiguousarray(feat[:N]), max_seconds=120.0)
    rep['layout_seconds']=round(time.time()-t1,1)
    q2=atk.layout_quality(lay,truecell)
    rep['layout_quality']={k:(round(v,4) if isinstance(v,float) else v) for k,v in q2.items()}
    print('layout', json.dumps(rep['layout_quality']), flush=True)
except Exception as e:
    print('recover_layout failed (%s); falling back to the robust QAP on the same'
          ' statistic'%e, flush=True)
    Wsim=np.clip(topology.partial_correlation(np.sqrt(np.clip(feat[:N],0,None))),0.0,None)
    lay,_=atk.layout_from_similarity(Wsim, max_seconds=60.0)
    rep['layout_seconds']=round(time.time()-t1,1); rep['recover_layout_crashed']=True
    q2=atk.layout_quality(lay,truecell)
    rep['layout_quality']={k:(round(v,4) if isinstance(v,float) else v) for k,v in q2.items()}
    print('layout(fallback)', json.dumps(rep['layout_quality']), flush=True)
imgs=topology.unpermute(np.ascontiguousarray(feat,dtype=np.float32), lay)
y_t=np.load(R+'/raw/pool_labels.npy')[study.train_indices(SEED,N)]; y_q=atk.dev_query_labels(SEED)
r=atk.train_cnn(imgs[:N], y_t, imgs[N:], epochs=EPOCHS, member_seed=101)
rep['cnn_error']=atk.error_rate(r['predictions'],y_q); rep['cnn_epochs']=EPOCHS
print('facet_attack_cnn N=%d epochs=%d err=%.4f'%(N,EPOCHS,rep['cnn_error']), flush=True)
rep['layout']=np.asarray(lay).tolist()
json.dump(rep, open('/tmp/whiten-attack/facet2_N%d.json'%N,'w'), indent=1)
np.save('/tmp/whiten-attack/facet2_feat_N%d.npy'%N, feat.astype(np.float32))
