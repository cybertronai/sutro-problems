import sys, json, time, warnings, numpy as np
warnings.filterwarnings('ignore')
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R); sys.path.insert(0,R+'/research')
import attack_ica_topographic as atk, study, topology, whiten
SEED=2026092301; T=whiten.default_transform()
U=np.asarray(study.pool_images(),dtype=np.float64); m=U.mean(0)
out={}
for N in (1000,10000):
    a=whiten.job_arrays(SEED,N,T)
    idx=np.concatenate([study.train_indices(SEED,N),study.query_indices(SEED)])
    Zc=np.concatenate([a['train_x'],a['query_x']]).astype(np.float64)
    X=U[idx]-m
    z=Zc-Zc.mean(0); x=X-X.mean(0)
    C=np.nan_to_num((z.T@x)/np.maximum(np.outer(np.linalg.norm(z,axis=0),np.linalg.norm(x,axis=0)),1e-300))
    cell=np.abs(C).argmax(1)
    t0=time.time()
    try:
        lay=topology.recover_layout(a['train_x'], max_seconds=120.0)
        crashed=False
    except Exception as e:
        print('  topology.recover_layout raised %s: %s -> falling back to a robust'
              ' second-order search on the same statistic' % (type(e).__name__, e), flush=True)
        crashed=True
        W=np.clip(topology.partial_correlation(np.asarray(a['train_x'],dtype=np.float64)),0.0,None)
        lay,_=atk.layout_from_similarity(W, max_seconds=60.0)
    q=atk.layout_quality(lay,cell)
    ch=atk.chance_layout_quality(cell,trials=100)
    out['N%d'%N]={'seconds':round(time.time()-t0,1),'recover_layout_crashed':bool(crashed),
                  'max_abs_corr_with_a_pixel_mean':round(float(np.abs(C).max(1).mean()),3),
                  'quality':{k:(round(v,4) if isinstance(v,float) else v) for k,v in q.items()},
                  'chance':{k:round(v,4) for k,v in ch.items() if isinstance(v,float)}}
    print(N, json.dumps(out['N%d'%N]), flush=True)
json.dump(out, open('/tmp/whiten-attack/secondorder.json','w'), indent=1)
