import sys, json, time, warnings, numpy as np
warnings.filterwarnings('ignore')
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R); sys.path.insert(0,R+'/research')
import torch; torch.set_num_threads(4)
import attack_ica_topographic as atk, study, topology, whiten
SEED=2026092301; N=int(sys.argv[1]); EPOCHS=int(sys.argv[2]); which=sys.argv[3].split(','); out=sys.argv[4]
y_t = np.load(R+'/raw/pool_labels.npy')[study.train_indices(SEED,N)]
y_q = atk.dev_query_labels(SEED)
res={}
def run(name, images, epochs=EPOCHS, ms=101):
    t0=time.time(); r = atk.train_cnn(images[:N], y_t, images[N:], epochs=epochs, member_seed=ms)
    e = atk.error_rate(r['predictions'], y_q)
    res[name]=dict(error=e, epochs=epochs, member_seed=ms, seconds=round(r['seconds'],1))
    print('%-28s N=%d epochs=%d err=%.4f (%.0fs)'%(name,N,epochs,e,time.time()-t0), flush=True)

if 'zca_true_layout' in which:
    U = np.asarray(study.pool_images(), dtype=np.float64)
    T0 = whiten.fit_transform(U, epsilon=1e-3, rotation_seed=20260923,
                              rotation='none', apply_permutation=False)
    idx = np.concatenate([study.train_indices(SEED,N), study.query_indices(SEED)])
    Z = whiten.apply(U[idx], T0).astype(np.float64)
    # feature j IS pixel j here: the true layout is the identity
    run('zca_true_layout', atk.sources_to_images(Z, np.arange(81), 'signed'))
if 'zca_true_layout_norescale' in which:
    U = np.asarray(study.pool_images(), dtype=np.float64)
    T0 = whiten.fit_transform(U, epsilon=1e-3, rotation_seed=20260923,
                              rotation='none', apply_permutation=False)
    idx = np.concatenate([study.train_indices(SEED,N), study.query_indices(SEED)])
    Z = whiten.apply(U[idx], T0).astype(np.float32)
    run('zca_true_layout_norescale', topology.unpermute(Z, np.arange(81)))
if 'ica_recovered_signed' in which or 'ica_random_signed' in which or 'ica_oracle_signed' in which:
    stem='logcosh_unit-variance_81_N%d_s%d'%(N,SEED)
    S=np.load('/tmp/whiten-attack/cache/sources_%s.npy'%stem).astype(np.float64)
    C=np.load('/tmp/whiten-attack/cache/correlation_%s.npy'%stem)
    cell=np.abs(C).argmax(1)
if 'ica_recovered_signed' in which:
    lay=np.load('/tmp/whiten-attack/cache/layout_%s_sqrt.npy'%stem)
    print('layout quality', json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in atk.layout_quality(lay,cell).items()}), flush=True)
    run('ica_recovered_signed', atk.sources_to_images(S, lay, 'signed'))
if 'ica_random_signed' in which:
    run('ica_random_signed', atk.sources_to_images(S, np.random.default_rng(12345).permutation(81), 'signed'))
if 'ica_oracle_signed' in which:
    run('ica_oracle_signed', atk.sources_to_images(S, atk.oracle_layout(cell), 'signed'))
if 'whitened_random' in which:
    T=whiten.default_transform(); a=whiten.job_arrays(SEED,N,T)
    Z=np.concatenate([a['train_x'],a['query_x']]).astype(np.float64)
    run('whitened_random', atk.sources_to_images(Z, np.random.default_rng(999).permutation(81), 'signed'))
if 'pixel_recovered' in which or 'pixel_true' in which:
    a=study.job_arrays(SEED,N); P=np.concatenate([a['train_x'],a['query_x']]).astype(np.float32)
if 'pixel_recovered' in which:
    lay,_=topology.recover_layout(a['train_x'], max_seconds=120.0, return_details=True)
    q=atk.layout_quality(lay, study.feature_permutation())
    res['pixel_layout_quality']={k:(float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in q.items()}
    print('pixel layout quality', json.dumps(res['pixel_layout_quality']), flush=True)
    run('pixel_recovered', topology.unpermute(P, lay))
if 'pixel_true' in which:
    run('pixel_true', topology.unpermute(P, study.feature_permutation()))
json.dump(res, open(out,'w'), indent=1)
