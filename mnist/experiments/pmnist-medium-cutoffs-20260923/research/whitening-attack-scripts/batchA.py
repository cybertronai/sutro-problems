import sys, json, time, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923')
import torch; torch.set_num_threads(4)
sys.path.insert(0,'/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923/research'); import attack_ica_topographic as atk, study, topology, whiten

SEED = 2026092301
N = int(sys.argv[1]); EPOCHS = int(sys.argv[2]); which = sys.argv[3].split(','); out = sys.argv[4]
y_train = np.load('/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923/raw/pool_labels.npy')[study.train_indices(SEED,N)]
y_query = atk.dev_query_labels(SEED)
tag = 'logcosh_unit-variance_81'
res = {}

def run(name, images, epochs=EPOCHS, member_seed=101):
    t0=time.time()
    r = atk.train_cnn(images[:N], y_train, images[N:], epochs=epochs, member_seed=member_seed)
    err = atk.error_rate(r['predictions'], y_query)
    res[name] = dict(error=err, epochs=epochs, member_seed=member_seed,
                     seconds=round(r['seconds'],1), final_train_loss=round(r['final_train_loss'],4))
    print('%-34s epochs=%d err=%.4f (%.0fs)' % (name, epochs, err, time.time()-t0), flush=True)

if any(w.startswith('ica') for w in which):
    S = np.load('/tmp/whiten-attack/sources_%s_N%d.npy'%(tag,N)).astype(np.float64)
    C = np.load('/tmp/whiten-attack/corr_%s_N%d.npy'%(tag,N))
    cell = np.abs(C).argmax(1)
if 'ica_recovered_signed' in which:
    lay = np.load('/tmp/whiten-attack/layout_%s_sqrt_N%d.npy'%(tag,N))
    run('ica_recovered_signed', atk.sources_to_images(S, lay, 'signed'))
if 'ica_recovered_abs' in which:
    lay = np.load('/tmp/whiten-attack/layout_%s_sqrt_N%d.npy'%(tag,N))
    run('ica_recovered_abs', atk.sources_to_images(S, lay, 'abs'))
if 'ica_random_signed' in which:
    lay = np.random.default_rng(12345).permutation(81)
    run('ica_random_signed', atk.sources_to_images(S, lay, 'signed'))
if 'ica_oracle_signed' in which:
    lay = atk.oracle_layout(cell)
    print('oracle layout quality', json.dumps({k:round(v,4) if isinstance(v,float) else v for k,v in atk.layout_quality(lay,cell).items()}), flush=True)
    run('ica_oracle_signed', atk.sources_to_images(S, lay, 'signed'))
if 'whitened_random' in which:
    T = whiten.default_transform()
    a = whiten.job_arrays(SEED, N, T)
    Z = np.concatenate([a['train_x'], a['query_x']]).astype(np.float64)
    lay = np.random.default_rng(999).permutation(81)
    run('whitened_random', atk.sources_to_images(Z, lay, 'signed'))
if 'pixel_recovered' in which or 'pixel_true' in which:
    a = study.job_arrays(SEED, N)
    P = np.concatenate([a['train_x'], a['query_x']]).astype(np.float32)
if 'pixel_recovered' in which:
    t0=time.time(); lay, det = topology.recover_layout(a['train_x'], max_seconds=120.0, return_details=True)
    perm = study.feature_permutation()
    q = atk.layout_quality(lay, perm)
    print('pixel recovery %.0fs quality %s' % (time.time()-t0, json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in q.items()})), flush=True)
    res['pixel_layout_quality'] = {k:(float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in q.items()}
    run('pixel_recovered', topology.unpermute(P, lay))
if 'pixel_true' in which:
    run('pixel_true', topology.unpermute(P, study.feature_permutation()))
json.dump(res, open(out,'w'), indent=1)
