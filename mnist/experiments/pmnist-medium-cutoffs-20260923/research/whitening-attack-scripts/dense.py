import sys, json, time, warnings, numpy as np
warnings.filterwarnings('ignore')
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R); sys.path.insert(0,R+'/research')
import torch; torch.set_num_threads(4)
import learners, study, whiten, attack_ica_topographic as atk
SEED=2026092301; N=int(sys.argv[1]); EPOCHS=int(sys.argv[2])
y_q = atk.dev_query_labels(SEED)
T = whiten.default_transform()
cfg = dict(family='mlp', widths=[1024,1024], dropout=0.2, input_noise_std=0.0,
           lr=0.001, weight_decay=0.0, epochs=EPOCHS, batch_size=128,
           warmup_fraction=0.0, members=1, normalization='standardize')
res={}
for name, arrays in (('whitened', whiten.job_arrays(SEED,N,T)),
                     ('permuted_pixels', study.job_arrays(SEED,N))):
    t0=time.time()
    out = learners.fit_predict(arrays['train_x'], arrays['train_y'], arrays['query_x'],
                               cfg, seed=101, deadline_unix=time.time()+3000, device='cpu')
    err = atk.error_rate(out['labels'], y_q)
    res[name]=dict(error=err, epochs=EPOCHS, seconds=round(time.time()-t0,1))
    print('mlp-standardize %-16s N=%d err=%.4f (%.0fs)'%(name,N,err,time.time()-t0), flush=True)
json.dump(res, open('/tmp/whiten-attack/dense_N%d.json'%N,'w'), indent=1)
