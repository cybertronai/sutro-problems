"""Is the MLP's -2.01 pp 'the rotation makes the task easier' a normalisation artifact?
Same recipe, permuted pixels, only std_floor / normalization changed."""
import sys, os, time, numpy as np
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R)
import torch; torch.set_num_threads(3)
import study, learners
SEED=2026092301; N=1000
a=study.job_arrays(SEED,N)
truth=np.load(study.POOL_LABELS_PATH)[study.query_indices(SEED)]
base=dict(family='mlp', widths=[1024,1024], dropout=0.3, input_noise_std=0.3,
          lr=0.002, weight_decay=0.01, epochs=300, batch_size=128,
          warmup_fraction=0.1, members=1)
for tag,extra in [('standardize std_floor=1e-5 (as reported)', dict(normalization='standardize', std_floor=1e-5)),
                  ('standardize std_floor=0.05',               dict(normalization='standardize', std_floor=0.05)),
                  ('4x-0.5 (study default)',                   dict(normalization='4x-0.5'))]:
    cfg=dict(base); cfg.update(extra)
    t0=time.time()
    out=learners.fit_predict(a['train_x'],a['train_y'],a['query_x'],cfg,seed=11,
                             deadline_unix=time.time()+3600.0, device='cpu')
    err=100.0*float((out['labels']!=truth).mean())
    print('permuted pixels, N=1000, %-38s error %5.2f%%  (%.0fs)'%(tag,err,time.time()-t0), flush=True)
