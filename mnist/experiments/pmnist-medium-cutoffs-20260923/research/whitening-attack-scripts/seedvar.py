import sys, json, time, warnings, numpy as np
warnings.filterwarnings('ignore')
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R); sys.path.insert(0,R+'/research')
import torch; torch.set_num_threads(3)
import attack_ica_topographic as atk, study
SEED=2026092301; N=1000; EPOCHS=60
y_t=np.load(R+'/raw/pool_labels.npy')[study.train_indices(SEED,N)]; y_q=atk.dev_query_labels(SEED)
S=np.load('/tmp/whiten-attack/sources_logcosh_unit-variance_81_N1000.npy').astype(np.float64)
lay_rec=np.load('/tmp/whiten-attack/layout_logcosh_unit-variance_81_sqrt_N1000.npy')
res={}
for name, lay in (('ica_recovered_signed', lay_rec),
                  ('ica_random_signed', np.random.default_rng(12345).permutation(81))):
    imgs=atk.sources_to_images(S, lay, 'signed')
    for ms in (102,103):
        r=atk.train_cnn(imgs[:N], y_t, imgs[N:], epochs=EPOCHS, member_seed=ms)
        e=atk.error_rate(r['predictions'], y_q)
        res['%s_seed%d'%(name,ms)]=e
        print('%s seed=%d err=%.4f'%(name,ms,e), flush=True)
json.dump(res, open('/tmp/whiten-attack/seedvar.json','w'), indent=1)
