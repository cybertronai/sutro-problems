import sys, time, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/tmp/whiten'); from common import *
import study, classical
SEED=2026092301
X=pool(); perm=np.asarray(study.feature_permutation())
order=study.draw_order(SEED); qu=order[10000:20000]
Y=np.asarray(study.pool_labels_unsafe()) if False else None
labels=np.load('/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923/raw/pool_labels.npy')
cfg={"family":"kernel_ridge","kernel":"arccos1","depth":3,
     "lambda_grid":[1e-7,1e-6,1e-5,1e-4,1e-3],"cv_subsample":4000,
     "predict_block":2000,"kernel_block":2000}
eps=1e-4; mu,W,Winv,ev=zca(X,eps); Q=rand_orth(777)
Z=(X-mu)@W; Yr=(Z@Q.T)[:,perm]
Xp=X[:,perm]
def run(tag,F,n):
    tr=order[:n]
    t=time.time()
    out=classical.fit_predict(F[tr].astype(np.float32),labels[tr].astype(np.uint8),
                              F[qu].astype(np.float32),cfg,1)
    err=(out['labels']!=labels[qu]).mean()*100
    print(f"{tag:42s} n={n:5d}  err={err:5.2f}%  lam={out['metrics'].get('lambda')}  {time.time()-t:.0f}s",flush=True)
for n in (1000,3162):
    run("raw permuted (protocol as-is)",Xp,n)
    run("ZCA(1e-4)+random rotation",Yr,n)
