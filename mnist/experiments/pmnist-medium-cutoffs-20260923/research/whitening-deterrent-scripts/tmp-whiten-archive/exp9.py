import sys, time, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/tmp/whiten'); from common import *
import study, classical
SEED=2026092301
X=pool(); perm=np.asarray(study.feature_permutation())
order=study.draw_order(SEED); qu=order[10000:20000]
labels=np.load('/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923/raw/pool_labels.npy')
cfg={"family":"kernel_ridge","kernel":"arccos1","depth":3,
     "lambda_grid":[1e-7,1e-6,1e-5,1e-4,1e-3],"cv_subsample":4000,
     "predict_block":2000,"kernel_block":2000}
def undo_norm(F):   # so that classical.normalize (4x-0.5) returns F unchanged
    return (F+0.5)/4.0
def run(tag,F,n):
    tr=order[:n]; t=time.time()
    out=classical.fit_predict(F[tr].astype(np.float32),labels[tr].astype(np.uint8),
                              F[qu].astype(np.float32),cfg,1)
    print(f"{tag:46s} n={n:5d}  err={(out['labels']!=labels[qu]).mean()*100:5.2f}%  {time.time()-t:.0f}s",flush=True)
for eps in (1e-4,1e-2):
    mu,W,Winv,ev=zca(X,eps); Z=(X-mu)@W; Q=rand_orth(777); Yr=(Z@Q.T)[:,perm]
    Yr=Yr/np.sqrt((Yr**2).sum(1).mean())*np.sqrt((X**2).sum(1).mean())
    for n in (1000,3162):
        run(f"ZCA(eps={eps:g})+rot, pipeline norm bypassed",undo_norm(Yr),n)
