import sys, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/tmp/whiten'); from common import *
import study, classical
X=pool(); perm=np.asarray(study.feature_permutation()); order=study.draw_order(2026092301); qu=order[10000:20000]
labels=np.load('/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923/raw/pool_labels.npy')
cfg={"family":"kernel_ridge","kernel":"arccos1","depth":3,"lambda_grid":[1e-7,1e-6,1e-5,1e-4,1e-3],
     "cv_subsample":4000,"predict_block":2000,"kernel_block":2000}
Q=rand_orth(777)
def run(tag,F,n):
    tr=order[:n]
    o=classical.fit_predict(F[tr].astype(np.float32),labels[tr].astype(np.uint8),F[qu].astype(np.float32),cfg,1)
    print(f"{tag:44s} n={n}  err={(o['labels']!=labels[qu]).mean()*100:5.2f}%",flush=True)
Xp=X[:,perm]
# rotation only, centred, normalisation bypassed on BOTH arms for a clean comparison
def undo(F): return (F+0.5)/4.0
mu=Xp.mean(0)
for n in (1000,3162):
    run("centred raw permuted (norm bypassed)",undo(Xp-mu),n)
    run("centred raw + random rotation  (norm bypassed)",undo((Xp-mu)@Q.T),n)
