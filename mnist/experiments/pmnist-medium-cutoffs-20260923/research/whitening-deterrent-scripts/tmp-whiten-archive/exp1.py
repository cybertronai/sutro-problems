import sys, time, json, numpy as np
sys.path.insert(0,'/tmp/whiten')
from common import *
import topology, study

SEED=2026092301
perm=np.asarray(study.feature_permutation())
X=pool()                                  # (60000,81) unpermuted, float64
order=study.draw_order(SEED)
tr=order[:10000]; qu=order[10000:20000]
vis=np.concatenate([tr,qu])               # 20000 rows a transductive learner sees

def report(name, F, perm_used):
    t=time.time()
    lay,det=topology.recover_layout(F.astype(np.float32), return_details=True)
    frac,cnt=edge_preservation(lay, perm_used)
    print(f"{name:34s} edges_kept={cnt:3d}/144 ({frac:.3f})  qap={det['qap_objective']:.1f} n_active={det['n_active']} t={time.time()-t:.0f}s", flush=True)
    return frac

# --- baseline: current protocol (raw, permuted) -------------------------------
raw_tr = X[tr][:,perm]
report("A raw permuted (N=10000)", raw_tr, perm)

# --- chance level -------------------------------------------------------------
rng=np.random.default_rng(0)
ch=[edge_preservation(rng.permutation(81).astype(np.int64),perm)[1] for _ in range(2000)]
print(f"chance edges_kept mean={np.mean(ch):.1f}/144 p95={np.percentile(ch,95):.0f}", flush=True)

# --- ZCA whitening on full pool + random rotation ----------------------------
for eps in (1e-4,):
    mu,W,Winv,evals=zca(X,eps)
    print(f"eps={eps} cov eigenvalues: max={evals.max():.4g} min={evals.min():.4g} "
          f"#<eps={int((evals<eps).sum())}", flush=True)
    Z=(X-mu)@W
    Q=rand_orth(777)
    Y=Z@Q.T                                 # dense random orthogonal after whitening
    np.save('/tmp/whiten/Y.npy',Y); np.save('/tmp/whiten/Z.npy',Z)
    np.save('/tmp/whiten/W.npy',W); np.save('/tmp/whiten/Winv.npy',Winv)
    np.save('/tmp/whiten/mu.npy',mu); np.save('/tmp/whiten/Q.npy',Q)
    report(f"B ZCA(eps={eps}) only, permuted", Z[tr][:,perm], perm)
    report(f"C ZCA+rand rotation (N=10000)", Y[tr], np.arange(81))
