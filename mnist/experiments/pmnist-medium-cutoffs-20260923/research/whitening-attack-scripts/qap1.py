import sys, json, time, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923')
sys.path.insert(0,'/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923/research'); import attack_ica_topographic as atk, topology
N = int(sys.argv[1])
combos = [('logcosh_unit-variance_81','sqrt'), ('logcosh_unit-variance_81','abs'),
          ('logcosh_unit-variance_81','partial_sqrt'), ('cube_unit-variance_81','partial_sqrt'),
          ('logcosh_False_81','sqrt')]
out=[]
for tag, mode in combos:
    S = np.load('/tmp/whiten-attack/sources_%s_N%d.npy'%(tag,N)).astype(np.float64)
    C = np.load('/tmp/whiten-attack/corr_%s_N%d.npy'%(tag,N))
    cell = np.abs(C).argmax(1)
    W = atk.energy_affinity(S, mode)
    t0=time.time()
    layout, rep = atk.layout_from_similarity(W, max_seconds=60.0)
    q = atk.layout_quality(layout, cell)
    row = dict(tag=tag, mode=mode, seconds=round(time.time()-t0,1),
               qap=round(rep['qap_objective'],3), qap_frac=round(rep['qap_objective_fraction'],4),
               winning_start=rep['winning_start'].get('embedding'),
               distinct_true_cells=int(np.unique(cell).size), **{k:(round(v,4) if isinstance(v,float) else v) for k,v in q.items()})
    out.append(row); print(json.dumps(row), flush=True)
    np.save('/tmp/whiten-attack/layout_%s_%s_N%d.npy'%(tag,mode,N), layout)
# chance baseline on the reference cell map
C = np.load('/tmp/whiten-attack/corr_logcosh_unit-variance_81_N%d.npy'%N)
cell = np.abs(C).argmax(1)
print('chance', json.dumps(atk.chance_layout_quality(cell, trials=200)), flush=True)
json.dump(out, open('/tmp/whiten-attack/qap1_N%d.json'%N,'w'), indent=1)
