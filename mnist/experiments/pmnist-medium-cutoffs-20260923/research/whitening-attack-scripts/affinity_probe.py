import sys, json, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923')
sys.path.insert(0,'/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923/research'); import attack_ica_topographic as atk, topology
N = int(sys.argv[1])
ADJ = topology.ADJACENCY > 0.5
RC = topology.ROW_COL.astype(np.int64)
for tag in ('logcosh_False_81','logcosh_unit-variance_81','cube_unit-variance_81','logcosh_unit-variance_64'):
    S = np.load('/tmp/whiten-attack/sources_%s_N%d.npy'%(tag,N)).astype(np.float64)
    C = np.load('/tmp/whiten-attack/corr_%s_N%d.npy'%(tag,N))
    cell = np.abs(C).argmax(1)
    d = np.abs(RC[cell][:,None,:]-RC[cell][None,:,:]).sum(-1)
    neigh = (d==1); far = (d>=3); off = ~np.eye(len(cell),dtype=bool)
    out = {'tag':tag}
    for mode in ('abs2','abs','sqrt','partial_abs','partial_sqrt'):
        W = atk.energy_affinity(S, mode)
        # separability: mean affinity on true-neighbour pairs vs far pairs
        out[mode] = [round(float(W[neigh&off].mean()),4), round(float(W[far].mean()),4),
                     round(float(W[off].mean()),4)]
        # rank statistic: for each component, fraction of its top-4 affinities that are true neighbours
        top4 = np.argsort(-W, axis=1)[:, :4]
        hit = neigh[np.arange(len(cell))[:,None], top4].mean()
        out[mode].append(round(float(hit),3))
    print(json.dumps(out), flush=True)
