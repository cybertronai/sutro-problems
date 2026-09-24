import sys, json, warnings, numpy as np, os
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923')
sys.path.insert(0,'/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923/research'); import attack_ica_topographic as atk, whiten
N = int(sys.argv[1]); out = sys.argv[2]
T = whiten.default_transform()
reports = []
for fun, wm in (('logcosh','unit-variance'), ('cube','unit-variance'), ('logcosh', False)):
    r = atk.run_attack(2026092301, N, T, fun=fun, whiten_mode=wm,
                       affinity_modes=('sqrt','partial_sqrt','abs2'),
                       qap_seconds=60.0, cache_dir='/tmp/whiten-attack/cache')
    reports.append(r)
    print(json.dumps({'fun':fun,'wm':str(wm),'N':N,
        'corr_max_mean':round(r['localisation']['max_abs_pixel_correlation_mean'],3),
        'syn_top1':round(r['localisation']['synthesis']['mean_top1_energy'],3),
        'best':r['best_affinity_mode'],
        'agree':{m:round(v['quality']['adjacency_agreement'],3) for m,v in r['layouts'].items()},
        'manh':{m:round(v['quality']['mean_manhattan_after_dihedral'],2) for m,v in r['layouts'].items()},
        'secs':round(r['seconds'])}), flush=True)
json.dump(reports, open(out,'w'), indent=1)
