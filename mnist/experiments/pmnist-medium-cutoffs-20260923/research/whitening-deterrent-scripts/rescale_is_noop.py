"""attack_ica.py rescales the ICA sources by the pixel-space mixing-column norms --
a quantity computed from the SECRET transform A.  Does that rescaling do anything?"""
import sys, numpy as np
R='/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923'
sys.path.insert(0,R)
import attack_ica, topology
rng=np.random.default_rng(0)
s=rng.standard_normal((500,81))*rng.uniform(0.2,5,81)
scale=rng.uniform(0.1,10,81)                      # any positive per-source rescale
f1=attack_ica.positive_part_features(s)
f2=attack_ica.positive_part_features(s*scale[None,:])
print('positive_part_features invariant to positive per-source scaling: max diff %.3e'%np.abs(f1-f2).max())
p1=topology.partial_correlation(np.sqrt(np.clip(f1,0,None)))
p2=topology.partial_correlation(np.sqrt(np.clip(f2,0,None)))
print('partial correlation fed to the QAP: max diff %.3e'%np.abs(p1-p2).max())
