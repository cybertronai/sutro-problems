# Ladder AMLP: transfer comparison

Accuracy is mean ± sample SD in percentage points. Reference rows use exactly the same draw indices as the candidate.

| Dataset | Draws | Candidate | linear-sgd | mlp64-sgd | mlp256-sgd | cnn16-sgd | cnn32-ensemble3 | reversible82-sgd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kmnist | 11 | 94.44 ± 0.17 | 79.45 ± 0.53 | 90.28 ± 0.25 | 92.85 ± 0.22 | 93.84 ± 0.34 | 96.28 ± 0.18 | 79.87 ± 0.35 |
| emnist_letters_aj | 11 | 93.34 ± 0.17 | 78.36 ± 0.69 | 88.86 ± 0.42 | 92.04 ± 0.43 | 93.09 ± 0.56 | 95.40 ± 0.16 | 78.32 ± 0.81 |
| qmnist_recovered | 11 | 95.68 ± 0.13 | 89.63 ± 0.38 | 94.31 ± 0.22 | 96.15 ± 0.21 | 96.43 ± 0.39 | 97.73 ± 0.13 | 89.39 ± 0.48 |
| fashion_mnist | 11 | 82.70 ± 0.36 | 78.79 ± 1.66 | 82.49 ± 1.02 | 85.04 ± 0.67 | 84.53 ± 1.09 | 87.21 ± 0.41 | 77.78 ± 3.04 |
| cifar10 | 11 | 38.65 ± 0.28 | 17.37 ± 3.31 | 29.85 ± 1.13 | 34.85 ± 1.06 | 37.56 ± 1.28 | 44.57 ± 0.51 | 23.95 ± 1.79 |

Positive differences below favor the candidate. These are paired descriptive differences across the matched draws.

| Dataset | linear-sgd | mlp64-sgd | mlp256-sgd | cnn16-sgd | cnn32-ensemble3 | reversible82-sgd |
|---|---:|---:|---:|---:|---:|---:|
| kmnist | +14.99 | +4.16 | +1.59 | +0.60 | -1.84 | +14.57 |
| emnist_letters_aj | +14.97 | +4.48 | +1.30 | +0.24 | -2.06 | +15.01 |
| qmnist_recovered | +6.05 | +1.38 | -0.47 | -0.74 | -2.05 | +6.30 |
| fashion_mnist | +3.91 | +0.20 | -2.34 | -1.83 | -4.52 | +4.91 |
| cifar10 | +21.28 | +8.81 | +3.81 | +1.09 | -5.91 | +14.71 |

- Reference CNNs use the original ordered 9x9 grid, whereas the candidate may use a fixed pixel permutation. This is a comparison of complete procedures, not an architecture-only or compute-matched ablation.
- These transfer tasks repartition curated pools; they are not official-test 28x28 pMNIST results.
- Models train from fresh initialization on each draw; this measures training-procedure transfer, not transfer of learned MNIST weights.
- Draws overlap and datasets share ancestry. Draw SD and paired-delta SD are descriptive, not confidence intervals or significance tests.
- A subset of tasks is an incomplete v1 fifteen-number suite even when all eleven draws are run for each selected task.

All selected tasks use all eleven draws: True. Complete fifteen-task suite: False.

The JSON records score-file, source, and available plan/prediction-manifest hashes. No model fitting or query-label reading occurs in this comparison script.
