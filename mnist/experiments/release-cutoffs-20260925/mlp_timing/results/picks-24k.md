| Level | Labels N | Cutoff | Eager MLP (dev error) | Time | Graph-captured MLP (dev error) | Time |
| ---: | ---: | ---: | --- | ---: | --- | ---: |
| 1 | 200 | 5.394% | `mlp-k1-w1024-s100-b512`, 5.22% | 154 ms | `mlpg-k1-w256-s200-b512`, 4.96% | 50 ms |
| 2 | 532 | 3.361% | `mlp-k16-w1024-s400-b512`, 3.21% | 886 ms | `mlpg-k4-w256-s800-b512`, 3.20% | 250 ms |
| 3 | 1,414 | 2.666% | none within 60 s | | | none within 60 s | | |
| 4 | 3,761 | 2.254% | none within 60 s | | | none within 60 s | | |
| 5 | 10,000 | 1.870% | none within 60 s | | | none within 60 s | | |

Host speed ratio, graphed container / eager container, on shared eager configurations: 0.935 (0.989, 0.933, 0.935)
Best eager MLP under 60 s: `mlp-k16-w256-s3200-b512`, 2.76% at 4,711 ms.
Best graphed MLP under 60 s: `mlpg-k16-w256-s3200-b512`, 2.75% at 1,641 ms.

Eager frontier:

| Configuration | Dev error | Time per call |
| --- | ---: | ---: |
| `mlp-k1-w256-s25-b128` | 19.84% | 38.5 ms |
| `mlp-k1-w1024-s25-b512` | 9.36% | 39.9 ms |
| `mlp-k4-w1024-s25-b512` | 8.55% | 41.6 ms |
| `mlp-k16-w1024-s25-b512` | 8.41% | 64.2 ms |
| `mlp-k16-w256-s50-b512` | 8.19% | 70.7 ms |
| `mlp-k4-w1024-s50-b512` | 6.94% | 79.7 ms |
| `mlp-k16-w1024-s50-b512` | 6.88% | 119.0 ms |
| `mlp-k16-w256-s100-b512` | 6.40% | 143.3 ms |
| `mlp-k1-w1024-s100-b512` | 5.22% | 154.4 ms |
| `mlp-k4-w1024-s100-b512` | 4.99% | 158.9 ms |
| `mlp-k16-w1024-s100-b512` | 4.96% | 228.1 ms |
| `mlp-k4-w256-s200-b512` | 4.62% | 290.5 ms |
| `mlp-k16-w256-s200-b512` | 4.55% | 296.9 ms |
| `mlp-k1-w1024-s200-b512` | 3.96% | 308.9 ms |
| `mlp-k4-w1024-s200-b512` | 3.72% | 315.3 ms |
| `mlp-k16-w1024-s200-b512` | 3.70% | 447.4 ms |
| `mlp-k16-w256-s400-b512` | 3.57% | 579.5 ms |
| `mlp-k1-w1024-s400-b512` | 3.36% | 601.2 ms |
| `mlp-k4-w1024-s400-b512` | 3.25% | 628.5 ms |
| `mlp-k16-w1024-s400-b512` | 3.21% | 885.7 ms |
| `mlp-k16-w256-s800-b512` | 3.13% | 1,151.0 ms |
| `mlp-k4-w1024-s800-b512` | 3.05% | 1,271.4 ms |
| `mlp-k16-w1024-s800-b512` | 2.99% | 1,768.7 ms |
| `mlp-k16-w256-s1600-b512` | 2.89% | 2,330.8 ms |
| `mlp-k4-w256-s3200-b512` | 2.85% | 4,613.1 ms |
| `mlp-k16-w256-s3200-b512` | 2.76% | 4,711.4 ms |

Graphed frontier:

| Configuration | Dev error | Time per call |
| --- | ---: | ---: |
| `mlpg-k1-w256-s25-b128` | 20.09% | 6.9 ms |
| `mlpg-k1-w256-s25-b512` | 11.86% | 7.3 ms |
| `mlpg-k4-w256-s25-b512` | 10.92% | 9.4 ms |
| `mlpg-k1-w1024-s25-b512` | 9.57% | 10.2 ms |
| `mlpg-k1-w256-s50-b512` | 9.16% | 13.4 ms |
| `mlpg-k4-w256-s50-b512` | 8.34% | 17.0 ms |
| `mlpg-k1-w1024-s50-b512` | 7.33% | 18.8 ms |
| `mlpg-k1-w256-s100-b512` | 6.89% | 25.6 ms |
| `mlpg-k4-w256-s100-b512` | 6.48% | 32.4 ms |
| `mlpg-k1-w1024-s100-b512` | 5.34% | 36.0 ms |
| `mlpg-k1-w256-s200-b512` | 4.96% | 49.8 ms |
| `mlpg-k4-w256-s200-b512` | 4.67% | 64.2 ms |
| `mlpg-k1-w1024-s200-b512` | 3.97% | 70.0 ms |
| `mlpg-k1-w256-s400-b512` | 3.89% | 98.2 ms |
| `mlpg-k4-w256-s400-b512` | 3.64% | 126.9 ms |
| `mlpg-k1-w1024-s400-b512` | 3.37% | 139.4 ms |
| `mlpg-k4-w256-s800-b512` | 3.20% | 249.5 ms |
| `mlpg-k1-w1024-s800-b512` | 3.11% | 285.7 ms |
| `mlpg-k4-w256-s1600-b512` | 2.95% | 500.8 ms |
| `mlpg-k16-w256-s1600-b512` | 2.91% | 822.2 ms |
| `mlpg-k4-w256-s3200-b512` | 2.82% | 1,000.5 ms |
| `mlpg-k16-w256-s3200-b512` | 2.75% | 1,641.3 ms |
