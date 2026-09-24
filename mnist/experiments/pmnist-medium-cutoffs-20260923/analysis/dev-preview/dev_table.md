# Development stage: candidates by training-set size

Pooled error is `100 * (sum(total) - sum(correct)) / sum(total)` over the development seeds, from integer counts. Sorted by error within each N.

## N = 1,000

| Candidate | Mean error | SD (pp) | Draws | Mean train s | Mean fit wall s | Truncated | Unlabeled queries |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `kr-arccos1-d3` | 5.7050% | 0.6293 | 2 | 0.3 | 2.5 | 0/2 | no |
| `kr-arccos1-d2` | 5.7250% | 0.5303 | 2 | 0.2 | 1.9 | 0/2 | no |
| `kr-ntk-d3` | 5.9150% | 0.5728 | 2 | 0.3 | 2.7 | 0/2 | no |
| `kr-rbf` | 6.1350% | 0.4313 | 2 | 0.0 | 2.4 | 0/2 | no |
| `svm-rbf` | 7.0500% | 0.4808 | 2 | 0.1 | 8.4 | 0/2 | no |
| `hgb` | 10.1350% | 0.1202 | 2 | 7.8 | 8.3 | 0/2 | no |
| `knn` | 11.1950% | 0.7849 | 2 | 0.0 | 0.2 | 0/2 | no |

## N = 10,000

| Candidate | Mean error | SD (pp) | Draws | Mean train s | Mean fit wall s | Truncated | Unlabeled queries |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `kr-arccos1-d3` | 2.3150% | 0.0212 | 2 | 21.2 | 49.0 | 0/2 | no |
| `kr-arccos1-d2` | 2.4000% | 0.1273 | 2 | 15.6 | 37.0 | 0/2 | no |
| `kr-rbf` | 2.5100% | 0.0141 | 2 | 6.3 | 42.1 | 0/2 | no |
| `kr-ntk-d3` | 2.5300% | 0.0141 | 2 | 23.6 | 54.1 | 0/2 | no |
| `svm-rbf` | 3.1100% | 0.0141 | 2 | 2.6 | 68.7 | 0/2 | no |
| `hgb` | 3.6850% | 0.2051 | 2 | 19.1 | 19.7 | 0/2 | no |
| `knn` | 4.7700% | 0.2121 | 2 | 0.0 | 1.1 | 0/2 | no |

## Per-seed errors

| Candidate | N | Seed | Error | Correct / total | Train s | Epochs | Truncated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `kr-arccos1-d3` | 1,000 | 2026092301 | 6.15% | 9385 / 10000 | 0.2 | None | False |
| `kr-arccos1-d3` | 1,000 | 2026092302 | 5.26% | 9474 / 10000 | 0.3 | None | False |
| `kr-arccos1-d2` | 1,000 | 2026092301 | 6.10% | 9390 / 10000 | 0.2 | None | False |
| `kr-arccos1-d2` | 1,000 | 2026092302 | 5.35% | 9465 / 10000 | 0.2 | None | False |
| `kr-ntk-d3` | 1,000 | 2026092301 | 6.32% | 9368 / 10000 | 0.3 | None | False |
| `kr-ntk-d3` | 1,000 | 2026092302 | 5.51% | 9449 / 10000 | 0.3 | None | False |
| `kr-rbf` | 1,000 | 2026092301 | 6.44% | 9356 / 10000 | 0.0 | None | False |
| `kr-rbf` | 1,000 | 2026092302 | 5.83% | 9417 / 10000 | 0.0 | None | False |
| `svm-rbf` | 1,000 | 2026092301 | 7.39% | 9261 / 10000 | 0.1 | None | False |
| `svm-rbf` | 1,000 | 2026092302 | 6.71% | 9329 / 10000 | 0.1 | None | False |
| `hgb` | 1,000 | 2026092301 | 10.05% | 8995 / 10000 | 7.6 | None | False |
| `hgb` | 1,000 | 2026092302 | 10.22% | 8978 / 10000 | 8.0 | None | False |
| `knn` | 1,000 | 2026092301 | 11.75% | 8825 / 10000 | 0.0 | None | False |
| `knn` | 1,000 | 2026092302 | 10.64% | 8936 / 10000 | 0.0 | None | False |
| `kr-arccos1-d3` | 10,000 | 2026092301 | 2.30% | 9770 / 10000 | 21.2 | None | False |
| `kr-arccos1-d3` | 10,000 | 2026092302 | 2.33% | 9767 / 10000 | 21.2 | None | False |
| `kr-arccos1-d2` | 10,000 | 2026092301 | 2.49% | 9751 / 10000 | 15.7 | None | False |
| `kr-arccos1-d2` | 10,000 | 2026092302 | 2.31% | 9769 / 10000 | 15.6 | None | False |
| `kr-rbf` | 10,000 | 2026092301 | 2.52% | 9748 / 10000 | 6.6 | None | False |
| `kr-rbf` | 10,000 | 2026092302 | 2.50% | 9750 / 10000 | 6.0 | None | False |
| `kr-ntk-d3` | 10,000 | 2026092301 | 2.54% | 9746 / 10000 | 24.0 | None | False |
| `kr-ntk-d3` | 10,000 | 2026092302 | 2.52% | 9748 / 10000 | 23.3 | None | False |
| `svm-rbf` | 10,000 | 2026092301 | 3.10% | 9690 / 10000 | 3.7 | None | False |
| `svm-rbf` | 10,000 | 2026092302 | 3.12% | 9688 / 10000 | 1.6 | None | False |
| `hgb` | 10,000 | 2026092301 | 3.83% | 9617 / 10000 | 18.8 | None | False |
| `hgb` | 10,000 | 2026092302 | 3.54% | 9646 / 10000 | 19.4 | None | False |
| `knn` | 10,000 | 2026092301 | 4.92% | 9508 / 10000 | 0.0 | None | False |
| `knn` | 10,000 | 2026092302 | 4.62% | 9538 / 10000 | 0.0 | None | False |
