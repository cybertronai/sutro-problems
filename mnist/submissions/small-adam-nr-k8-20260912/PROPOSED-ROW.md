# Proposed row (MNIST-small, 67% target)

```markdown
| 2026-09-14 | 67.35% ± 1.76 pp | 2,600 | 110 | 0.19 | 1,800 | [NR-K8 Adam MLP](submissions/small-adam-nr-k8-20260912/README.md) |
```

Two significant figures in the row; exact values (2,560.31 mJ / 106.93 ms,
0.185228620462 mJ / 1,798.760521 ms) are in the evidence JSONs. Grid columns are
provisional pending review of the scorer extension, which is requested as a
merge condition. The A100 advantage is largely the fused runtime; a
same-container interleave measures the H32 learner at 44.5 ms / 1,062 mJ versus
106.9 ms / 2,424 mJ for this entry, and no accuracy-superiority claim is made
(+0.27 pp; ±1.76 pp is a sample SD).
