> Superseded: the K5-only source below failed one independent Modal review run. See [graph1250-repair](graph1250-repair/README.md) for the replacement and all three passing energy-enabled evaluations.

# Historical K5-only difficulty 3 candidate

Adds a standalone native BF16 difficulty 3 candidate: eight width-256 MLPs, three RMSNorm/ReLU hidden layers, 1,250 updates in a CUDA graph, and five-neighbor embedding voting.

Three independent required-sandbox evaluations passed for the exact submitted source. Each completed 11 fresh MNIST and four holdout calls with no evaluator problems.

| Run | MNIST error | Ranked time |
|---|---:|---:|
| 1 | 2.624545% | 377.125 ms |
| 2 | 2.618182% | 377.431 ms |
| 3 | 2.560909% | 377.411 ms |

Median evaluator-ranked time: **377.411 ms**. Source size: 20208 bytes, within 20 KiB. Source SHA-256: `6c00a6f593d3e4015c9e6d0ce126fac7fd0519654906421682d631c394750083`. Raw JSON and the validation manifest accompany the source.

These are local SSH-hosted sandbox evaluations. The independent Modal review of this K5-only source passed only two of three runs; run 3 had 2.729090909% mean MNIST error, 32 correct labels short of the 2.70% limit. All three outcomes are in [historical-review](historical-review/verification.json). This historical source does not qualify a leaderboard record; the repaired graph readout above is separately qualified.

## Reproduction

```bash
MNIST_SANDBOX=required python mnist.py submissions/native-mlp-d3-20261007/mlp_rms_knn_d3_specialized.py:classify --difficulty 3 --no-energy --json evaluation.json
```

Repeat three times in fresh processes with distinct output paths.
