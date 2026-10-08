# Native MLP candidate for difficulty 3

Eight width-256 MLPs with three RMS-normalized hidden layers, 1,250 updates, five-neighbor embedding readout. The softmax heads train the embeddings; neighbor voting over the labelled training embeddings makes the final prediction. Native CUDA kernels and GEMMs execute training in a CUDA graph. The standalone source embeds its CUDA implementation for the benchmark source-size constraint.

## Local evidence

- MNIST error: 2.624545% over 11 draws (110,000 predictions).
- Evaluator-ranked time: 377.125 ms on an A100-SXM4-80GB.
- Sandbox enabled: True.
- Evaluator problems: [].
- Source size: 20208 bytes; SHA-256: `6c00a6f593d3e4015c9e6d0ce126fac7fd0519654906421682d631c394750083`.
- Raw evaluator output: [evaluation.json](evaluation.json).

The upstream established time at base commit a2ed895 is 1157.1 ms; this local measurement is 3.07 times faster. This is a comparison of local evidence, not an official record or median of three official runs. Energy has not been measured. One local sandbox evaluation passed; official median-of-three verification remains outstanding.

## Reproduction

From `mnist-a100`, on an A100 with CUDA development tools:

```bash
python run_modal.py submissions/native-mlp-d3-20261007/mlp_rms_knn_d3_specialized.py:classify --difficulty 3 --runs 3
```

This is the upstream official verification command, not a claim that it has been run. No leaderboard row is changed by this draft.
