# Native MLP candidate for difficulty 2

Two 60-256-256-10 MLPs, 120 training updates, three-neighbor embedding readout. The softmax heads train the embeddings; neighbor voting over the labelled training embeddings makes the final prediction. Native CUDA kernels and GEMMs execute training in a CUDA graph. The standalone source embeds its CUDA implementation for the benchmark source-size constraint.

## Local evidence

- MNIST error: 3.276364% over 11 draws (110,000 predictions).
- Evaluator-ranked time: 9.452 ms on an A100-SXM4-80GB.
- Sandbox enabled: False.
- Evaluator problems: [].
- Source size: 20406 bytes; SHA-256: `60701d26a105d8200b7be885d2ef4d0525336a2636a049ee8c9de21c9cdef639`.
- Raw evaluator output: [evaluation.json](evaluation.json).

The upstream established time at base commit a2ed895 is 38.3 ms; this local measurement is 4.05 times faster. This is a comparison of local evidence, not an official record or median of three official runs. Energy has not been measured. Sandbox qualification is still required.

## Reproduction

From `mnist-a100`, on an A100 with CUDA development tools:

```bash
python run_modal.py submissions/native-mlp-d2-20261007/mlp_neighbor3_register_d2.py:classify --difficulty 2 --runs 3
```

This is the upstream official verification command, not a claim that it has been run. No leaderboard row is changed by this draft.
