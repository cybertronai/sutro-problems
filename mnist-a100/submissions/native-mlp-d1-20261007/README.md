# Native MLP candidate for difficulty 1

Two 60-256-256-10 MLPs, 30 training updates, three-neighbor embedding readout. The softmax heads train the embeddings; neighbor voting over the labelled training embeddings makes the final prediction. Native CUDA kernels and GEMMs execute training in a CUDA graph. The standalone source embeds its CUDA implementation for the benchmark source-size constraint.

## Local evidence

- MNIST error: 4.960000% over 11 draws (110,000 predictions).
- Evaluator-ranked time: 3.653 ms on an A100-SXM4-80GB.
- Sandbox enabled: False.
- Evaluator problems: [].
- Source size: 20405 bytes; SHA-256: `90653ffbb06dd4b01abe6252b8536cf3648b76d573c5083283e3427f9c5dffd5`.
- Raw evaluator output: [evaluation.json](evaluation.json).

The upstream established time at base commit a2ed895 is 61.7 ms; this local measurement is 16.89 times faster. This is a comparison of local evidence, not an official record or median of three official runs. Energy has not been measured. Sandbox qualification is still required.

## Reproduction

From `mnist-a100`, on an A100 with CUDA development tools:

```bash
python run_modal.py submissions/native-mlp-d1-20261007/mlp_neighbor3_register_d1.py:classify --difficulty 1 --runs 3
```

This is the upstream official verification command, not a claim that it has been run. No leaderboard row is changed by this draft.
