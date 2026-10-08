> Historical research snapshot. The revised standalone candidate and three passing sandbox evaluations are in [native-mlp-d4-20261007](../native-mlp-d4-20261007/README.md). Results below describe this earlier snapshot.

# Difficulty 4 research candidate — not qualified

This draft preserves a reproducible multi-file research setup for review. It is not a valid standalone submission and does not claim a difficulty-4 pass or change the leaderboard.

Two independently initialized teachers (eight width-256, three-hidden-layer ReLU/RMSNorm MLPs; 1,000 and 250 updates) produce confidence-filtered pseudo-labels for the released query inputs. A four-member width-512, three-hidden-layer SiLU/RMSNorm student trains for 1,000 updates. The final readout uses five-neighbor label voting and query-embedding graph propagation. Only the original 10,000 labelled examples supply dictionary labels; query ground truth is used only for evaluation. No reconstruction objective or generated input points are used.

Weights, activations, gradients and momentum use BF16; Tensor Core accumulation and RMS statistics use FP32. Training uses native CUDA kernels, cuBLAS/cuBLASLt GEMMs and CUDA graphs. This is not a single fused megakernel. Python handles pseudo-label selection and orchestration.

## Fresh-draw evidence

The included full-wrapper 11-draw evaluation gives **2.360% mean error at 984.241 ms**, which misses the **2.30%** difficulty-4 threshold. Earlier exploratory pools passed, but this result prevents treating the current wrapper as qualified. Reset reproducibility and input immutability passed. These measurements are not sandboxed or an official median of three runs; energy is unmeasured.

The current upstream established difficulty-4 time is 2,882.5 ms, but a speedup claim requires passing accuracy and qualification first.

## Research reproduction

The `research/` snapshot includes the wrapper, imported local Python modules and native CUDA source dependencies. On an A100 with PyTorch and CUDA development tools, put `research/` on `PYTHONPATH` and import:

```python
from candidates.mlp_rms_graph_selftrain_silu_fourstudent_shortteacher_rcp_nonoise_softlookup import classify
labels = classify(train_x, train_y, test_x)
```

Required next steps: settle a fresh-draw passing learner, package within the 20 KiB standalone-source limit, run the sandbox evaluator, then official timing and energy verification.

## Longer-training variant: additional fresh evidence

A subsequent 11-draw paired study on a separate fresh pool compared three and four SiLU/RMSNorm hidden layers at 2,000 student updates. The three-layer candidate reached **2.110%** at **1,562.302 ms**, versus **2.150%** at **2,183.778 ms** for four layers (paired difference +0.040 percentage points, SE 0.01784). Retain three layers. The included `mlp_rms_graph_selftrain_silu_fourstudent_2000_research.py` freezes the longer three-layer variant.

The longer variant passes the D4 mean error threshold on this research pool. Its reported time sums warmed teacher/student training and readouts; it excludes some pseudo-label selection and concatenation orchestration, and is not complete-wrapper qualification. It remains multi-file, unsandboxed and unofficial. The 1,000-update complete-wrapper result above remains unchanged. Neither result establishes a D5 pass.
