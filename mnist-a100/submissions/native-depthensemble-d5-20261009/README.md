# D5 native BF16 mixed-depth MLP ensemble

Three initial independent canonical Modal A100-80GB sandbox evaluations passed. All initial records are included unchanged; no failed qualification runs were replaced. Median ranked latency is **7,022.198 ms**; median energy is **1,978,525.562 mJ**. This PR supplies a qualifying candidate for review and does not edit the leaderboard.

| Run | MNIST error | Ranked ms | mJ per call |
|---|---:|---:|---:|
| 1 | 1.851818% | 7022.198 | 1978525.562 |
| 2 | 1.887273% | 7013.174 | 1955308.067 |
| 3 | 1.818182% | 7060.753 | 2019724.753 |

## Method

Two BN/ReLU teachers (eight width-256 MLPs, 1,000 and 250 updates) produce hard pseudo labels for query rows whose averaged graph-readout probability is at least 0.999. Four width-512 students with three hidden layers and four with four hidden layers each train for 3,000 updates. Positive first-to-final residual gain is 0.25. Training uses Nesterov momentum 0.95, cosine LR 0.4, dropout 0.1, weight decay 0.001, mixup 0.2, two views, and final-quarter SWA. Parameters, gradients and momentum are BF16; accumulation, normalization statistics and SWA are FP32.

Prediction concatenates true-prefix-centered, half-normalized middle/final student embeddings after matching the two depth groups' row norms. Five-neighbor voting against true training labels supplies a prior, followed by three query-graph propagation steps. Query labels never enter the method. The package contains source code only; no fitted weights or data.

Native CUDA/cuBLASLt operations run in CUDA graphs. This is not a fully fused megakernel. The two student ensembles dominate runtime; a separate approximately two-second research variant has not qualified for D5 and is not included here.

## Reproduction and source identity

From `mnist-a100`:

```sh
python run_modal.py submissions/native-depthensemble-d5-20261009/submission.py:classify --difficulty 5 --runs 3 --json results/d5-review
```

The submission is 20,428 bytes, SHA256 `86a1a749f5b12c34be86c2e5025487827bda6b8b20f55fac72bb2911dcaa9066`. Its source-only archive has 30 readable files in `source/`, checked against `source_manifest.json`. The canonical runner used Torch 2.12, CUDA 13.3 and Python 3.13. Records include all 15 calls per evaluation, sandbox state, source hash, energy windows, and distinct GPU UUIDs. `qualification-audit.json` independently rechecks each record using the canonical judge. All three initial evaluations must pass to claim the median.
