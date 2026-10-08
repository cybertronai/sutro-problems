# Difficulty 1 native BF16 MLP

Two 60-256-256-10 MLPs train for 30 updates in a CUDA graph. The softmax heads train the embeddings; three-neighbor voting over labelled training embeddings makes the final prediction. Native CUDA and cuBLAS implement the computation. The standalone source embeds its CUDA code.

## Three independent local sandbox evaluations

All three evaluations passed for the exact same source: **3.563617 ms** median ranked time, **4.864545%** mean MNIST error across evaluations. Each evaluation used 11 fresh MNIST calls and four holdout calls in fresh sandboxed processes (330,000 MNIST predictions total). Sandbox was explicitly required; all records report sandbox enabled and no problems.

| Evaluation | MNIST error % | Ranked ms | Raw evidence |
| --- | ---: | ---: | --- |
| 1 | 4.939091 | 3.451706 | [JSON](evaluation-run1.json) |
| 2 | 4.839091 | 3.665033 | [JSON](evaluation-run2.json) |
| 3 | 4.815455 | 3.563617 | [JSON](evaluation-run3.json) |

Source SHA-256: `d165b1cb8c395e3b2f3aa2f228c101818eaf389893b271b5d6d15358aaba8987`; size 20422 bytes (limit 20,480). The source directory now includes the process UID to avoid a permission collision with a preexisting development compilation directory. Learning and native kernels are unchanged by that loader fix.

Device: NVIDIA A100-SXM4-80GB. PyTorch: 2.12.1+cu129. Evaluator: mnist-a100/1.2.0. The upstream established D1 time at base a2ed895 is 61.7 ms; the local median is 17.31 times faster.

These are three local SSH-hosted sandbox evaluations, not three official Modal verification runs. Energy was skipped and has not been measured. The draft does not change the leaderboard or claim an established official record.

## Reproduction

From `mnist-a100`, invoke each time in a fresh process on an A100 with CUDA development tools and a working sandbox installation:

```bash
MNIST_SANDBOX=required python mnist.py submissions/native-mlp-d1-20261007/mlp_neighbor3_register_d1.py:classify --difficulty 1 --no-energy --json evaluation.json
```

Repeat three times with distinct output paths. This benchmark randomly releases fresh draws each time. The official upstream alternative is `python run_modal.py submissions/native-mlp-d1-20261007/mlp_neighbor3_register_d1.py:classify --difficulty 1 --runs 3`.
