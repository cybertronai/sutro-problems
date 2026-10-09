# Difficulty 1 native BF16 MLP

Two 60-256-256-10 MLPs train for 30 updates in a CUDA graph. The softmax heads train the embeddings; three-neighbor voting over labelled training embeddings makes the final prediction. Native CUDA and cuBLAS implement the computation. The standalone source embeds its CUDA code.

## Official qualification — 2026-10-08

**Three of three fresh official Modal A100-80GB runs passed** for the unchanged source. Median ranked time is **6.705614625 ms per call**; median energy above idle is **529.576891 mJ per call**. The 33 MNIST calls together predicted **314,013 / 330,000** labels correctly, for **4.844545% aggregate error**, below the difficulty-1 bound of 5.40%.

| Run | GPU / power limit | MNIST error % | Ranked ms | Energy mJ above idle | Raw evidence |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | A100 80GB PCIe, 300 W | 4.898182 | 6.705615 | 529.576891 | [JSON](official/run-1.json) |
| 2 | A100-SXM4-80GB, 400 W | 4.877273 | 7.178109 | 501.677797 | [JSON](official/run-2.json) |
| 3 | A100-SXM4-80GB, 400 W | 4.758182 | 6.131708 | 760.733904 | [JSON](official/run-3.json) |

The canonical runner sampled three distinct boards: **one PCIe and two SXM4**. The reported median includes all three initially requested runs, with no retries or result selection. Each run used 11 MNIST and four KMNIST holdout calls, with fresh sandboxed workers and a foreign warmup for each timed call. All canonical accuracy, timing, sandbox and energy gates passed. Official software was PyTorch 2.12.0+cu130 with scorer `mnist-a100/1.2.0`.

The ranked time includes the scorer's independent parent-process clock floor, after its empty-method overhead adjustment; CUDA-event time alone does not determine the score. For example, run 1 averaged 3.426769 ms of MNIST CUDA-event time and 3.475200 ms on KMNIST, while its canonical ranked time was 6.705615 ms. The historical SSH-hosted measurements below used another environment and do not replace this official result.

The exact source is SHA-256 `d165b1cb8c395e3b2f3aa2f228c101818eaf389893b271b5d6d15358aaba8987`, 20,422 bytes. Source identity, every canonical judge result, ranked time and full energy record were independently reconstructed from the saved records in [verification.json](official/verification.json). Execution provenance is in [manifest.json](official/manifest.json); exact copied evidence hashes are in [evidence-sha256.json](official/evidence-sha256.json). The learner, author local records and author validation manifest are unchanged.

## Historical local sandbox evaluations

All three evaluations passed for the exact same source: **3.563617 ms** median ranked time, **4.864545%** mean MNIST error across evaluations. Each evaluation used 11 fresh MNIST calls and four holdout calls in fresh sandboxed processes (330,000 MNIST predictions total). Sandbox was explicitly required; all records report sandbox enabled and no problems.

| Evaluation | MNIST error % | Ranked ms | Raw evidence |
| --- | ---: | ---: | --- |
| 1 | 4.939091 | 3.451706 | [JSON](evaluation-run1.json) |
| 2 | 4.839091 | 3.665033 | [JSON](evaluation-run2.json) |
| 3 | 4.815455 | 3.563617 | [JSON](evaluation-run3.json) |

Source SHA-256: `d165b1cb8c395e3b2f3aa2f228c101818eaf389893b271b5d6d15358aaba8987`; size 20422 bytes (limit 20,480). The source directory now includes the process UID to avoid a permission collision with a preexisting development compilation directory. Learning and native kernels are unchanged by that loader fix.

Device: NVIDIA A100-SXM4-80GB. PyTorch: 2.12.1+cu129. Evaluator: mnist-a100/1.2.0. The upstream established D1 time at base a2ed895 is 61.7 ms; the local median is 17.31 times faster.

These three SSH-hosted sandbox evaluations preceded the official qualification. Energy was skipped in these local runs. Their records and validation manifest remain historical evidence; the leaderboard now uses the official three-run medians above.

## Reproduction

From `mnist-a100`, with Modal configured, reproduce the canonical three-run timing and energy protocol:

```bash
python run_modal.py submissions/native-mlp-d1-20261007/mlp_neighbor3_register_d1.py:classify \
  --difficulty 1 --runs 3 --json /tmp/native-mlp-d1-official
```

To reproduce a historical local evaluation in a fresh process on an A100 with CUDA development tools and a working sandbox installation:

```bash
MNIST_SANDBOX=required python mnist.py submissions/native-mlp-d1-20261007/mlp_neighbor3_register_d1.py:classify --difficulty 1 --no-energy --json evaluation.json
```

Repeat the local command three times with distinct output paths. Both protocols randomly release fresh draws each time; the official command also measures the canonical energy column.
