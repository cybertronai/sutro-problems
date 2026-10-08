# Difficulty 2 native BF16 MLP

The standalone native difficulty 2 MLP passed three fresh official Modal evaluations for the exact submitted source. The per-UID temporary-directory fix preserves the native arithmetic and permits required sandbox execution.

Two 60–256–256–10 MLPs train for 120 updates, followed by a three-neighbor embedding readout. The source is within the 20 KiB limit. All three evaluations used the required sandbox, completed 11 MNIST and four holdout calls, and reported no problems.

## Official qualification — 2026-10-08

**Three of three fresh canonical Modal A100-80GB runs passed** for the unchanged source. Median ranked time is **14.600212215 ms per call**; median energy above idle is **1,332.631133937 mJ per call**. The 33 MNIST calls together predicted **319,164 / 330,000** labels correctly, for **3.283636% aggregate error**, below the difficulty-2 bound of 3.40%.

| Run | GPU / power limit | MNIST error % | Margin, correct labels | Ranked ms | Energy mJ above idle | Raw evidence |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | A100-SXM4-80GB, 400 W | 3.249091 | +166 | 14.656805 | 1,383.597654 | [JSON](official/run-1.json) |
| 2 | A100-SXM4-80GB, 400 W | 3.312727 | +96 | 14.600212 | 1,332.631134 | [JSON](official/run-2.json) |
| 3 | A100-SXM4-80GB, 400 W | 3.289091 | +122 | 14.227056 | 1,326.673694 | [JSON](official/run-3.json) |

All three official runs used **distinct A100-SXM4-80GB boards**. Each included 11 MNIST and four holdout calls, with fresh sandboxed workers and a foreign warmup for each timed call. The holdout was KMNIST in runs 1 and 3 and Fashion-MNIST in run 2. All canonical accuracy, timing, sandbox and energy gates passed. The median includes all three initially requested runs, with no retries or result selection. Official software was PyTorch 2.12.0+cu130 with scorer `mnist-a100/1.2.0`.

Ranked time includes the canonical scorer's independent parent-process clock floor, after its empty-method overhead adjustment. CUDA-event time alone and the historical local timing below do not replace this official score.

The earlier local runs 1 and 2 cleared the mean-error threshold by only **15 and 3 correct labels**, respectively. The fresh official margins are **166, 96 and 122 labels** above the exact 106,260 / 110,000 threshold. These observed passes do not establish a failure probability or guarantee qualification on future draws.

The exact source is SHA-256 `3f949ff304de18b537e660fcdbea24bf60697d2799294d8af348bff4f4971023`, 20,423 bytes. Source identity, every canonical judge result, ranked time and full energy record were independently reconstructed in [verification.json](official/verification.json). Execution provenance is in [manifest.json](official/manifest.json); exact copied evidence hashes are in [evidence-sha256.json](official/evidence-sha256.json). The learner, original local records and original validation manifest are unchanged.

## Historical local sandbox evaluations

| Run | MNIST error | Evaluator-ranked time | Raw evidence |
| ---: | ---: | ---: | --- |
| 1 | 3.386364% | 9.586 ms | [JSON](evaluation-run1.json) |
| 2 | 3.397273% | 10.158 ms | [JSON](evaluation-run2.json) |
| 3 | 3.267273% | 9.689 ms | [JSON](evaluation-run3.json) |

Median local evaluator-ranked time: **9.689 ms**. Run 2 exceeded the author's 10 ms development target at 10.158 ms; this is retained in the evidence. Energy was skipped in these earlier local runs. Their raw JSON and [validation manifest](validation-manifest.json) remain historical evidence; the leaderboard now uses the official three-run medians above.

## Reproduction

From `mnist-a100`, with Modal configured, reproduce the canonical three-run timing and energy protocol:

```bash
python run_modal.py submissions/native-mlp-d2-20261007/mlp_neighbor3_register_d2.py:classify \
  --difficulty 2 --runs 3 --json /tmp/native-mlp-d2-official
```

To reproduce a historical local evaluation in a fresh process on an A100 with CUDA development tools and a working sandbox installation:

```bash
MNIST_SANDBOX=required python mnist.py submissions/native-mlp-d2-20261007/mlp_neighbor3_register_d2.py:classify --difficulty 2 --no-energy --json evaluation.json
```

Repeat the local command in three fresh processes with distinct output paths. Both protocols randomly release fresh draws each time; the official command also measures the canonical energy column.
