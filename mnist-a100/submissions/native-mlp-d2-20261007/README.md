# Difficulty 2 native BF16 MLP

Adds the standalone native difficulty 2 MLP candidate with three independent sandbox evaluations of the exact submitted source. The per-UID temporary-directory fix preserves the native arithmetic and permits required sandbox execution.

Two 60–256–256–10 MLPs train for 120 updates, followed by a three-neighbor embedding readout. The source is within the 20 KiB limit. All three evaluations used the required sandbox, completed 11 MNIST and four holdout calls, and reported no problems.

| Run | MNIST error | Evaluator-ranked time |
|---|---:|---:|
| 1 | 3.386364% | 9.586 ms |
| 2 | 3.397273% | 10.158 ms |
| 3 | 3.267273% | 9.689 ms |

Median evaluator-ranked time: **9.689 ms**. Run 2 exceeded the 10 ms target at 10.158 ms; this is retained in the evidence. These are local independent sandbox results. Official energy measurements and the official median protocol remain outstanding. No leaderboard change is included.

Source SHA-256: `3f949ff304de18b537e660fcdbea24bf60697d2799294d8af348bff4f4971023`. Raw JSON and the validation manifest accompany the candidate.

## Reproduction

```bash
MNIST_SANDBOX=required python mnist.py submissions/native-mlp-d2-20261007/mlp_neighbor3_register_d2.py:classify --difficulty 2 --no-energy --json evaluation.json
```

Repeat in three fresh processes with distinct output paths.
