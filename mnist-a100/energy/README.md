# Energy of the MNIST-on-A100 methods under a minute

Measured 2026-09-26 with the scorer's energy column (`mnist.py` 1.2.0): every
known method that passes a difficulty in under 60 s per call, three runs each,
each run in its own Modal A100-80GB container, sandbox on. Energy is the GPU
board's energy per call above idle; time is the ranked time per call. Medians
of three runs.

| Method | Difficulty | ms per call | mJ per call above idle | W above idle in a call | MNIST |
| --- | :-: | ---: | ---: | ---: | ---: |
| CUDA-graph MLP, 60-256-256-10, 200 steps ([PR #96](https://github.com/cybertronai/sutro-problems/pull/96)) | 1 | 61.6 | 2,429 | 48 | 95.03% |
| eager MLP, 60-1024-1024-10, 100 steps | 1 | 191.3 | 2,648 | 13 | 94.98% |
| [`example.py`](../example.py), eager MLP, 60-1024-1024-10, 400 steps | 1 | 762.4 | 29,662 | 42 | 96.77% |
| CUDA-graph ensemble, 4 x 60-256-256-10, 800 steps | 2 | 252.1 | 14,436 | 57 | 96.87% |
| eager ensemble, 16 x 60-1024-1024-10, 400 steps | 2 | 927.3 | 200,870 | 217 | 96.78% |

* **The fastest method at each difficulty also uses the least energy:** about
  2.4 J per call at difficulty 1 and 14 J at difficulty 2, both with the whole
  training step in a CUDA graph.
* **Time and energy rank differently.** The eager MLP takes three times as long
  as the CUDA-graph MLP for about the same energy. It is launch-bound, so the GPU
  mostly waits: 13 W above idle during a call, against 48 W.
* **The example costs twelve times the fastest method's energy.** Its matmuls
  run in full FP32; every other method here enables TF32.
* **The eager ensemble costs fourteen times the CUDA-graph ensemble's energy**
  for the same band. It runs near the PCIe card's 300 W limit (298 W on average).
* **The board moves energy more than time.** Modal's A100-80GB is sometimes an
  A100 80GB PCIe card and sometimes an A100-SXM4-80GB board. In the energy window
  the CUDA-graph MLP took about 50 ms per call on both, but read 1.39 J on a PCIe
  card (1.44 J in the smoke-test run) against 2.43-2.64 J on SXM4 boards. The example ran faster
  on its SXM4 host (486 ms against 762-793 ms) and used more energy (37.9 J against
  28.8-29.7 J). A record takes the median of its runs for this reason.

## Every run

| Method | Run | GPU | ms per call | mJ per call | Energy window | Idle W | Round trip mJ | Reference J/TFLOP |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| CUDA-graph MLP (PR #96) | smoke | A100 80GB PCIe | 51.0 | 1,441 | 364 calls, 20.0 s, 49.6 ms each | 78.2 | 17.2 | 7.21 |
| CUDA-graph MLP (PR #96) | 1 | A100 80GB PCIe | 51.2 | 1,393 | 359 calls, 20.0 s, 49.7 ms each | 75.9 | 18.9 | 7.74 |
| CUDA-graph MLP (PR #96) | 2 | A100-SXM4-80GB | 61.6 | 2,637 | 355 calls, 20.0 s, 50.1 ms each | 69.3 | 13.2 | 9.22 |
| CUDA-graph MLP (PR #96) | 3 | A100-SXM4-80GB | 62.0 | 2,429 | 343 calls, 20.0 s, 50.3 ms each | 68.6 | 16.6 | 8.94 |
| eager MLP | 1 | A100 80GB PCIe | 168.2 | 3,258 | 116 calls, 20.2 s, 168.1 ms each | 80.9 | 13.3 | 7.61 |
| eager MLP | 2 | A100-SXM4-80GB | 191.3 | 2,370 | 100 calls, 20.1 s, 194.6 ms each | 69.8 | 14.1 | 8.10 |
| eager MLP | 3 | A100-SXM4-80GB | 196.2 | 2,648 | 97 calls, 20.1 s, 197.1 ms each | 67.7 | 17.4 | 8.78 |
| `example.py` | 1 | A100 80GB PCIe | 793.0 | 29,662 | 29 calls, 20.5 s, 702.8 ms each | 82.2 | 19.0 | 7.42 |
| `example.py` | 2 | A100 80GB PCIe | 762.4 | 28,810 | 25 calls, 20.1 s, 784.9 ms each | 72.4 | 18.6 | 7.56 |
| `example.py` | 3 | A100-SXM4-80GB | 486.5 | 37,930 | 38 calls, 20.1 s, 512.3 ms each | 68.0 | 12.7 | 7.98 |
| CUDA-graph ensemble | 1 | A100 80GB PCIe | 251.8 | 14,333 | 77 calls, 20.2 s, 255.4 ms each | 72.9 | 19.1 | 7.40 |
| CUDA-graph ensemble | 2 | A100 80GB PCIe | 252.1 | 14,524 | 77 calls, 20.0 s, 255.2 ms each | 74.0 | 20.1 | 7.33 |
| CUDA-graph ensemble | 3 | A100 80GB PCIe | 253.4 | 14,436 | 79 calls, 20.2 s, 250.8 ms each | 80.0 | 18.6 | 7.69 |
| eager ensemble | 1 | A100 80GB PCIe | 927.3 | 202,072 | 22 calls, 20.5 s, 928.2 ms each | 78.2 | 21.0 | 7.89 |
| eager ensemble | 2 | A100 80GB PCIe | 943.8 | 198,165 | 22 calls, 20.5 s, 927.9 ms each | 85.6 | 20.4 | 7.43 |
| eager ensemble | 3 | A100 80GB PCIe | 922.0 | 200,870 | 22 calls, 20.4 s, 925.8 ms each | 81.0 | 16.5 | 7.60 |

Every run passed its difficulty, and every energy passed every check. The
sixteen runs landed on fifteen boards (one SXM4 board served two). The smoke-test
run is not in the medians above.

## Why the column is measured this way

`probe_nvml.py` measured the telemetry on one A100-SXM4-80GB
([`results/probe-a100-80gb.json`](results/probe-a100-80gb.json)):

| Measured | Consequence in the scorer |
| --- | --- |
| The energy counter moves every 100 ms (40 changes in 4 s, gaps 95-106 ms) | No call is read on its own: a 20 s window of back-to-back calls |
| Thirty isolated 74 ms bursts of 8 FP32 matmuls read 8.1-15.4 J against the 16.5 J the same work costs in a long window; their edges alone read -5.0 to 11.1 J | The same |
| Idle drew 60.4 W with no CUDA context, 67.3 W with an idle context and 68.8 W with that process stopped | Idle is measured with the method's process frozen and its context open |
| A read of the counter takes 3.1 ms, and polled in a tight loop the counter advanced at about 90 W against 60-67 W read sparsely | The counter is read only at window edges |
| The same FP32 matmul drew 351 W (15.0 J/TFLOP above idle) on random operands; the earlier harness's constant operands read 8.1-8.8 J/TFLOP on six boards | The telemetry check uses constant operands, whose band is known: 6-11 J/TFLOP |

The fifteen boards here read 7.2-9.2 J/TFLOP at 17.2-18.4 TFLOP/s on that check,
PCIe cards 7.2-7.9 and SXM4 boards 8.0-9.2. The round trip without the method
cost 13-21 mJ per call, at most 1.4% of any method's energy. In the energy window
the GPU ran the method 87-90% of the time for the 50 ms CUDA-graph MLP and
96-99.7% for the others.

## What the number leaves out

* **Idle.** Counting the board's idle draw too, a CUDA-graph MLP call costs
  5.6-6.6 J, not 1.4-2.6 J; the record for each run is in its JSON.
* **Everything but the GPU board:** host CPU, memory, power supply, cooling.
* **The first call in a process.** Energy calls run back to back in one warm
  process. On SXM4 hosts the CUDA-graph MLP took about 50 ms per call there,
  against 61-62 ms in the fresh-process timed calls.

## Reproduce

From `mnist-a100/`, with Modal configured (the three ports are generated from the
cutoff study's `mnist/experiments/release-cutoffs-20260925/mlp_timing/submissions`):

```bash
python energy/probe_nvml.py --out energy/results/probe-a100-80gb.json
python energy/port.py
python run_modal.py energy/entries/fast_mlp.py:fast_mlp --difficulty 1 --runs 3 --json energy/results/fast_mlp
python run_modal.py energy/entries/mlp_k1_w1024_s100_b512.py:mlp --difficulty 1 --runs 3 --json energy/results/mlp_k1_w1024_s100_b512
python run_modal.py example.py:mlp --difficulty 1 --runs 3 --json energy/results/example
python run_modal.py energy/entries/mlpg_k4_w256_s800_b512.py:mlp --difficulty 2 --runs 3 --json energy/results/mlpg_k4_w256_s800_b512
python run_modal.py energy/entries/mlp_k16_w1024_s400_b512.py:mlp --difficulty 2 --runs 3 --json energy/results/mlp_k16_w1024_s400_b512
python energy/summarize.py   # the tables above, recomputed from every run's raw energy windows
```

`entries/fast_mlp.py` is byte-identical to PR #96's submission (sha256
`44cb0c03d4ac869e99186c584d234c8743c12565e0a3b49cf3172bd5c8a82325`). The ports
change only the signature: `custom_kernel(data)` becomes `mlp(train_x, train_y, test_x)`.

Cost: $2.35 billed by Modal for the seven apps (the probe $0.09, the smoke test
$0.12, the fifteen scored runs $2.13).
