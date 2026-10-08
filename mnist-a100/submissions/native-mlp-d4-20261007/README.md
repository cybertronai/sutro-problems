# Native BF16 self-training candidate for difficulty 4

Qualification: **3/3 independent local sandbox evaluations passed**. The local push gate is satisfied. Official timing and energy verification remain outstanding. This package does not modify the leaderboard.

Two independent ensembles, each eight 60–256–256–256–10 ReLU/RMSNorm MLPs, train for 1,000 and 250 updates and pseudo-label the released query inputs. Accepted predictions have averaged confidence at least 0.999. Accepted query rows are cyclically repeated to exactly the query count; if none qualify, true-labelled training rows fill that count. The constant training size allows reuse of the student CUDA graph across the evaluator's foreign warmup and measured MNIST draw.

Four 60–512–512–512–10 SiLU/RMSNorm student MLPs train for 1,000 updates with batch 2,048 and two mixed views. They concatenate final and penultimate embeddings for a five-neighbor readout over the original true-labelled training examples, followed by three query-graph propagation steps. Query ground truth is never available to the learner. This is transductive self-training on released inputs, without generated input points or reconstruction.

Model state uses BF16; GEMM accumulation and RMS statistics use FP32. Training uses native CUDA, cuBLAS/cuBLASLt Tensor Core GEMMs and CUDA graphs. It is not a whole-model fused megakernel. Python orchestrates selection and readouts.

## Frozen-source evidence

`submission.py` is 19,515bytes; SHA256 `8b7790e61ce658ba5ee0409c6b59014e4312331ae2a2f892ec57cbabeda81175`. Its encoded archive contains Python/CUDA source only. The readable `source/` tree and source manifest are included for inspection; no fitted weights or dataset are embedded.

| Independent evaluation | MNIST error | Ranked ms | Result |
|---|---:|---:|---|
| 1 | 2.200909% | 990.631 | Passed |
| 2 | 2.237273% | 991.149 | Passed |
| 3 | 2.167273% | 991.037 | Passed |

Every evaluation includes 11 MNIST and 4 holdout calls in fresh processes with sandbox enabled. Compilation/autotuning warmup is outside the method timing. Raw records and exact-source validation manifest accompany this package. Local sandbox passes do not establish official energy measurements.

Run from the benchmark directory:

```sh
MNIST_SANDBOX=required python3 mnist.py submissions/native-mlp-d4-20261007/submission.py:classify --difficulty 4 --no-energy --json evaluation.json
```

The earlier variable-size candidate failed its first sandbox evaluation at 2.346364%; that record remains in the research workspace and is not counted toward this revised source.
