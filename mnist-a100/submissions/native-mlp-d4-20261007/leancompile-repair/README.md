# D4 cold-compilation repair

The earlier source failed two independent review runs at the second-call90-second KMNIST warmup limit. Historical local passes remain preserved in the parent directory. [Independent review](https://spacesheep.dev/@yaroslavvb/sutro-pr-review-20261008).

This replacement keeps the exact teacher/student learner and all archived CUDA kernels unchanged. C++ binding headers use tensor types and pybind casters, with host-O0/-g0 flags, to reduce cold compilation cost. The same exhaustive GEMM tuner remains. All three canonical Modal sandbox runs completed warmup and passed accuracy, timing and energy gates. Runtime attribution between compilation and tuning was not independently measured.

| Run | Error | Ranked ms | Above-idle mJ |
|---|---:|---:|---:|
| 1 | 2.227273% | 985.074 | 286669.683 |
| 2 | 2.220000% | 984.546 | 283665.268 |
| 3 | 2.228182% | 986.252 | 296628.249 |

Median **985.074ms**, median energy **286669.683mJ**, aggregate error **2.225152%**. Three distinct boards, unchanged source. No failing qualification run replaced.

Initial no-energy lean/bounded checks passed. The pruned-compilation alternative completed warmup but missed accuracy by two predictions; its failed record is retained and it is not the submitted source.

Source SHA256 `c5ed1d8ffb115cc58a6a6b4ac468b07c780d773b3f8d7387ec3f295b6f1b3eb1`, 19538bytes; source-only archive without fitted data. CUDA graphs/cuBLAS and custom kernels, not a whole-model megakernel. This PR does not itself change the leaderboard.

```bash
python run_modal.py submissions/native-mlp-d4-20261007/leancompile-repair/submission.py:classify --difficulty 4 --runs 3 --json out/d4-leancompile
```
