# D3 graph-readout repair

The earlier K5-only submission failed one independent review run at2.72909% error. These historical local records remain in the parent directory; they do not establish qualification. [Independent review](https://spacesheep.dev/@yaroslavvb/sutro-pr-review-20261008).

This replacement keeps E8 width256, three RMSNorm/ReLU hidden layers and1250updates, and adds K5/query-graph propagation (alpha0.9, three steps). It uses CUDA graphs with native kernels and cuBLAS; it is not a whole-model megakernel. Source-only archive, no fitted data.

| Run | MNIST error | Ranked ms | Above-idle mJ |
|---|---:|---:|---:|
| 1 | 2.608182% | 415.527 | 91822.431 |
| 2 | 2.633636% | 422.589 | 91854.397 |
| 3 | 2.530000% | 422.287 | 99702.583 |

All three initial energy-enabled canonical Modal runs passed. Median **422.287ms**, median energy **91854.397mJ**, aggregate error **2.590606%**. Initial no-energy diagnostic also retained. No failing run was replaced.

Source SHA256: `c6082659a49a6bd942077e359db42f28016626bb49303d68032467710d765a7e`; 17973bytes. Independently recomputed all scorer verdicts/ranked times and checked energy problems. This PR does not itself change the leaderboard.

Reproduce from mnist-a100:
```bash
python run_modal.py submissions/native-mlp-d3-20261007/graph1250-repair/submission.py:classify --difficulty 3 --runs 3 --json out/d3-graph1250
```
