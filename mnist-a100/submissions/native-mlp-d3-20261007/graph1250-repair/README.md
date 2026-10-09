# D3 graph-readout repair

## Independent reviewer qualification — 2026-10-08

The repaired source passed **all three initially requested fresh canonical Modal A100-80GB runs**, including full energy measurements. Median ranked time is **416.134586331 ms per call**; median GPU energy above idle is **98,226.023073203 mJ per call**. The 33 MNIST calls predicted **321,509 / 330,000** labels correctly, for **2.573030% aggregate error**. Each individual run passed the difficulty-3 2.70% mean-error gate and every other canonical timing, holdout, sandbox and energy gate.

| Run | GPU | MNIST error % | Margin, correct labels | Ranked ms | Energy mJ above idle | Raw evidence |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | NVIDIA A100 80GB PCIe | 2.573636 | +139 | 384.850889 | 83843.534463 | [JSON](official/run-1.json) |
| 2 | NVIDIA A100-SXM4-80GB | 2.479091 | +243 | 416.134586 | 98226.023073 | [JSON](official/run-2.json) |
| 3 | NVIDIA A100-SXM4-80GB | 2.666364 | +37 | 420.447074 | 100261.515947 | [JSON](official/run-3.json) |

All three runs used distinct GPU UUIDs and completed 11 MNIST and four holdout calls. Ranked time includes the unchanged scorer's parent-clock floor; GPU energy is idle-adjusted board energy, not whole-system energy. These observed passes do not estimate future qualification probability.

The submitted learner remains byte-identical at SHA-256 `c6082659a49a6bd942077e359db42f28016626bb49303d68032467710d765a7e`, 17,973 bytes. The source-only archive and all 15 decoded files were reviewed for label isolation and reset of weights, momentum and SWA on every graph replay; no fitted-data or retained-trained-state blocker was found. [Human source review](official/human-source-review.json) records the resolved encoded-source advisory flags.

The unchanged scorer and runner identities, Modal application and reviewed PR head are in [manifest.json](official/manifest.json). Every complete scorer verdict, ranked time and full energy report was independently reconstructed in [verification.json](official/verification.json). Copied evidence hashes are in [evidence-sha256.json](official/evidence-sha256.json).

The author's three energy runs and initial no-energy diagnostic below are retained separately and independently rescored in [author-record-audit.json](official/author-record-audit.json). They are not mixed into the reviewer median. The earlier K5-only source failed the previous independent set; all three of those outcomes are retained in [historical-review](../historical-review/verification.json). This repair changes the readout and therefore starts a new three-run qualification set. No failed run was replaced or rerun for either source version.


The earlier K5-only submission failed one independent review run at2.72909% error. These historical local records remain in the parent directory; they do not establish qualification. [Independent review](https://spacesheep.dev/@yaroslavvb/sutro-pr-review-20261008).

This replacement keeps E8 width256, three RMSNorm/ReLU hidden layers and1250updates, and adds K5/query-graph propagation (alpha0.9, three steps). It uses CUDA graphs with native kernels and cuBLAS; it is not a whole-model megakernel. Source-only archive, no fitted data.

| Run | MNIST error | Ranked ms | Above-idle mJ |
|---|---:|---:|---:|
| 1 | 2.608182% | 415.527 | 91822.431 |
| 2 | 2.633636% | 422.589 | 91854.397 |
| 3 | 2.530000% | 422.287 | 99702.583 |

All three author-supplied initial energy-enabled canonical Modal runs passed. Median **422.287ms**, median energy **91854.397mJ**, aggregate error **2.590606%**. Initial no-energy diagnostic also retained. No failing run was replaced.

Source SHA256: `c6082659a49a6bd942077e359db42f28016626bb49303d68032467710d765a7e`; 17973bytes. Independently recomputed all scorer verdicts/ranked times and checked energy problems. The leaderboard uses the independent reviewer medians above; this table preserves the author-supplied results.

Reproduce from mnist-a100:
```bash
python run_modal.py submissions/native-mlp-d3-20261007/graph1250-repair/submission.py:classify --difficulty 3 --runs 3 --json out/d3-graph1250
```
