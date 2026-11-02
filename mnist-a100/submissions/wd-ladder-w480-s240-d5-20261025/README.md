# Weight-decayed width-480 Ladder, 240 updates: difficulty 5

Entrant: [@spartanhaden](https://github.com/spartanhaden).

This is @sjbaebae's merged [codesign D5](../codesign-d4-d5-20261024/README.md) (#140, `d5_244`) with decoupled weight decay on its eight encoder/decoder matrices. With the decay it reaches the same error in 240 Adam updates instead of 315. Nothing else in the learner changes.

Three sandboxed Modal A100-80GB runs of the unchanged scorer (`mnist-a100/1.2.0`, stock `run_modal.py --difficulty 5 --runs 3`) all passed. **Median ranked time: 229.800 ms; median energy: 31,490 mJ per call.**

| Run | GPU | MNIST error | Margin (labels) | Hold-out | Ranked ms | mJ per call |
| ---: | --- | ---: | ---: | --- | ---: | ---: |
| [1](evidence/official/run-1.json) | A100-SXM4-80GB | 1.678% | +244 | KMNIST 97.32% | 229.800 | 33,029.3 |
| [2](evidence/official/run-2.json) | A100-SXM4-80GB | 1.723% | +195 | fashion 88.72% | 228.705 | 31,490.0 |
| [3](evidence/official/run-3.json) | A100-SXM4-80GB | 1.662% | +262 | KMNIST 97.42% | 242.864 | 31,225.2 |

An [earlier set](evidence/official-no-energy/) of the same file, without the energy column, also passed 3 of 3: median 224.542 ms, runs 230.390, 224.542 and 195.011 ms (the last on a PCIe host).

The current record, #140's reviewer median, is 273.889 ms and 40,390 mJ. This entry is 1.19× faster and uses 22% less energy, though the host mix differs. A paired in-process test on one GPU type isolates the change:

| 24 fresh pool splits × 11 calls, A100-SXM4-80GB | Mean error | Split sd | Worst split | Median ms |
| --- | ---: | ---: | ---: | ---: |
| #140 `d5_244`, unchanged (315 updates) | 1.746% | 0.075 | 1.887% | 239 |
| this file (240 updates, decay 0.5) | 1.731% | 0.079 | 1.877% | 190 |

The paired difference is −0.015 ± 0.007 points (17 of 24 splits better) at −20% time. Split *s* is a `mnist.Pool` from `default_rng(1027000 + s)` with its 11 draws ([records](evidence/fresh-splits/confirm-seed1027000.jsonl), [runner](evidence/fresh-splits/runner_modal.py)).

`submission.py` is 20,461 bytes, SHA-256 `132a5e02097ae2d8a776d0e5dd9706bb935a3d5a39ae27ffe025e299511f7b44`, matching `source_sha256` in every record ([manifest](evidence/official/manifest.json)). It passes `check_source`.

## Method

- **Optimizer.** `torch.optim.AdamW` with `weight_decay=0.5` on the four encoder and four decoder matrices. The batch-norm shifts and scales and the combinator parameters are not decayed. On our own Ladder, decaying every parameter erased the gain.
- **Updates.** 240 instead of 315. Every schedule in the file is a fraction of the step count, so the schedules rescale with it.
- **Bytes.** Two changes keep the file under the limit: `.022857142857142857` becomes `.16/7` (the same double), and `index_select(0,i).squeeze(0)` becomes `index_select(0,i)[0]` (the same view).

Development used a separate set of 12 splits (seed 1026000, [records](evidence/fresh-splits/development-seed1026000.jsonl)). Error at each setting, relative to the unchanged 315-update file:

| Updates | Decay 0 | Decay 0.3 | Decay 0.5 | Decay 0.8 |
| ---: | ---: | ---: | ---: | ---: |
| 270 | +0.061 | −0.034 | −0.069 (12/12 splits better) | — |
| 240 | — | — | −0.019 | −0.009 |
| 225 | — | — | — | +0.051 |
| 210 | — | — | — | +0.163 |

The weight decay came out of our D6 probes, prompted by the Telegram regularization discussion. On our long-budget Ladder it cut single-model error by about 0.15 points; dropout and DropConnect did not help.

## Credits

- Everything except the optimizer and step count is #140 by @sjbaebae. See its README for the Ladder and kernel lineage; the source's credit line is unchanged.

Co-developed with Claude.
