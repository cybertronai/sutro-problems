# Kernel-taught Ladder with a fused Triton combinator backward: difficulty 5

## Independent reviewer reproduction — October 19, 2026

The first frozen reviewer group passed **3/3 fresh canonical sandboxed Modal A100-80GB runs**, with valid above-idle energy measurements (3 recorded GPU UUIDs). Median ranked time is **633.529424 ms/call** and median energy is **107,885.521303 mJ/call**; pooled MNIST error is **1.750000%** (324,225/330,000 correct).

| Reviewer run | MNIST error | Margin, labels | Ranked ms | Above-idle mJ |
| - | -: | -: | -: | -: |
| 1 | 1.750000% | +165 | 630.967276 | 118,481.918772 |
| 2 | 1.710909% | +208 | 633.529424 | 107,885.521303 |
| 3 | 1.789091% | +122 | 636.385705 | 106,205.624267 |

Source is unchanged at SHA-256 `cc423b95ed940a5bcd5dc3b08f71502dddbcee3b42cbc34fc5322819378aef2a`. The scorer and runner are unchanged. [Verification](evidence/reviewer-20261019/verification.json) reconstructs every complete canonical score and energy result from the [raw records](evidence/reviewer-20261019/); [source review](evidence/reviewer-20261019/source-review.json) covers per-call fitting, label isolation and cached-state reset. No reviewer outcome was replaced or retried.

This is @spartanhaden's expanded kernel-taught Ladder variant: the accepted [Ladder/kernel teacher](../ladder-kernel-d5-20261012/README.md) and [D4 teacher](../ladder-kernel-d4-20261012/README.md), @SethTS/Rasmus et al.'s Ladder work and @islamborghini's [kernel-PCG](../kernel-pcg-20260929/README.md) remain credited. It also credits @islamborghini's [batch/step tuning](../ladder-kernel-tuned-d5-20261017/README.md), @likelytobelaura's [tangent jitter](../tangent-crossteach-d5-20261012/README.md), and @sigkillme0's held #127 for grouping steps in graph replays. The branchless guard chooses between already-computed predictions; it does not skip the teacher or any of the 600 student updates. Exact-source author records use working filename `a_k2t_600.py` and reconstruct separately. Development-call counts, probability estimates and historical kernel chronology below are author assertions, not independently reproduced qualification evidence.

The leaderboard uses the independent reviewer medians above. Author results below are retained separately. The published instructions specify three official Modal runs and their median; this independent reproduction is review evidence, not a newly imposed second-batch submission rule.

Using Python 3.13 as in the canonical runner, recheck saved records from this submission directory with `python evidence/reviewer-20261019/verify_records.py evidence/reviewer-20261019 submission.py --difficulty 5 --function ladder_kernel --output /tmp/pr129-verification.json`. This reconstructs recorded verdicts without executing the learner.

## Original author report (as submitted)

Present-tense record and pending-review claims below describe the submission date; the dated reviewer section above gives the accepted review status.


Entrant: [@spartanhaden](https://github.com/spartanhaden).

Three sandboxed Modal A100-80GB runs of the unchanged scorer (`mnist-a100/1.2.0`, stock `run_modal.py --difficulty 5 --runs 3`) all passed.
- **Median ranked time:** 632.089 ms
- **Median energy above idle:** 112,581 mJ per call

| Run | GPU | MNIST error | Margin (labels) | Worst draw | Hold-out | Ranked ms | mJ above idle |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| [1](evidence/official/run-1.json) | A100-SXM4-80GB | 1.850% | +55 | 2.05% | KMNIST 97.26% | 632.089 | 112,581 |
| [2](evidence/official/run-2.json) | A100 80GB PCIe | 1.708% | +211 | 1.98% | KMNIST 97.27% | 636.150 | 110,725 |
| [3](evidence/official/run-3.json) | A100-SXM4-80GB | 1.636% | +290 | 1.89% | KMNIST 97.34% | 630.646 | 119,454 |

Compared with the current record (897.5 ms, 154,289 mJ; @islamborghini's [tuning of our Ladder](../ladder-kernel-tuned-d5-20261017/README.md)), this entry is **1.42× faster** and uses 27% less energy per call.

`submission.py` is 20,474 bytes with SHA-256 `cc423b95ed940a5bcd5dc3b08f71502dddbcee3b42cbc34fc5322819378aef2a`, identical to `source_sha256` in all three records (which name it by its working name, `a_k2t_600.py`). It passes `check_source`. `evidence/official/manifest.json` lists the hash of every file.

## Method

- **Teacher.** Learned-metric kernel ridge, as in our [d4 entry](../ladder-kernel-d4-20261012/README.md): two AGOP metric updates, Nyström-preconditioned CG and a pseudo-label refit.
- **Student.** The same Ladder network, 60-1024-512-256-10 on AGOP^0.5 inputs, now trained for 600 Adam steps of batch 2,500 (LR 0.011). Labelled rows take random steps along 4 local tangents (offsets to their 10 nearest same-class rows). In the second half, the 20% of test rows the teacher is least sure of are sampled twice as often.
- **Guard.** A rare bad training path (about 1 call in 2,000) can flip a whole class. One branchless check per call compares the student with the teacher's confident labels and with the training labels; if it trips, the teacher's confident labels are used. It caught every bad path in our testing. It also trips on every Fashion hold-out call, which still scores 88–89%.
- **Speed.** The sampler, score update and Adam run inside the compiled step, 30 steps are captured per CUDA-graph replay, and activations are bf16. One hand-written Triton kernel, `comb_bwd_kernel`, computes the backward pass of the three wide combinator layers. It saves about 100 ms per call at the same accuracy.

Certified before the official set on 4,301 fresh in-process A100 calls: mean error 1.739%, P(3/3) ≈ 0.99.

## Credits

- Our merged [ladder-kernel-d5](../ladder-kernel-d5-20261012/README.md) / [d4](../ladder-kernel-d4-20261012/README.md) entries (#121) are the base. They follow @SethTS's [Ladder entries](../ladder-sampled-20260929/README.md) (Rasmus et al. 2015) and build the teacher on @islamborghini's [kernel-pcg](../kernel-pcg-20260929/README.md).
- @islamborghini's #124 retuned that base to fewer, bigger steps. We had moved the same way: an official batch-2,500 set on 2026-10-16 (904.6 ms, `7ece8e50`, unsubmitted).
- Local tangent jitter: @likelytobelaura's [tangent-crossteach-d5](../tangent-crossteach-d5-20261012/README.md) (#118).
- Several steps per CUDA-graph replay: after @sigkillme0's held #127. Nothing else from #127 is used. Our Triton kernel was written on 2026-10-16, before #127 was posted, and is independent of its fused kernels.

## Notes

- Transductive: test rows are used without labels, in the unlabelled pool, as distillation targets and by the guard.
- Training is nondeterministic on the GPU, so reruns on the same draw flip a few labels.
- Cold warm-up (compile, Triton build, graph capture) takes about 33 s.
- The docstring's "as the difficulty-4 entry" refers to our merged ladder-kernel-d4. We left the wording so the file stays byte-identical to the official source.
- Earlier official sets of related d5 files, all 3/3: the same learner without the Triton kernel (727.970 ms, `4418dc26`, 2026-10-19, unsubmitted), the 10-16 set above, a 2,300-step variant of our #121 file (1,817.1 ms, `720948f8`, 2026-10-05) and our #121 entry's own set.
- Co-developed with Claude.

Reproduce from `mnist-a100/`:

```sh
python run_modal.py submissions/ladder-k2-d5-20261019/submission.py:ladder_kernel --difficulty 5 --runs 3 --json /tmp/d5-verification
```
