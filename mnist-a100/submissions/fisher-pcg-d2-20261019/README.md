# RBF kernel ridge in a Fisher metric on a compressed training set: difficulty 2

## Independent reviewer reproduction — October 19, 2026

The first frozen reviewer group passed **3/3 fresh canonical sandboxed Modal A100-80GB runs**, with valid above-idle energy measurements (3 recorded GPU UUIDs). Median ranked time is **3.184645 ms/call** and median energy is **207.682235 mJ/call**; pooled MNIST error is **3.087576%** (319,811/330,000 correct).

| Reviewer run | MNIST error | Margin, labels | Ranked ms | Above-idle mJ |
| - | -: | -: | -: | -: |
| 1 | 2.987273% | +454 | 3.184645 | 189.029881 |
| 2 | 3.060000% | +374 | 3.918747 | 207.682235 |
| 3 | 3.215455% | +203 | 2.691631 | 281.843872 |

Source is unchanged at SHA-256 `de16bb0926ae35895400758697ba68f73e7f5ec366f2b84495b74c0abba89804`. The scorer and runner are unchanged. [Verification](evidence/reviewer-20261019/verification.json) reconstructs every complete canonical score and energy result from the [raw records](evidence/reviewer-20261019/); [source review](evidence/reviewer-20261019/source-review.json) covers per-call fitting, label isolation and cached-state reset. No reviewer outcome was replaced or retried.

This plain-PyTorch D2 port credits @sigkillme0's accepted [D3 Fisher/compression method](../kernel-d3-20261018/README.md), @spartanhaden's [earlier D2 kernel](../rfm-pcg-d2-20261012/README.md) and @islamborghini's [kernel-PCG](../kernel-pcg-20260929/README.md). The Fisher metric, log-radius feature, class-restricted compression and extrapolated PCG derive from #125; the triangular-solve preconditioner, side-stream factorization and D2 retuning are the submitted changes. Exact-source author records use working filename `kk_d2_final.py` and reconstruct separately. Development-call counts, probability estimates and optimization comparisons below were not independently reproduced by this review.

The leaderboard uses the independent reviewer medians above. Author results below are retained separately. The published instructions specify three official Modal runs and their median; this independent reproduction is review evidence, not a newly imposed second-batch submission rule.

Using Python 3.13 as in the canonical runner, recheck saved records from this submission directory with `python evidence/reviewer-20261019/verify_records.py evidence/reviewer-20261019 submission.py --difficulty 2 --function kc_d2 --output /tmp/pr131-verification.json`. This reconstructs recorded verdicts without executing the learner.

## Original author report (as submitted)

Present-tense record and pending-review claims below describe the submission date; the dated reviewer section above gives the accepted review status.


Entrant: [@spartanhaden](https://github.com/spartanhaden).

This entry ports @sigkillme0's merged d3 record ([#125](../kernel-d3-20261018/README.md)) to difficulty 2, in plain PyTorch, with a faster preconditioner solve and retuned constants.

Three sandboxed Modal A100-80GB runs of the unchanged scorer (`mnist-a100/1.2.0`, stock `run_modal.py --difficulty 2 --runs 3`) all passed.
- **Median ranked time:** 2.874 ms
- **Median energy above idle:** 282 mJ per call

| Run | GPU | MNIST error | Margin (labels) | Worst draw | Hold-out | Ranked ms | mJ above idle |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| [1](evidence/official/run-1.json) | A100-SXM4-80GB | 3.165% | +259 | 3.58% | Fashion 86.76% | 2.931 | 400 |
| [2](evidence/official/run-2.json) | A100 80GB PCIe | 3.113% | +316 | 3.38% | Fashion 86.67% | 2.874 | 255 |
| [3](evidence/official/run-3.json) | A100 80GB PCIe | 3.023% | +415 | 3.30% | KMNIST 95.69% | 2.699 | 282 |

Compared with the current record, our own [rfm-pcg-d2](../rfm-pcg-d2-20261012/README.md) (4.6 ms, 603 mJ), this entry is **1.59× faster** and uses 53% less energy per call.

`submission.py` is 16,400 bytes with SHA-256 `de16bb0926ae35895400758697ba68f73e7f5ec366f2b84495b74c0abba89804`, identical to `source_sha256` in all three records (which name it by its working name, `kk_d2_final.py`). It passes `check_source`. `evidence/official/manifest.json` lists the hash of every file.

## Method

No backprop. The whole call is one CUDA graph.

- **Metric (closed form).** A mix of LDA and the Fisher information of a tempered softmax over class Gaussians. Rows are mapped into it, put on a sphere and given a log-radius coordinate.
- **Compression.** Class-restricted k-means (24 centres per class). The 4,096 lowest-margin training rows are kept, and the rest are replaced by count-weighted centroids.
- **Fit.** RBF kernel ridge on the 4,336 weighted points by 5 preconditioned CG steps plus an extrapolated last direction. The preconditioner is Nyström on the centres plus a diagonal.
- **Predict.** Argmax of the test rows' kernel scores, through PyTorch's flash-attention kernel.

**Our changes to #125's method:**
- the preconditioner is applied through one triangular solve instead of an explicit inverse;
- fewer kernels;
- the factorisation chain runs on a high-priority side stream;
- constants retuned for the looser band.

Together these save 0.72 ms of GPU time per call against our first port, at +0.018 points of error. Not used from #125: its Triton kernels, the natural-gradient step and the HKNN readout.

Certified before the official set on 1,056 fresh calls: mean error 3.109%, worst draw 3.71%, P(3/3) ≈ 0.995.

## Credits

- @sigkillme0's [#125](../kernel-d3-20261018/README.md): the method. That is the Fisher metric, log-radius feature, k-means compression, FITC-style preconditioner, extrapolated PCG step and untimed warm-up replays. We re-implemented it and did not run their code.
- Our [rfm-pcg-d2](../rfm-pcg-d2-20261012/README.md), built on @islamborghini's [kernel-PCG](../kernel-pcg-20260929/README.md): kernel ridge with Nyström-preconditioned CG, flash-attention kernel products and the one-graph call.

## Notes

- Inductive: test rows are only row-scaled and featurised.
- On SXM4 hosts, expect about 2.9–3.2 ms, occasionally about 4 ms when the scorer's clock reads one hold-out call high.
- Earlier official sets of d2 files, all 3/3:
  - our first #125 port, 4.101 ms (`2fed8090`, 2026-10-19, unsubmitted);
  - our merged rfm-pcg-d2, 4.736 ms (`d2e38e10`);
  - an earlier d2 file, 4.637 ms (`96edf53a`, 2026-10-05).

  Earlier still, there were four single-run development probes on 10-04.
- Co-developed with Claude.

Reproduce from `mnist-a100/`:

```sh
python run_modal.py submissions/fisher-pcg-d2-20261019/submission.py:kc_d2 --difficulty 2 --runs 3 --json /tmp/d2-verification
```
