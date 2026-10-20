# RBF kernel ridge in a Fisher metric on a compressed training set: difficulty 2

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
