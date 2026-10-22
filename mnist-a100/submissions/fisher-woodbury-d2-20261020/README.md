# Fisher-metric kernel ridge with a standard-form Woodbury preconditioner: difficulty 2

Entrant: [@spartanhaden](https://github.com/spartanhaden).

This entry speeds up our merged [fisher-pcg-d2](../fisher-pcg-d2-20261019/README.md) (#131). The learner and all its constants are unchanged; only the linear algebra and the stream layout around it are restructured.

Three sandboxed Modal A100-80GB runs of the unchanged scorer (`mnist-a100/1.2.0`, stock `run_modal.py --difficulty 2 --runs 3`) all passed.
- **Median ranked time:** 2.696 ms
- **Median energy above idle:** 234 mJ per call

| Run | GPU | MNIST error | Margin (labels) | Worst draw | Hold-out | Ranked ms | mJ above idle |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| [1](evidence/official/run-1.json) | A100 80GB PCIe | 3.083% | +349 | 3.32% | Fashion 86.83% | 2.602 | 234 |
| [2](evidence/official/run-2.json) | A100-SXM4-80GB | 3.063% | +371 | 3.51% | KMNIST 95.59% | 2.696 | 340 |
| [3](evidence/official/run-3.json) | A100-SXM4-80GB | 3.094% | +337 | 3.32% | Fashion 86.78% | 2.986 | 147 |

Compared with the current record, our own fisher-pcg-d2 (reviewer median 3.18 ms, 208 mJ), this entry is **1.18× faster**. Per-run energy varied from 147 to 340 mJ across hosts, and the median is above the record's, so we make no energy claim.

`submission.py` is 17,238 bytes with SHA-256 `311c1d980e162691e3872e7bafe18a357548d3c3442237e788c06570fcf6dde6`, identical to `source_sha256` in all three records (which name it by its working name, `kd2_k.py`). It passes `check_source`. `evidence/official/manifest.json` lists the hash of every file.

## Method

The learner is the one in fisher-pcg-d2: an LDA + QDA Fisher metric with a log-radius coordinate, class-restricted k-means compression to 4,096 hard rows plus 240 weighted centroids, and RBF kernel ridge by 5 preconditioned CG steps plus an extrapolated direction. No backprop. The whole call is one CUDA graph.

**What changed.** Everything below is exact up to rounding, except the last item.
- The QDA class moments are summed on a side stream while LDA runs.
- The shrunk class covariances are inverted by Woodbury from LDA's 10×10 pieces.
- **Standard-form Woodbury preconditioner.** A⁻¹ ≈ D⁻¹ − D⁻¹K_w (K_mm + K_wᵀD⁻¹K_w)⁻¹ K_wᵀD⁻¹, with one Cholesky of the 240×240 inner matrix and one triangular solve. It replaces the landmark Cholesky, a triangular inverse and a GEMM on the critical side chain.
- The 10×10 Choleskys are batched with an identity so cuSOLVER takes its faster batched path.
- The QDA precision is formed as L⁻ᵀL⁻¹ from the triangular factor.
- One 123-column operand is zero-padded to 128 columns, so its GEMMs get aligned kernels.
- The preconditioner products run in fp16 with fp32 accumulation.

In-process on one A100-SXM4, paired over 24 draws, the call takes 2.31 ms against 2.66 ms for the merged file (−13%).

Certified on 2,112 fresh in-process A100 calls in two batches: mean error 3.117%, worst 11-draw split 3.39%, P(3/3) ≈ 0.99. Against the merged file the error rises by 0.002–0.004 points.

## Credits

- Builds on our [fisher-pcg-d2](../fisher-pcg-d2-20261019/README.md) (#131; see its credits).
- The Woodbury identity is textbook. Its use here, the batched-path trick, the fp16 preconditioner products and the stream layout are ours.

## Notes

- Inductive: test rows are only row-scaled and featurised.
- The scorer's clock sets most of the ranked time on calls this short. The CUDA-event MNIST means were 2.36, 2.53 and 2.73 ms. Expect about 2.6–3.0 ms on SXM4 hosts.
- Related files developed for this PR also had official sets: this file with 1,024 Fisher rows and 3,584 hard rows passed 3 of 3 with median 2.900 ms (`ff0b0cb2`, unsubmitted). Our fisher-pcg-d2 README lists the earlier sets.
- During development we also ran three single scorer runs of this exact source in our own sandboxed Modal container, without energy. All passed.

Reproduce from `mnist-a100/`:

```sh
python run_modal.py submissions/fisher-woodbury-d2-20261020/submission.py:kc_d2 --difficulty 2 --runs 3 --json /tmp/d2-verification
```

Co-developed with Claude.
