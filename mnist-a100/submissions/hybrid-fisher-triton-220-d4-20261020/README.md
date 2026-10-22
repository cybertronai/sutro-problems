# Fisher-PCG kernel teachers distilled into MLP students, with a Triton student step (220 steps): difficulty 4

## Independent reviewer reproduction — October 22, 2026

The first frozen reviewer group passed **3/3 fresh canonical sandboxed Modal A100-80GB runs**. Median ranked time is **38.695741 ms/call**; median above-idle GPU board energy is **6,664.381444 mJ/call**. Pooled MNIST error is **2.117576%** (323,012/330,000 correct). Each complete run independently passes all accuracy, holdout and timing gates; pooled accuracy is descriptive.

| Reviewer run | MNIST error | Margin, labels | Ranked ms | Above-idle mJ |
| - | -: | -: | -: | -: |
| 1 | 2.097273% | +223 | 36.206193 | 5998.131722 |
| 2 | 2.214545% | +94 | 39.126181 | 6664.381444 |
| 3 | 2.040909% | +285 | 38.695741 | 6781.826246 |

The submitted source is unchanged, SHA-256 `1c25eb4ed6ef06cf94c1a34ca80eecd16e90b33c0a6091da0c2bd791db18f133`; scorer and stock runner are unchanged. All 45 timed calls in the three returned records completed. All returned records and controller provenance are retained. No maintainer-launched replacement run was made. Modal automatically restarted an input after preemption; that platform restart is disclosed in the retained controller log. [Raw records and executable verifier](evidence/reviewer-20261022/), [reconstruction](evidence/reviewer-20261022/verification.json), [decision](evidence/reviewer-20261022/review-decision.json) and [source review](evidence/reviewer-20261022/source-review.json) are retained. Optional energy is assessed separately from score qualification and excludes host/system energy and initial compilation.

This builds on @spartanhaden's tangent hybrid and its inherited @islamborghini, @likelytobelaura, @sjbaebae, RFM/HKNN and modded-nanogpt credits. Fisher-PCG teacher ideas credit @sigkillme0; the two-kernel Triton student step is Haden's. The submitted reports' large development batches, sanitizer results, numerical-drift comparisons and unsubmitted 200-step history are author claims unless separately supported in the retained reviewer audits; they are not substitutes for this exact-source execution group.

The leaderboard uses these independent reviewer medians; the original author report below remains separate. Published instructions specify three official runs and their median; this reviewer reproduction supplies evidence, without adding a permanent source exclusion or a new mandatory second-batch submission rule. No D6 qualification is claimed here.

Using Python 3.13 from this submission directory: `python evidence/reviewer-20261022/verify_records.py evidence/reviewer-20261022 submission.py --difficulty 4 --function hybrid_d4 --output /tmp/pr138-verification.json`. This audits saved records without executing the learner.

## Original author report (as submitted)

Present-tense record and pending-review claims below describe the author submission date; the dated reviewer section above gives accepted status.


Entrant: [@spartanhaden](https://github.com/spartanhaden).

This entry keeps the pipeline of our merged [hybrid-tangent-d4](../hybrid-tangent-d4-20261019/README.md) (#130): a kernel teacher, three tangent-jittered MLP students, then HKNN. New are our d3 Fisher-PCG learner as the teacher and a two-kernel Triton student step.

Three sandboxed Modal A100-80GB runs of the unchanged scorer (`mnist-a100/1.2.0`, stock `run_modal.py --difficulty 4 --runs 3`) all passed.
- **Median ranked time:** 41.348 ms
- **Median energy above idle:** 6,593 mJ per call

| Run | GPU | MNIST error | Margin (labels) | Worst draw | Hold-out | Ranked ms | mJ above idle |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| [1](evidence/official/run-1.json) | A100-SXM4-80GB | 2.076% | +246 | 2.45% | Fashion 88.53% | 41.348 | 6,528 |
| [2](evidence/official/run-2.json) | A100-SXM4-80GB | 2.094% | +227 | 2.21% | Fashion 88.63% | 40.071 | 6,593 |
| [3](evidence/official/run-3.json) | A100-SXM4-80GB | 2.084% | +238 | 2.31% | Fashion 88.40% | 42.397 | 6,613 |

Compared with the current record, @sjbaebae's [hybrid-tangent-250-d4](../hybrid-tangent-250-d4-20261019/README.md) (#132; reviewer median 78.334 ms, 12,845 mJ), this entry is **1.89× faster** and uses 49% less energy per call.

`submission.py` is 20,479 bytes, 1 under the size limit, with SHA-256 `1c25eb4ed6ef06cf94c1a34ca80eecd16e90b33c0a6091da0c2bd791db18f133`, identical to `source_sha256` in all three records (working name `k4_p220g5.py`). It passes `check_source`, and `evidence/official/manifest.json` hashes every file.

## Method

Each call trains from scratch, as one CUDA graph captured in the warm-up.

- **Teacher.** Two of our d3 Fisher-PCG learners on cyclic shifts of the training rows, on concurrent CUDA streams, each with HKNN (λ 8, temperature 1.5); their softmax(s / 0.25) outputs are averaged.
- **Students.** Three 60-256-256-10 RMSNorm/ReLU MLPs (zero-initialised head) on member 0's half-power-metric rows: 220 Adam steps (β 0.8 / 0.95) on 1,024 labelled and 1,024 test rows with teacher targets, cross-teaching at chunks 7 and 15, and jitter along the top-4 tangents to each row's 10 nearest same-class rows (exact two-pass top-k) plus noise 0.1.
- **Predict.** 0.75 students + 0.25 teacher, then HKNN.
- **Triton student step.** `k_rows` (16 rows of one student per program) fuses sampling, noise, jitter, the bf16 forward and the backward to each layer's pre-activations; `k_adam` computes each 64-wide weight-gradient tile in fp32, with Adam and the bf16 weight copy in its epilogue.

Against the compiled-PyTorch step (2,112 paired fresh A100 calls), the kernels cut in-process time from about 41 to 34.2 ms, with accuracy changes of +0.001 ± 0.002 and −0.001 ± 0.002 points in two batches. Parameter drift is no larger than between two reference runs; no NaN.

Certified on 2,112 fresh in-process A100 calls: mean error 2.102%, 11-draw split sd 0.064, 0 of 192 splits over the band, worst call 2.59%, P(3/3) ≈ 0.997.

Under compute-sanitizer on a Modal A100, `k_rows` and `k_adam` had 0 memcheck errors in this file, and 0 racecheck hazards and 0 synccheck errors in our 200-step file (same kernel text and shapes).

## Credits

- Builds on our [hybrid-tangent-d4](../hybrid-tangent-d4-20261019/README.md) (#130; see its credits).
- **Teacher:** @sigkillme0's [#125](../kernel-d3-20261018/README.md) method (LDA + QDA Fisher metric with rank-48 subspace and natural-gradient step, log radius, k-means compression, FITC preconditioner, extrapolated PCG step, class-slot tables, Newton–Schulz square root), re-implemented as in our [fisher-pcg-d2](../fisher-pcg-d2-20261019/README.md) (#131).
- The Triton student step is ours.

## Notes

- Transductive: test rows are used without labels, as distillation targets, in the neighbour search (with the teacher's class) and in cross-teaching.
- The file uses 1-space indentation and no function docstrings to fit the size limit.
- Warm-up is about that of our 200-step twin: 22–31 s cold, 5–8 s warm.
- Modal logged one input-fetch error in an idle container after run 2; run 3 then ran normally, with no relaunch.
- Related files developed for this PR also had official sets: this file at 200 steps, chunks (6, 13) (`k4_p200g5.py`, SHA-256 `12cad60b…`), passed 3 of 3, median 37.526 ms, margins +82 / +88 / +212. Our hybrid-tangent-d4 README lists earlier sets.
- We also ran one development scorer run of this exact source in our own Modal sandbox (no energy); it passed.

Reproduce from `mnist-a100/`:

```sh
python run_modal.py submissions/hybrid-fisher-triton-220-d4-20261020/submission.py:hybrid_d4 --difficulty 4 --runs 3 --json /tmp/d4-verification
```

Co-developed with Claude.
