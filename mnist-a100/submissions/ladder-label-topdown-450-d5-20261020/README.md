# Kernel-taught Ladder with a label top-down input, a hidden-space HKNN readout and a noise bank (450 steps): difficulty 5

Entrant: [@spartanhaden](https://github.com/spartanhaden).

This entry builds on our merged [ladder-k2-d5](../ladder-k2-d5-20261019/README.md) (#129) at #135's width of 768, with 450 Adam steps instead of 480, and changes the teacher, noise source, decoder top-down input and readout.

Three sandboxed Modal A100-80GB runs of the unchanged scorer (`mnist-a100/1.2.0`, stock `run_modal.py --difficulty 5 --runs 3`) all passed.
- **Median ranked time:** 405.073 ms
- **Median energy above idle:** 64,777 mJ per call

| Run | GPU | MNIST error | Margin (labels) | Worst draw | Hold-out | Ranked ms | mJ above idle |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| [1](evidence/official/run-1.json) | A100-SXM4-80GB | 1.692% | +229 | 1.94% | KMNIST 97.41% | 405.073 | 69,649 |
| [2](evidence/official/run-2.json) | A100-SXM4-80GB | 1.720% | +198 | 1.86% | KMNIST 97.30% | 403.997 | 64,250 |
| [3](evidence/official/run-3.json) | A100-SXM4-80GB | 1.720% | +198 | 1.92% | KMNIST 97.37% | 405.940 | 64,777 |

Compared with the current record, @sjbaebae's [ladder-width768-d5](../ladder-width768-d5-20261019/README.md) (#135; reviewer median 492.843 ms, 77,090 mJ), this entry is **1.22× faster** and uses 16% less energy per call.

`submission.py` is 20,474 bytes, 6 under the size limit, with SHA-256 `98e310b12031555088ee47e6f8170ad001fad1b6f0be28dcead493ff7682f5c3`, identical to `source_sha256` in all three records (working name `x_k5cFdt450.py`). It passes `check_source`, and `evidence/official/manifest.json` hashes every file.

## Method

- **Teacher.** As in #129 (Nyström-PCG kernel ridge, AGOP^0.75 metric, two fits), but **without the pseudo-label refit**: −19 ms and −0.018 ± 0.005 points. Its PCG right-hand sides are padded from 10 to 16 columns to align the fp16 GEMMs (−2.5 ms).
- **Student.** Ladder 60-768-512-256-10 on AGOP^0.5 inputs, 450 foreach-Adam steps of batch 2,500 at LR 0.01467 (our 0.011 · 600 / steps rule); otherwise as in #129.
- **Label top-down input.** On training rows, the decoder's top combinator gets the one-hot label instead of the student's softmax, straight-through (`v + (label − v).detach()`); test rows keep the softmax. At 480 steps it gives **−0.048 ± 0.002 points** for +1 ms (2,112 paired fresh A100 calls). At 450 steps this file is still −0.028 ± 0.002 points better than the 480-step file without it, and 24.5 ms (5.7%) faster.
- **Noise bank.** The Gaussian corruption is read at random offsets from a per-call bf16 table of 2²² N(0, 1) values, not drawn fresh each step: −22 ms, +0.001 ± 0.006 points (n 420).
- **Launch configs.** Retuned tile and warp settings for our Triton `comb_bwd_kernel`, with its body unchanged: −10 ms, +0.000 ± 0.008 points.
- **Readout.** HKNN on the 1,024 lowest-margin test rows in the student's layer-2 pre-activation space, rescaled to the input norm (40 neighbours per class, top 3 classes, Woodbury K×K solve): −0.040 ± 0.003 points, about 2× the gain of input-space HKNN. #129's teacher-fallback guard follows unchanged.

Certified on 2,112 fresh in-process A100 calls: mean error 1.743%, 11-draw split sd 0.058, 0 of 192 splits over the band (worst 1.896%), P(3/3) ≈ 0.990. No call went over the per-draw limit after the guard, which tripped once.

Under compute-sanitizer on a Modal A100, `comb_bwd_kernel` showed 0 memcheck errors, 0 racecheck hazards and 0 synccheck errors on our 480-step file, whose kernel text, launch configs and shapes are identical.

## Credits

- Builds on our [ladder-k2-d5](../ladder-k2-d5-20261019/README.md) (#129; see its credits).
- **Label top-down input:** @sigkillme0's #127 (open), whose Ladder feeds known targets into the decoder's top level. Ours is our own labels-only re-implementation.
- **Width 768:** as in @sjbaebae's [#135](../ladder-width768-d5-20261019/README.md); our steps, LR and Adam variant differ.
- **Noise bank:** ours, the same idea as the shared noise table of Salimans et al. 2017.
- Dropping the refit, the padding, the launch configs and hidden-space HKNN are ours.

## Notes

- Transductive: test rows are used without labels, as distillation targets, in the unlabelled pool, as HKNN vertices and by the guard.
- The guard trips on every Fashion hold-out call, which still scores about 88–89%. All three runs of this set drew KMNIST.
- The file drops the spaces around binary operators and the blank lines to fit the size limit.
- Cold warm-up is about 30–50 s (our k5_cF file, which shares its kernel and graphs except one op, took 43–48 s).
- Related files developed for this PR also had official sets, both 3 of 3: this file at 480 steps (`k5_cFdt480.py`, SHA-256 `218e3643…`, median 429.512 ms) and that file without the label input (`k5_cF.py`, SHA-256 `e6b3ac01…`, median 431.666 ms). Our ladder-k2-d5 README lists the earlier sets.

Reproduce from `mnist-a100/`:

```sh
python run_modal.py submissions/ladder-label-topdown-450-d5-20261020/submission.py:ladder_kernel --difficulty 5 --runs 3 --json /tmp/d5-verification
```

Co-developed with Claude.
