# PR draft

**Title:** MNIST-small NR-K8 Adam submission — request Adam grid-scorer review

**Body:**

- Adds `mnist/submissions/small-adam-nr-k8-20260912/` with a 9-32-10 MLP trained
  by NR-K8 Adam (bias-corrected Adam; `sqrt(vhat)` lowered to existing ISA ops
  via 8 Newton-Raphson iterations), 1,000/1,000 3x3, 11 draws (seeds
  20261201-211). Adds one row to the MNIST-small (67%) table.
- Accuracy: 7,409/11,000 = 67.35% +/- 1.76 pp (sample SD). Two-phase
  freeze/score evidence; per-draw indices and input hashes; all 11 draws
  re-verified on GPU against the frozen predictions.
- A100: 2,560 mJ / 107 ms (median of 3; two sessions retained).
- Grid: 0.185228620462 mJ / 1,798.760521 ms. **Provisional**: this PR extends
  the grid scorer with an Adam path (SGD path reproduces the official score
  exactly). Scorer review/approval is requested as a merge condition.
- Disclosure: the A100 advantage over the published H32 row is largely the
  fused runtime (same-container interleave: H32 learner 44.5 ms / 1,062 mJ vs
  106.9 ms / 2,424 mJ here). No accuracy-superiority claim (+0.27 pp).
- Reproduce from the submission directory: `python run.py prepare/freeze/score`,
  `python verify.py`, `modal run gpu_benchmark.py`.
