# Ported Sutro entries

Each file is self-contained, exposes `custom_kernel`, takes its device from the
input tensors, and runs on a CPU as well as a GPU. They are reference points and
smoke tests for the harness, not an endorsement of any particular approach.

| File | Source | Band it meets on pixels | Accuracy observed on CPU (pixels) | A100 time reported upstream |
| --- | --- | --- | ---: | ---: |
| `ncm_baseline.py` | `reference.py` | none | 79.0% - 79.5% | under 1 ms |
| `pca_qda.py` | `mnist/submissions/medium-pca-qda-20260915` | 5% | 95.65% over 3 draws | ~3.3 ms |
| `mlp512.py` | `mnist/submissions/medium-affine-20260911` | 5% | 96.71% on one full draw | not measured |
| `cg_pair.py` | `mnist/submissions/medium-cg-pair-20260916` | 2% | 98.20% on one full draw | ~260 ms |

The "band it meets" column is a pixel-release verdict. On the 60-column release
the same files score what the table further down measures, where only `cg_pair`
and `mlp512` are near 5% and nothing is near 2% or 3%.

CPU numbers come from `run_modal.py --local` on this harness with the 10,000 /
10,000 draw, seed 20260922; they are accuracy checks, not timings. A CPU call
takes about 0.6 ms (`ncm_baseline`), 70 ms (`pca_qda`), 23 s (`mlp512`) and
10 s (`cg_pair`) on an Intel MacBook Pro. Every accuracy in the table above was
measured on the **pixel** release, i.e. before harness 1.2.0.

## Harness 1.2.0: the linear release

Draws now arrive as `(N, 60)` features, `z = Q W (x - mu)`, instead of
`(N, 1, 9, 9)` pixels (see the README and DESIGN.md D11). Each file here now
detects which release it was handed -- a pixel release is 4-dimensional, a
linear release is 2-dimensional -- and behaves exactly as before on pixels.
What changed, and why:

| File | Change on a linear release |
| --- | --- |
| `ncm_baseline.py` | none. It already flattened and took its width from the tensor. |
| `pca_qda.py` | feature count read off the tensor; the arcsine (Newton square root) transform is skipped, because the release is signed and the iteration does not converge on negative values; the 40-dimensional subspace iteration is skipped, because the release is already the top-60 principal subspace and is white, so there is no variance ranking left to find and projecting would discard a third of the signal. QDA itself is untouched. |
| `mlp512.py` | input width read off the tensor (so the seeded PCG64(101) initialization is uniform(-1/sqrt(D), 1/sqrt(D)) at whatever D arrives); the `x * 4 - 0.5` pixel rescaling becomes `x * 0.25`. It is *not* enough to skip the rescaling: the release is zero-mean and unit-variance, but that spreads the energy over all 60 directions instead of the ~18 the pixel scaling left, and the fixed lr = 0.1 then diverges on some draws. See below. |
| `cg_pair.py` | the convolutional half has no lattice to convolve over, so on a linear release it uses 4,608 frozen random ReLU features -- the same width the 512 filters plus 3x3 pooling produced, from the same PCG64(0) stream. The arcsine transform is skipped, and the RBF bandwidth is set from the data (`gamma = 3.0 / mean pairwise squared distance`, which reproduces the upstream `gamma = 0.3` on arcsine pixels, where that mean is about 10). A sweep of that constant over 0.75 to 12 on a 2,000-example release draw peaked at 3.0. |

### Accuracy on each release

CPU, this harness, 10,000/10,000 draws of case seed 101, `release_dims: 60`.
Each cell is the **mean over five MNIST draws** with the range in brackets, and
the hold-out column is two Fashion draws. Five draws matter: a single draw hid a
divergence in `mlp512` that only two of eleven draws showed (see below). These
are still accuracy checks on a CPU, not band verdicts.

| File | Pixel release | Linear release (60) | Hold-out, pixels | Hold-out, release |
| --- | ---: | ---: | ---: | ---: |
| `ncm_baseline.py` | 80.1% (79.7-80.5) | **85.0%** (84.4-85.6) | 67.0% | 77.6% |
| `pca_qda.py` | 95.5% (95.4-95.6) | 90.3% (89.8-90.6) | 78.0% | 75.4% |
| `mlp512.py` | 96.4% (96.0-96.8) | 95.7% (95.6-95.8) | 85.9% | 86.5% |
| `cg_pair.py` | 98.0% (97.9-98.2) | 95.8% (95.6-95.9) | 88.7% | 86.5% |

Nearest class mean *gains* 5 points: whitening turns its Euclidean metric into
something closer to a Mahalanobis one. `pca_qda` loses 5.2, `cg_pair` 2.2 (and
4.3 at N = 2,000, so the gap closes with data), and the plain MLP 0.7 -- roughly
the pattern the rotation study predicted (dense learners pay under a point,
methods that rely on the pixel metric or a pixel-scale nonlinearity pay more). A
linear check confirms the release itself is faithful: ridge regression on
one-hot targets scores 84.2% on pixels and 84.0% on the release at N = 10,000,
as an affine-equivariant learner must.

**`mlp512` needed a scale, not just a skipped one.** Handing the release to it
unchanged -- it is already zero-mean and unit-variance, so the pixel rescaling
looked redundant -- put the fixed lr = 0.1 squared-error update past its
stability limit on some draws: over the 11 timed draw seeds of case seed 101 at
N = 10,000 it collapsed onto a single class twice (10.8% and 8.9%) and averaged
79.7%, which would fail the per-draw floor on the first ranked call. The file
now scales the release by `RELEASE_SCALE = 0.25`, which removes the collapse
(the numbers above) and is worth 2 to 3 points on the draws that did not
collapse at N = 2,000.

`mlp512.py` is a plain-PyTorch transcription of the upstream architecture,
hyperparameters and update rule rather than a copy of its ordered-FP32 NumPy
arithmetic; see its docstring. The other two carry their upstream learners
essentially unchanged.
