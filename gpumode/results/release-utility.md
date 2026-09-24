# Utility of the linear release for honest entrants (CPU dry runs)

Harness `sutro-mnist-medium-time/1.2.0`. One 10,000 / 10,000 draw per cell,
`--mode test`, secret seed **20260924**, band `mnist-medium-5pct`, device CPU
(Intel MacBook Pro, torch 2.2.2, numpy 1.26.4, python 3.11.13).

**These are accuracy measurements, not timings.** A CPU dry run says nothing
about where an entry lands on a time leaderboard: the per-call numbers below
(1.5 ms to 28 s) are CPU wall time and are three to four orders of magnitude
away from the A100 times in `DESIGN.md`. Only the accuracy columns transfer.

"Pixels" is `--case release_dims=0`, the harness 1.1.1 release: `(N, 1, 9, 9)`
float32 in [0, 1]. "Release-60" is the default in `bands.json`:
`z = Q W (x - mu)`, `(N, 60)` float32, exact PCA whitening onto the top 60
principal directions of *that draw's own* training rows followed by a secret
per-draw Haar rotation.

## Results

| Entry | Pixels | Release-60 | Delta | Band on pixels | Band on release-60 | Notes |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `submissions/ncm_baseline.py` | 79.13% | 84.24% | **+5.11 pp** | none (control) | none | Gains. Whitening turns its Euclidean metric into a near-Mahalanobis one. Still below the 12% band (88%), which is what this control is for. |
| `submissions/pca_qda.py` | 95.33% | 89.86% | **-5.47 pp** | 5% | 12% only | The big loser. Exact whitening flattens the variance spectrum its isotropic shrinkage was implicitly using to damp noisy directions; the pixel-specific arcsine transform is also unavailable on a signed release. It drops out of 5% and out of 8% (92%). |
| `submissions/mlp512.py` | 96.26% | 95.84% | **-0.42 pp** | 5% | 5% | Under a point, exactly what the rotation study predicted for a dense learner. Note it fails the pixel hold-out on an A100 (47.8%); on the release the hold-out rose to 86.5% in the five-draw sweep in `submissions/README.md`, so the release makes this entry *more* likely to qualify, not less. |
| `submissions/cg_pair.py` | 97.98% | 95.77% | **-2.21 pp** | 3% (fails 2% on 11 draws, `DESIGN.md`) | 5% | The kernel half pays for the lost pixel metric, and the conv half has no lattice: on a release it substitutes 4,608 frozen random ReLU features. Drops two bands. |

Band thresholds are mean accuracy over 11 ranked draws: 2% -> 98%, 3% -> 97%,
5% -> 95%, 8% -> 92%, 12% -> 88%, plus a per-draw floor 1.5 pp below the band.
The band columns above are what a single CPU draw at this seed implies; they
are not leaderboard verdicts (11 draws, hold-out, timing, GPU).

## Leaderboard-style run, `pca_qda` only, `--mode benchmark` (3 draws)

| Release | Band run | Verdict | Accuracy, 3 draws | Per draw |
| --- | --- | --- | ---: | --- |
| pixels | `mnist-medium-5pct` | pass | 95.82% (28,745 / 30,000, needs 28,500) | 9,553 / 9,600 / 9,592 |
| release-60 | `mnist-medium-5pct` | **fail**, per-draw floor | draw 0 scored 9,063, floor is 9,350 | run aborts at draw 0 |
| release-60 | `mnist-medium-12pct` (to get all 3 draws past the floor) | pass | 90.84% (27,252 / 30,000) | 9,063 / 9,126 / 9,063 |

The 5% run on the release stops on the first draw, so the 12% run is how the
three-draw release number was obtained. `pca_qda` under the linear release is a
12%-band entry, not a 5%-band entry.

## Consistency with the existing five-draw sweep

`submissions/README.md` reports a five-draw CPU sweep at case seed 101 / secret
seed 20260922: ncm 80.1 -> 85.0, pca_qda 95.5 -> 90.3, mlp512 96.4 -> 95.7,
cg_pair 98.0 -> 95.8. The independent seed here (20260924) reproduces every
delta to within about half a point, including the sign.

## Exact commands

Pool cache prepared once:

```
mkdir -p /tmp/gpumode-pool   # already held the four MNIST/Fashion gz files
```

Per entry, both releases:

```
MNIST_POOL_CACHE=/tmp/gpumode-pool /tmp/penv/bin/python \
  /Users/yaroslavvb/git/sutro-problems/gpumode/run_modal.py \
  --band mnist-medium-5pct \
  --submission /Users/yaroslavvb/git/sutro-problems/gpumode/submissions/<name>.py \
  --mode test --local --seed 20260924 \
  --output /Users/yaroslavvb/git/sutro-problems/gpumode/results/cpu-release-<name>-60.json

MNIST_POOL_CACHE=/tmp/gpumode-pool /tmp/penv/bin/python \
  /Users/yaroslavvb/git/sutro-problems/gpumode/run_modal.py \
  --band mnist-medium-5pct \
  --submission /Users/yaroslavvb/git/sutro-problems/gpumode/submissions/<name>.py \
  --mode test --local --seed 20260924 --case release_dims=0 \
  --output /Users/yaroslavvb/git/sutro-problems/gpumode/results/cpu-release-<name>-0.json
```

for `<name>` in `ncm_baseline`, `pca_qda`, `cg_pair`, `mlp512`; and the same
with `--mode benchmark` (and with `--band mnist-medium-12pct`) for `pca_qda`,
writing `cpu-release-pca_qda-bench-<dims>.json` and
`cpu-release-pca_qda-bench12-<dims>.json`.

**Interpreter path gotcha.** `/tmp/penv` is a symlink to the intended
`/tmp/pmnist-env`. The submission-process tripwire in `utils.py`
(`DATASET_FRAGMENTS`) denies any `open` whose path contains `mnist`, so
importing torch from an interpreter living under `/tmp/pmnist-env` raises
`DatasetFileDenied(".../torch/__init__.py")` and every run fails at child
start-up with exit 112. This is a false positive of a deliberately blunt
tripwire on a local path, not an evaluator bug (the Modal image has no such
path), but it is worth knowing before debugging a mysterious start-up failure.

## Result files

`results/cpu-release-{ncm_baseline,pca_qda,cg_pair,mlp512}-{0,60}.json`,
`results/cpu-release-pca_qda-bench-{0,60}.json`,
`results/cpu-release-pca_qda-bench12-{0,60}.json`.

## Best-known dense ceiling on the release

| Entry | Pixels (3 draws) | Release-60 (3 draws) | Delta |
| --- | ---: | ---: | ---: |
| arc-cosine depth-3 kernel ridge (`redteam/release/dense_ceiling.py`) | 97.47% | 96.29% | -1.18 pp |

Source: `results/release-dense-ceiling.json`; draws are the evaluator's own (case seed 101, timed draw seeds 114/127/140).
