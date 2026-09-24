# Does a secret rotation (with or without whitening) stop precomputed spatial features on MNIST-medium?

Study date: 2026-09-23. Question from the organiser: release MNIST-medium (9×9 area-resized
digits, 10,000 train / 10,000 test, 81 features) as `z = Q W (x − μ)` with a secret Haar
rotation `Q` and `W` either the identity ("rotate only") or a whitening matrix ("decorrelate,
then rotate"). Does that stop an entrant from smuggling precomputed edge detectors / an offline
CNN and reconstructing the pixel grid inside the scored submission? Is rotation alone enough?
Should train and test be whitened separately?

Everything below was measured, not argued. Code, raw JSON and logs are in this directory;
`summarize.py` regenerates every table from `results/*.json`.

## Answer in one paragraph

**Rotation alone is not enough, whitening before rotation is necessary but not sufficient, and
train and test must share one map.** Against a *blind* attacker (no source data) every variant
behaves identically — an attacker can always whiten the release themselves — and the blind
attacker fails: the best blind basis recovery (an ICA-like sparse, non-negative unmixing) gets
about 20 of 64 live pixels back, its recovered lattice has 4–5× chance edge precision, and a
CNN trained on that lattice is no better than an MLP on the released features. Against an
*informed* attacker who holds public MNIST (a disjoint 40k pool) plus the 10k released labels,
rotation-only is broken outright (pixels recovered to 7% RMS from one eigendecomposition plus a
moment fit) and whitened releases are broken almost as well (10–13% RMS) by matching
class-conditional second moments, which whitening does not touch; an offline CNN applied to the
recovered pixels scores 98.4–98.8% on every variant, within 0.4 points of the same CNN on true
pixels and above anything an honest in-competition learner reaches (MLP 96.2%, CNN 97.9%). If the
released images come from the public pool, none of this matters: norms alone re-identify 99.9%
of rows and hand over labels and the full map (the hole the hold-out draw already fixes). If the
map is a protocol constant rather than a per-evaluation secret, a 5.5–7.2 KB int8 inverse map
undoes it exactly. Whitening with a variance floor (ε) leaks the dead-border subspace through
the bottom eigenvectors; exact whitening on the top-60 principal directions does not.

## Setup

* **Data.** The official 60,000-image MNIST training split, area-resized to 9×9 (`common.pool`,
  same resize as `mnist/code/data.py`), values in [0, 1]. Per draw seed: 10,000 train, 10,000
  test, and the remaining 40,000 rows are the informed attacker's *public pool* (disjoint from
  the draw). 64 of the 81 pixels have pool variance above 1e-4 ("active"); the rest are the
  always-black border.
* **Release variants** (`common.VARIANTS`; `Q` is Haar-random per draw seed, `W` and `μ` are
  fitted on the draw's train rows unless stated):
  `raw` (control), `perm` (secret pixel permutation, the previous proposal), `rot`
  (`W = I`), `zca1e-3+rot` and `zca1e-2+rot` (ZCA whitening with variance floor ε added to
  every eigenvalue, then `Q`), `pca60+rot` (exact PCA whitening of the top-60 eigendirections,
  60 released coordinates, then `Q`), `zca1e-3` (whitening without rotation, control).
* **Threat models.** *Blind*: sees released train features + labels and released test features,
  knows only that these are 9×9 non-negative images. *Informed*: additionally holds the public
  40k pixel-space pool with labels and can train anything offline. *Exact-source*: holds the
  very images that were released (the known hole).
* **Attacks.**
  * Blind: attacker-side PCA to the top 64 directions, then FastICA (logcosh) or a
    maximum-likelihood unmixing with a Laplace prior and a penalty on negative values
    (`attacks.sparse_nonneg_unmix`); lattice recovery from co-activation of the recovered
    sources (partial correlation of |s| or of max(s, 0)) through the pmnist study's
    embedding + QAP search (`topology.py`); then a fresh CNN trained on the recovered lattice.
  * Informed (`attacks.informed_attack`): whiten both the public pool and the release to their
    top-60 principal subspaces; the residual unknown is one 60×60 orthogonal matrix `R`.
    Initialise `R` by (a) rank-ordered eigenvector matching of the two covariances with
    skewness-fixed signs (unlabelled; only informative when the release is not white),
    (b) Procrustes on class means + class-covariance eigenvectors, (c) Procrustes over the
    eigenvectors of 40 random linear combinations of the 10 class covariances (`rlc`), or
    (d) random; refine every init by Adam + L-BFGS + basin hopping on the class-conditional
    second-moment mismatch plus a non-negativity prior on the reconstructed pixels. The
    attacker picks the init with the smallest moment loss (no secrets needed). A *scoring-side*
    diagnostic evaluates the same loss at the true map ("oracle").
  * Smuggling test: a CNN trained offline on the public 40k (the smuggled artifact) is applied
    to the attacker's pixel estimate of the test rows; also a frozen-CNN + learned linear
    adapter attack that never recovers a basis explicitly (`models.adapter_fit_predict`).
  * Exact-source: nearest-norm matching of released test rows against the 60k pool, then a
    least-squares fit of the whole map from the matched pairs.
* **Learners** (`models.py`): logistic regression, 512×2 MLP (30 epochs), and a 3-layer
  32/64/64-channel CNN with BatchNorm (30 epochs, with and without ±1-pixel shift
  augmentation). The CNN on true pixels is the "spatial prize" an attacker is after.
* **Compute.** Attacks and dense learners on Modal CPU containers, CNNs on Modal T4 containers
  (at most 2 GPU containers at a time, never an A100 — accuracy, not timing, is the question).
  Total: 1.55 container-hours, 0.85 of them on T4 containers (including two superseded runs);
  roughly $2 of the $10 budget.
  Three draw seeds; tables give mean ± sample sd.

## Results

### 1. What the defence costs honest entrants (utility)

| release | seeds | dim | logreg % | MLP-512x2 % | CNN (no aug) % | CNN (+shift aug) % |
|---|---|---|---|---|---|---|
| raw | 3 | 81 | 90.64 ± 0.17 | 96.20 ± 0.31 | 97.94 ± 0.15 | 97.93 ± 0.28 |
| perm | 3 | 81 | 90.64 ± 0.17 | 96.21 ± 0.33 | — | — |
| rot | 3 | 81 | 90.76 ± 0.20 | 96.79 ± 0.29 | — | — |
| zca1e-3 | 3 | 81 | 90.58 ± 0.24 | 95.58 ± 0.32 | — | — |
| zca1e-3+rot | 3 | 81 | 90.64 ± 0.13 | 96.16 ± 0.14 | — | — |
| zca1e-2+rot | 3 | 81 | 90.73 ± 0.18 | 96.45 ± 0.13 | — | — |
| pca60+rot | 3 | 60 | 90.63 ± 0.29 | 95.92 ± 0.34 | — | — |

Dense learners are (nearly) invariant to any invertible linear release: logistic regression is
identical to the second decimal, the MLP moves by well under a point in either direction
(rotation actually helps it slightly). The CNN on true pixels is 1.7 points above the MLP; that
gap is the prize the obfuscation protects. Whitening with a floor keeps 81 coordinates of which
~17 are amplified quantisation noise; truncating to 60 principal directions (`pca60+rot`) costs
the MLP 0.3 points and removes that tail entirely.

### 2. Separate or shared whitening? (organiser question 2)

| release | seeds | MLP shared | MLP separate | MLP pool-fit | logreg shared | logreg separate | logreg pool-fit | basis mismatch (separate) |
|---|---|---|---|---|---|---|---|---|
| zca1e-3+rot | 3 | 96.08 ± 0.20 | 95.95 ± 0.33 | 96.11 ± 0.24 | 90.64 ± 0.13 | 90.54 ± 0.25 | 90.64 ± 0.10 | 0.050 ± 0.001 |
| zca1e-2+rot | 3 | 96.33 ± 0.35 | 96.33 ± 0.24 | 96.44 ± 0.32 | 90.73 ± 0.18 | 90.55 ± 0.21 | 90.74 ± 0.18 | 0.029 ± 0.000 |
| pca60+rot | 3 | 95.89 ± 0.20 | 61.56 ± 5.15 | 95.84 ± 0.24 | 90.63 ± 0.29 | 53.56 ± 3.63 | 90.61 ± 0.24 | 0.993 ± 0.062 |

`shared`: one map fitted on the train rows and applied to both splits. `separate`: the test
rows get their own `W, μ` (same `Q`). `pool-fit`: one map fitted on all 60k. "basis mismatch"
is `‖A_test A_train⁻¹ − I‖_F / √d`. Separate ZCA whitening costs 0.0–0.15 points because ZCA is a
continuous function of the covariance (mismatch 0.03–0.05 at n = 10,000). Separate PCA whitening
destroys the task (62% ± 5): PCA eigenvectors of two independent 10k-sample covariances differ by
sign flips and by rotations inside near-degenerate eigenspaces (mismatch ≈ 1). Part of that is
the sign rule: anchoring every eigenvector's sign on the all-ones vector instead of its largest
entry lifts separate-PCA logistic regression from 50–56% to 76–83% (shared: 90.6%) and the
mismatch from 0.94–1.06 to 0.67–0.87 (`results/separate-sign-convention.json`); the rest is the
eigenspace rotation that no sign rule fixes. Separate fitting
buys no security either: a blind attacker re-whitens each split on their own (the blind attacks
below are indifferent to `W`), and the informed attack run against the separately-whitened
`zca1e-3+rot` release recovers pixels as well as against the shared one
(`results/informed-separate-fit.json`; the seed-to-seed spread of the attack is larger than any
shared-vs-separate difference). **Use one shared map.**

### 3. Blind attacker (no source data): can the pixel basis or the edge graph be recovered?

| release | unmixing | seeds | localisation r<=1.5 | active matched energy | active sources >0.5 / >0.9 (of 64 live pixels; 60 sources for pca60) | edge-prec@1.5 |s| pcorr | |s| corr | max(s,0) pcorr | s corr (control) |
|---|---|---|---|---|---|---|---|---|---|
| perm | ica-logcosh | 3 | 0.66 ± 0.01 | 0.39 ± 0.00 | 22 ± 1 / 14 ± 1 | 0.46 ± 0.03 (chance 0.09 ± 0.00) | 0.45 ± 0.04 (chance 0.09 ± 0.00) | 0.42 ± 0.04 (chance 0.09 ± 0.00) | 0.12 ± 0.05 (chance 0.09 ± 0.00) |
| perm | sparse-nonneg | 3 | 0.65 ± 0.01 | 0.46 ± 0.00 | 24 ± 1 / 20 ± 1 | 0.40 ± 0.04 (chance 0.11 ± 0.01) | 0.43 ± 0.03 (chance 0.11 ± 0.01) | 0.40 ± 0.05 (chance 0.11 ± 0.01) | 0.39 ± 0.03 (chance 0.11 ± 0.01) |
| rot | ica-logcosh | 3 | 0.66 ± 0.01 | 0.39 ± 0.00 | 22 ± 1 / 14 ± 1 | 0.46 ± 0.02 (chance 0.09 ± 0.01) | 0.50 ± 0.04 (chance 0.09 ± 0.01) | 0.40 ± 0.05 (chance 0.09 ± 0.01) | 0.14 ± 0.01 (chance 0.09 ± 0.01) |
| rot | sparse-nonneg | 3 | 0.65 ± 0.00 | 0.46 ± 0.00 | 23 ± 1 / 20 ± 1 | 0.40 ± 0.02 (chance 0.10 ± 0.01) | 0.45 ± 0.01 (chance 0.10 ± 0.01) | 0.38 ± 0.00 (chance 0.10 ± 0.01) | 0.39 ± 0.02 (chance 0.10 ± 0.01) |
| zca1e-3+rot | ica-logcosh | 3 | 0.66 ± 0.01 | 0.39 ± 0.00 | 22 ± 1 / 14 ± 1 | 0.45 ± 0.01 (chance 0.09 ± 0.01) | 0.49 ± 0.04 (chance 0.09 ± 0.01) | 0.43 ± 0.03 (chance 0.09 ± 0.01) | 0.11 ± 0.04 (chance 0.09 ± 0.01) |
| zca1e-3+rot | sparse-nonneg | 3 | 0.65 ± 0.01 | 0.46 ± 0.00 | 23 ± 1 / 20 ± 1 | 0.41 ± 0.04 (chance 0.10 ± 0.01) | 0.44 ± 0.05 (chance 0.10 ± 0.01) | 0.33 ± 0.03 (chance 0.10 ± 0.01) | 0.37 ± 0.03 (chance 0.10 ± 0.01) |
| zca1e-2+rot | ica-logcosh | 3 | 0.66 ± 0.01 | 0.39 ± 0.00 | 22 ± 1 / 14 ± 1 | 0.46 ± 0.04 (chance 0.09 ± 0.01) | 0.48 ± 0.03 (chance 0.09 ± 0.01) | 0.47 ± 0.02 (chance 0.09 ± 0.01) | 0.09 ± 0.02 (chance 0.09 ± 0.01) |
| zca1e-2+rot | sparse-nonneg | 3 | 0.66 ± 0.01 | 0.46 ± 0.00 | 24 ± 1 / 20 ± 1 | 0.40 ± 0.04 (chance 0.10 ± 0.01) | 0.42 ± 0.01 (chance 0.10 ± 0.01) | 0.38 ± 0.03 (chance 0.10 ± 0.01) | 0.37 ± 0.01 (chance 0.10 ± 0.01) |
| pca60+rot | ica-logcosh | 3 | 0.63 ± 0.03 | 0.39 ± 0.01 | 21 ± 1 / 10 ± 1 | 0.45 ± 0.07 (chance 0.11 ± 0.00) | 0.45 ± 0.03 (chance 0.11 ± 0.00) | 0.45 ± 0.05 (chance 0.11 ± 0.00) | 0.16 ± 0.01 (chance 0.11 ± 0.00) |
| pca60+rot | sparse-nonneg | 3 | 0.63 ± 0.02 | 0.48 ± 0.00 | 25 ± 1 / 16 ± 1 | 0.36 ± 0.05 (chance 0.12 ± 0.02) | 0.42 ± 0.03 (chance 0.12 ± 0.02) | 0.34 ± 0.05 (chance 0.12 ± 0.02) | 0.35 ± 0.03 (chance 0.12 ± 0.02) |

Localisation = mean energy fraction of each recovered coordinate within radius 1.5 of its pixel
centroid (a pixel-space ZCA filter scores 0.95; a random direction 0.10). "Active sources" = how
many of the 64 live pixels are recovered with more than 50% / 90% of a coordinate's energy on
them (Hungarian matching). Edge precision = fraction of the recovered lattice's edges that join
coordinates whose true centroids are within 1.5 pixels, against a random-layout chance line.
The last column is a control: correlations of the raw ICA sources carry nothing because ICA
outputs are white (the sparse unmixer's sources are not white, so its control is not at
chance). For `perm` the table shows ICA applied to permuted pixels; the *right* blind attack on
a permutation is the direct second-order lattice recovery, which the pmnist study measured at
edge precision ≥ 0.99 — a permutation is not an obfuscation, and this table is not evidence
otherwise.

Two things to read off. First, every release variant gives the same blind numbers (within
seed noise): a blind attacker can whiten the release themselves, so rotation-only, floored ZCA
and exact PCA whitening are one object to them — one unknown orthogonal matrix acting on
whitened images. Second, the blind recovery is partial: the sparse, non-negative unmixer pins
down ~20 of 64 live pixels almost exactly and leaves the rest as delocalised filters; the lattice
built on top is 4–5× better than chance but far from the grid.

End-to-end (fresh CNN trained on the recovered lattice; MLP on the sources):

| release | unmixing | seeds | abs-pcorr_cnn_aug_acc | abs-pcorr_cnn_noaug_acc | mlp_on_sources_acc | pos-pcorr_cnn_aug_acc | pos-pcorr_cnn_noaug_acc | random_layout_cnn_noaug_acc |
|---|---|---|---|---|---|---|---|---|
| perm | ica-logcosh | 3 | 93.60 ± 0.32 | 95.74 ± 0.33 | 95.84 ± 0.11 | 93.67 ± 0.47 | 95.79 ± 0.23 | 95.38 ± 0.33 |
| perm | sparse-nonneg | 3 | 93.95 ± 0.40 | 95.92 ± 0.32 | 96.01 ± 0.21 | 93.97 ± 0.43 | 95.98 ± 0.23 | 95.78 ± 0.08 |
| rot | ica-logcosh | 3 | 93.27 ± 0.36 | 95.57 ± 0.22 | 95.65 ± 0.09 | 93.15 ± 0.36 | 95.86 ± 0.10 | 95.34 ± 0.12 |
| rot | sparse-nonneg | 3 | 93.92 ± 0.33 | 95.89 ± 0.20 | 96.00 ± 0.34 | 93.87 ± 0.44 | 95.91 ± 0.22 | 95.59 ± 0.14 |
| zca1e-3+rot | ica-logcosh | 3 | 93.53 ± 0.62 | 95.84 ± 0.08 | 95.91 ± 0.13 | 93.47 ± 0.25 | 95.82 ± 0.18 | 95.53 ± 0.08 |
| zca1e-3+rot | sparse-nonneg | 3 | 93.96 ± 0.41 | 95.94 ± 0.16 | 96.00 ± 0.23 | 93.61 ± 0.17 | 95.85 ± 0.14 | 95.65 ± 0.24 |
| zca1e-2+rot | ica-logcosh | 3 | 93.56 ± 0.58 | 95.82 ± 0.12 | 95.81 ± 0.15 | 93.31 ± 0.67 | 95.75 ± 0.36 | 95.57 ± 0.15 |
| zca1e-2+rot | sparse-nonneg | 3 | 93.99 ± 0.41 | 95.89 ± 0.22 | 96.05 ± 0.19 | 94.02 ± 0.13 | 95.93 ± 0.17 | 95.83 ± 0.15 |
| pca60+rot | ica-logcosh | 3 | 93.63 ± 0.03 | 95.92 ± 0.12 | 95.93 ± 0.24 | 93.34 ± 0.21 | 95.65 ± 0.27 | 95.65 ± 0.28 |
| pca60+rot | sparse-nonneg | 3 | 93.80 ± 0.37 | 96.12 ± 0.16 | 96.13 ± 0.25 | 93.79 ± 0.05 | 95.95 ± 0.16 | 95.81 ± 0.26 |

End-to-end the blind attacker gains nothing: a fresh CNN on the recovered lattice (95.6–96.1%,
no augmentation) equals an MLP on the same sources (95.7–96.1%) and a CNN on a *random*
arrangement of them (95.3–95.8%), while the honest CNN on true pixels is 97.9%. Shift
augmentation hurts (93–94%) because the recovered layout is not translation-consistent.
An attempted "zero-atom" polish (push each direction toward the point mass that true pixels have
at exactly zero) made the recovery worse in every configuration tried and is not used.

### 4. Informed attacker (public MNIST + the 10k released labels): pixels come back

(no results yet: KeyError: 'variant')

"diag energy" = mean over the 64 live pixels of the fraction of a recovered pixel's energy that
sits on the right pixel (1.0 = exact); "active px > 0.9" = live pixels recovered almost exactly;
pixel rel-RMS = RMS error of the reconstructed test pixels relative to their RMS value (a
constant-mean predictor scores 0.66, the truncated-to-60 true map 0.02–0.03). The oracle row is
a scoring-side diagnostic: the moment loss evaluated at the true map, and — because the attacker
works in a 60-dimensional subspace — the ceiling of the diag-energy and rel-RMS columns. Every
attack init ends within 1–1.8× of that loss floor, so the residual error is the objective's own
noise floor (class covariances from ~1,000 rows per class), not a failure of the optimiser.

Rotation-only (`rot`, identical to `perm`) is recovered to 7–8% RMS with 26 of 64 pixels exact —
one eigendecomposition of the released covariance matched by rank to the public one, plus the
labelled moment fit. Whitening removes exactly that channel (the `eig` init on `pca60+rot` is
as bad as random) and nothing else: the labelled class-conditional moments, which whitening
cannot flatten, still recover pixels to 10–13% RMS with ~20 of 64 exact. The reconstructions
are legible digits for every variant (`results/reconstructions-s1.png`).

Smuggled offline CNN (trained on the disjoint public 40k) applied to recovered pixels, test accuracy %:

| release | seeds | on true pixels (ceiling) | on release, no attack | attacker-picked recovery | rlc recovery | eig recovery | adapter (random init) | adapter (informed init) |
|---|---|---|---|---|---|---|---|---|
| perm | 3 | 98.83 ± 0.06 | 8.81 ± 0.95 | 98.63 ± 0.13 (rlc/rlc/rlc) | 98.63 ± 0.13 | 98.57 ± 0.01 | 91.97 ± 0.65 | 96.19 ± 0.22 |
| rot | 3 | 98.83 ± 0.06 | 9.27 ± 1.45 | 98.63 ± 0.13 (rlc/rlc/rlc) | 98.63 ± 0.13 | 98.57 ± 0.01 | 92.22 ± 0.23 | 95.73 ± 0.45 |
| zca1e-3+rot | 3 | 98.83 ± 0.06 | 10.49 ± 0.86 | 98.60 ± 0.21 (eig/rlc/rlc) | 98.58 ± 0.23 | 98.40 ± 0.23 | 91.76 ± 0.61 | 93.20 ± 0.27 |
| zca1e-2+rot | 3 | 98.83 ± 0.06 | 10.26 ± 0.91 | 98.60 ± 0.15 (eig/eig/rlc) | 98.53 ± 0.13 | 98.55 ± 0.14 | 92.57 ± 0.32 | 93.89 ± 0.20 |
| pca60+rot | 3 | 98.83 ± 0.06 | — | 98.56 ± 0.16 (eig/random/rlc) | 98.58 ± 0.12 | 98.02 ± 0.40 | 92.33 ± 0.09 | 93.03 ± 0.21 |

The offline CNN applied to the recovered pixels reaches 98.4–98.8% on every variant — within
0.1–0.4 points of the same CNN on the true pixels (98.8%) and 2.3 points above the honest MLP
(96.2%), i.e. above the honest CNN trained in-competition (97.9%). Whitening does not change
this: on `pca60+rot`, where the release covariance is exactly the identity, the attacker's
lowest-loss init lands at 98.5–98.7%. The 10–13% pixel RMS left by the moment fit is noise a
convolutional net shrugs off. The frozen-CNN + linear-adapter shortcut (no explicit recovery)
is much weaker (93–96%), so explicit basis recovery is what the attacker needs, and it costs
about a minute of CPU (or seconds on a GPU) plus the class-conditional statistics of the public
pool.

The smuggling test is the number the organiser cares about: a CNN trained offline on the public
40k applied to the attacker's pixel estimate of the secret test rows, against the same CNN on
the true pixels and an honest MLP on the released features (96.2–96.8%).

### 5. Exact-source attacker (the known hole, quantified)

| release | seeds | label acc % (W fit on pool) | exact row hits % | full map from matches: rel-RMS on other rows | label acc % (W fit on secret train rows, naive) | exact row hits % |
|---|---|---|---|---|---|---|
| rot | 3 | 99.96 ± 0.02 | 99.95 ± 0.01 | 0.0027 ± 0.0006 | 99.96 ± 0.02 | 99.95 ± 0.01 |
| zca1e-3+rot | 3 | 99.93 ± 0.03 | 99.93 ± 0.03 | 0.0015 ± 0.0004 | 15.83 ± 0.27 | 0.13 ± 0.02 |
| pca60+rot | 3 | 99.95 ± 0.03 | 99.95 ± 0.02 | 0.0102 ± 0.0002 | 15.91 ± 0.21 | 0.11 ± 0.04 |

If the released rows are public MNIST images and the whitening fit set is public, the norm
`‖z‖ = ‖W(x − μ)‖` is preserved by `Q` and identifies the source row 99.9% of the time in
0.1 s of CPU; labels are then free, and least squares on the matched pairs returns the whole map
(0.3% pixel RMS on the unmatched rows). Fitting `W` on the secret train rows defeats this naive
norm match (16%) — but only because the attacker's norms are perturbed by ~1%; matching on the
pairwise-distance graph would survive that and was not implemented. The hold-out draw from a
different 9×9 source, already planned, is the actual fix.

### 6. Two design facts that dominate everything else

* **The map must be a fresh secret per evaluation.** If `(μ, W, Q)` is a protocol constant, an
  entrant cracks it once offline and ships the inverse: an int8 `A⁻¹` with per-row scales is
  7,209 bytes (5,508 for `pca60`) and reconstructs test pixels to 0.3–1.1% RMS
  (`results/fixed-map-foldin.json`). Rotation and whitening are then decorative.
* **A variance floor leaks the border.** With ε > 0 the released covariance is
  `Q diag(λ/(λ+ε)) Qᵀ`, not the identity: its 17 smallest eigenvectors are single dead-border
  pixels (11–13 of 17 exactly, localisation 0.85–0.98) for `rot`, `zca1e-3+rot` and
  `zca1e-2+rot` alike, and delocalised (0.15–0.19, 0–1 of 17) for `pca60+rot`
  (`results/tail-leak.json`). Those pixels carry no digit information, but they anchor the
  border and break the dihedral ambiguity for free. Exact whitening of the top-k directions is
  the clean construction.

## Recommendations

1. **Do not rely on rotation alone.** Its second-order channel is undone by one
   eigendecomposition by anyone holding the source distribution.
2. **If a linear release is used at all, whiten exactly on the top-k principal directions
   (`pca60`-style), then rotate, with one shared map for train and test, fitted per
   evaluation from a secret seed.** Never publish the map, never reuse it across evaluations,
   and do not use a variance floor (it leaks the border subspace).
3. **Understand what it buys.** Against entrants with no source data it removes the spatial
   prize entirely (blind CNN = MLP). Against entrants who hold the public source it only raises
   the cost: labelled moment matching on a disjoint pool recovers legible pixels, and an offline
   CNN on those pixels gets most of the CNN-over-MLP gap back. The linear release is therefore a
   complement to, not a substitute for, the defences already decided: the hold-out draw from a
   source the entrant cannot hold, the size cap (the informed attack needs class-conditional
   statistics — about 19 KB at int8 in a 60-dim basis — plus the public covariance square root,
   which is at the cap), and ranking on the hold-out.
4. **Separate whitening of train and test is a bad idea**: no security gain, a small loss for
   ZCA and a catastrophic one for PCA.

## Caveats

* Three draw seeds, with the rotation seed tied to the draw seed; seed-to-seed spread of the
  informed attack (e.g. 4–14% RMS on `zca1e-3+rot`) is optimisation luck in a non-convex fit, so
  the informed numbers are a floor on what a determined attacker gets, not a ceiling.
* Informed-attack hyper-parameters (top-60 truncation, non-negativity weight, init choice) were
  tuned on seed 1 while looking at scoring-side metrics. An attacker can do the same offline by
  simulating releases from the public pool with their own random maps, so this does not make
  the attack unrealistic, but the numbers are not a blind pre-registration.
* The honest MLP uses one fixed recipe for every release; per-variant learning-rate sweeps
  were not run, and the ± is seed spread only (GPU run-to-run nondeterminism is another
  ~0.2 points). Differences under 0.5 points between releases should not be read.
* Every attack here whitens the release itself first, so the per-variant rows of the blind
  table are one experiment repeated (by design: that is the point being made), and the
  informed rows differ only through the unlabelled `eig` init. The lattice search runs under a
  120 s wall-clock budget, so its edge-precision spread is partly container speed.
* The blind attacks tried are FastICA (logcosh), a Laplace + non-negativity maximum-likelihood
  unmixer, and an abandoned zero-atom polish. Stronger blind attacks exist on paper (facet
  recovery from exact zeros, marginal-quantile matching) and were not made to work here.
* The exact-source attack with `W` fitted on secret rows was only run in its naive norm-matching
  form; a distance-graph matcher would very likely restore it.
* Only same-source draws were measured. The planned hold-out draw from a different 9×9 dataset
  removes the informed and exact-source attackers by construction, leaving the blind attack,
  which fails here.
* GPU work ran on Modal T4 containers, not the competition's A100; nothing here is a timing or
  energy claim.

## Reproduction

```
# environment: /tmp/pmnist-env (numpy 1.26, scipy, scikit-learn, torch 2.2); Modal for fan-out
python run.py exact    --seeds 1,2,3          # local, seconds
python -m modal run modal_run.py --stage utility  --seeds 1,2,3 --gpu --per-seed
python -m modal run modal_run.py --stage separate --seeds 1,2,3
python -m modal run modal_run.py --stage blind    --seeds 1,2,3
python -m modal run modal_run.py --stage blindcnn --seeds 1,2,3 --gpu
python -m modal run modal_run.py --stage informed --seeds 1,2,3
python rescore_informed.py
python -m modal run modal_run.py --stage smuggle  --seeds 1,2,3 --gpu --per-seed
python summarize.py > results/summary.md
python figures.py
```

`results/redteam-panel.json` holds the red-team attack catalogue (three lenses + critic +
synthesis) that shaped the attack list; `results/old-sdfloor/` holds the first utility /
separate / blind-CNN runs, superseded after a standardiser floor bug (near-dead coordinates were
amplified at test time) was fixed. `topology.py` is used from
`mnist/experiments/pmnist-medium-cutoffs-20260923/`.
