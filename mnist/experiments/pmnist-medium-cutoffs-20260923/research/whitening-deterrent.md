# Can whitening remove the ability to recover the pixel grid?

Study `pmnist-medium-cutoffs-20260923`. Synthesis of three independent CPU
investigations (dense cost, ICA/topographic attack, adversarial review) plus a
literature pass, all on **dev seed 2026092301 only**. Final seeds
2026092001..2026092011 were never read. No Modal, no GPU, nothing committed.

> **Every accuracy number in this document is a single-draw, single-seed, reduced-epoch
> CPU pilot.** One training draw, one 10,000-row query set, learner seed 11, no
> repetitions, no error bars. Binomial SE at 3% error on 10,000 queries is 0.17 pp;
> measured cnn-09 member-seed spread is ~0.16 pp. Treat gaps below ~0.4 pp (N=10,000)
> and ~1.0 pp (N=1,000) as noise. These numbers rank variants; they are **not**
> comparable to the study's frozen 1.53% / 2.32% headline figures.

---

## 1. Short answer

No. Whitening does not remove the ability, and it is not even the part of your
proposal that does the work. Three things came out of the measurements, in
descending order of how much they should change your mind. **(a)** The whole
protective effect comes from the *random rotation*, not the whitening: rotation
alone drives the existing second-order attack to chance (3.5% adjacency precision
vs 4.4% chance) and is exactly free for the RBF kernel and within noise for the
arc-cosine kernel, whereas ZCA whitening *without* a rotation leaves the attack at
44.4% and whitening *on top of* a rotation adds no security at all (an attacker
can whiten the released data himself) while costing the kernel baselines ~1 pp at
eps=1e-2 and 1.3-4.1 pp at eps=1e-4. **(b)** Against the residual attacks we could
run without external data — FastICA, a topographic energy-correlation sort, a
non-negativity/cone-facet search — the rotation holds: the best end-to-end attack
ties the *no-attack* baseline exactly (10.59% vs 10.58% at N=1,000; 4.48% vs 4.44%
at N=10,000), even though a third of the pixel basis is genuinely recovered.
**(c)** But none of that is the binding constraint, because your pool is the public
MNIST training split. Any *fixed invertible* map of a public dataset is invertible
in seconds: matching released rows to the public pool by a rotation-invariant
fingerprint re-identified **99.94-99.98% of 20,000 rows**, least squares then
recovered the rotation, and `topology.recover_layout` on the reconstructed pixels
returned the true grid at **92.6% exact placement — identical to the unobfuscated
protocol — in 17-26 seconds**, for `rotate_only`, eps=1e-3 and eps=1e-2 alike. The
same is true of today's permutation-only protocol (sorted-pixel fingerprints play
the role of the norm). So the honest conclusion is: **topology secrecy in this
benchmark is enforced by the rule, not by the data transform, and no linear
transform can change that.** What a transform *can* do is close the cheap
rule-abiding route, and if you want that, `rotate_only` buys it for ~0.1-0.2 pp.

---

## 2. Why a fixed invertible transform is nearly free for dense learners — and where it is not

An orthogonal `Q` changes nothing for any learner that depends on the data only
through inner products and norms. That is provable, not empirical: `<Qx, Qy> =
<x, y>` and `||Qx - Qy|| = ||x - y||`, so RBF, arc-cosine and NTK kernels produce a
**bit-identical** Gram matrix, hence identical predictions. Ng's classification of
*rotationally invariant* algorithms (L2-regularised logistic regression, SVMs,
backprop nets with Gaussian init) covers most of the dense ladder
(https://robotics.stanford.edu/~ang/papers/icml04-l1l2.pdf); Grinsztajn et al.
(NeurIPS 2022 D&B) show empirically that MLPs are nearly rotation-invariant while
trees are not.

Whitening is a different animal. It is invertible but **not** an isometry: it
replaces Euclidean geometry with Mahalanobis geometry and amplifies the near-null
pixel directions. The 9x9 pool covariance is very ill-conditioned (max eigenvalue
0.4252, min 1.511e-8, 17 eigenvalues below 1e-4 — the always-black frame and its
near-duplicates), so `epsilon` decides how much pure area-resize quantisation noise
gets rescaled to unit variance, and that shows up directly as accuracy.

### 2.1 Measured dense cost (dev seed 2026092301, single draw, learner seed 11)

Query error, percent:

| method | N | permuted pixels (current) | rotation only | ZCA eps=1e-2 + rot | ZCA eps=1e-4 + rot | exact whitening, top-64 |
|---|---|---|---|---|---|---|
| kernel ridge, arc-cosine d3 | 1,000 | 6.15 | 6.38 | 7.34 | 9.68 | 9.65 |
| kernel ridge, arc-cosine d3 | 10,000 | 2.30 | 2.44 | 3.17 | 3.60 | 3.49 |
| kernel ridge, RBF | 1,000 | 6.44 | 6.44 | 7.43 | 10.28 | 10.53 |
| kernel ridge, RBF | 10,000 | 2.52 | 2.52 | 2.83 | 4.10 | 4.41 |
| MLP 1024-1024, `standardize` (see 2.2) | 1,000 | 8.85 | 6.84 | 7.36 | 9.64 | n/a |
| MLP 1024-1024, `standardize` (see 2.2) | 10,000 | 3.09 | 2.66 | 2.86 | 3.29 | n/a |

Deltas vs the current protocol (pp; positive = worse):

| method | N | rotation only | ZCA eps=1e-2 | ZCA eps=1e-4 | exact top-64 |
|---|---|---|---|---|---|
| arc-cosine d3 | 1,000 | **+0.23** | +1.19 | +3.53 | +3.50 |
| arc-cosine d3 | 10,000 | **+0.14** | +0.87 | +1.30 | +1.19 |
| RBF | 1,000 | **+0.00** (exact) | +0.99 | +3.84 | +4.09 |
| RBF | 10,000 | **+0.00** (exact) | +0.31 | +1.58 | +1.89 |

The RBF zeros are not rounding: the CV picked gamma 0.02 vs 0.019999999993 and the
predictions are identical. The arc-cosine deltas are inside single-draw noise.
`learners.py` hard-codes 81 inputs, so the top-64 variant cannot run the MLP family
at all without a protocol change — itself a small warning about how invasive this is.

**Epsilon is an accuracy dial with no security content** (see 3.3). Train-only
5-fold CV on the arc-cosine kernel at N=1,000 picks eps=1e-2: CV error 8.50% vs
9.80% (1e-3), 10.70% (1e-4), 11.10% (1e-5 and 0), against 7.00% for permuted pixels.

### 2.2 Correction: the MLP's apparent *gain* from rotation is a normalisation artifact

The dense write-up reports the rotation *helping* the standardized MLP by 2.01 pp at
N=1,000 and concludes "the rotation changes the task, not just its obfuscation".
That conclusion does not survive a control. `learners.py` `standardize` divides by
`std + std_floor` with `std_floor = 1e-5`, and the 9x9 pool has 6 pixels with
std < 1e-3 (min 1.32e-4) and 17 with std < 1e-2 — so on *permuted pixels* the MLP is
handed up to 17 features of quantisation noise amplified by up to 7,020x, which a
rotation happens to mix away. Same recipe, permuted pixels, N=1,000, 300 epochs,
learner seed 11, only the normalisation changed:

| normalisation on permuted pixels | error |
|---|---|
| `standardize`, `std_floor=1e-5` (as reported) | 8.83% |
| `standardize`, `std_floor=0.05` | 6.91% |
| `4x-0.5` (the study default) | 6.44% |
| *(rotated arm, `standardize`, `std_floor=1e-5`)* | *6.84%* |

Against the study's own default normalisation the **permuted** arm is 0.40 pp
*better* than the rotated one. 1.92 of the claimed 2.01 pp is one hyperparameter.
So: a rotation does change which per-feature tricks matter, but there is no measured
evidence that it changes which dense recipe wins.

### 2.3 What a rotation does break

Unaffected by `Q`: RBF / arccos / NTK kernels (exactly), L2 weight decay, isotropic
Gaussian input noise, Gaussian-init MLPs. Broken by `Q`: per-feature normalisation
(`4x-0.5` and `standardize` both become meaningless as *pixel* operations), input
dropout, any L1 / sparsity / feature-selection prior, kNN with per-feature scaling,
trees, and anything relying on pixel non-negativity or the `[0,1]` range. Whitening
additionally changes the feature scale and sign (outputs are signed, zero-mean, and
for RBF the gamma grid had to be rescaled by 0.083 at eps=1e-2 / 0.043 at eps=1e-4 or
the kernel is all-zero).

---

## 3. What whitening + rotation kills, and what survives

### 3.1 Killed outright: the second-order attack in `topology.py`

`topology.recover_layout` reads a ridge-regularised partial correlation matrix that
is essentially non-zero only on lattice edges. Neighbour-correlation diagnostic,
20,000 rows:

| variant | mean abs r, true neighbours | non-neighbours | ratio |
|---|---|---|---|
| permuted pixels (current) | 0.3497 | 0.0674 | **5.2** |
| ZCA, **no rotation**, eps=1e-3 | — | — | **10.5** |
| whitened + rotated | 0.0558 | 0.0571 | **0.98** |
| exact whitening + rotated | 0.0172 | 0.0191 | 0.90 (sampling floor 1/sqrt(M)=0.0071) |

End-to-end layout recovery (ground truth = each feature's most-correlated raw pixel):

| variant | N | `recover_layout` | adjacency precision | chance | exact cells |
|---|---|---|---|---|---|
| permuted pixels | 1,000 | ok | 88.9% | 4.4% | 90.1% |
| permuted pixels | 10,000 | ok | 92.4% | 4.4% | **92.6% (all 75 variance-carrying features)** |
| rotation only | 1,000 | **crashes** | 3.5% | 4.4% | 3.7% (ceiling 64.2%) |
| rotation only | 10,000 | **crashes** | 5.6% | 4.4% | 3.7% (ceiling 64.2%) |
| ZCA eps=1e-2 + rot | 10,000 | **crashes** | 11.1% | 12.5% | 4.9% (ceiling 35.8%) |
| ZCA eps=1e-4 + rot | 10,000 | **crashes** | 18.1% | 13.1% | 2.5% (ceiling 34.6%) |
| ZCA eps=1e-2, **no rotation** | 1,000 | crashes | **44.4%** | 4.4% | 7.4% |

Three caveats on that table, all of which the raw write-ups under-state.
*(i)* The "exact cells" column is **not comparable across rows**: it counts features
placed on their peak pixel, and for the rotated variants the delocalised synthesis
filters claim only 28-52 distinct peak pixels, capping the column at 34.6-64.2%
rather than 100%. Read the adjacency column against its own recomputed chance level
instead. *(ii)* On that basis, eps=1e-4 at N=10,000 (18.1% vs 13.1% chance on 144
pairs, z ~ 1.8) is "not significant", not "at chance". *(iii)* `recover_layout`
raises `ValueError` on every rotated variant (NaN in the stress-majorisation step at
`topology.py:305` when the mutual-kNN graph disconnects); the reported precisions
come from running the decisive QAP stage alone from 9 starts, a choice that favours
the attacker.

**Conclusion: the rotation is the active ingredient.** Whitening alone is *worse*
than useless (ratio 10.5, precision 44.4%) because ZCA is by construction the
whitening that stays closest to the original variables (Kessy, Lewandowski &
Strimmer, https://arxiv.org/abs/1512.00809): its filters are localised
centre-surround, so each coordinate still *is* its pixel.

### 3.2 Survives, partially: ICA, topographic sorting, non-negativity

After whitening, second-order structure is gone but *energy* correlations are not,
which is exactly the setting topographic ICA was invented for (Hyvärinen, Hoyer &
Inki, Neural Computation 13(7):1527-1558,
https://direct.mit.edu/neco/article/13/7/1527/6531/Topographic-Independent-Component-Analysis).
Two agents ran the family independently and agree.

How pixel-like are the recovered components (max abs correlation with a raw pixel,
20,000 unlabelled rows = train + query images):

| basis | mean | median | >0.9 | >0.99 |
|---|---|---|---|---|
| released features, no attack | 0.278 | 0.274 | 0 | 0 |
| FastICA logcosh | **0.695** | 0.800 | 31/81 | 8 |
| FastICA cube | 0.690 | 0.743 | — | — |
| zero-atom / cone-facet search (400 restarts, ~100 s) | — | 0.861 | 16/55 | **12** |
| *ceiling*: ZCA, the closest white basis to raw pixels | 0.896 | 0.892 | 81/81 | — |

Layout quality from those components (topographic affinity + the same lattice QAP):

| attack | N | adjacency agreement | chance | mean Manhattan | chance | exact cells |
|---|---|---|---|---|---|---|
| ICA logcosh + energy affinity + QAP | 1,000 | 0.444 | 0.046 | 4.60 | 5.49 | 13.6% |
| ICA cube + partial-corr affinity + QAP | 1,000 | **0.486** | 0.046 | 4.57 | 5.55 | 7.4% |
| ICA logcosh + energy affinity + QAP | 10,000 | 0.438 | 0.044 | 4.81 | 5.51 | 2.5% |
| ICA + **oracle** layout of the same components | 1,000 | 0.764 | — | 0.23 | — | 85.2% |

Ten times chance on adjacency, but Manhattan displacement barely below chance: the
map is locally right and globally folded.

Downstream — the number that matters, with the controls (cnn-09, CPU, 1 member,
10,000 query rows; **single-draw pilots**):

| arm, N=1,000 (60 ep) | error | N=10,000 (15 ep) | error |
|---|---|---|---|
| permuted pixels + true grid | 4.61% | — | — |
| permuted pixels + `recover_layout` grid (**status-quo attack works**) | 4.56% | same | **1.97%** |
| ZCA-whitened pixels, no rotation, true grid | 5.71% | same | 2.24% |
| ICA sources + *oracle* layout | 9.81% | — | — |
| **ICA sources + recovered layout (the attack)** | **10.59%** (3 seeds: 10.52/10.56/10.68) | same | **4.48%** |
| **released features + random layout (NO attack)** | **10.58%** | same | **4.44%** |
| facet/cone attack, all channels | 49.66% | same | 46.07% |

**The attack is worth 0.0 +/- 0.1 pp.** The gap it had to close was 4.9 pp at
N=1,000 and 2.2 pp at N=10,000. Three reasons this is structural rather than a
tuning failure: (1) every FastICA output is exactly white and raw pixels are
correlated, so **no** ICA solution can be the pixel basis — the ceiling for any
white basis is 0.896; (2) FastICA's answer scores *higher* on its own contrast than
the pixel-aligned basis does (logcosh 3.33 vs 2.73; cube 4.45e7 vs 3.59e7), so the
pixel basis is not the optimum being searched for, and a better optimiser moves
*away* from the grid; (3) doubling the unlabelled pool from 11,000 to 20,000 rows
moves max abs r from 0.692 to 0.695. This is Comon's identifiability result read in
the defender's favour: the leftover rotation is identified only by non-Gaussianity,
and the direction non-Gaussianity points in here is not the pixel axes
(https://www.sciencedirect.com/science/article/abs/pii/0165168494900299).

Also worth noting: **partial basis recovery gives no partial credit.** 31 of 81
pixels recovered almost exactly buys zero accuracy, because a 3x3 convolution needs
whole neighbourhoods, not scattered pixels. The best-effort facet attack, which
keeps only high-atom channels, is *catastrophically worse* than no attack (49.66%
vs 10.58%) because it discards 60-80% of the information.

Two measurement caveats. The ICA "null" and "ceiling" columns in the dense write-up
use different estimators from the measurement itself; computed like-for-like
(Hungarian, same row count, active columns only) the honest null is 0.0649
(eps=1e-2) / 0.0778 (rotate_only) rather than 0.108 / 0.086, and the ceiling that
actually bounds a 64-component FastICA is rank-64 exact ZCA at 0.8257, not 0.9446.
Direction of the error is conservative — ICA's margin over chance is slightly
*larger* than reported — so no conclusion flips. And one step in `attack_ica.py`
(lines 217-219) rescales the sources using the secret `A`; it is an exact no-op
(verified: downstream statistics differ by 0.000e+00, because everything downstream
is per-feature affine invariant) but it should be deleted rather than left as a path
by which a future edit silently becomes oracle-assisted.

### 3.3 Correction: `epsilon` has no security content

`whiten.py`'s stated argument is that after whitening `Cov(z) = I`, so the
second-order statistics contain "zero bits" about `Q` and the residual leak is
"confined to the tail". That premise is false at both operating points anyone used.
Measured on the canonical pool: at **eps=1e-2** the achieved per-direction variances
`lambda/(lambda+eps)` run from 1.51e-6 to 0.977 (median 0.328), **all 81** are below
0.99, `cond(Cov_z) = 6.5e5` and max off-diagonal 0.153; at **eps=1e-3** the median is
0.830 and 74/81 are below 0.99. `Cov(z)` is nowhere near the identity, and because
every eigenvalue is separated, its eigenvectors *are* the columns of `PQU`.

The correct argument is different and stronger: for **every** epsilon including 0,
and for `rotate_only` too, `Cov(z) = A Sigma A^T` determines `A` only up to an
unknown orthogonal factor, and that residual `O(81)` ambiguity is what protects the
grid. Consequence: epsilon buys no security whatsoever; it is purely an accuracy tax.
Relatedly, the attack write-up's observation that exact rank-75 whitening "resisted
the facet attack better" (median abs r 0.666 vs 0.901) is a **search-dimension
confound**, not an epsilon effect. At matched dimension (24 restarts, 400 Adam steps,
20,000 rows):

| variant, attacker's chosen subspace | median abs r | #>0.99 |
|---|---|---|
| eps=1e-3, attacker keeps eig>0.3 (56 dims) | **0.995** | 17 |
| eps=1e-3, top-75 | 0.649 | 0 |
| eps=1e-3, all 81 | 0.579 | 0 |
| eps=0 rank-75 exact, all 75 | 0.666 | 0 |
| eps=0 exact, matched 56 dims | **0.999** | 14 |
| rotate_only, top-56 | 0.990 | — |

The attacker picks the subspace; the defender cannot. Shipping rank-75 buys nothing.

---

## 4. If you adopt it anyway: the exact spec, and what it does to the ladder

The variant the measurements support is **rotation only, no whitening**
(`whiten.py`, `method='rotate_only'`).

1. **Fit on the full public pool, not on the draw.** `X = raw/pool_images.npy`
   (60,000 x 81 canonical UNPERMUTED float32, already hash-pinned in
   `raw/data_manifest.json`); `mu = X.mean(0)` in float64. Fitting on all 60,000 rows
   is what keeps the map identical for every seed, level and row, and keeps it a
   protocol decision (like `perm`) rather than learner leakage — it uses no labels and
   no train/query split. It does mean the map is a property of the pool and must be
   published with the protocol.
2. **Rotation.** `g = Generator(PCG64(ROT_SEED)).standard_normal((81,81))`;
   `Q, R = qr(g)`; `Q = Q * sign(diag(R))` (the sign fix makes `Q` reproducible across
   LAPACK builds). Commit `sha256(ROT_SEED)` in `protocol.draft.json` now; publish
   `ROT_SEED` when the study closes. Publishing the seed with the data is equivalent
   to publishing the transform — which, per section 5, is roughly where you already are.
3. **Apply identically everywhere.** `y = (x - mu) @ Q.T` in float64, cast to float32,
   for every train and query row of every seed; feed `y[:, perm]` so the existing
   permutation plumbing and hashes still apply. Note in the protocol that `perm` is now
   **cosmetic**: `P Q` is itself Haar, so the permutation adds nothing.
4. **Re-pin hashes** as a *second, parallel* `data_manifest` entry. Do not mutate the
   live study.
5. **If you insist on whitening**, use ZCA with `eps = 1e-2` (the train-only CV
   optimum) and budget +0.9 to +1.2 pp on the kernels, +1.3 to +4.1 pp at smaller eps.
   Understand that you are buying accuracy loss and no security.
6. **Declare what changed for learners**: inputs are no longer in `[0,1]`, are signed
   and zero-mean, and every candidate assuming `4*x - 0.5`, non-negativity or per-pixel
   dropout must be re-tuned.

### Effect on the study's ladder

Today the study publishes a clean pair: dense ladder A `kr-arccos1-d3` at 2.32%
(N=10,000) and unrestricted ladder B `topo-cnn09-x3` at 1.53%, against a true-grid
spatial reference of 1.51% — i.e. **recovering the grid is worth ~0.8 pp and costs
6-9 s of CPU inside the fit**, and the recovered grid is statistically
indistinguishable from the real one (+0.0355 pp paired, 11 final seeds).

Under `rotate_only`, ladder B as currently constituted disappears: the topology
route is at chance, the best attack we could build ties no-attack, and the two
ladders collapse into one. The dense ladder becomes *the* measured quantity, at a
cost of roughly +0.1 to +0.2 pp on the kernel (inside single-draw noise) and with
every per-feature MLP recipe needing a re-tune. That is a coherent thing to want. It
is also strictly *less* informative than what you already have, because the
0.8 pp figure — the actual answer to "how much is 2-D structure worth?" — is only
measurable when topology recovery is allowed.

**The cheaper alternative, which the evidence actually supports:** leave the data
alone, keep `features.permutation_disclosure`'s existing requirement that topology
recovery be *declared*, audit the declarations, and publish the two ladders you
already have. You get the honest number without the whitening tax and without a
security claim you cannot back. This is also how the prior art does it — the NIPS
2003 Feature Selection Challenge hid Gisette's MNIST origin by randomising feature
order *and withholding the attribute metadata*, not by a clever transform
(https://proceedings.neurips.cc/paper/2004/file/5e751896e527c862bf67251a474b3819-Paper.pdf).

---

## 5. What no data transformation can enforce

**Can be enforced by a transform:** that a *named* statistic is uninformative.
"The partial correlation matrix of the released features carries no neighbourhood
signal" is checkable, falsifiable, and true of `rotate_only` (ratio 0.98 vs 5.2).
That is worth something: it turns "we hope nobody recovers the grid" into "the cheap
route is closed, here is the number".

**Cannot be enforced by a transform:** that topology is unrecoverable. Every
invertible map is by definition information-preserving; hiding is cost, never
impossibility. Formally: applying a uniformly random element of a group `G` hides
information only to the extent the data distribution is `G`-invariant, and for
`G = O(81)` that means Gaussian (Comon). MNIST is strongly non-Gaussian, so the only
perfectly-hiding linear map is one whose output is exactly `N(0, I)` — which carries
no task. Any transform that preserves enough structure for a CNN to beat an MLP
preserves enough structure for an attacker to find.

**And in your case the binding constraint is not even that.** The pool is the public
MNIST training split. Reproduced twice, independently:

| variant | rows re-identified (of 20,000) | max abs pixel error after inversion | `recover_layout` exact placement | wall time |
|---|---|---|---|---|
| `rotate_only` | **99.94%** | 4.8e-3 (float32 release precision) | **92.6%** | 16.9 s |
| ZCA eps=1e-3 + rot | **99.98%** | 4.0e-3 | **92.6%** | 18.1 s |
| ZCA eps=1e-2 + rot | **99.96%** | 5.0e-3 | **92.6%** | 25.9 s |

The attacker recomputes `mu` and `W` himself from public MNIST (both are published
in the spec), matches each released row to a pool row by the *rotation-invariant*
norm `||W(x - mu)||`, solves least squares for `Q`, and runs the study's own
`topology.recover_layout` on the reconstruction — getting **exactly the same grid as
on unobfuscated data**. Script:
`research/whitening-deterrent-scripts/reidentify_public_mnist.py`. A tighter variant
(matching on whitened norms only, plus orthogonal Procrustes) reaches
`||X_hat - X||/||X|| = 2.4e-15`.

This is not new; it is the standard break of distance-preserving privacy maps (Liu,
Giannella & Kargupta, PKDD 2006; Giannella et al., https://arxiv.org/abs/0911.2942).
And it applies equally to **today's** permutation-only protocol, where a sorted-pixel
fingerprint is the rotation-invariant statistic. Adding whitening does not make this
worse — but it removes the entire security rationale for paying the accuracy cost.

**So the enforcement mechanism is, and will remain, the rule.** `learner_contract.forbidden`
already lists "the official MNIST test split" and "any pretrained weights or external
data"; that clause, plus declaration and audit, is what stands between you and a
30-line inversion. That is a perfectly respectable position — it is how every
benchmark with a public source dataset works — but it should be stated as a rule,
not implied to be a mechanism.

Non-invertible alternatives do remove the ability, and they change the task rather
than hide it. A per-example permutation reduces each image to its intensity multiset
and deletes all spatial information. Per-example random linear mixing was tried for
privacy as InstaHide and broken by Carlini et al.
(https://arxiv.org/abs/2011.05315).

---

## 6. Limitations, and what a stronger attacker would do

**Limitations.** One dev draw (2026092301), one query slice, learner seed 11, no
repetitions; cnn-09 pilots at 15-60 epochs and 1 member versus the protocol's
100-125 epochs and 3 members; the MLP at N=10,000 used 60 epochs rather than 150
(hence 3.09% vs the ~2.7% on record) — identical in recipe across variants, but
under-trained in absolute terms. Ground truth for "which pixel is this component" is
its most-correlated raw pixel, which is itself noisy at r ~ 0.7 and biases the oracle
numbers *downwards*, i.e. against the attacker. Attack coverage is FastICA with three
contrasts, two whitening modes, three component counts, plus a crude cone/facet
search; the QAP was capped at 60 s. The two agents used different default epsilons
(1e-2 dense, 1e-3 attack), so the dense and attack tables are not strictly paired. A
pool-independent transform (fitted on training rows only) was never measured. The
RBF gamma grid had to be extended with a scale-matched copy for the non-pixel
variants. `test_whiten.py` passes 18/18 — but note that
`test_whitened_pool_covariance_is_identity_up_to_epsilon` asserts only that the
spectrum matches `lambda/(lambda+eps)` and `mean(diag) > 0.5`, not closeness to `I`,
so it does not test the property its name claims.

**What a stronger attacker would do**, roughly in order of expected damage:

1. **Download MNIST.** Section 5. Already exact, already reproduced, ~20 s. Only the
   rule prevents it.
2. **Topographic ICA proper** (Hyvärinen-Hoyer-Inki), which optimises the lattice
   energy-correlation structure *jointly* instead of sorting ICA outputs post hoc.
   Not run by anyone here, and the single most obvious untested member of the family.
3. **A serious minimum-volume / separable-NMF solver** (SISAL/MVES with restarts,
   deflation and a joint refit, exploiting the 59% exactly-zero pixel values). MNIST
   pixels are non-negative with a large mass at zero, which is precisely the
   separability condition under which the containing simplicial cone — hence the
   linear map — is unique (Donoho & Stodden NIPS 2003; Lin et al.,
   https://arxiv.org/abs/1406.5273). A one-minute untuned version already recovers 12
   pixel functionals at abs r > 0.99, and a 7-minute `log|det B|` ascent in a parallel
   run put 0.948 median energy on a single pixel per axis and restored top-pair
   precision to 0.591. Neither converted to accuracy; a real solver plausibly would.
4. **Learned symmetry discovery.** Efe & Ozakin, "Blind Recovery of Latent Domains via
   Unsupervised Symmetry Discovery" (https://arxiv.org/abs/2606.17782, June 2026)
   reports recovering the latent domain of permuted MNIST under a *general invertible
   linear* operator at r=0.995 (15x15, 225 dims) and r=0.947 (27x27, 729 dims) from
   ~1,100 samples, using stationarity + locality + InfoMax regularisation of a shallow
   group-conv net. Your setting is 81 dims and 20,000 samples — strictly easier on both
   axes. Nobody reproduced it. **This is the single experiment that would settle the
   question**, and until it is run, "no cheap attack found" is the strongest claim
   available.

Prior art says recovery is the default expectation, not an exotic exploit: Le Roux
et al., "Learning the 2-D Topology of Images" (NIPS 2007,
https://papers.nips.cc/paper_files/paper/2007/hash/7fa732b517cbed14a48843d74526c11a-Abstract.html)
and Rahman & Nemenman (Phys. Rev. E 108, 034410, https://arxiv.org/abs/2305.04386).

---

## 7. Files and how to re-run

New files created for this question (nothing in the live study was modified):

| path | what |
|---|---|
| `whiten.py` | the transform (`fit_transform`, `default_transform`, `job_arrays`, `filter_peaks`) |
| `test_whiten.py` | 18 unit tests, all passing |
| `attack_ica.py` | FastICA attack + localisation metrics (dense agent) |
| `research/whitening-dense-results.{md,json}` | dense cost + second-order attack tables |
| `research/whitening-dense-scripts/` | `run_dense.py`, `attack_topology.py`, `eps_cv.py`, `collate.py` |
| `research/whitening-attack-results.{md,json}` | ICA / topographic / cone attack |
| `research/attack_ica_topographic.py` | attack agent's independent implementation |
| `research/whitening-attack-scripts/` | 21 stage scripts (`stage1.py`, `atom.py`, `facet*.py`, ...) |
| `research/whitening-deterrent.md` | this document |
| `research/whitening-deterrent-scripts/` | the verification/refutation runs below |

Re-run (all CPU, `/tmp/pmnist-env/bin/python`, from the study root):

```bash
P=/tmp/pmnist-env/bin/python
R=/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923

$P -m unittest discover -s $R -p test_whiten.py -t $R                    #  ~3 s, 18/18

# the decisive break: public-MNIST re-identification + Procrustes, 3 variants
$P $R/research/whitening-deterrent-scripts/reidentify_public_mnist.py    # ~60 s total

# refutation of the 'rotation helps the MLP' claim (normalisation control)
$P $R/research/whitening-deterrent-scripts/mlp_normalisation_control.py  # ~4 min

# refutation of 'exact rank-75 resists the facet attack better' (dimension confound)
$P $R/research/whitening-deterrent-scripts/facet_dimension_confound.py   # ~4 min

# released-covariance spectrum + like-for-like ICA nulls and ceilings
$P $R/research/whitening-deterrent-scripts/covariance_and_ica_nulls.py   # ~1 min

# power check: does the neighbour diagnostic fire on ZCA-without-rotation?  (yes, 10.5)
$P $R/research/whitening-deterrent-scripts/neighbour_diagnostic_power.py # ~1 min

# proof that attack_ica.py's secret-A rescale is an exact no-op
$P $R/research/whitening-deterrent-scripts/rescale_is_noop.py            # ~20 s

# the two agents' own pipelines
$P $R/research/whitening-dense-scripts/run_dense.py --help
$P $R/research/whitening-attack-scripts/stage1.py
```

Scratch for the original runs lives in `/tmp/whiten/` (dense, `exp1-13.py`) and
`/tmp/whiten-review/` (review); both are ephemeral, which is why the load-bearing
scripts were copied into `research/whitening-deterrent-scripts/`.
