# Whitening / rotation vs the permutation: dense-method cost and attack resistance

Study `pmnist-medium-cutoffs-20260923`. All numbers below are **single-draw, single-seed CPU pilots** on dev seed 2026092301 with learner seed 11: one training draw, one query set of 10,000 rows, no repetitions and no error bars. The binomial standard error on a 10,000-row query set at 3% error is 0.17 pp, so treat gaps under ~0.4 pp (N=10,000) and ~1.0 pp (N=1,000) as noise. The eleven FINAL seeds 2026092001..2026092011 were never touched; no file belonging to the live study was modified.

Variants (all share the study's draws exactly: `order = PCG64(SeedSequence(seed).spawn(2)[0]).permutation(60000)`, train = order[:N], query = order[10000:20000]):

- **permuted pixels** - the current protocol, `x[:, perm]`.
- **random rotation only** - `z = P Q (x - mean)`, `Q` Haar orthogonal (QR of a PCG64 Gaussian, seed 20260923). No whitening.
- **whitened+rotated, eps** - `z = P Q U diag(1/sqrt(lambda+eps)) U^T (x - mean)`, pool covariance. eps is in variance units of [0,1] pixels; eps=1e-2 is the value chosen by the train-only CV sweep at the bottom of this file.
- **exact, top-64** - drop the 17 near-null eigen-directions and whiten the rest exactly; releases 64 features, so its covariance is exactly the identity and there is no low-variance tail to leak. The `mlp` family hard-codes 81 inputs, so it cannot run on this variant without a protocol change (marked `-`).

## Dense methods: query error (%) on dev seed 2026092301, single draw, learner seed 11

| method | N | permuted pixels (current protocol) | random rotation only, no whitening | whitened+rotated, ZCA eps=1e-2 (81 feat) | whitened+rotated, ZCA eps=1e-4 (81 feat) | whitened+rotated, exact, top-64 components |
|---|---|---|---|---|---|---|
| kernel ridge, arc-cosine depth 3 | 1000 | 6.15 | 6.38 | 7.34 | 9.68 | 9.65 |
| kernel ridge, arc-cosine depth 3 | 10000 | 2.30 | 2.44 | 3.17 | 3.60 | 3.49 |
| kernel ridge, RBF | 1000 | 6.44 | 6.44 | 7.43 | 10.28 | 10.53 |
| kernel ridge, RBF | 10000 | 2.52 | 2.52 | 2.83 | 4.10 | 4.41 |
| MLP 1024-1024, standardized | 1000 | 8.85 | 6.84 | 7.36 | 9.64 | - |
| MLP 1024-1024, standardized | 10000 | 3.09 | 2.66 | 2.86 | 3.29 | - |

## Deltas vs the permuted-pixel protocol (percentage points; positive = the preprocessing hurts the dense learner)

| method | N | permuted pixels (current protocol) | random rotation only, no whitening | whitened+rotated, ZCA eps=1e-2 (81 feat) | whitened+rotated, ZCA eps=1e-4 (81 feat) | whitened+rotated, exact, top-64 components |
|---|---|---|---|---|---|---|
| kernel ridge, arc-cosine depth 3 | 1000 | 0 (ref) | +0.23 | +1.19 | +3.53 | +3.50 |
| kernel ridge, arc-cosine depth 3 | 10000 | 0 (ref) | +0.14 | +0.87 | +1.30 | +1.19 |
| kernel ridge, RBF | 1000 | 0 (ref) | +0.00 | +0.99 | +3.84 | +4.09 |
| kernel ridge, RBF | 10000 | 0 (ref) | +0.00 | +0.31 | +1.58 | +1.89 |
| MLP 1024-1024, standardized | 1000 | 0 (ref) | -2.01 | -1.49 | +0.79 | - |
| MLP 1024-1024, standardized | 10000 | 0 (ref) | -0.43 | -0.23 | +0.20 | - |

## Pixel-topology attack (second order), dev seed 2026092301

| variant | N | recover_layout | adjacency precision | chance | exact placement | cnn-09 (1 member, 25 epochs) error |
|---|---|---|---|---|---|---|
| permuted pixels (current protocol) | 1000 | ok | 88.9% | 4.4% | 90.1% | 5.10% |
| random rotation only, no whitening | 1000 | CRASHED | 3.5% | 4.4% | 3.7% | 8.29% |
| whitened+rotated, ZCA eps=1e-2 (81 feat) | 1000 | CRASHED | 12.5% | 12.5% | 2.5% | 9.58% |
| whitened+rotated, ZCA eps=1e-4 (81 feat) | 1000 | CRASHED | 16.0% | 13.1% | 2.5% | 11.59% |
| ZCA whitened eps=1e-2, permuted, NO rotation | 1000 | CRASHED | 44.4% | 4.4% | 7.4% | - |
| permuted pixels (current protocol) | 10000 | ok | 92.4% | 4.4% | 92.6% | - |
| random rotation only, no whitening | 10000 | CRASHED | 5.6% | 4.4% | 3.7% | - |
| whitened+rotated, ZCA eps=1e-2 (81 feat) | 10000 | CRASHED | 11.1% | 12.5% | 4.9% | - |
| whitened+rotated, ZCA eps=1e-4 (81 feat) | 10000 | CRASHED | 18.1% | 13.1% | 2.5% | - |

## ICA attack on the whitened variant (FastICA, 20,000 unlabelled rows)

| transform | eps | components | contrast | mean peak-energy of the composite map | random-rotation control (null) | ZCA control (ceiling) | matched energy on active pixels | pixels matched >0.9 | recover_layout on the sources |
|---|---|---|---|---|---|---|---|---|---|
| rotate_only | 0 | 64 | logcosh | 0.549 | 0.086 | 0.839 | 0.390 | 15/64 | CRASHED |
| zca | 0.01 | 64 | logcosh | 0.556 | 0.108 | 0.945 | 0.395 | 15/64 | CRASHED |
| zca | 0.01 | 64 | cube | 0.497 | 0.108 | 0.945 | 0.407 | 17/64 | CRASHED |
| zca | 0.01 | 81 | logcosh | 0.859 | 0.108 | 0.945 | 0.037 | 0/64 | CRASHED |

Why `recover_layout` cannot be chained after ICA: every FastICA output is exactly white (`Cov(S) = I`) and correlation is invariant to per-coordinate affine rescaling, so no amount of post-hoc rescaling of the sources can put second-order neighbour structure back. The second-order attack is structurally unavailable downstream of ICA; what ICA can give an attacker is the pixel *basis*, measured by the matched-energy column. `matched energy` is a Hungarian one-to-one matching of sources to the 64 active pixels on the row-normalised energy of the composite map `W_ica A`; 1.0 means every source is exactly one pixel. The ZCA control is the ceiling for any white basis (no rotation of whitened data can reproduce raw pixels, because raw pixels are correlated).

Read of the ICA numbers: off-the-shelf FastICA on 20,000 unlabelled rows recovers a real but partial piece of the pixel basis -- matched energy 0.39-0.41 against a 0.086-0.108 null and a 0.84-0.94 ceiling, with 15-17 of the 64 active pixels isolated above 0.9 in about 25 s. It is not a working attack (the recovered basis is too noisy to hand the lattice search a usable input, and the 81-component run collapses onto the 17 near-null quantisation directions, matched energy 0.037), but it is not nothing either. The reason plain ICA stalls is that its model is wrong for this data: ICA insists the sources are independent, pixels are strongly dependent, and no white basis can be the pixel basis. The method designed for exactly that mismatch -- topographic ICA (Hyvarinen, Hoyer & Inki 2001), which models dependence between *neighbouring* components and recovers a 2-D topography from natural image patches -- is the attack this defence should be assumed to face. It was not run here. Non-negative ICA (Plumbley) is a second untested route: the pixels are non-negative and 44% zero on the median active pixel, so the released cloud is a linear image of a shifted orthant whose 81 extreme rays are the pixel directions.

## Train-only 5-fold CV used to choose epsilon (arc-cosine depth 3, N=1000)

| variant | CV error (%) | chosen lambda |
|---|---|---|
| permuted-pixels (baseline) | 7.00 | 1e-06 |
| whitened zca eps=0.01 | 8.50 | 1e-07 |
| whitened zca eps=0.01 (no 4x-0.5) | 8.70 | 1e-07 |
| whitened zca eps=0.001 | 9.80 | 0.0001 |
| whitened zca eps=0.001 (no 4x-0.5) | 9.90 | 0.0001 |
| whitened zca eps=0.0001 | 10.70 | 0.0001 |
| whitened zca eps=0.0001 (no 4x-0.5) | 10.80 | 0.0001 |
| whitened zca eps=1e-05 | 11.10 | 0.0001 |
| whitened zca eps=1e-05 (no 4x-0.5) | 11.20 | 0.0001 |
| whitened zca eps=0 | 11.10 | 0.0001 |
| whitened zca eps=0 (no 4x-0.5) | 11.20 | 0.0001 |
| whitened exact, top-48 components | 10.10 | 1e-05 |
| whitened exact, top-64 components | 10.70 | 0.0001 |
| whitened exact, top-72 components | 11.30 | 1e-07 |

Scripts: `research/whitening-dense-scripts/`. The attack side of the same question (topographic ICA and friends) was worked in parallel by a second agent and lands in `research/whitening-attack-results.*`.

## Which ingredient does the work

- **Whitening alone is not a defence.** ZCA whitening filters are localised centre-surround, so after whitening-without-rotation each coordinate still essentially *is* its pixel: the attack keeps 44.4% adjacency precision against a 4.4% chance rate.
- **The rotation alone is a full defence against this attack, and it is free.** Rotation without whitening puts the attack at chance (3.5% vs 4.4%) while costing the RBF kernel exactly nothing (a rotation preserves pairwise distances) and the arc-cosine kernel +0.23 pp at N=1,000 / +0.14 pp at N=10,000, both inside the single-draw noise. It *helps* the standardized MLP (-2.01 pp at N=1,000, -0.43 pp at N=10,000): a rotation spreads each pixel's signal over all 81 coordinates, which suits per-feature standardisation, Gaussian input noise and dropout better than the raw sparse pixels do. That is a change to the task, not a free lunch -- it shifts which dense recipe wins -- and one draw is not enough to size it.
- **Whitening on top of the rotation buys no extra security.** An attacker can whiten the released data himself: `Q x` and `Q W x` differ by a linear map he can estimate from the sample covariance, so both leave him with exactly the same residual problem (recover an unknown orthogonal factor from higher-order statistics). What whitening does change is what the *learner* sees, and there it is a straight cost: +0.9 to +1.2 pp for the kernels at eps=1e-2 and +1.3 to +4.1 pp at eps=1e-4 or with exact whitening.
- **The variance floor is a security/accuracy dial with the wrong shape.** Large eps keeps accuracy but leaves the released covariance far from the identity (at eps=1e-2 only the top handful of directions are whitened at all, so the low-variance tail is still second-order identifiable); small eps whitens properly but amplifies ~17 directions of area-resize quantisation noise to unit variance and costs 3-4 pp at N=1,000. Since the rotation already does the security work, there is no reason to pay this.


## Notes on the method configurations

- Kernel ridge uses the study's `plans/classical_candidates.json` grids: lambda in {1e-7,1e-6,1e-5,1e-4,1e-3}, `cv_subsample=4000`, arc-cosine depth 3.
- `classical.fit_predict` applies the study-wide `4x-0.5` map to kernel inputs and it cannot be disabled. For the arc-cosine kernel this is harmless: the kernel is homogeneous and the gram matrix is renormalised by its mean diagonal, so only the -0.5 shift matters, and on whitened features (per-feature sd 1, scaled to 4) that shift is negligible. Measured directly in the CV table below: pre-descaling the inputs so that `4x-0.5` reproduces z exactly changes the CV error by 0.1-0.2 pp at every eps.
- The RBF gamma grid IS in `4x-0.5` pixel units and does not transfer: whitened features have ~27x larger mean squared pairwise distance, so the study's gammas would give an all-zero kernel. The grid used on every non-pixel variant is the study grid PLUS a scale-matched copy (study gamma times the ratio of mean squared pairwise distances); the existing train-only CV chooses between them and always picked the scale-matched values.
- MLP: widths [1024,1024], dropout 0.3, input noise 0.3, lr 0.002, weight decay 0.01, batch 128, warmup 0.1, 1 member, `normalization='standardize'`. 300 epochs at N=1,000. At N=10,000 the requested 150 epochs needs ~24 min per fit on this CPU, so **60 epochs** were used instead (~7 min); the N=10,000 MLP numbers are therefore under-trained in absolute terms but identical across variants, which is what the comparison needs.
- Topology attack: `topology.recover_layout` verbatim. On every rotated variant it raises `ValueError: matrix contains invalid numeric entries` - the mutual-kNN graph of a near-identity partial correlation matrix is disconnected, so the hop distances are infinite. The reported adjacency precision then comes from the decisive stage run on its own (`topology._search`, the QAP that on real pixels reaches the true layout exactly) from 9 starts, and the cnn-09 fit is given that layout. Both choices favour the attacker.
- `adjacency precision` = of the feature pairs the recovered layout puts on adjacent lattice cells, the fraction whose pixel-space synthesis filters (columns of `A_inv`) peak on adjacent pixels. `chance` is the density of adjacent pairs among the peak-pixel assignment actually realised, so it is above 4.4% when the delocalised filters share peak pixels.

