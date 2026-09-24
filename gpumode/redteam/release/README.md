# Red team: the linear release

Harness `sutro-mnist-medium-time/1.2.0` stops handing submissions 9x9 pixels.
Every draw is now released as

```
z = Q W (x - mu)                                (eval.release_map / eval.make_draw)
```

with `W` exact PCA whitening onto the top `release_dims = 60` principal
directions of *that draw's own* 10,000 training rows (no variance floor), `mu`
those rows' mean, and `Q` a Haar-random 60x60 orthogonal matrix drawn fresh per
draw from the secret seed (`RELEASE_SALT`). One map is fitted on the training
rows and applied to both halves; the map is never returned, logged or sent to
the child. A submission sees `(10000, 60)` float32 with zero mean and identity
covariance, and a secret permutation of the ten labels on top.

Why that construction, and not a rotation alone: a rotation preserves the pixel
covariance, and one eigendecomposition matched by rank against public MNIST
undoes it (the [rotation-obfuscation
study](../../../mnist/experiments/rotation-obfuscation-20260923/README.md)
recovers pixels to 7-8% RMS that way). Whitening deletes that channel exactly --
the released covariance is the identity, so there is nothing to match. A
variance floor (`eps > 0`) would leak it back through the dead-border subspace,
which is why the whitening is exact on the top 60 directions. And the map is
per-draw and secret because a protocol-constant map is cracked once offline and
shipped as a 5.5 KB int8 inverse.

This directory is the evidence for the *remaining* attack surface. Every script
gets its data from the real evaluator -- `eval.make_draw(pool, seed, n_train,
n_test, universes, release_dims=60)`, the same call a ranked run makes -- and
only uses the secret map for scoring, recovered by least squares from the draw's
true pixels (`harness.Draw._fit_oracle`, residual 2.5e-08, which also proves the
rows we scored against are the rows the evaluator drew). `Draw.check_oracle`
also verifies the construction independently: refitting the documented recipe
with the study's `common.whitener` and reading off `Q = A W+` gives an
orthogonal `Q` to 3.0e-09, an offset equal to `-A mu` to 6.3e-10, and a
smallest kept eigenvalue of 2.2e-04 (the evaluator's rank-deficiency guard
trips at 1e-10).

## Summary

| # | attack | what the attacker holds | result | what stops it |
|---|---|---|---|---|
| a | blind unmixing + lattice recovery | one draw, nothing else | 4.0x chance on lattice edges, **-0.1 points** vs the honest MLP | nothing needed: the recovered lattice is worthless |
| b | informed moment matching | public MNIST (40k rows) + the draw | pixels back at **0.077 rel-RMS**, offline CNN **98.52%** vs honest MLP 95.79% (**+2.73**) | the size cap (c); *not* the hold-out (see below) |
| c | packing (b) into a submission | -- | payload **48.1x** the 20,480-byte cap; the largest artifact that fits **loses 5.0 points** | `max_source_bytes` / `max_literal_bytes` |
| d | (b) fired at a Fashion hold-out draw | as (b) | **57.12%**, below the 70% floor (honest MLP 85.95%) | the hold-out, against a *non-adaptive* smuggler |
| e | norm re-identification of source rows | the whole 60k pool | **15.9%** labels, 0.06% row hits; table is 1.8x the cap | per-draw fit of `(mu, W)` + the size cap |

Bottom line: the release does what it was chosen to do. A blind entrant gets no
spatial prize at all. An informed entrant still can invert the map -- that was
never in doubt -- but cannot carry the offline knowledge it takes through the
door, and exact whitening is what makes the knowledge *large*: because the
released covariance is exactly the identity there is no eigenvalue ordering to
truncate against, so the attacker must work in the full 60 dimensions and carry
the full 60-dimensional class statistics. Halving that (k = 30, 8,416 bytes)
does not shrink the attack, it destroys it (46.49%).

## (a) Blind attack -- `blind_attack.py`

No source data. Unmix the release blindly, arrange the recovered sources on a
9x9 lattice from their `|s|` partial correlations (pmnist embedding + QAP
search), convolve.

| unmixer | edge precision @1.0 | chance | @1.5 | chance | ratio @1.5 | source localisation r<=1.5 | sources >0.9 of 60 | unmix / QAP |
|---|---|---|---|---|---|---|---|---|
| sparse-nonneg | 0.118 | 0.048 | 0.265 | 0.105 | 2.5x | 0.65 | 17 | 14 s / 20 s |
| FastICA logcosh | 0.235 | 0.061 | 0.451 | 0.113 | 4.0x | 0.65 | 9 | 18 s / 22 s |

So the lattice *is* partly recovered -- 4x chance is not nothing, and 17 of 60
sources are near-single pixels. It buys nothing:

| learner on the same draw | test accuracy |
|---|---|
| CNN on the sparse-nonneg lattice | 95.69% |
| CNN on the FastICA lattice | 95.66% |
| CNN on a **random** layout (control) | 95.68% |
| honest MLP on the released features | **95.79%** |

Every CNN, including the one on a random layout, lands within 0.13 points of the
MLP: at 9x9 with 10,000 examples the spatial prior is worth about nothing once
you have to find it yourself, and the best blind attack does not even recover
the layout well enough to beat the control. The attacker also has no sound way
to choose between the two unmixers (the QAP objectives are not comparable), so
the table above is generous to them -- we trained a CNN on both and report the
best. Total 4-5 min of CPU.

## (b) Informed attack -- `informed_attack.py`

The attacker holds the 40,000 pool rows the draw did not use, with labels.
Three steps, no secrets:

1. **Undo the secret label permutation.** Whiten both sides to their own top-60
   subspace; the two spaces then differ by an unknown orthogonal map, under
   which class-mean norms and class-covariance spectra are invariant, so the
   pairing is a 10x10 assignment problem. **10 of 10 classes correct in 0.2 s.**
   The per-draw relabel is not an obstacle to an informed attacker.
2. **Recover the map** by matching class-conditional means and second moments
   (`attacks.informed_attack`, several inits, 1.5-2.5 min of CPU).
3. **Cash in** with a CNN trained offline on the public 40,000 pixel rows.

| init | pixel rel-RMS, init -> refined | moment loss | pixels recovered >0.9 (of 64 live) | offline CNN on the recovery |
|---|---|---|---|---|
| eig (unlabelled) | 0.945 -> 0.132 | 131.6 | 13 | 98.05% |
| cls | 0.428 -> 0.182 | 144.8 | 13 | 97.87% |
| random | 0.919 -> 0.104 | 98.9 | 12 | 98.16% |
| **rlc** (attacker's pick, lowest loss) | 0.608 -> **0.077** | **65.4** | 21 | **98.52%** |
| oracle (scoring diagnostic: the true map) | 0.028 | 76.6 | 58 | -- |

Note the `eig` init, which is the cheap second-order attack that breaks a
rotation-only release, starts at 0.945 rel-RMS -- i.e. at the constant-mean
baseline. Whitening did delete that channel. What survives is the labelled
moment fit, which whitening cannot flatten.

| | test accuracy |
|---|---|
| offline CNN on the **true** test pixels (ceiling) | 98.64% |
| offline CNN on the **recovered** test pixels | **98.52%** |
| honest MLP on the released features | 95.79% |
| **gain** | **+2.73 points** |

Two points of accuracy at this level is the whole leaderboard, so this attack
would win if it could be shipped. It cannot -- see (c). Total 5-6 min of CPU.

## (c) Can it be shipped? -- `artifact_size.py`

Caps (`bands.json` defaults, enforced in `eval.py` before the submission is
imported): `max_source_bytes` 20,480, `max_literal_bytes` 4,096. Sizes below are
"bytes added to `submission.py`": int8 with one fp16 scale per row, zlib if it
helps, base64, chopped into <=4,096-byte literals.

**What the informed attack must carry.**

| object | numbers | int8 in a submission | fp16 | vs the 20,480 cap |
|---|---|---|---|---|
| public mean, top-60 eigenbasis + eigenvalues, 10 class means, 10 class covariances | 23,901 | **22,364** | 60,496 | 1.09x, *alone* |
| offline CNN weights (study CNN, 32/64/64) | 721,034 | **962,824** | -- | 47.0x |
| verbatim source of the attack functions | -- | 14,751 | -- | 0.72x |
| **total** | | **985,188** | | **48.1x** |

The CNN is the obvious blocker, but the statistics alone already miss. Shrinking
the CNN does not rescue it (16/32/32: 464,008 bytes; 8/16/16: 228,520).

**Why the statistics cannot be shrunk.** The k-sweep says k = 50 statistics
would fit alone (17,708 bytes) and k = 30 comfortably (8,416):

| k | 10 | 15 | 20 | 25 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|---|
| int8 bytes in a submission | 1,932 | 3,176 | 4,692 | 6,428 | 8,416 | 12,908 | 17,708 | 22,364 |

but truncation is exactly what exact whitening forbids. The released covariance
is the identity, so its "top 30 eigenvectors" are noise directions with no
correspondence to the attacker's top-30 public subspace. Measured, on the same
draw: the k = 30 recovery reaches pixel rel-RMS **0.700** (a constant-mean
predictor scores 0.66) and the tiny CNN applied to it scores **46.49%**. The
attacker must work in the full 60 dimensions, which is over the cap.

**What DOES fit, and what it is worth.** The only cap-compliant smuggled
artifact is a small CNN with no statistics at all, used through a linear adapter
fitted in-competition on the released labels (freeze the CNN, learn
`x_hat = z H^T + c` by cross-entropy):

| | value |
|---|---|
| TinyCNN (8/16/16, 2x2 pool, 6,218 params), int8 in a submission | **8,100 bytes** (fits) |
| its accuracy on true 9x9 pixels | 97.86% |
| through the learned adapter, on the release | **90.90%** |
| honest MLP on the release | 95.94% |
| **gain** | **-5.04 points** |

The minimal *combined* payload -- k = 30 statistics (8,416) plus the TinyCNN
(8,100) -- is 16,516 bytes, leaving 3,964 for code against 14,751 bytes of
verbatim attack source, and it scores 46.49%. Total 2-3 min of CPU.

## (d) The Fashion hold-out -- `holdout_check.py`

`holdout_draws` calls per ranked run come from Fashion-MNIST, released through
their own map, at positions the secret seed picks, and must clear
`holdout_min_bp` = 3000 bp (70%).

| | accuracy on the Fashion hold-out draw (seed 9203) | verdict |
|---|---|---|
| smuggler: MNIST statistics + MNIST offline CNN, full pipeline | **57.12%** | **FAIL** (floor 70%) |
| honest MLP on the same released features | **85.95%** | PASS |
| the MNIST CNN on the *true* Fashion pixels | 14.08% | -- |

The label-permutation step already degrades to 4 of 10 classes correct. The
57.12% is worth reading correctly: it is not smuggled MNIST knowledge working on
Fashion (that scores 14.08%), it is the moment-matching map acting as a badly
parameterised classifier *fitted in-competition on the released Fashion labels*.
The hold-out floor is 12.9 points above it, and an honest learner clears the
floor by 16 points, which is what a floor has to do.

**Residual risk, measured.** The smuggler can tell which kind of draw it is on:
the best moment-matching residual is **65.4** on the MNIST draw and **752.9** on
the Fashion draw, a factor of **11.5**. An adaptive submission reads its own
residual and falls back to an honest learner on hold-out calls. So the hold-out
is not what stops the informed attack -- the size cap is. The hold-out's job is
to stop the cheaper smugglers, and it does.

## (e) Exact-source re-identification -- `exact_source.py`

`Q` is orthogonal, so `||z|| = ||W (x - mu)||` is a rotation-invariant
fingerprint of the source image. Refit the published recipe, compute every pool
row's norm, match by nearest norm, read off the label.

| where the attacker fits (mu, W) | exact row hits | label accuracy | median rank of the true row | true row in top 10 / top 100 | median relative norm error |
|---|---|---|---|---|---|
| all 60k pool rows | 0.12% | 16.33% | 463 | 1.9% / 13.3% | 9.5e-03 |
| the public 40k rows | 0.06% | **15.89%** | 521 | 1.3% / 11.1% | 1.1e-02 |
| the draw's secret 10k train rows (**oracle**) | 99.96% | 99.96% | 0 | 100% / 100% | 3.8e-09 |

The oracle row is the point: the channel is wide open if you know the fit set,
and it is closed only because the harness fits `(mu, W)` on the draw's own
secret training rows. That perturbs the attacker's norms by ~1%, against a
median pool norm spacing of 1.3e-05, which leaves the true row at median rank
~500 -- about **9 bits** short. A matcher using pairwise distances (also
`Q`-invariant) could plausibly find those 9 bits; it is not implemented here,
and that is the honest residual risk of this channel.

The table is the second door, and it is the tighter one:

| what has to ship | in a submission | vs the cap |
|---|---|---|
| 60,000 pool norms, fp16 | 129,136 | 6.3x |
| 60,000 labels in norm order, 4-bit packed + zlib | 31,896 | 1.6x |
| 4,096 norm quantiles (replaces the norms) | 4,260 | 0.21x |
| **cheapest working table** (quantiles + labels) | **36,156** | **1.8x** |
| entropy floor of the label column alone (3.32 bits/row, base64) | **33,200** | **1.6x** |

The last line is the one that cannot be argued with: 60,000 labels carry 3.32
bits each, 24,900 bytes entropy-coded, 33,200 after base64, against a
20,480-byte file. No coder beats it. Total 2 s of CPU.

## Reproduction

Local CPU only; no Modal, no GPU. Python `/tmp/pmnist-env/bin/python` (3.11,
numpy 1.26.4, torch 2.2.2 CPU, scipy, scikit-learn). The scripts import `eval`
from `gpumode/` and `attacks.py`, `models.py`, `common.py` from the
rotation-obfuscation study; `harness.py` wires both up.

```bash
export MNIST_POOL_CACHE=/tmp/gpumode-pool           # mnist_data.load_pool caches the 4 gz files here
cd /Users/yaroslavvb/git/sutro-problems/gpumode/redteam/release
/tmp/pmnist-env/bin/python ./blind_attack.py        # ~4 min
/tmp/pmnist-env/bin/python ./informed_attack.py     # ~5 min
/tmp/pmnist-env/bin/python ./artifact_size.py       # ~2 min
/tmp/pmnist-env/bin/python ./holdout_check.py       # ~4.5 min
/tmp/pmnist-env/bin/python ./exact_source.py        # ~1 s
/tmp/pmnist-env/bin/python ./summarize.py           # merges the five into results.json
```

Each script writes `results-<attack>.json`; `summarize.py` merges them into
`results.json` with a `summary` block and the cross-file hold-out detectability
number. Every script takes `--seed` (the case seed; the draw seed is derived the
way `eval.run_case` derives it) and `--release-dims`.

## Caveats

* One draw, one seed (case seed 101, MNIST draw seed 114, Fashion hold-out draw
  seed 9203). The rotation-obfuscation study ran three seeds on an equivalent
  release and its spreads were 0.1-0.4 points on accuracies and ~0.05 on edge
  precision, so the conclusions are not seed-sensitive, but the exact numbers
  here are single-draw.
* CPU budget forced shorter training than the study: 12 epochs for every CNN
  (the study used 15-30), 30 for the MLP. The honest MLP baseline (95.79-95.94%)
  is therefore a little below a fully trained one, which if anything flatters
  every attack.
* The distance-graph refinement of (e) was not implemented; the 9-bit gap is
  reported, not closed.
* `artifact_size.py` measures the verbatim source of the attack functions. A
  determined minifier would cut that figure; it does not change any verdict,
  because the payload alone is over the cap.
* Every numpy stage reproduces bit for bit (unmixing, lattice recovery, the
  moment fit: pixel rel-RMS 0.077 and moment loss 65.4 in both full runs). The
  torch CPU training does not: across two identical runs the accuracies moved by
  0.1 points (offline CNN 98.45 / 98.52), 0.4 (hold-out smuggler 57.51 / 57.12)
  and 0.8 (the adapter route 91.68 / 90.90). The tables report the second run,
  which is the one in `results.json`; no verdict is close enough to the line for
  that spread to matter.

## Not an attack: the dense ceiling on the release

`dense_ceiling.py` runs the pmnist study's arc-cosine depth-3 kernel ridge on the evaluator's own draws: 97.47% on pixels, 96.29% on the 60-dimensional release (three draws). It is band-calibration evidence, recorded in `results/release-dense-ceiling.json` and `results/release-utility.md`.
