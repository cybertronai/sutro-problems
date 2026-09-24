# Does whitening + a random rotation remove the ability to recover the 9x9 topology?

Second opinion on the whitened variant, produced independently of (and concurrently
with) the agent that wrote `whiten.py` and the other `attack_ica.py`.  Everything
below is CPU-only, dataset seed **2026092301** (a dev seed), no Modal, no GPU, no
final seed.  Query labels were read only through `raw/pool_labels.npy` indexed by
`study.query_indices(2026092301)`, and only to score pilots.

Code: `research/attack_ica_topographic.py`; scripts and provenance in
`research/whitening-attack-scripts/`; machine-readable numbers in
`research/whitening-attack-results.json`.

**Every accuracy below is a single draw, a single cnn-09 member and a reduced epoch
budget (60 epochs at N=1000, 15 at N=10000, versus the protocol's 100-125 and three
members).** They rank methods; they are not protocol numbers and they are not
comparable to the study's 1.53 % headline.

---

## Short answer

1. **Whitening + rotation does kill the existing attack, completely.** The
   second-order cue that `topology.recover_layout` lives on is gone: on the
   whitened variant it recovers the grid at chance, and at N=1000 it does not even
   run (NaN in its MDS embedding).
2. **The ICA attack works in the sense that it finds real structure, and fails in
   the sense that it buys nothing.** FastICA components each track one pixel at
   |r| ~ 0.69, the topographic layout it produces is ~10x above chance, and the
   end-to-end CNN error is **identical to doing no attack at all** (10.59 % vs
   10.58 % at N=1000; 4.48 % vs 4.44 % at N=10000).
3. **The reason is structural, not a matter of data or tuning.** ICA outputs are
   white, raw pixels are not, and at N=10000 FastICA's solution scores *higher* on
   the ICA contrast than the pixel-aligned basis does — the pixel basis is simply
   not the optimum being searched for. Going from 11,000 to 20,000 unlabelled
   images changes nothing.
4. **A real part of the pixel basis *is* recoverable, and it still buys nothing.**
   FastICA lands 31-32 of its 81 components on a raw pixel at |r| > 0.9 (8 at
   > 0.99); a crude non-negativity / zero-atom search — a prior that does *not*
   assume independence — recovers 15-16 at |r| > 0.9 with **12 at |r| > 0.99** in
   about a minute. Higher precision, lower coverage. Roughly a third of the pixel
   basis comes back and the downstream CNN gains nothing from it, because a
   convolution needs whole neighbourhoods, not scattered pixels.
5. **The deterrent has a real but modest cost for the dense methods being
   measured** — at the `epsilon` the sister analysis recommends (1e-2) kernel ridge
   loses ~1 point and the MLP loses nothing; at the `epsilon = 1e-3` default used
   here the cost is larger (see `research/whitening-dense-results.md`).

---

## 1. The variant under test

`whiten.default_transform()`: `z = A (x - mean)`, `A = P Q W`, `W` = ZCA whitening
with variance floor `epsilon = 1e-3`, `Q` a Haar rotation (`qr_pcg64`, seed
20260923), `P` the study's fixed 81-feature permutation.

| property | value |
| --- | --- |
| condition number of `A` | 20.6 |
| invertible | yes — **no information is destroyed; the defence is purely computational** |
| released covariance | *not* exactly `I`: 29 of 81 eigenvalues < 0.5, 14 < 0.05, min 1.5e-5 |

That last row matters twice. It is a small second-order leak (the released
covariance eigenvalues are `lambda/(lambda+eps)`, from which the pool's covariance
spectrum is recoverable), and it creates ~25 near-degenerate coordinates that are
mostly quantisation noise. Those coordinates are a trap for every attack metric
below: a filter coefficient on an always-black pixel looks large but contributes
nothing. All headline numbers therefore use **correlation between a component and
a raw pixel**, which needs no filter algebra and is immune to that trap.

Sanity checks (`verify_conventions`, `synthetic_check`): the forward map reproduces
`whiten.job_arrays` to 2.1e-7, `A A_inv = I` to 1.2e-15, and on synthetic data with
*known* independent sources the same pipeline recovers the true mixing columns at
mean |cosine| 0.9995 (min 0.9989, 16/16 distinct). The algebra is right; the
failures below are not bookkeeping errors.

## 2. The existing second-order attack, applied to the whitened variant

Ground truth for "which pixel is this feature" is the feature's most-correlated raw
pixel (evaluation only).

| variant | N | adjacency agreement | chance (mean / p95) | exact cells | note |
| --- | --- | --- | --- | --- | --- |
| whitened + rotated | 1000 | **0.111** | 0.066 / 0.104 | 3.7 % | `recover_layout` **crashed**; number from a robust re-implementation of the same statistic |
| whitened + rotated | 10000 | **0.056** | 0.068 / 0.097 | 1.2 % | ran, output at chance |
| permuted pixels (status quo) | 1000 | 0.882 | — | 90.1 % | 73/81 exact; 9 features are constant in that draw |
| permuted pixels (status quo) | 10000 | 0.924 | — | 92.6 % | **75/75 variance-carrying features exact**; the 6 misses are the 6 constant features |

The released features correlate with their best raw pixel at only |r| = 0.33
(mean), versus 1.0 for permuted pixels. **The cheap attack is dead.** That part of
the user's intuition is exactly right, and it is the rotation that does it:
whitening alone leaves ZCA nearly diagonal, and rotation alone leaves
`Q Sigma Q^T`, whose eigenstructure still encodes the lattice.

## 3. The residual ICA attack

Label-free: FastICA on `concat(train_x, query_x)` (11,000 rows at N=1000; 20,000 at
N=10000), 81 components, then a topographic affinity between components, then the
same lattice QAP `topology.py` uses.

### 3a. How pixel-like are the components?

| N (unlabelled rows) | contrast | whitening | max abs r with a pixel (mean / median) | synthesis top-1 energy | distinct peak pixels |
| --- | --- | --- | --- | --- | --- |
| 1000 (11,000) | logcosh | re-whitened | 0.692 / 0.785 | 0.204 | 70 |
| 1000 (11,000) | cube | re-whitened | 0.683 / 0.707 | 0.168 | 68 |
| 1000 (11,000) | logcosh | assumed white | 0.677 / 0.673 | 0.290 | 62 |
| 10000 (20,000) | logcosh | re-whitened | **0.695 / 0.800** | 0.205 | 69 |
| 10000 (20,000) | cube | re-whitened | 0.690 / 0.743 | 0.171 | 67 |
| 10000 (20,000) | logcosh | assumed white | 0.683 / 0.697 | 0.286 | 60 |
| — | *no attack* (released features) | — | 0.278 / 0.274 | — | — |
| — | *ceiling*: ZCA, the closest white basis to pixels | — | **0.896 / 0.892** | 0.899 | 81 |
| — | random direction (null) | — | — | 0.092 | — |

FastICA gets ~77 % of the way to the ceiling, and **doubling the unlabelled data
changes the third decimal**. `whiten=False` (taking the released data to be
already white, as the task sketch suggested) is consistently the weakest setting —
the `epsilon` floor makes the released data not-quite-white, so letting sklearn
re-whiten is the better attack.

### 3b. The recovered layout

| N | contrast | affinity | adjacency agreement | chance | mean Manhattan (dihedral-aligned) | chance | exact cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | logcosh | corr of sqrt abs s | 0.444 | 0.046 | 4.60 | 5.49 | 13.6 % |
| 1000 | cube | partial corr of sqrt abs s | **0.486** | 0.046 | 4.57 | 5.55 | 7.4 % |
| 1000 | logcosh (assumed white) | corr of sqrt abs s | 0.354 | 0.043 | 4.85 | 5.45 | 11.1 % |
| 10000 | logcosh | corr of sqrt abs s | 0.438 | 0.044 | 4.81 | 5.51 | 2.5 % |
| 10000 | cube | partial corr of sqrt abs s | 0.438 | 0.045 | 4.12 | 5.56 | 0.0 % |
| 10000 | logcosh (assumed white) | corr of sqrt abs s | 0.347 | 0.048 | 5.10 | 5.45 | 2.5 % |
| — | *oracle* layout of the same components | — | 0.65-0.76 (0.76 for logcosh at N=1000) | — | 0.23 | — | 85 % |

Read this carefully: adjacency agreement is **10x chance**, but mean Manhattan
displacement is barely below chance and exact-cell placement is near zero. The
recovered map is *locally* right and *globally* scrambled — folded and warped, not
a dihedral image of the grid. The energy affinity does carry the signal (mean
0.27-0.31 on true-neighbour pairs vs 0.035-0.07 on far pairs; 51 % of each
component's top-4 affinities are true neighbours, vs 4.4 % chance), so the QAP is
not the bottleneck: even the **oracle** layout of these components tops out at 0.76
agreement, because the components are not pixels in the first place.

### 3c. Does any of it help downstream? No.

cnn-09 (`spatial_learner.build_model`, the `_fit_topo_cnn` training arithmetic),
CPU, 1 member, 10,000 query rows.

**N=1000, 60 epochs**

| arm | query error | what it shows |
| --- | --- | --- |
| permuted pixels + true grid | **4.61 %** | what a CNN is worth here |
| permuted pixels + `recover_layout` grid | **4.56 %** | the status-quo attack, fully successful |
| ZCA-whitened pixels, **no rotation**, true grid | **5.71 %** | whitened values are still fine CNN inputs |
| ICA sources + *oracle* layout | 9.81 % | a perfect topographic sort of ICA components |
| ICA sources + recovered layout | **10.52 %** (seeds 101/102/103: 10.52 / 10.56 / 10.68, mean **10.59 %**) | the attack, end to end |
| released features + random layout (**no attack**) | **10.58 %** | the honest baseline |
| dense MLP (standardise) on whitened features | 10.65 % | what the benchmark is meant to measure |
| dense MLP (standardise) on permuted pixels | 10.49 % | same, un-whitened |
| ICA sources + random layout | 11.75 % (seeds: 11.75 / 12.15 / 12.36, mean 12.09 %) | ICA alone *hurts* a CNN |
| ICA abs(s) + recovered layout | 21.61 % | rectifying the sources is much worse |

**N=10000, 15 epochs**

| arm | query error |
| --- | --- |
| permuted pixels + `recover_layout` grid | **1.97 %** |
| ZCA-whitened pixels, no rotation, true grid | **2.24 %** |
| ICA sources + recovered layout | 4.48 % |
| released features + random layout (no attack) | **4.44 %** |

The whole attack is worth **-0.01 points at N=1000 and -0.04 at N=10000**, i.e.
nothing, against a member-seed spread of about 0.16 points. The gap it would have
had to close is 4.9 points at N=1000 (10.58 -> 5.71) and 2.2 at N=10000.

The `ZCA-whitened, no rotation, true grid` row is the important control: whitened
pixel values cost the CNN only ~1.1 points when laid out correctly. So **the entire
defence rests on the rotation being unfindable**, not on whitening degrading the
signal. And the `ICA sources + oracle layout` row shows that improving the QAP or
the affinity cannot rescue the attack: the components themselves are wrong.

## 4. Why ICA fails — and why more data will not fix it

Every ICA output is white; raw MNIST pixels are strongly correlated; therefore **no
ICA solution can be the pixel basis.** The best a white basis can do is ZCA, which
correlates 0.896 with its own pixel. FastICA lands at 0.695, and it lands there for
a reason that is not optimisation failure:

| quantity (N=10000, sample-white coordinates) | FastICA solution | pixel-aligned (ZCA) basis | untouched released |
| --- | --- | --- | --- |
| logcosh contrast, sum of squares | **3.33** | 2.73 | 0.28 |
| cube (kurtosis) contrast, sum of squares | **4.45e7** | 3.59e7 | 1.13e4 |
| mean excess kurtosis of the coordinates | **918** | 894 | — |

FastICA's answer is *more* non-Gaussian than the pixel basis. The pixel basis is
therefore not the global optimum of the criterion, so a better optimiser or more
data moves the attack away from the grid, not towards it. (The pixel directions
were mapped into the sample-white coordinates and Löwdin-orthogonalised to make
them feasible; `|B - Q|_max = 0.364`.)

This is the sharpest thing I can say in the defender's favour: against
**independence-based** attacks, whitening + rotation is not merely expensive to
break, it is aimed at a criterion whose optimum is somewhere else.

## 5. How much of the pixel basis comes back, and by which prior

### 5a. What ICA already recovers

Before looking for a better attack, note what the "failed" one achieves:

| | components matching a raw pixel at &#124;r&#124; > 0.8 | > 0.9 | > 0.99 |
| --- | --- | --- | --- |
| FastICA logcosh, N=1000 (11,000 rows) | 40 / 81 | 32 | 8 |
| FastICA logcosh, N=10000 (20,000 rows) | 40 / 81 | 31 | 8 |

**A third of the pixel basis is recovered almost exactly and the CNN still gains
nothing** (section 3c). That is the most striking single fact here: partial basis
recovery is not partial credit. A 3x3 convolution needs whole neighbourhoods; 31
scattered pixels mixed with 50 wrong ones gives it nothing to stand on.

### 5b. The prior that does not need independence

Independence is the wrong assumption for pixels. **Non-negativity is not.** Every
MNIST pixel is >= 0 and 59 % of all pixel values are exactly 0, so in pixel space the
data lies on the facets of a translated simplicial cone with 81 facets. A linear map
carries cones to cones, so the facets — the pixel functionals — are identifiable
from the released data with **no independence assumption at all**.

Operationally: there exists a direction whose projection has a large atom (a point
mass), one per pixel, and random directions have essentially none.

| smoothed atom mass of a projection | value |
| --- | --- |
| true pixel directions (12 active pixels) | 0.764 (min 0.307) |
| random directions | 0.059 |

Gradient ascent on that atom mass, run in the 56-dimensional subspace where the
released covariance eigenvalue exceeds 0.3 (the `epsilon` floor's degenerate
directions have to be excluded or they hijack the objective):

| run | result |
| --- | --- |
| 24 restarts, 14 s | median |r| with a raw pixel **0.901**; best six: 1.000, 0.999, 0.999, 0.991, 0.974, 0.937 |
| 400 restarts, N=10000, ~100 s | 55 distinct directions; **16 match a raw pixel at |r| > 0.9, 12 at |r| > 0.99** |
| 400 restarts, N=1000, ~70 s | 63 distinct directions; 15 at |r| > 0.9, 11 at |r| > 0.99 |

So the trade against FastICA is **precision up, coverage down**: 12 pixels at
|r| > 0.99 versus FastICA's 8, but only 15-16 above 0.9 versus FastICA's 31. In one
minute, with a prior that does not care that pixels are dependent.

What I could *not* do in the time available is turn it into a learner. The search
saturates at ~15-16 pixels out of 75; the other 40-plus recovered directions are
high-atom junk, and with them included the layout recovery is at chance (agreement
0.049) and the CNN is far worse than no attack. Selecting channels by atom score
(a label-free signal that is strongly predictive of being a real pixel) is the
obvious fix and is measured in section 6.

Two more notes on this family:

* A Plumbley-style non-negative-ICA rotation search (400 steps, concave penalty
  above a low quantile) barely beat the null — best-match |r| 0.31 vs 0.29 for the
  untouched features, against 0.695 for FastICA. I report it as a weak
  implementation, not as evidence against the family.
* Removing the `epsilon` floor does **not** obviously help the defender in the way
  one might hope, but it did help here: on an exact rank-75 whitening (drop the null
  directions, whiten exactly, rotate) the same facet probe reached median |r| 0.666,
  max 0.917, and nothing above 0.99 — noticeably worse for the attacker than the
  `epsilon = 1e-3` variant's 0.901 median. That is one configuration of one quick
  attack and should not be over-read, but if the variant ships, `epsilon` deserves a
  deliberate choice rather than a default.

## 6. Best-effort facet attack, end to end

Channels selected by atom score (label-free), laid out by `recover_layout` where it
survives and by the robust QAP on the same statistic where it does not, then cnn-09.
N=1000, 60 epochs, 1 member.

| channels kept | of which are true pixels (&#124;r&#124; > 0.9) | query error |
| --- | --- | --- |
| all 63 | 15 | 49.66 % |
| top 40 by atom score | 13 | 55.01 % |
| top 25 by atom score | 10 | 60.44 % |
| top 15 by atom score | 9 | 62.46 % |
| — *no attack* (released features, random layout) | — | **10.58 %** |

And at N=10000, 15 epochs, all 55 channels: 46.07 % versus 4.44 % for no attack.

So the facet attack, as implemented, is **catastrophically worse than doing
nothing**: it throws away 60-80 % of the information (it keeps at most 56 of 81
dimensions and then only the high-atom ones) in exchange for making ~15 of them
interpretable, and a CNN would rather have all 81 dimensions in a random order.
`recover_layout` crashed on these channels too, and the robust fallback placed them
at chance (adjacency agreement 0.049).

This is the honest state of that attack: **a real and surprising leak at the level
of individual pixel functionals, with no path yet from there to accuracy.** The
missing step is not the layout search — it is recovering enough of the basis,
which needs proper deflation and a joint refit rather than 400 independent
gradient ascents.

## 7. What this means for the benchmark

* **The obfuscation works against everything I could actually run.** The strongest
  end-to-end attack on the whitened variant ties the no-attack baseline.
* **It is a computational defence, not an information-theoretic one.** `A` is
  invertible with condition number 20.6; an oracle that inverts it hands back the
  original 1.5 %-error problem. Anything claimed here is "no cheap attack found",
  never "no attack exists".
* **It costs the dense methods something, and how much depends on `epsilon`.** My
  own control (standardised MLP, 100 epochs, N=1000) moved 10.49 % -> 10.65 %, i.e.
  nothing; but the MLP is the least sensitive method. The sister analysis
  (`research/whitening-dense-results.md`, same dev seed) measures kernel ridge
  losing 1.19 points at `epsilon = 1e-2` and 3.53 points at `epsilon = 1e-4` at
  N=1000, and recommends `epsilon = 1e-2` on train-only CV. **My attack numbers are
  all for the `epsilon = 1e-3` default**; the attack should be re-run at whichever
  `epsilon` ships. What does disappear is the spatial CNN's 4-5 point advantage,
  which is the intent.
* **Declare it, do not rely on secrecy.** The right framing is "second-order
  topology recovery is provably useless here; higher-order recovery is an open
  problem and a legitimate candidate contribution", not "the topology cannot be
  recovered".
* **Watch the non-negativity channel.** It is the one prior that does not care
  about the pixels being dependent, and a crude version of it already recovers a
  dozen pixels exactly. If this variant ships, a stated bounty on "recover k pixels
  from the released features" would surface the real attack faster than any
  internal red-teaming.

## 8. Limitations

* One dataset seed (2026092301), one member seed for most arms, reduced epochs.
  Member-seed spread where measured is ~0.16 points, which is smaller than every
  gap the conclusions rest on except the attack-vs-no-attack comparison — and that
  one is a *null* result, so the spread is the right unit: the attack is worth
  0.0 +- 0.1 points.
* "True cell of a component" is defined by the largest correlation with a raw
  pixel. For components correlating ~0.7 with their best pixel that assignment is
  itself noisy, and only 67-70 of 81 components claim distinct pixels; the layout
  metrics inherit that noise. It biases the oracle numbers **downwards**, i.e. in
  the attacker's disfavour, so it does not weaken the negative conclusion.
* The QAP search was capped at 60 s per affinity with the same start construction
  `topology.py` uses plus random restarts; a longer search would raise the QAP
  objective but cannot exceed the oracle bound in 3b.
* FastICA was run with three contrasts, two whitening modes and three component
  counts; topographic ICA proper (Hyvärinen-Hoyer-Inki), which optimises the
  lattice energy-correlation structure jointly rather than post hoc, was **not**
  run and is the most obvious untested member of the ICA family.
* The facet/cone attack is under-developed here: no proper deflation, no joint
  refit, a fixed kernel width, and only two subspace choices tried.
