# RFF-ridge on MNIST-medium — non-record research

This archive contains @mikech76's exploratory RFF-ridge study from [PR #110](https://github.com/cybertronai/sutro-problems/pull/110), frozen at `95368c5469238ff5d6c3c0633bf8831947454c84`. It is included as research, without a record-table entry. The learner, protocol, energy harness and original evidence JSON are unchanged.

The author's eleven accuracy draws were already seen during development. The requested A100 diagnostic was completed on **8 October 2026**: one existing draw, three measurement rounds, on an **A100-SXM4-40GB**. It is not a fresh eleven-draw qualification or an A100-80GB record result. Corrected reviewer measurements, source hashes and limitations are in [reviewer-evidence.json](reviewer-evidence.json).

The main caveats are precision labeling, incorrect historical measurement-scope/sham descriptions, and cross-platform image-array hashes. The corrections below describe the retained code; no old measurement is silently relabeled as a new run.

---

## Result

Method: random Fourier features with ridge regression on one-hot targets,
with the three hyperparameters `(p, gamma, lam)` selected **per draw** on an
inner-validation split of that draw's own training subset. Eleven previously seen exploratory
dataset seeds `20261001…20261011`, profile `medium-error-targets-v1`, 10,000
train / 10,000 test, 9×9 images. Official generator and official evaluator.

| D | correct / 110,000 | accuracy | SD (ddof=1) | 3 % band (106,700) | 2 % band (107,800) |
|---:|---:|---:|---:|---:|---:|
| 2448 | 106,735 | 97.0318 % | 0.2175 pp | PASS (+35) | FAIL (−1,065) |
| 4000 | 106,900 | 97.1818 % | 0.2530 pp | PASS (+200) | FAIL (−900) |
| 8000 | 107,169 | 97.4264 % | 0.2057 pp | PASS (+469) | FAIL (−631) |
| **12000** | **107,240** | **97.4909 %** | **0.2012 pp** | **PASS (+540)** | **FAIL (−560)** |
| PCA-QDA reference, same 11 draws | 105,224 | 95.6582 % | 0.2365 pp | FAIL (−1,476) | FAIL |

These are the author’s archived results, not a fresh reviewer sweep. Independent
review reconstructed all four aggregate counts, exact band comparisons and sample
SDs, and the D=12000 selection table. The author reports that D=2448 and D=4000
prediction hashes match earlier runs. That is reproducibility evidence on seen
draws, rather than independent confirmation.

### Efficiency metrics: what is measured and what is not

| metric | D=2448, f32 products + f64 accumulation | D=12000, float64 |
|---|---:|---:|
| A100 diagnostic energy above idle (mJ) | 19,854.014 counter / 19,900.217 sampled power | — |
| A100 diagnostic CUDA-event time (ms) | 193.524 | — |
| A100 diagnostic peak allocated bytes | 220,134,912 | — |
| grid-model energy (mJ) | — | — |
| grid-model time (ms) | — | — |
| time to score (s) | — | — |
| RTX 2060 energy per draw (mJ, **not an A100**) | 47,541 counter / 56,218 sampled power | — |
| RTX 2060 time per draw (ms, **not an A100**) | 877 | — |
| RTX 2060 peak device memory (bytes, **not an A100**) | 220,134,912 | — |

The A100 column is the completed one-draw diagnostic, not an eleven-draw or
three-independent-run median. RTX values below are historical author
measurements on different hardware. Grid-model cost and the D=12000 A100 cost
were not measured. No figure here is a record-table result.

## Completed reviewer A100 diagnostic

Configuration: D=2448, float32 products, float64 Gram accumulation, chunk 2048;
learner seed 42. Three rounds each repeated the full per-draw eighteen-point
selection, refit and prediction 31 times after warmup. The median over rounds was
**193.524099 ms** by CUDA events, **193.519069 ms** by wall time, and
**19,854.014292 mJ above idle** by the NVML counter. The sampled-power readout
gave 19,900.217387 mJ; it shares the board’s sensors and is not an independent
calibration. Peak allocated CUDA memory was 220,134,912 bytes.

| Round | Counter above-idle mJ/task | Sampled-power above-idle mJ/task | CUDA-event ms/task | Corrected counter sham residual mJ/nominal task |
|---:|---:|---:|---:|---:|
|1|19,395.951952|19,329.415790|192.387947|-1,216.870552|
|2|20,122.273577|20,270.212848|193.524099|-481.274963|
|3|19,854.014292|19,900.217387|193.635285|-249.915157|

This is warm repeated execution of the full learner invocation. **CPU projection
generation, host/device transfers and CUDA allocations occur inside each fit.**
Dataset file loading is outside. Energy is GPU board energy, including GPU idle
while host work occurs; it does not measure host CPU/RAM or wall-socket energy.
The original harness scope string says otherwise and remains unchanged as a
historical implementation defect.

On existing draw 20261001 the independent score was**9684/10000**, with the same
prediction hash and selected parameters as the author. This one-draw result is
below 9700; it does not determine an eleven-draw aggregate by itself. Recreated
label/index hashes matched, but image-array/NPZ hashes did not match the author’s
files. The original image arrays are absent, so exact pixel differences cannot
be quantified. The matching prediction is a limited observation, not proof of
bitwise input reproduction.

See [reviewer-evidence.json](reviewer-evidence.json) for full-precision interval
summaries and hash provenance, and [energy/README.md](energy/README.md) for
commands and the corrected interpretation of the unchanged harness. The original
measurement request is complete; this archive does not request another run.

## Files

| path | what it is |
|---|---|
| [rff_ridge.py](rff_ridge.py) | the frozen learner: fit and predict, NumPy and PyTorch backends |
| [run_protocol.py](run_protocol.py) | eleven-draw protocol driver; writes predictions, selection records, and hashes, then calls the official evaluator |
| [energy/nvml_energy.py](energy/nvml_energy.py) | NVML idle-adjusted energy harness |
| [energy/README.md](energy/README.md) | corrected measurement boundaries, commands and legacy-field interpretation |
| [reviewer-evidence.json](reviewer-evidence.json) | sanitized exact-source A100 diagnostic, independent arithmetic checks, hashes and limitations |
| [evidence/audit_evidence.json](evidence/audit_evidence.json) | per-draw correct/total for every D, the PCA-QDA reference on the same draws, SHA-256 of every prediction file, dataset seeds and dataset hashes, validation-isolation counts |
| [evidence/draw_manifest.json](evidence/draw_manifest.json) | SHA-256 of each generated `medium.npz` and of each array inside it |
| [evidence/accuracy_D*.json](evidence/) | full per-draw records: selected hyperparameters, inner-validation counts, evaluator output per error band, prediction hash, overlap counts |
| [evidence/selection_protocol_D12000.json](evidence/selection_protocol_D12000.json) | all 18 grid points with their inner-validation counts for all 11 draws, so the selection is fully auditable |
| [evidence/energy_rtx2060.json](evidence/energy_rtx2060.json) | the RTX 2060 sweep as machine-readable records: every variant, both readouts, all rounds, sham noise, peak device bytes, plus the caveats and the list of what was not measured |

## Model

```
u    = x ** p                                     81-d, 9x9 pixels in [0,1]
W    = default_rng(42).standard_normal((81, D)) * sqrt(2 * gamma)
b    = default_rng(42).uniform(0, 2*pi, D)         same generator, next draw
Z    = [cos(U W + b), 1]                          D+1 columns
A    = solve(Z^T Z + lam I, Z^T Y)                 ridge, Y one-hot
pred = argmax(Zq A)                                first maximum wins
```

`W` has shape `(81, D)`, so basis sizes are **separately generated**, not nested: D=12000
is not a prefix of D=8000. This is a fact about the code, and it is why the
D-curve near the band boundary is not monotonic — see *Findings that constrain
the frontier*.

### Selection procedure, frozen

Per draw, an 18-point grid:

| parameter | values |
|---|---|
| `p` | 0.25, 0.50 |
| `gamma` | 0.01, 0.03, 0.10 |
| `lam` | 0.10, 1.00, 10.00 |

Selected on an inner split of **this draw's own training rows only**:

```python
idx = np.random.default_rng(42).permutation(10000)
inner_train, inner_val = idx[:8000], idx[8000:]
```

Metric is inner-validation correct count. Ties break on smaller `lam`, then
smaller `gamma`, then smaller `p`. The selected point is refit on all 10,000
train rows of that draw, then test images are projected and scored.

Learner seed 42 in every run. No learned state crosses draws or basis sizes.

### Validation isolation

`rff_ridge.load_draw` deliberately does not return `test_labels`. Every run
asserts, before fitting, that the inner-validation rows and the full training
rows have zero index intersection with the test rows. Across the 44 runs behind
the table above the counts are:

| check | result |
|---|---|
| inner-validation ∩ test indices | **0 rows, 44/44 runs** |
| all train ∩ test indices | **0 rows, 44/44 runs** |
| test-label values used by fitting or selection | no; loader reads only the array length |

## Evidence

Per-draw correct counts, in seed order `20261001 … 20261011`:

| D | per-draw correct / 10,000 |
|---:|---|
| 2448 | 9684, 9695, 9712, 9722, 9688, 9663, 9727, 9720, 9693, 9696, 9735 |
| 4000 | 9692, 9711, 9736, 9739, 9687, 9677, 9746, 9723, 9714, 9719, 9756 |
| 8000 | 9735, 9737, 9749, 9757, 9703, 9719, 9768, 9756, 9735, 9738, 9772 |
| 12000 | 9727, 9746, 9766, 9753, 9740, 9733, 9765, 9757, 9725, 9736, 9792 |
| PCA-QDA | 9557, 9538, 9555, 9612, 9584, 9569, 9584, 9557, 9538, 9543, 9587 |

Every prediction file is hashed. Dataset hashes are in
[evidence/draw_manifest.json](evidence/draw_manifest.json); prediction hashes and
per-draw correct counts are in
[evidence/audit_evidence.json](evidence/audit_evidence.json). The four accuracy
tables are author evidence from this directory; no new eleven-draw reviewer
execution is implied. The D=2448 and D=4000 hash matches are the author’s
reported comparison to earlier predictions.

Accuracy is reported as `sum(correct) / 110,000` with the exact integer
comparison against the band threshold, not a rounded percentage. `SD` is
`ddof=1` over the eleven draws; all eleven are included.

---

## Measurements on non-A100 hardware, and what they do and do not establish

Everything in this section was measured on an **NVIDIA GeForce RTX 2060**
(12,288 MiB, driver 617.14, power limit 184 W), with torch 2.5.1+cu121, CUDA
12.1, Python 3.10, NumPy 2.2.6. **These numbers do not go in the A100 table.**
Different silicon, different idle draw, different telemetry. The A100 rows in the
main README cannot be compared with them, and we are not claiming a position
relative to them.

Idle baseline on that card, 40 one-second windows, no CUDA context open in the
sampling process: counter median **46.3 W**, min/max 45.8 / 51.0 W, SD 1.27 W;
`nvmlDeviceGetPowerUsage` median 47.1 W; utilization median 1 %. The idle windows
that bracket the runs themselves read 38.0–41.7 W, and NVML reports 31
compute processes on the device, so other processes share this GPU. That is why
the harness subtracts each active window from its own bracketing idle windows
rather than from one global baseline. The observed idle variation is not an uncertainty interval for active energy.
The counter and sampled power share the same sensors, so they are not independent
checks. The legacy `mean_util_percent` field averages accumulated history rather
than just the named interval; no interval-specific utilization claim is made.

Median of 3 rounds, mJ per draw, one draw = `20261001`:

| variant | mJ counter | mJ sampled power | ms/draw | peak device bytes | correct / 110,000 | 3 % band |
|---|---:|---:|---:|---:|---:|---:|
| **RFF D=2448, f32 products + f64 accumulation, chunk 2048** | **47,541** | **56,218** | **877** | **220,134,912** | 106,747 | PASS (+47) |
| RFF D=2448, float64 on device | 187,644 | 223,985 | 4,016 | 477,570,560 | 106,735 | PASS (+35) |
| RFF D=2448, blockwise Gram, float64, chunk 2048 | 196,251 | 219,942 | 4,238 | 428,965,376 | 106,735 | PASS (+35) |
| RFF D=4000, float64 on device | 530,358 | 616,151 | 10,791 | 808,220,672 | 106,900 | PASS (+200) |
| RFF D=2448, f32 rows + f64 Gram (f64 in products too) | 167,814 | 197,102 | 3,847 | 362,055,680 | 106,735 | PASS (+35) |
| PCA-QDA reference, float32 on device | 141 | 166 | 130 | 137,191,936 | 105,224 | FAIL (−1,476) |
| RFF D=2448, NumPy float32, CPU-resident | ≈ 0 (−2,182) | −534 | 6,912 | — | 106,746 | PASS (+46) |
| RFF D=2448, NumPy float64, CPU-resident | ≈ 0 (5,472) | −826 | 11,866 | — | 106,735 | PASS (+35) |

The CPU rows describe GPU telemetry while host computation runs; they do not
measure CPU energy. Their small signed above-idle values are not measured zero
and do not provide a calibrated uncertainty bound. The original field names
`*_noise_mj_per_nominal_task` hold **gross idle energy divided by nominal
repeats**, not baseline-subtracted noise. The first GPU variant’s gross counter
shams are 56,263.571,53,775.000 and55,792.857 mJ, exceeding its 47,541.078 mJ net
active result; the original claim that shams were 4–7 times smaller was incorrect.
Corrected A100 sham residuals are separately labelled above and in the reviewer
summary; neither gross shams nor corrected residuals are confidence intervals.

**Precision caveat:** `run_protocol.py` ignores `--dtype` on its NumPy branch and
uses the float64 default. Calling `fit_predict_np(..., dtype=float32)` directly
still promotes its design/Gram through float64 projection/offset arrays. The
legacy NumPy-float32 table label must not be read as a validated float32
arithmetic path. These code defects are documented rather than changed here.

Historical author comparisons on this card (descriptive, not fresh qualification):

- **The mixed-precision Gram is the cheapest GPU configuration that passes the
  3 % band.** 47,541 mJ against 187,644 mJ for the float64 device path is 3.95×
  (−74.7 %), and 877 ms against 4,016 ms is 4.58×. Accuracy is not worse:
  106,747 against 106,735, +12 correct.
- **Cost follows D².** Energy ratio D=4000 / D=2448 = 2.827 against a
  theoretical `(4001/2449)²` = 2.670, i.e. 6 % above prediction.
- **A pure float32 Gram breaks the solve.** Cholesky fails with "leading minor
  of order 2090/2194/2250 is not positive definite" on all 11 draws.
- **Blockwise Gram by itself costs energy**, 196,251 mJ against 187,644 mJ,
  +4.6 %, at 409 MiB peak instead of 455 MiB. The memory goal is met and the
  energy goal is not.
- **Training on a train subset does not reach the 3 % band**, even at 6,000 of
  10,000 rows: 106,054, which is FAIL (−646).

These RTX comparisons do not establish the same ratios on A100. The completed
A100 diagnostic measures one configuration only, so it cannot validate a
cross-hardware ranking of all variants.

### Why the grid columns are em dashes

The affine IR of the canonical grid submission (`affine.py`) defines

```
ARARIES = {"set":0, "recv":0, "send":1, "copy":1,
           "add":2, "sub":2, "mul":2, "cmp":2, "select":3, "div":2}
```

There is no `cos`, no `sin`, no `exp` — and no `sqrt`, which the file's own
comment notes was replaced by Newton–Raphson. The defining feature of this method
is `cos(U W + b)`. Approximating it by a polynomial is possible, but it is a
different numerical model with a different cost, and the README requires costing
the model as declared. Nested loops are not the obstacle — `loop(...)` nests and
`build_mlp` already nests five deep, so `ZtZ = Ztrᵀ·Ztr` is expressible
structurally. The obstacle is the absent opcode. We therefore did not generate a
grid-IR program for this method rather than publish a polynomial surrogate's cost
as if it were ours.

### Why our MLP row could not be measured on this side

The upstream `medium96-grid-20260912` submission is written against **Triton**
(four `@triton.jit` kernels) with its own payload format and its own draw seeds.
Triton is not installable in our environment: no module named `triton`, no
official Windows wheels for torch-cu121, and no `nvcc`. Running their frozen
learner on our card would require rewriting it, which is outside a measurement
exercise. Separately, that row is a 96 % / 5 % band entry, so it would not be a
like-for-like comparison against a 3 % band method in any case.

---

## Findings that constrain the frontier

Recorded because they are negative results with numbers attached, not because they
support the submission.

**The band boundary is narrow and non-monotonic.** Near the 3 % threshold,
changing D by single digits moves the total by ±35 correct, which is comparable
to the margin itself. D=2448 PASSes at +35; D=2464 FAILs at −51; D=2466 FAILs at
−3; D=2468 PASSes at +20. Because the projection shape changes with basis size, these are
separate bases and not nested prefixes. Any claim of the form "D=2448 clears
the 3 % band" is true for these 11 draws and this grid, and does not mean the
margin is robust.

**No D ≤ 2048 reaches the 3 % band.** D=256 → 102,762; D=512 → 104,680;
D=1024 → 105,756; D=2048 → 106,558, which is FAIL (−142). Every one of the 11
archived draws is worse than at D=2448. This describes this seen-draw comparison,
not a generalization test.

**Accuracy saturates with D.** The gain per doubling falls: +1.74 pp (512→1024),
+0.98 pp (1024→2048), +0.31 pp (2048→4000), +0.24 pp (4000→8000), +0.06 pp
(8000→12000). The paired totals improve in all 11 archived draws at each step up to
D=12000, but this was not an independent test of the ordering. The increments
do not close the observed2% gap: D=16000 was screened on three draws and not run on all eleven
within our budget, so no claim is made about it.

**An ensemble of RFF-ridge members adds less than a larger basis.** Three members
at D=4000 with different `gamma` gave 107,032 (+132 over the single D=4000),
while a single D=8000 gave 107,169 (+269). It also won in only 8 of 11 draws.
That archived comparison does not identify an asymptotic ensemble-gain law or
establish independent member errors.

**Multi-scale RFF plus a raw-pixel block is worse than the control**, 106,829
versus 106,900, winning in 3 of 11 draws.

**A quadratic polynomial basis was rejected on screening.** Explicit degree-2
features (3,484 of them) scored 1935/2000 inner-validation on one screen draw
against 1939 for the control, and the best `lam` sat at the edge of the grid
(100), so it wanted more regularisation. It was not tuned further.

**A softmax head on the same basis was rejected on screening**, 1831/2000 on one
screen draw. That single run predates a bug fix described below, so its absolute
value is not strictly comparable, so it is retained only as a historical screen, not independent rejection
evidence after the fix.

**A train subset of 6,000 rows does not reach the 3 % band** (106,054, FAIL
−646), though it is the cheapest GPU variant measured at 39,822 mJ. It qualifies
for the 5 % band at +1,554 and is kept as a candidate for that band, not for this
one.

**A trainable CNN was not measured.** It was out of budget; this report makes no
claim about it.

### Author-reported historical screen-cache bug

The cache key for the design matrix `Z` was `(p,)`, missing `gamma` and `D`, so
every grid point after the first silently reused the first point's basis. It was
caught because `smD4000` produced identical inner-validation counts (1831, 1801,
1649 …) at three different `gamma` values, which is impossible with correct code.
The key now carries every parameter the basis depends on,
`(kind, p, D, D1, D2, gamma, gamma1, gamma2)`, and every candidate whose number
was affected was re-run. The author reports rerunning affected candidates. The separate `smD4000`
screen remains flagged. This historical screen-cache fix does not resolve the
precision and measurement-reporting defects documented in the reviewer note.

---

## Disclosures

**The eleven dataset seeds were already seen.** Seeds `20261001…20261011` were
generated and scored by earlier experiments of ours before the final procedure was
frozen. The instructions require predeclaring the seeds before inspecting any of
the eleven test results, and that predeclaration cannot be reconstructed for
seeds whose labels have already been read. This is a structural limitation of the
situation, not a defect of the procedure. What the procedure does satisfy, and
what is checked by assertion in the code, is the requirement that any validation
come only from the current draw's training subset: the overlap counts are zero in
44 of 44 runs.

**The hyperparameter grid space was informed by earlier results of ours.** The
grid `p ∈ {0.25, 0.5}` was designed after seeing an earlier experiment's numbers.
No grid point was ever selected using test labels — selection is inner-validation
accuracy only, and the complete 18-point table for every draw is published in
[evidence/selection_protocol_D12000.json](evidence/selection_protocol_D12000.json).
The screen that promoted candidates between basis sizes used mean inner-validation
accuracy, not test accuracy; the screen's spread of 1–3 correct out of 6,000 is
within its own standard error of about 7.6, so the screen served only to discard
clearly weaker branches and its ordering should not be relied on.

**The author’s original energy sweep is RTX 2060 evidence.** The added reviewer
summary is one A100 40GB diagnostic. These different-hardware results are clearly
separated and none is promoted to the record table.

**No claim of optimality or leadership is made.** We report the frontier we
measured on the disclosed, previously seen seeds, with the negative results attached. Where the
margin is inside the noise (D=2448 at +35), we say so.

**The D=12000 run is expensive.** About 210 s per draw on 16 CPU threads, roughly
39 minutes for the eleven-draw protocol, and the Gram matrix is 12,001 × 12,001.
D=2448 takes about 12 s per draw and is the point on the frontier that trades the
least accuracy for the least cost.

## Reproduction

```bash
# from the repository root
python -m mnist.code.data --profile medium-error-targets-v1 \
    --output <draws-dir>/data-20261001 --seed 20261001     # repeat for each seed

cd mnist/submissions/mnist-medium-rff-ridge-20261006

# accuracy (CPU, NumPy only)
OMP_NUM_THREADS=16 python run_protocol.py --D 2448 \
    --data-root <draws-dir> --outdir out --records out/records.json \
    --bands 2,3,5,8,12 --repo-root <repo-root>

# accuracy (GPU, float32 products with float64 accumulation)
python run_protocol.py --D 2448 --data-root <draws-dir> --outdir out \
    --records out/records.json --backend torch --device cuda \
    --dtype float32 --chunk 2048 --gram-dtype float64 --f32-products \
    --bands 2,3,5,8,12 --repo-root <repo-root>

# energy, on whichever card you want to report
python energy/nvml_energy.py --data-root <draws-dir> --idle-baseline \
    --out energy/results/idle_baseline.json
python energy/nvml_energy.py --data-root <draws-dir> --D 2448 \
    --dtype float32 --chunk 2048 --gram-dtype float64 --f32-products \
    --rounds 3 --shams 1 --idle-seconds 10 \
    --out energy/results/energy-rff-cuda-f32p-chunk 2048-D2448.json
```

`run_protocol.py` writes the per-draw prediction file, its SHA-256, the selected
hyperparameters with their inner-validation counts, the full grid, the overlap
counts, and the verbatim evaluator output for each error band. It performs no
accuracy arithmetic beyond summing the `correct` fields copied from the official
evaluator, and it compares band totals against the required total as exact
integers.

## Model revision

The canonical grid scorer used to reproduce the MLP row during this work was
pinned at SPEC_COMMIT `01a0bd5e0d2564825b0f53dd766f763c82dbc7c0`, with the
spatial-computer clone at `5803d89`. It was used for reading the affine IR only.
No generator was added to it and no file in it was modified.