# NR-K8 Adam MLP: MNIST-small, 67% target

Submission date: September 14, 2026 (UTC). Contributor: OpenCode.

This submission reports **7,409 / 11,000 = 67.35%** mean accuracy with sample
standard deviation **1.76 pp** over eleven independently sampled datasets,
meeting the 67% target (7,370 required). Each draw contains 1,000 training and
1,000 disjoint test images reduced to 3x3 from the official 60,000-image
training split. The learner is a 9-32-10 MLP trained with NR-K8 Adam.

**Review request:** this submission includes an Adam optimizer path in the
shared grid scorer (`../grid-mlp-scoring-20260912/`). The grid columns below
are provisional until that extension is reviewed; scorer approval is requested
as a merge condition. The SGD path is unchanged and reproduces the official
scores byte-for-byte.

## Accuracy evidence

Draws use `Generator(PCG64(seed)).permutation(60000)`; rows 0-999 are training
and rows 1000-1999 are test, disjoint within each draw. Predictions were
written and hashed before any evaluation-label slice was opened (two-phase
`freeze`/`score`); `draw_manifest.json` records seeds, indices and input
hashes, and `prediction_manifest.json` records per-draw prediction hashes.

| Draw | Dataset seed | Correct / total | Accuracy |
| ---: | ---: | ---: | ---: |
| 0 | 20261201 | 675 / 1,000 | 67.5% |
| 1 | 20261202 | 693 / 1,000 | 69.3% |
| 2 | 20261203 | 688 / 1,000 | 68.8% |
| 3 | 20261204 | 670 / 1,000 | 67.0% |
| 4 | 20261205 | 636 / 1,000 | 63.6% |
| 5 | 20261206 | 669 / 1,000 | 66.9% |
| 6 | 20261207 | 676 / 1,000 | 67.6% |
| 7 | 20261208 | 676 / 1,000 | 67.6% |
| 8 | 20261209 | 676 / 1,000 | 67.6% |
| 9 | 20261210 | 698 / 1,000 | 69.8% |
| 10 | 20261211 | 652 / 1,000 | 65.2% |

The exact mean is `7409 / 11000 = 0.6735454545454546`; sample standard
deviation is `1.7642922866484638` pp (`ddof=1`). Qualification uses the exact
count against the 7,370 required. The learning procedure (width 32, 100
epochs, Newton-Raphson count 8) was fixed on disjoint pilot seeds
20261006-20261015 (67.47%) before the official draws were evaluated; no
evaluation results influenced the procedure.

## Frozen learner: NR-K8 Adam

Architecture 9-32-10, batch 25, learning rate 0.2, 100 epochs, fixed supplied
sample order, fresh seed-101 initialization per draw, ordered FP32 arithmetic,
no state transfer between draws. Gradients are computed from pre-update
weights and summed in ascending order. With `step = lr / batch = 0.008` and
`t` the global minibatch index starting at 1:

```
m    = 0.9*m + 0.1*g
v    = 0.999*v + 0.001*g*g
mhat = m / (1 - 0.9**t)
vhat = v / (1 - 0.999**t)
y    = vhat + 1e-6 ; repeat 8x: y = 0.5*(y + vhat/y)     # Newton-Raphson sqrt
p    = p - step * mhat / (y + 1e-8)                       # IEEE division
```

The square root lowering uses only existing instruction-set operations
(`div`, `add`, `mul`). This is a defined optimizer variant, not exact-sqrt
Adam; both `reference.py` (CPU) and `gpu_benchmark.py` (GPU) implement exactly
this recurrence and are bitwise-equal on all evaluated cases.

## Measured A100 cost

| Accuracy | Energy on A100 | Time on A100 |
| --- | ---: | ---: |
| 67.35% +/- 1.76 pp | 2,560 mJ | 107 ms |

Medians of three cyclic rounds of CUDA-graph replays on an A100-SXM4-40GB:
session 2 is 2,560.3 mJ / 106.9 ms; session 1 is 2,483 mJ / 107.1 ms (both
retained). Each replay resets parameters, optimizer moments and step state,
normalizes inputs, runs all 4,000 minibatches and produces 1,000 predictions;
`fused` runs the whole task in one launch (7-node graph, second variant 4,006
nodes). GPU parameters, scores and predictions are bitwise equal to the
ordered CPU reference in eager and graph modes for canonical, changed-labels
and changed-queries inputs, and all 11 official draws were re-run on the GPU
matching the frozen predictions. Scope excludes transfers, allocation, JIT,
graph capture and cold start; energy is NVML board energy with paired idle
subtraction, matching the existing small entry's protocol.

**Disclosure:** a same-container interleave shows the advantage over the
published H32 row is largely the fused runtime. Under matched execution, the
H32 learner runs 44.5 ms / 1,062 mJ versus 106.9 ms / 2,424 mJ for this entry.
No accuracy-superiority claim is made (+0.27 pp; the quoted +/- 1.76 pp is a
sample SD).

## Grid-model cost (provisional pending review)

| Energy in grid model | Time in grid model | Peak scratch | Time to score |
| ---: | ---: | ---: | ---: |
| 0.185228620462 mJ | 1,798.760521 ms | 128,352 bytes | 0.123 s |

Exact totals: 185,228,620,462 fJ and 1,798,760,521 cycles for the serialized
schedule under spatial-computer revision
`01a0bd5e0d2564825b0f53dd766f763c82dbc7c0`. The program and score are in
`../grid-mlp-scoring-20260912/small-adam-nr-k8-20260912/`; the scorer extension
adds the Adam path while leaving the SGD path unchanged. The SGD regression
reproduces the official price exactly (0.242990182992 mJ / 3073.508456 ms,
byte-identical program). Review is requested for the Adam state
initialization, bias correction, Newton-Raphson arithmetic and all
memory/tape accounting.

## Verification and limitations

`verify.py` re-checks draw seeds, train/test indices, input hashes, prediction
hashes and the label-derived accuracy total; `run.py freeze`/`score` establish
the freeze-before-labels order. Limitations: one GPU class (A100-SXM4-40GB),
two sessions; grid columns provisional pending scorer review; the accuracy
margin over the existing entry is small (30 predictions) and is not claimed as
statistically established.

## Reproduce and audit

Run from this directory (set `SUTRO_REPO` and `SUTRO_RAW` if the checkout and
raw MNIST gz files are not found by walking up):

```sh
python run.py prepare        # official draws, indices and input hashes
python run.py freeze         # train + hash predictions (no evaluation labels)
python run.py score          # verify every prediction hash first, then count
python run.py gpu-payload    # regenerate generated/*.npz for the GPU runner
python verify.py             # re-check manifests, inputs and accuracy
modal run gpu_benchmark.py   # one fused A100 session
```

`generated/` is regenerable and gitignored. Evidence:
[accuracy](evidence/accuracy/accuracy.json),
[draw manifest](evidence/accuracy/draw_manifest.json),
[prediction manifest](evidence/accuracy/prediction_manifest.json),
[evaluation freeze](evidence/accuracy/evaluation_freeze.json),
[GPU results](results/gpu_results.json),
[grid score](../grid-mlp-scoring-20260912/small-adam-nr-k8-20260912/grid-score.json).
