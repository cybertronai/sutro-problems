# Research archive: D5 width-128 cross-taught Muon ensemble

**Research inclusion; D5 qualification remains unresolved.** Both existing author sets (2/3, then 3/3; later-set median **1,490.895 ms**) remain separately preserved. The initial independent reviewer group passed **1/3**: two complete runs missed D5 by **28 and 74 labels**, with all 45 timed calls retained. There is no qualifying reviewer-group median and no leaderboard promotion. This inclusion preserves the source, original author evidence and attribution. See [reviewer evidence](reviewer-evidence.json).

Entrant: [@likelytobelaura](https://github.com/likelytobelaura). The joint train+query graph-propagation readout is
[@sjbaebae](https://github.com/sjbaebae)'s, from the D3 record (see [Credits](#credits)).

**Two official sets were run for this exact source; the first failed by one label and is disclosed below.** The
second set of three canonical Modal A100-80GB sandbox evaluations **all passed difficulty 5** (mean MNIST error at most
1.90%): median ranked time **1,490.895 ms per call**, median energy above idle **190,385.8 mJ per call**, 323,965 /
330,000 MNIST labels correct (**1.829% aggregate error**).

| Set 2 run | MNIST error | Margin, labels | Hold-out | Ranked ms | mJ above idle | Record |
|---|---:|---:|---|---:|---:|---|
| 1 | 1.830% | +77 | Fashion 88.94% | **1,490.895** | 186,415 | [json](official/run-1.json) |
| 2 | 1.833% | +74 | Fashion 89.01% | 1,473.153 | 200,918 | [json](official/run-2.json) |
| 3 | 1.824% | +84 | KMNIST 96.97% | 1,496.069 | 190,386 | [json](official/run-3.json) |

Three distinct GPU UUIDs; every call within 2% of its dataset's median time. [verification.json](official/verification.json)
recomputes every verdict and the ranked times from the raw per-call data.

## Disclosure: the first official set failed

The first set of three runs of the same source (identical SHA-256) passed 2 of 3. Run 3 labelled 107,909 of 110,000
correctly, one fewer than the 107,910 the band requires (1.901% error), and was disqualified. No run of either set was
dropped or replaced; the second set was run afterwards as a separate, complete set of three.

| Set 1 run | MNIST error | Margin, labels | Ranked ms | Record |
|---|---:|---:|---:|---|
| 1 | 1.837% | +69 | 1,535.147 | [json](official-set1-failed/run-1.json) |
| 2 | 1.809% | +100 | 1,533.598 | [json](official-set1-failed/run-2.json) |
| 3 | 1.901% | **-1** | disqualified | [json](official-set1-failed/run-3.json) |

Across all six official runs the source labelled 647,863 / 660,000 correctly (1.839% aggregate error); five of six
runs passed. [Set 1 verification](official-set1-failed/verification.json). The margin is thinner than the width-256
entries', so an occasional failed run is to be expected.

Source: `submission.py`, 15,223 bytes, SHA-256 `d3bc8a2bf543c8db7f90e57d9b974532eb4e15d3f5415c1f79015ac34a2075a9`
(identical in all six records). `mnist.check_source` passes and `mnist.review_flags` reports nothing. Source only:
scalar hyperparameters, no arrays, no fitted data.

## Method

- **Members.** Eight MLPs 64-128-128-128-10: inputs zero-padded from 60 to 64 (the padding columns stay exactly zero
  through the data-sampled init, the tangents and every update) so the first-layer GEMM is 16-byte aligned; hidden
  layers `x W^T + b` -> RMSNorm without gain -> ReLU -> dropout 0.05. 2,200 updates of 2,048 labelled rows (+1,024
  pseudo-labelled rows once cross-teaching starts), cosine schedule, EMA of the weights.
- **Optimizer.** Muon (momentum 0.95, 3-step Newton-Schulz in BF16, all square matrices in one batched call) on every
  hidden matrix; AdamW (0.9/0.99, decoupled weight decay) on the hidden biases and the head, as one small compiled
  graph (numerically equal to `torch.optim.AdamW`).
- **Local tangent jitter.** Once per call, from the draw's own labelled rows: for each row, the offsets to its 10
  nearest same-class rows give a 10x10 Gram matrix whose top 4 eigenvectors yield the row's local tangent directions,
  scaled to the rms offset. Every update moves each sampled labelled row by an N(0, 1) step along each tangent, then
  applies mixup (Beta(0.5, 0.5), partner = next row of the batch).
- **In-graph dropout.** Masks from a 32-bit counter hash of (element, update, layer) inside the compiled step.
- **Cross-teaching on disjoint query pools.** Two halves of four members; the query rows split once into two pools.
  From 20% of training, every 200 updates, each half reads out all query rows, and the 90% it is surest of become
  hard pseudo-labels for the other half, which trains on them only within its own pool.
- **Readout.** Centred EMA embeddings with a signed power 0.25 per coordinate, L2-normalised, concatenated over
  members; a nearest-labelled-neighbour prior (temperature 20) feeds three propagation steps over the kNN graph of
  labelled rows (clamped) plus query rows (alpha 0.9). This propagation readout is @sjbaebae's from the D3 record. In
  the final readout each member's embedding is widened with its middle hidden layer (both normalised, concatenated),
  10 neighbours are used, and similarities are hub-corrected (each row's mean similarity to its 10 nearest query
  rows, x0.3, subtracted). Neighbours are computed as two 10,000 x 10,000 blocks with merged top-k lists.
- **Engineering.** Training steps are CUDA graphs (loss/backward, Muon+EMA and AdamW compiled by Inductor), captured
  during the warm-up call. Every call recomputes the tangents and re-initialises all weights, optimizer state, EMA,
  pseudo-labels and pools.

## Rules review

- **Trains from scratch on every call.** `classify` runs `_load` (new draw, its tangents) and `_reset` (weights, EMA,
  Muon momentum, AdamW moments, update counter, pseudo-labels, pools, init seed) before any update; the warm-up call
  only compiles and captures graphs, and every call runs in a fresh process.
- **Uses only `train_x`, `train_y`, `test_x`.** Query rows are used without labels, as in the D3 record.
- **Invariant to the release.** Classes are handled symmetrically; tangents, neighbours and the data-sampled init are
  rotation-invariant functions of the draw.
- **No fitted data or offline prior.** Only scalar hyperparameters, chosen on MNIST development draws (seeds 401-416
  and 501-516 of `mnist.draw`) scored outside the method. The unused RMSNorm gain tensors stay at their reset value 1.
- **Sandbox and files.** All six runs were sandboxed; the only side effect is pointing the Triton and Inductor caches
  at `/tmp` at call time.
- **Same work per call.** Fixed update count and cross-teaching schedule.

## Development (A100, `dev/quick.py`, not official scores)

| Step (seeds 401-416 unless noted) | Mean error | ms per call |
|---|---:|---:|
| Width 256, 2,200 updates ([#120](https://github.com/cybertronai/sutro-problems/pull/120)) | 1.838-1.849% | ~1,900 |
| Width 128, dropout 0.1 | 2.024% | 1,327 |
| + dropout 0.05 + widened final readout | 1.872% (two runs); 1.881% on fresh seeds 501-516 | ~1,290 |
| + hidden bias (this file) | 1.841% / 1.842% (two runs) | ~1,440 |

With width 128, more updates (to 5,000), stronger tangent jitter, 10-16 members, and re-tuning Muon learning rate,
weight decay, EMA decay, cross-teaching coverage and start, mixup strength and the tangent neighbourhood gave nothing
beyond noise.

## Credits

- Joint train+query kNN graph propagation readout (k 5, alpha 0.9, three steps): by
  [@sjbaebae](https://github.com/sjbaebae), from the D3 record
  ([graph1250-repair](../native-mlp-d3-20261007/graph1250-repair/README.md)).
- Everything else here (tangent jitter, cross-teaching on disjoint pools, Muon training, the width-128 ReLU members
  with hidden bias, in-graph hash dropout, padded inputs, signed-power and widened readout, hub correction):
  [@likelytobelaura](https://github.com/likelytobelaura). Tangents follow the tangent-distance / tangent-prop line of
  work by Simard et al.; Muon and its Newton-Schulz coefficients are by Keller Jordan et al.

## Reproduce

From `mnist-a100`:

```bash
python run_modal.py submissions/tangent-relu-w128-bias-d5-20261013/submission.py:classify --difficulty 5 --runs 3 --json out/d5-w128-bias
```
