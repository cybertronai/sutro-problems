# D5: tangent-jitter ensemble with less frequent cross-teaching

This submission is a credited derivative of [@likelytobelaura's PR #118](https://github.com/cybertronai/sutro-problems/pull/118), pinned at `436b80bb40e4c346792ddad71670c7a5dbc8c5e9`. It reduces training to 2,000 updates, refreshes cross-teaching pseudo-labels every 400 updates, and lets both groups sample all query rows rather than disjoint query halves. The original joint train/query graph-propagation readout is @sjbaebae's. Muon and its Newton–Schulz coefficients are credited to Keller Jordan et al.

All three original official Modal A100 sandbox evaluations passed D5. Median ranked runtime is **2,580.846 ms**, median energy above idle **536,878.610 mJ**, and pooled MNIST error **1.848182%** (323,901 / 330,000 correct). These are measured submission results, not a leaderboard update. No qualification reruns replaced failures.

| Run | MNIST error | Margin in labels | Ranked ms | Energy mJ |
|---|---:|---:|---:|---:|
| [1](official/run-1.json) | 1.827273% | +80 | 2580.846 | 536878.610 |
| [2](official/run-2.json) | 1.824545% | +83 | 2577.045 | 514395.945 |
| [3](official/run-3.json) | 1.892727% | +8 | 2724.036 | 562568.582 |

The tightest run has only eight labels of margin. Passing this fixed three-run evaluation does not establish that every future batch will pass. Raw records preserve scorer stdout, stderr, all 15 timed calls, energy windows, device UUIDs, sandbox status and source hash. `verify.py` checks their consistency against the unchanged source.

## Model and learning

Eight batched MLPs, 60–256–256–256–10, use RMSNorm/GELU and dropout 0.1. Hidden matrices use Muon (three Newton–Schulz steps); gains, biases and output heads use AdamW. Training uses a cosine schedule, EMA, and mixup Beta(0.5,0.5). Each labelled row has four local tangent directions computed from its ten nearest same-class training rows; fresh Gaussian tangent jitter is applied online. True-label batches contain 2,048 rows. From update 400, batches also contain 1,024 pseudo-labelled query rows. The two halves of four models cross-teach at updates 400, 800, 1200 and 1600, selecting their top 90% confident query predictions. Both halves sample all queries. There is no separately trained teacher/student stage.

The final readout centers last-hidden EMA embeddings, applies signed power 0.25, normalizes and concatenates members, then uses a K5 labelled-neighbor prior and three joint train/query graph-propagation steps. Final similarities use hub correction 0.3. Training is BF16-oriented compiled PyTorch with CUDA graphs, not the earlier native BN teacher/student kernel implementation.

Each call recomputes tangents and initializes weights, optimizer state, EMA, query pools and pseudo-labels from scratch. Only the supplied `train_x`, `train_y` and unlabelled `test_x` enter the learner; there is no offline fitted data or query-label access.

## Accuracy comparison and earlier failures

`comparison-2500/` contains the separately frozen 2,500-update, every-400-refresh variant and its three original official evaluations. All passed: errors 1.802727%, 1.800909%, 1.767273%; pooled error **1.790303%**; median **3,385.827 ms** and **674,307.053 mJ**. This is a slower accuracy-margin comparison, not the submitted timing.

`research/paired-development.json` contains eleven paired fresh development draws. The 2,000-step one-refresh control scored 1.747273% at 2484.646 ms; periodic refresh scored 1.723636% at 2579.867 ms. At 2,500 steps, one refresh scored 1.699091% at 3249.113 ms and periodic refresh 1.668182% at 3368.803 ms. These development draws were easier than the official set; their absolute accuracy is not a qualification claim.

The earlier 2,000-step **one-refresh** candidate passed only 2/3 official evaluations: 1.880000%, 1.954545%, 1.799091%. All initial records and its source are retained in `research/once2000/`. It is not qualified and is not substituted for this submission.

## Reproduce

From `mnist-a100`:

```bash
python run_modal.py submissions/tangent-refresh400-d5-20261012/submission.py:classify --difficulty 5 --runs 3 --json out/d5-refresh400
python submissions/tangent-refresh400-d5-20261012/verify.py
```

`submission.py` is 10,883 bytes, SHA-256 `6f0e1a109ece2679f8ee1e0527cd98abc6a58e664d11fc6033740765ba4a990f`, matching every primary official record. The 2,500-step comparison hash is `86d7caab67e4c9de30a698652fab6195599e5184113c359a83872f5135851018`. Both executable ASTs were verified against the development variants before launching official evaluation.
