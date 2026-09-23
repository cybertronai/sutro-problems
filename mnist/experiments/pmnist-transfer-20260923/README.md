# Fixed-permutation MNIST and transfer of the Ladder training procedure

Research and experiment date: 2026-09-23. This directory is a self-contained report and agent handoff. Start a new agent with [HANDOFF.md](HANDOFF.md), then use the evidence and commands below. No previous conversation or neighboring checkout is required.

## Main findings

The strongest eligible published result verified in this search is **99.431% mean accuracy** for **Ladder AMLP [2,2]** (Pezeshki et al., ICML 2016), corresponding to 0.569% error with standard error 0.010 percentage points over ten runs. This is effectively tied with the original Ladder result of 99.43%. This is the best source-supported result found, not a claim that an exhaustive, continuously updated leaderboard exists. [Final paper, Table 2](https://proceedings.mlr.press/v48/pezeshki16.pdf), [original Ladder paper](https://arxiv.org/pdf/1507.02672).

Our fresh PyTorch implementation achieved **99.47%** on the canonical **60,000-train / 10,000-test** split: **53 errors**, versus the paper's mean 56.9 errors. Its accuracy differs from the published mean by **+0.039 percentage points**. This is one prespecified seed, with no test-guided tuning; it is not the paper's ten-run result or its uncertainty estimate.

The same frozen recipe was trained from scratch on **five of the fifteen** Aminist 21 transfer tasks, using all **eleven fixed draws** per selected task. Every draw contains 10,000 training and 10,000 query examples at 9×9 resolution. These are procedure-transfer results, not evaluation of a pretrained MNIST digit classifier on unrelated classes.

| Transfer task | Ladder AMLP, mean ± SD | Existing MLP256 | Existing CNN ensemble | Ladder − MLP256, pp |
|---|---:|---:|---:|---:|
| KMNIST | 94.44 ± 0.17 | 92.85 | 96.28 | +1.59 |
| EMNIST letters A–J | 93.34 ± 0.17 | 92.04 | 95.40 | +1.30 |
| QMNIST recovered | 95.68 ± 0.13 | 96.15 | 97.73 | -0.47 |
| Fashion-MNIST | 82.70 ± 0.36 | 85.04 | 87.21 | -2.34 |
| CIFAR-10 | 38.65 ± 0.28 | 34.85 | 44.57 | +3.81 |

The strong original pMNIST score did **not** make this frozen recipe the strongest transfer procedure. It improved on MLP256 for KMNIST, EMNIST A–J and CIFAR-10, but was worse for recovered QMNIST and Fashion-MNIST. The CNN ensemble remained ahead on all five. This is evidence that success on original MNIST alone does not predict the ordering of these procedures across the selected transfer tasks; it does not establish that the Ladder architecture could never do better after separately authorized tuning.

All values are accuracy percentages; SD is the sample SD across the eleven draws. References use the exact same draws and query labels. The CNN ensemble sees the original ordered 9×9 grid; our Ladder receives a fixed permutation and uses no spatial prior. These comparisons are between complete procedures, with different parameter counts and compute, not controlled architecture-only or equal-budget comparisons. All six reference recipes, paired differences, and per-draw values are retained in the comparison files.

## Benchmark interpretation

One PCG64 permutation with seed `20260923` is applied to all examples, identically at train and test: 784 features for canonical MNIST and 81 for the transfer suite. The learner gets the full feature vector and no inverse permutation or original coordinate map. A dense model can absorb a fixed feature permutation into its input weights, so the relevant literature calls this **permutation-invariant MNIST**. This is distinct from causal/sequential pMNIST and continual-learning suites of multiple permutations. Gaussian noise is coordinatewise; no rotations, translations, elastic distortions, spatial convolution, or external examples are used.

The five transfer datasets were selected for breadth before training or inspecting query results: Japanese characters, Latin letters, recovered digits, clothing, and objects. They retain the suite's curated pools, classes, preprocessing, draw seeds `20261101`–`20261111`, and 10k/10k disjoint-within-draw sampling. The suite's pools can repartition original releases, so these are not official-test results for KMNIST/Fashion-MNIST/CIFAR-10. Draws may overlap; some task families share ancestry. Reported SDs are descriptive, not confidence intervals. **This is an incomplete fifteen-task benchmark, with complete eleven-draw coverage of each selected task.**

## Method and reproducibility

The encoder is `input → 1000 → 500 → 250 → 250 → 250 → 10`, with batch normalization and ReLU hidden units. Each decoder coordinate has its own `3 → 2 → 2 → 1` leaky-ReLU combinator. The published full-label settings are noise SD 0.3 at every level, input reconstruction coefficient 2000 and zero at other levels, and combinator weight initialization SD 0.025. Adam uses learning rate 0.002, batch size 100, 100 constant-rate epochs plus 50 decay epochs. Final checkpoints are the last fixed epoch. [Supplementary Tables 4–5](https://proceedings.mlr.press/v48/pezeshki16-supp.pdf).

No authors' AMLP implementation was found. This is a paper-based PyTorch reimplementation informed by the original public Ladder code. Numerical/random-number implementation, Adam's epsilon placement, zero combinator biases, and independent reconstruction-stream shuffling are documented adaptation choices. The schedule follows the paper. Train-only clean batch-statistic calibration precedes evaluation. CUDA graph warmup is erased from model, optimizer, and random state before recorded training; arithmetic is FP32 with TF32 disabled.

Each of the 56 fits starts with fresh weights and optimizer state. Query labels are absent from GPU payloads. All 55 transfer prediction archives and the canonical MNIST predictions were saved before scoring. The evaluator checked source snapshots, all completed epoch counts, checkpoint/logit/prediction hashes, and agreement between saved logits and classes before reading official MNIST test labels. The suite scorer additionally rechecked dataset hashes, draw indices, disjoint native-image identities and matching input hashes. Correctness tests cover coordinate-permutation equivalence of the objective/gradients, clean inference, nodewise combinators and normalization calibration.

The fit workers reported **57.0 GPU-worker minutes** including adapter setup/calibration/inference, of which **56.2 minutes** were training. The entire app lasted **15.5 minutes**, with at most four workers requesting A100-40GB. Actual reported hardware: **NVIDIA A100 80GB PCIe, NVIDIA A100-SXM4-40GB**. Using the higher published A100-80GB rate conservatively, adapter time corresponds to approximately **$2.61**, and charging four complete workers for the app's full elapsed time gives **$2.85**; neither is a provider invoice. This corrects the launch record's estimate based on the requested 40GB rate. A separate synthetic GPU smoke test is recorded. The app had a 3,000-second absolute deadline and no user-code retries. Runtime is not energy consumption.

### Running and rescoring

All report tables, per-draw scores, predictions, logits, configurations, source snapshots and reference score tables are committed here. The exact 57 model checkpoints (56 final fits plus one synthetic smoke fit) are a **370 MB GitHub release download**, verified against [assets-manifest.json](assets-manifest.json). Dataset files remain external downloads with pinned hashes and retained rights. The report itself can be read completely without downloading either.

The bundled [Aminist package and manifests](vendor/aminist21/) replace the original adjacent checkout; [vendor/provenance.json](vendor/provenance.json) pins its upstream revision and every copied file. Six complete baseline score tables are included. The model and training mathematics are unchanged. The current driver only adds portable paths and a separate output directory; [executed source](results/all/source/) and [original review scripts](results/all/review-source/) retain the exact historical bytes and hashes.

**Inspect and verify from a fresh checkout, without launching GPUs:**

```bash
git clone https://github.com/cybertronai/sutro-problems.git
cd sutro-problems/mnist/experiments/pmnist-transfer-20260923
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-review.txt
python -m unittest -v test_model.py test_fetch_assets.py
python fetch_assets.py --data --checkpoints
python audit_results.py --output runs/audit.json
python compare_results.py --scores results/all/transfer/scores.json --output runs/comparison --label 'Ladder AMLP'
```

The review requirements target Linux/Python 3.11; older Intel macOS installations can use PyTorch 2.2.2 with NumPy 1.26.4 instead. The recorded CPU checkpoint replay used that combination. An optional `--cpu-python /path/to/python` selects another replay interpreter. The audit never scores query labels. To independently rescore without rewriting archived results, copy `results/all` to `runs/rescore` and run `python score_study.py --study runs/rescore` after fetching data and checkpoints. The new scores may have different timestamps; counts and accuracies should match exactly.

**Intentionally retrain the same 56-fit experiment:**

The commands below assume the review environment and dataset downloads above are already complete.

```bash
python -m pip install -r requirements-controller.txt
# Authenticate the Modal CLI for your account, then:
modal run run_study.py --mode smoke --output-root runs/reproduction
modal run run_study.py --mode all --output-root runs/reproduction
python audit_results.py --study runs/reproduction/all --output runs/reproduction/audit.json
python score_study.py --study runs/reproduction/all
```

Retraining requests up to four cloud GPUs and incurs charges; it is optional and was not rerun when publishing this report. The container image is pinned by digest with PyTorch 2.5.1/CUDA 12.4 and NumPy 2.2.6. All training choices remain fixed. Reusing an existing launch directory is rejected. Keep published `results/` immutable; new work belongs under ignored `runs/`. Defaults are bundled `vendor/aminist21` and `data/mnist`; `AMINIST21_ROOT` and `PMNIST_MNIST_DIR` can override the data locations. No local absolute path in historical provenance is needed for execution. `build_report.py` rebuilds this report from the published results.

The complete checkpoint archive is attached to [the versioned release](https://github.com/cybertronai/sutro-problems/releases/tag/pmnist-transfer-20260923). Its SHA-256 and all member hashes are recorded in the asset manifest; the downloader rejects unexpected entries, unsafe paths and conflicting existing files. Code and data terms are described in the bundled license documents. The vendored package is an exact subset; its upstream README also describes features and files outside that subset, so use this report's commands for this study.

## Files

- [Agent handoff and remaining work](HANDOFF.md)
- [Literature audit and excluded claims](research/literature.md)
- [Detailed model protocol and porting choices](research/model-protocol.md)
- [Transfer dataset inventory](research/transfer-inventory.md)
- [Canonical pMNIST score](results/all/official/scores.json)
- [Transfer scores and per-draw confusion matrices](results/all/transfer/scores.json)
- [Transfer score CSV](results/all/transfer/scores.csv)
- [All-six-reference comparison](results/all/comparison/comparison.md)
- [Comparison data and provenance](results/all/comparison/comparison.json)
- [Independent audit](results/all/audit.json)
- [Standalone publication validation](results/publication-validation.json)
- [Frozen source snapshot](results/all/source/)
- [All-predictions verification record](results/all/all-predictions-verified.json)
- [Execution record](results/all/execution.json)
- [Original review scripts](results/all/review-source/)
- [Checkpoint download manifest](assets-manifest.json)
- [Vendored dependency provenance](vendor/provenance.json)
- [Conservative cost estimate](results/all/cost-estimate.json)
- [Stopped-app verification](results/lifecycle.json)
