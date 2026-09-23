# Handoff: fixed-permutation MNIST and transfer

Study date: 2026-09-23. The requested research and initial transfer experiment
are complete. This document is enough to understand the state without the
original conversation; use the neighboring README for installation, downloads,
and the published package's reproduction commands.

## What the user asked

Find the strongest published result for ten-class MNIST with one deterministic
pixel permutation shared by training and test, retaining the original 60,000
training and 10,000 test examples. Try that learning method on some of the
fifteen transfer datasets from the earlier **Mnist transfer** work. The user
subsequently requested a GitHub publication that another agent could inspect
and continue without access to either conversation or the author's machine.

The implemented interpretation is full-vector, unknown-layout
**permutation-invariant MNIST**. The learner receives all 784 pixels at once
in a fixed order. This is separate from causal/sequential pMNIST and
continual-learning tasks that switch between multiple permutations. A dense
first layer can absorb a fixed feature permutation into its weights. The
experiment does not supply an inverse permutation, geometric augmentation,
convolutions, original coordinates, pretrained weights, or outside examples.

## Completed result

The selected family is **Ladder AMLP [2,2]**. The strongest eligible result
verified in the bounded literature search was **99.431% mean accuracy**
(0.569% error, standard error 0.010 percentage points, ten runs), from
Pezeshki et al., ICML 2016, Table 2. It is effectively tied with the earlier
Ladder result, 99.43%; the numerical gap between the papers is not evidence
of a meaningful improvement. This is the best source-supported eligible
result found, not a certified exhaustive 2026 world record. Higher numbers
from spatially augmented dense models, convolutional methods, binary tasks,
or continual-learning aggregates were not mixed into this ranking.

A fresh PyTorch implementation obtained **99.47% accuracy: 9,947 correct and
53 errors** on the official 10,000 test examples after using all 60,000
training examples. The single prespecified training seed was 11. This is
one run, not a reproduction of a ten-run mean or a new record claim. Its
+0.039 percentage-point difference from the paper's mean is only 3.9 test
examples in an average. No test-guided hyperparameter or checkpoint choice
was made for this run.

The same fixed training recipe was evaluated from fresh initialization on
five breadth-selected transfer tasks, eleven draws apiece. Original MNIST
is an additional control, not one of those fifteen transfer tasks.

| Benchmark | Resolution and input length | Training / query per fit | Fits completed |
|---|---|---|---:|
| Official MNIST | 28×28; 784 permuted features | Official 60,000 / 10,000 | 1 |
| Selected Aminist 21 tasks | 9×9 grayscale; 81 permuted features | 10,000 / 10,000 from each curated pool | 5 × 11 = 55 |

Transfer here means **transfer of a training procedure**. Each fit constructs
new model and optimizer state and learns the task's own ten classes. No digit
classifier checkpoint was applied directly to letters, clothing, or objects.
The transfer pools can repartition source releases, so their scores are not
those datasets' official-test accuracies. Draws may overlap and related
datasets share ancestry; sample SDs are descriptive, not confidence intervals.

| Completed transfer task | Ladder mean ± sample SD | Historical MLP256 mean | Historical CNN ensemble mean |
|---|---:|---:|---:|
| KMNIST | 94.4364 ± 0.1677 | 92.85 | 96.28 |
| EMNIST Letters A–J | 93.3373 ± 0.1743 | 92.04 | 95.40 |
| QMNIST recovered | 95.6845 ± 0.1275 | 96.15 | 97.73 |
| Fashion-MNIST | 82.6955 ± 0.3586 | 85.04 | 87.21 |
| CIFAR-10 grayscale | 38.6536 ± 0.2795 | 34.85 | 44.57 |

All numbers in the table are percentages. References use the same task IDs
and eleven draw indices. The CNN sees the ordered 9×9 spatial grid; Ladder
sees a fixed permutation. Architectures, parameter counts, epoch budgets,
ensembling and compute differ. Thus this compares complete procedures,
not isolated architecture changes or equal compute budgets. The published
MNIST leader did not become the strongest procedure on these transfer tasks:
it beat the MLP256 reference on three of five tasks but trailed the CNN
ensemble on all five. Scores for all six historical references are retained.

## Frozen protocol and evidence

- Feature permutation: `Generator(PCG64(20260923)).permutation(D)`, with
  `D=784` or `81`; the same saved permutation is used for every train/query
  example of a given input dimension.
- Transfer draws: dataset seeds `20261101` through `20261111`;
  `PCG64(SeedSequence(seed).spawn(2)[0])` permutes the whole curated pool;
  rows 0–9,999 train and rows 10,000–19,999 query. No class rebalancing.
- Encoder: `D → 1000 → 500 → 250 → 250 → 250 → 10`, batch normalization,
  ReLU hidden units; nodewise decoder combinators `3 → 2 → 2 → 1` with
  leaky-ReLU slope 0.1. Parameters: 3,127,018 for MNIST and 1,709,067 for
  the transfer model.
- Gaussian noise SD 0.3; input reconstruction coefficient 2000 and zero
  hidden reconstruction coefficients; combinator weight initialization SD
  0.025. Paired labeled/reconstruction batches come from independent
  shuffles of the same training pool.
- Adam, learning rate 0.002, batch size 100, 150 epochs: 100 constant-rate
  epochs followed by 50 decay epochs. FP32, TF32 disabled. All fits use seed
  11 and the final fixed epoch, followed by training-only clean BN calibration.
- All 56 prediction sets were saved before scoring. Query labels were absent
  from remote learner payloads. The separate audit verified 560,000 saved
  predictions against logits and regenerated input/split/permutation hashes.
- Two checkpoint replays used CPU PyTorch 2.2.2 against GPU PyTorch 2.5.1
  output. On the first 100 official-MNIST and KMNIST-draw-0 queries, maximum
  logit differences were respectively `7.6294e-6` and `3.8147e-6`, below the
  declared absolute tolerance `1e-4`; no predicted classes differed.
- Training image: digest pinned in the runner, PyTorch 2.5.1+cu124 and
  NumPy 2.2.6. Actual workers reported A100 80GB PCIe and A100-SXM4-40GB,
  despite the A100-40GB request. Per-record metadata has exact software.

No authors' AMLP-specific implementation or checkpoint was located. This is
a paper-based port informed by `CuriousAI/ladder` revision
`5a8daa1760535ec4aa25c20c531e1cc31c76d911`, not a conversion of authors'
weights. Porting choices include PyTorch RNG/arithmetic, Adam epsilon
placement, zero combinator biases, independent reconstruction shuffles,
and the paper's intended learning-rate schedule. Preserve those stated
limitations when reporting the reproduction.

The complete app ended at `2026-09-23T13:08:07Z`, after 931.4 seconds wall
time with at most four workers. Both this study's full run and synthetic
smoke app were verified **stopped with zero tasks** at
`2026-09-23T13:08:53Z`. There is no background training, queued extension,
or GPU job to resume for this study. Other users' or sessions' applications
are outside this handoff; do not stop unrelated apps. The approximately
$2.61 adapter-time / $2.85 full-app conservative resource estimates are
not provider invoices and are not energy measurements.

## All fifteen transfer tasks

These are the canonical version-1 order and clean pool counts. **Completed**
means this Ladder study has all eleven draws. **Unevaluated** means this
Ladder study has no result; the existing six-recipe reference panel does
cover the task. The ten unevaluated tasks are optional future scope, not
unfinished jobs or promised work from the original request to test “some.”

| # | Dataset ID | Clean pool count | Ladder study status |
|---:|---|---:|---|
| 1 | `kmnist` | 60,000 | Completed: 11 draws |
| 2 | `emnist_letters_aj` | 47,998 | Completed: 11 draws |
| 3 | `emnist_letters_kt` | 47,999 | Unevaluated |
| 4 | `emnist_balanced_aj` | 23,998 | Unevaluated |
| 5 | `emnist_digits` | 240,000 | Unevaluated |
| 6 | `emnist_mnist` | 60,000 | Unevaluated |
| 7 | `qmnist_recovered` | 50,000 | Completed: 11 draws |
| 8 | `k49_10` | 60,000 | Unevaluated |
| 9 | `kannada_digits` | 60,000 | Unevaluated |
| 10 | `devanagari_digits` | 20,000 | Unevaluated |
| 11 | `madbase` | 59,944 | Unevaluated |
| 12 | `notmnist_large` | 461,751 | Unevaluated |
| 13 | `fashion_mnist` | 60,000 | Completed: 11 draws |
| 14 | `svhn` | 73,255 | Unevaluated |
| 15 | `cifar10` | 50,000 | Completed: 11 draws |

Every task uses 10,000 training and 10,000 query examples per draw, even
when its pool is much larger. Input preprocessing, selected classes,
orientation/polarity corrections, native-image deduplication, and dataset
rights are defined by Aminist 21's versioned manifest and dataset notices.
USPS, DiG-MNIST and the earlier small Omniglot selection were excluded from
the common 20,000-example protocol; do not substitute them for these IDs.

## What another agent should do first

1. Read this handoff and the README. Use the bundled score records to inspect
   results before downloading datasets or allocating GPUs. Rebuilding a
   table is not a reason to retrain completed models.
2. Follow the README's portable environment and data/asset-fetch commands.
   Native datasets and checkpoint binaries are separate from small source
   and score records where the package manifest says so. Check their pinned
   hashes before auditing or replaying. A clone alone must not be mistaken
   for an already populated data cache.
3. Keep `results/all/source/` and recorded plans, metadata, manifests and
   predictions immutable. They define the executed experiment. Portable
   wrappers may differ from the frozen source only in documented I/O/setup
   behavior; never silently rewrite source hashes to make old results pass.
4. Distinguish lightweight checks, dataset-backed score recounts, checkpoint
   replay and new GPU training. Use a new output directory for any new
   experiment: the portable runner defaults to `runs/all`, separate from
   published `results/all`. The original runner refuses an existing launch
   record. Do not delete historical results to bypass its guard.

The portable package uses these paths:

| Item | Location or retrieval |
|---|---|
| Pinned suite code and dataset manifests | `vendor/aminist21/aminist21/` and `vendor/aminist21/datasets/` |
| Six original reference score snapshots | `vendor/aminist21/reference-results/<recipe>/scores.json` |
| Suite license, dataset terms and recipe catalog | `vendor/aminist21/LICENSE`, `DATASETS.md`, `dataset-licenses.json`, `baseline-catalog.json` |
| Original MNIST gzip files | `data/mnist/`, populated by `python fetch_assets.py --data` |
| Prepared transfer pools | Suite data location populated by `python fetch_assets.py --data`; see README |
| Predictions, logits, metadata and score evidence | Tracked under `results/` in the Git checkout |
| All 57 checkpoints, including the smoke checkpoint | `python fetch_assets.py --checkpoints`; archive in GitHub release tag `pmnist-transfer-20260923`, asset `checkpoints.tar.gz` |
| Download identities and hashes | `assets-manifest.json` |
| New experiment output | `runs/`, preserving `results/` |

The vendored suite is pinned to upstream commit
`4bf0559754e7974fe1b17fe132d002726596f5ce`; the package records its file hashes.
Its canonical dataset manifest SHA-256 is
`08e6f58dad660f8bb931a73b0239030cd20c76cd3e8cde344cf3d4c535748167`.
Preserve the included dataset rights notices when copying the handoff.
The approximately 21 MB of prediction/logit NPZ evidence is separate from
the approximately 398 MB of checkpoint files before archive compression.
Reading scores and rebuilding comparison tables needs neither dataset
downloads nor checkpoint downloads.

The prepublication portability audit identified original machine-specific
defaults for the adjacent suite, official MNIST cache, reference-score path,
and CPU Python executable. The portable scripts use the bundled suite and
`data/mnist` defaults; `AMINIST21_ROOT` and `PMNIST_MNIST_DIR` support explicit
overrides. See the README for choosing the CPU verification environment.
Another machine does not need to recreate `/Users/yaroslavvb`. Historical
absolute paths remain in immutable provenance records as evidence of the
original run, and the exact executed source remains unchanged.

The source has these distinct dependency levels: report construction uses
the Python standard library; array audits/comparisons and pool fetching use
NumPy and the suite's download support; model checks/checkpoint replay need
PyTorch; remote training additionally needs Modal and authentication. The
training runtime is pinned independently from the local controller. Do not
claim bitwise equality across GPU types, PyTorch versions or CPU/GPU systems.

## Optional follow-up directions

No follow-up experiment is running or scheduled. Choose a direction only
when it serves a new user request, and freeze its scope before new scores.

1. **Complete the fifteen-task vector:** run the ten unevaluated task IDs with
   this exact configuration on all eleven draws, adding 110 fresh fits.
   Preserve the existing 55 results; retain canonical task order and clearly
   record the extended prediction freeze. This is the most direct extension.
   The current downloader fetches only the five completed tasks, and the
   runner, scorer and auditor assume five tasks / 55 transfer fits. Fetch the
   additional pools with the bundled suite and create an expanded plan and
   corresponding runner/scoring/audit support before launching an extension.
2. **Quantify original-MNIST training variation:** predeclare additional
   training seeds and retain the official 60k/10k split. Report all runs,
   their mean and variability instead of selecting the best seed. The
   existing single run cannot supply a ten-run uncertainty estimate.
3. **Investigate transfer failure modes:** separately predeclare budget-
   matched baselines or train-only-validation ablations of input scaling,
   reconstruction loss, epoch budget or network width. The five transfer
   query scores are already known; changing the recipe in response creates
   a new study, not continuation of a blind frozen evaluation.
4. **Extend the literature search:** accept a challenger only after its
   primary source establishes ten classes, original 60k training/10k test,
   full-vector non-spatial learning and no geometry-restoring augmentation.
   The absence of convolution by itself does not establish eligibility.
5. **Study sequential pMNIST separately if requested:** causal/sequence input
   constraints require a new question and comparison table. Do not rebrand
   the existing full-vector Ladder result as a sequence-model record.

## Artifact navigation

- [README: report, package setup and reproduction](README.md)
- [Pinned asset downloads](assets-manifest.json) and [fetch utility](fetch_assets.py)
- [Bundled validation suite](vendor/aminist21/README.md),
  [dataset definitions and terms](vendor/aminist21/DATASETS.md), and
  [reference score snapshots](vendor/aminist21/reference-results/)
- [Literature audit and excluded claims](research/literature.md)
- [Detailed model protocol and adaptations](research/model-protocol.md)
- [Transfer inventory and original local execution context](research/transfer-inventory.md)
- [Canonical MNIST score and confusion matrix](results/all/official/scores.json)
- [Transfer scores, per-draw counts and confusion matrices](results/all/transfer/scores.json)
- [Transfer CSV](results/all/transfer/scores.csv)
- [Six-reference comparison](results/all/comparison/comparison.md)
- [Comparison JSON and source-score hashes](results/all/comparison/comparison.json)
- [Independent 56-fit audit and checkpoint replay](results/all/audit.json)
- [Standalone publication validation](results/publication-validation.json)
- [All-predictions verification gate](results/all/all-predictions-verified.json)
- [Official frozen plan](results/all/official/plan.json)
- [Transfer frozen plan](results/all/transfer/plan.json)
- [Transfer prediction manifest](results/all/transfer/prediction-manifest.json)
- [Exact executed source snapshot](results/all/source/)
- [Run execution record](results/all/execution.json)
- [Conservative cost estimate](results/all/cost-estimate.json)
- [Verified stopped-app lifecycle](results/lifecycle.json)
- [Audit utility](audit_results.py), [comparison utility](compare_results.py),
  [training runner](run_study.py), [model checks](test_model.py)
- [Primary selected paper](https://proceedings.mlr.press/v48/pezeshki16.pdf)
  and [supplement](https://proceedings.mlr.press/v48/pezeshki16-supp.pdf)
- [Aminist 21 Validation repository](https://github.com/cybertronai/aminist-21-validation)
