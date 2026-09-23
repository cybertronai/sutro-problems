# Transfer dataset and execution inventory

Inspected 2026-09-23. The other Codex thread is **Mnist transfer**, ID
`01a09635-9f04-7280-893b-db6682e2abd6`. Its requested fifteen-dataset package is
the adjacent checkout `/Users/yaroslavvb/git/aminist-21-validation`, public
repository `https://github.com/cybertronai/aminist-21-validation`, inspected at
commit `4bf0559754e7974fe1b17fe132d002726596f5ce`. The checkout was clean.

This is the historical inventory from the initial experiment. The published package bundles the required code and reference tables; use [the report](../README.md) for current portable paths and commands. Local paths below record the original execution context and are not dependencies.

## Protocol boundary

The package measures transfer of a **training procedure**, initializing a
fresh model and optimizer for each task/draw. It does not transfer an MNIST
checkpoint. All inputs are float32, grayscale, shape `(N,1,9,9)` in `[0,1]`;
all tasks have ten classes. Each of eleven fixed draws contains 10,000 train
and 10,000 disjoint query examples. The official MNIST 60,000/10,000,
28×28 pMNIST test is a separate experiment and must be reported separately.

Dataset seeds are `20261101` through `20261111`. The exact splitter is
`PCG64(SeedSequence(seed).spawn(2)[0])`; the first 10,000 permuted pool rows
train and the next 10,000 query. Draws are unstratified and may overlap.
The benchmark's baseline learner seed is 11. Models must receive only
training images/labels and query images. Freeze one numerical configuration
before evaluating transfer tasks and freeze predictions before scoring.

For fixed-permutation full-vector transfer, flatten the 9×9 images and apply one
saved deterministic permutation to all examples. An 81-pixel vector differs
from the original 784-pixel pMNIST vector; report that adaptation explicitly.
The existing reference CNNs see the ordered spatial grid, so comparisons with
a permuted candidate compare full procedures, not architecture alone.

## Exact inventory

Every listed pool uses 10,000 train / 10,000 query examples in every draw.
The final column gives the highest of the six historical reference means:
`cnn32-ensemble3` accuracy percent over all eleven draws. This is historical
context, not a criterion for selecting new models or datasets.

| # | Dataset ID | Clean pool size | Selected source pool | CNN ensemble % |
|---:|---|---:|---|---:|
| 1 | kmnist | 60,000 | KMNIST official train, ten classes | 96.2764 |
| 2 | emnist_letters_aj | 47,998 | EMNIST Letters official train, classes 1–10 | 95.4018 |
| 3 | emnist_letters_kt | 47,999 | EMNIST Letters official train, classes 11–20 | 97.4955 |
| 4 | emnist_balanced_aj | 23,998 | EMNIST Balanced official train, classes 10–19 | 97.1364 |
| 5 | emnist_digits | 240,000 | EMNIST Digits official train | 98.3318 |
| 6 | emnist_mnist | 60,000 | EMNIST MNIST official train | 98.3527 |
| 7 | qmnist_recovered | 50,000 | Recovered test rows 10,000–59,999, repartitioned | 97.7327 |
| 8 | k49_10 | 60,000 | K49 official train, classes 0,1,2,5,7,9,10,11,15,18 | 96.3436 |
| 9 | kannada_digits | 60,000 | Kannada-MNIST official train | 98.6536 |
| 10 | devanagari_digits | 20,000 | 17,000 source train + 3,000 test digits, repartitioned | 99.2973 |
| 11 | madbase | 59,944 | MADBase official train | 98.7036 |
| 12 | notmnist_large | 461,751 | notMNIST large archive, A–J fonts | 87.5773 |
| 13 | fashion_mnist | 60,000 | Fashion-MNIST official train | 87.2145 |
| 14 | svhn | 73,255 | Cropped digit official train, no extra split | 80.5927 |
| 15 | cifar10 | 50,000 | CIFAR-10 official train | 44.5655 |

The original MNIST control and USPS, DiG-MNIST, and the old small Omniglot
selection are not part of these fifteen. Related EMNIST/QMNIST and KMNIST/K49
tasks share ancestry. Native-image exact duplicates were removed; near
duplicates, writer/font overlap and potential downsampling collisions remain.

## Data location and verified state

All fifteen ready-to-use NPZ assets are present in
`/Users/yaroslavvb/git/aminist-21-validation/data/pools/<dataset_id>.npz`
(approximately 332 MB total). Each contains exactly `images`, `labels`,
`example_hashes`, `source_indices`, and `pool_indices`.

Executed the following read-only verification successfully on all fifteen:

```bash
cd /Users/yaroslavvb/git/aminist-21-validation
.venv/bin/python -m aminist21 verify-data
```

Verification checks asset and array SHA-256, tensor shape/dtype/pixel range,
label mapping, pool counts and unique native-image hashes. The manifest is
`datasets/manifest.json`, SHA-256
`08e6f58dad660f8bb931a73b0239030cd20c76cd3e8cde344cf3d4c535748167`.

Original archives also exist in `data/raw/` (EMNIST gzip IDX files, KMNIST/K49
NPZ files, QMNIST IDX files, SVHN MAT, CIFAR tarball, Kannada NPZ, Devanagari
ZIP, MADBase RAR/extracted files, notMNIST tarball). `data/raw_pools/` contains
precuration 9×9 pools, not retained full-resolution image tensors. The canonical
resized data are sufficient for the agreed transfer benchmark.

Official MNIST originals are already cached under
`/Users/yaroslavvb/git/sutro-problems/mnist/experiments/official-mnist-sample-curve-20260922/raw/source/`:
`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`,
`t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`.
The neighboring `raw/data_manifest.json` records canonical source MD5/SHA-256
and 60,000/10,000 tensor shapes. Reuse these bytes read-only; do not overwrite
the completed sample-curve experiment.

## Available baselines and comparison script

All six complete 165-draw score/prediction sets are local under
`reference-results/<recipe>/`, with `scores.json`, `scores.csv`, frozen
prediction manifests, per-draw prediction NPZ and metadata. Recipes are:

| Recipe | Architecture | Parameters | Epochs | Ensemble |
|---|---|---:|---:|---:|
| linear-sgd | 81→10 linear | 820 | 8 | 1 |
| mlp64-sgd | 81→64→10 | 5,898 | 8 | 1 |
| mlp256-sgd | 81→256→256→10 | 89,354 | 12 | 1 |
| cnn16-sgd | Two 16-channel convolutions, 64-unit head | 86,106 | 8 | 1 |
| cnn32-ensemble3 | Three 32-channel convolutions, 128-unit head | 351,914/member | 8 | 3 |
| reversible82-sgd | Two 82-state reversible coupling blocks | 7,554 | 2 | 1 |

`REPORT.md` has the full accuracy/SD matrix. `baseline-catalog.json` and
`aminist21/baselines.py` freeze the recipes; `reference-results/protocol.json`
records their execution protocol. The panel already received an independent
recount of 9.9 million predictions. Historical adapter times were 1.0, 1.4,
2.7, 2.3, 7.4 and 1.0 seconds per draw respectively on A100; these do not
predict a sequence model's runtime.

The new study's `compare_results.py` accepts an Aminist-compatible `scores.json`,
validates score counts and aggregates, matches the exact task/draw IDs against
all six historical references, and writes comparison JSON, CSV and Markdown.
It reads no evaluation labels and performs no fitting. It includes paired
draw differences, source/score hashes, subset completeness and interpretation
caveats. A 15-task self-reference check produced exactly zero paired deltas;
changed counts and seeds were correctly rejected.

```bash
python compare_results.py \
  --scores /absolute/new-run/scores.json \
  --references /Users/yaroslavvb/git/aminist-21-validation/reference-results \
  --output /absolute/new-run/comparison \
  --label 'Fixed-permutation candidate'
```

## Executable reuse path

1. Import `aminist21.data.load_pool`, `learner_inputs`, and `common.DATASETS`,
   `SEEDS`, `array_sha`, `sha`. Use the exact selected draw arrays returned by
   `learner_inputs`; avoid reimplementing the split.
2. Expose `train_predict(train_images, train_labels, test_images, config)`.
   It must return int64 classes or `(classes, metadata)`, preserve inputs,
   and create fresh weights/optimizer every invocation.
3. The CPU CLI supports a fresh child process per draw and arbitrary absolute
   adapter paths. For CUDA, copy the Modal scaffolding into this new study and
   adapt the worker's `invoke` call, leaving historical scripts unchanged.
4. Prefer frozen breadth-selected tasks with all eleven draws: KMNIST,
   EMNIST Letters A–J, recovered QMNIST, Fashion-MNIST, CIFAR-10. This is 55
   fits and a clearly labeled subset; all fifteen require 165 fits.
5. Save plan/source/data hashes, each NPZ and metadata, then call the reusable
   `aminist21.suite.freeze_predictions` and `score` after all planned fits
   finish. Run the independent comparison script only after that scoring.

The existing driver `tools/run_modal.py` is a clear template: compressed
three-array payloads, strict label exclusion, per-call provenance, `.starmap`
execution, maximum four A100 40 GB workers, frozen predictions before scoring.
Its pinned image is
`ghcr.io/ab-10/wikitext-bench@sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f`.
The image records Linux/Python 3.11.15/PyTorch 2.5.1+cu124/CUDA 12.4;
`numpy==2.2.6` is installed by the driver. It supplies 4 CPU cores and 8 GiB
host memory per worker, with `max_containers=4`, `min_containers=0`,
`buffer_containers=0`, `scaledown_window=2`, timeout 1,800 seconds.
Use a new application name and stop only this study's apps.

The adjacent `.venv/bin/python` is Python 3.11.4 with NumPy 2.2.6 and Modal
1.5.5; authenticated `modal app list --json` succeeded. Its `.venv-cpu`
uses Python 3.11.13, NumPy 1.26.4, PyTorch 2.2.2, MPS available, CUDA absent.
The latter is useful for interface checks, not numerically identical reference
reproduction. No new GPU tasks were launched by this inventory.

At inspection, unrelated `sutro-mnist-submit` and
`mnist-nine-a100-throughput-20260923` applications had active tasks; leave
them untouched. Cloud availability is dynamic and should be rechecked by the
runner when launching its own jobs.
