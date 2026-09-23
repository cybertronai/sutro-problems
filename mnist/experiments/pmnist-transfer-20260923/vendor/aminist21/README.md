# Aminist 21 Validation

**Aminist in the 21st century** — fifteen numbers for checking whether an image-classification **training procedure** works beyond its original task.

Take a solution, train it from fresh initialization on each dataset, and obtain a fixed-order vector of 15 mean accuracies. Every task has ten classes, 9×9 grayscale inputs, 10,000 training examples, and 10,000 evaluation examples. The default evaluation repeats each task on 11 fixed draws: **165 fresh fits per procedure**. All prepared datasets are included as checksum-pinned assets of this repository's `data-v1` release; evaluation does not depend on the original dataset hosts.

The [reference report](REPORT.md), also [published on SpaceSheep](https://spacesheep.dev/@yaroslavvb/aminist-21-validation), compares six fixed neural-network procedures and gives empirical performance ranges. These ranges describe the measured references; they are not universal pass/fail thresholds for a new solution. Machine-readable ranges are in [reference-ranges.json](reference-ranges.json).

## Run a solution and obtain the 15 numbers

The reference environment is Linux, Python 3.11, NumPy 2.2.6, PyTorch 2.5.1, CUDA 12.4, and an A100 40 GB GPU. Use a source checkout and an editable installation: the CLI reads the versioned manifests relative to this checkout.

```bash
git clone https://github.com/cybertronai/aminist-21-validation.git
cd aminist-21-validation
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[baselines]'
aminist21 list
aminist21 fetch
aminist21 verify-data
aminist21 run --recipe mlp64-sgd --device cuda --output runs/mlp64
```

`fetch` downloads the prepared NPZ assets from this GitHub repository and checks their SHA-256 hashes. It reuses matching local files. Once fetched, the evaluation can run offline. `verify-data` also checks individual arrays, shapes, label mappings, pixel ranges, and native-image uniqueness. Use `--data-dir /path/to/data` consistently on `fetch`, `verify-data`, `run`, and `score` to keep data elsewhere.

The command prints `fifteen_numbers` and writes:

- `runs/mlp64/scores.json`: the vector in `dataset_order`, each dataset's mean accuracy in percent and sample SD in percentage points, every draw's score, and confusion matrices.
- `runs/mlp64/scores.csv`: one row per dataset, including mean, SD, minimum, and maximum draw accuracy.
- `runs/mlp64/plan.json`, `prediction-manifest.json`, and per-draw prediction archives/metadata: the frozen configuration, inputs, source hashes, predictions, and runtime records needed to audit or rescore the result.

All predictions are frozen before the suite is scored. A stopped run can resume with the same command and output directory; mismatching source/configuration/data are rejected. Use a new directory after changing a procedure. Recompute scores without retraining with:

```bash
aminist21 score --output runs/mlp64
```

For a quick interface check, restrict the selection:

```bash
aminist21 run --recipe linear-sgd --device cpu --datasets kmnist --draws 0 --output runs/smoke
```

This is a **partial** result (`complete_v1_suite: false`), not the 15-number benchmark. Omit `--datasets` and `--draws` for a complete result. CPU execution uses the same training protocol; exact numerical agreement with the canonical GPU environment is not promised. The baseline dependency pins target Linux. Older Intel-macOS PyTorch builds require their own compatible NumPy environment and cannot be substituted into the pinned reference installation unchanged.

## Plug in an existing solution

Expose this callable in a Python module or file:

```python
def train_predict(train_images, train_labels, test_images, config):
    # Construct a fresh model and optimizer, train, then predict query classes.
    # Return predictions, or (predictions, JSON-serializable metadata).
    ...
```

The input contract is strict:

| Argument | Contents |
|---|---|
| `train_images` | NumPy `float32`, shape `(10000, 1, 9, 9)`, finite pixels in `[0,1]` |
| `train_labels` | NumPy `int64`, shape `(10000,)`, classes `0` through `9` |
| `test_images` | NumPy `float32`, shape `(10000, 1, 9, 9)`, finite pixels in `[0,1]` |
| `config` | A JSON object, fixed across all datasets and draws |
| Return value | Integer predictions of shape `(10000,)`, values `0` through `9`; optional JSON metadata |

There is no evaluation-label argument. Treat inputs as read-only. Keep architecture, initialization rule, epochs, optimizer, augmentation, and all other training choices fixed across the suite. Each call must start with fresh weights and optimizer state; do not reuse an MNIST checkpoint or fit preprocessing statistics on the evaluation images. A local run starts a new subprocess for every draw. Source and input hashing provide an audit trail, not a security sandbox: the adapter is trusted code and must obey the no-label-access contract.

[examples/adapter.py](examples/adapter.py) is a complete NumPy-only nearest-centroid example. It builds ten class centroids from the current training subset, then predicts the query set. It is an interface demonstration, separate from the six-network reference panel. Run it without installing PyTorch:

```bash
python -m pip install -e .
printf '{}\n' > centroid-config.json
aminist21 run --adapter examples/adapter.py:train_predict --config centroid-config.json --output runs/centroid
```

To use your learner, replace the example's body with your model construction, training, and inference; point `--adapter` at `your_package.your_module:train_predict` or `/absolute/path/adapter.py:train_predict`, and pass its fixed JSON configuration with `--config`. Keep imported dependencies versioned too: the runner hashes the adapter file but does not automatically archive every third-party dependency.

## What the 15 numbers mean

The order is part of the version-1 contract:

| # | Dataset ID | Ten-class task |
|---:|---|---|
| 1 | `kmnist` | Ten Kuzushiji characters |
| 2 | `emnist_letters_aj` | EMNIST Letters A–J |
| 3 | `emnist_letters_kt` | EMNIST Letters K–T |
| 4 | `emnist_balanced_aj` | EMNIST Balanced A–J |
| 5 | `emnist_digits` | EMNIST Digits |
| 6 | `emnist_mnist` | EMNIST MNIST |
| 7 | `qmnist_recovered` | Recovered digits beyond the standard MNIST test set |
| 8 | `k49_10` | Ten additional Kuzushiji-49 classes |
| 9 | `kannada_digits` | Kannada digits |
| 10 | `devanagari_digits` | Devanagari digits |
| 11 | `madbase` | Arabic handwritten digits |
| 12 | `notmnist_large` | Rendered font letters A–J |
| 13 | `fashion_mnist` | Clothing categories |
| 14 | `svhn` | Street-view house-number digits |
| 15 | `cifar10` | Object categories |

These are fifteen tasks, not fifteen independent data populations. Several EMNIST/QMNIST tasks share ancestry, and KMNIST/K49 share a collection. Original MNIST is not included as a control; USPS, DiG-MNIST, and the earlier small Omniglot subset are excluded because they cannot support this same 20,000-example protocol. notMNIST has only A–J, so no second-half-letter notMNIST task is invented.

Native images are decoded with the recorded orientation corrections. Curation uses hashes of these native image bytes: keep the first same-label exact duplicate and remove every member of a conflicting-label image group. The selected rows retain the recorded intensity polarity and use grayscale conversion where necessary, then exact box-area averaging to 9×9. Native-image deduplication does not remove near-duplicates, enforce writer/font independence, or guarantee that different originals remain distinct after downsampling.

Draw seeds are `20261101` through `20261111`, inclusive. For each seed the splitter uses `PCG64(SeedSequence(seed).spawn(2)[0])`, permutes the curated pool, takes the first 10,000 rows for training and the next 10,000 for evaluation. The two subsets are disjoint within a draw. The 11 draws are independently generated and may overlap; they are not eleven mutually disjoint 20,000-example blocks. Draws are not class-balanced by resampling. The standard learner seed is separately fixed at `11`.

A score is mean classification accuracy across the eleven draws, with sample SD across those draws. SD and the report's mean ± 2 SD bands are descriptive variation, not confidence intervals or universal acceptance boundaries. The original dataset train/test partitions are not the evaluation protocol here; some pools combine or repartition source splits. Do not compare these numbers directly with official-test leaderboard accuracies. Full pool definitions, class mappings, preprocessing, exclusions, and provenance are in [DATASETS.md](DATASETS.md) and [datasets/manifest.json](datasets/manifest.json).

## Reproduce the reference panel

The six recipes are fixed in [baseline-catalog.json](baseline-catalog.json): `linear-sgd`, `mlp64-sgd`, `mlp256-sgd`, `cnn16-sgd`, `cnn32-ensemble3`, and `reversible82-sgd`. No recipe is tuned separately to a dataset. The CNN ensemble is a native PyTorch reference inspired by the earlier MNIST solution; it is not claimed to reproduce that submission's ordered arithmetic bit for bit. The reversible classifier reconstructs its coupling-core activations during backward.

Any single recipe can be run with the ordinary CLI above. To retrain the entire panel, work in a fresh clone and first move its checked-in `reference-results` directory to `published-reference-results`; otherwise the driver verifies and resumes the published files instead of repeating completed fits. The checked-in Modal driver reproduces the full six-recipe panel on at most four ephemeral A100 workers. A Modal account and its normal authentication are required:

```bash
python -m pip install -e '.[baselines,experiments]'
modal setup
modal run tools/run_modal.py --mode smoke
mv reference-results published-reference-results
modal run tools/run_modal.py --mode run
python tools/build_report.py
```

The driver pins the container image digest and records hardware/software versions and all recipe sources in `reference-results/protocol.json` before fitting. It saves all six sets of predictions before scoring any of them. Completed files are reused only when provenance matches. The six-procedure panel contains 990 procedure/draw results; the ensemble trains three models per result, so there are 1,320 individual fresh model fits. Workers are configured to scale to zero; after execution or interruption, check `modal app list` and stop any still-running task app with `modal app stop APP_ID`. Runtime measurements are wall time, not energy measurements.

For local interface checks:

```bash
python -m pip install -e '.[baselines,test]'
python -m pytest -q
```

## Rebuild or archive the data

The normal path is `aminist21 fetch`: it retrieves the exact versioned inputs without contacting upstream hosts. To reconstruct the pools from their original archives instead:

```bash
python -m pip install -e '.[build-data]'
python tools/build_data.py --datasets all --workers 6
python tools/build_manifest.py
aminist21 verify-data
```

Raw rebuilding also needs system `curl` and `bsdtar` (from `libarchive-tools` on Linux) to decode the MADBase RAR archive. The preparation implementation and source checksums are versioned in the repository. Rebuilding requires access to the upstream sources and can take much longer than fetching the prepared assets. Keep this checkout, the `data-v1` NPZ assets, manifests, dataset notices, your adapter/configuration, and result directory together for an offline archive. The release manifest hashes both the compressed assets and their arrays.

Code is MIT-licensed. Dataset terms are separate: notably, SVHN is restricted to non-commercial use, and some other original sources do not state a standard redistribution/reuse license. No blanket license over all fifteen datasets is asserted. See [DATASETS.md](DATASETS.md) and [dataset-licenses.json](dataset-licenses.json) for the source-specific terms, attribution, and retained notices.
