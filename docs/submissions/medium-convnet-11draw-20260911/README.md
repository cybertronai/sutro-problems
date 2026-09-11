# MNIST-medium: ConvNet submission across 11 datasets

This submission fixes a three-member ConvNet ensemble before training on
11 new MNIST-medium datasets. The inclusive accuracy requirement is **98% mean**,
or **64,680 / 66,000** correct predictions. The separate report contains the
measured mean, sample standard deviation, and all individual results.

Each dataset contains 6,000 training and 6,000 test images, resized to 9 × 9,
drawn without replacement from the original 60,000 training examples. Training
and test rows are disjoint within a draw; independently sampled draws may
overlap. Dataset seeds are **20261001 through 20261011**. These are integer RNG
seeds, not experiment dates. The same learner seeds **101, 102, 103** are used
on every draw, with fresh weights, optimizer state, and normalization statistics.

Every member has three padded 3 × 3, 64-channel convolution layers with
BatchNorm and GELU, then a 256-unit GELU head, dropout 0.2, and ten logits.
The fixed 71-epoch training procedure uses AdamW, a 100-epoch cosine learning
rate schedule, batch size 128, and mild affine augmentation. FP32 member logits
are averaged in FP64 before argmax. There is no validation, checkpoint
selection, or tuning on these 11 draws.

The architecture and epoch count came from the preceding training-only search.
The ensemble was chosen for this attempt after observing that study's disclosed
single-dataset test results. No earlier weights, labels, or learned state are
transferred. All choices were frozen before the new 11-dataset evaluation.

Contributors: Yaroslav Bulatov (requirements), Codex (implementation,
experiments, auditing, and reporting). No W&B runs were created.

## Reproduce the accuracy evaluation

Use Python 3.11 and run from the repository root. The source is on branch
`codex/mnist-medium-convnet-submission` until its submission PR is merged.
Configured Modal credentials and `uv` are required for the supplied A100 runner.

```sh
python3.11 -m venv .venv-convnet
.venv-convnet/bin/python -m pip install -r mnist/submissions/medium-convnet-11draw-20260911/requirements.txt
S=mnist/submissions/medium-convnet-11draw-20260911
mkdir -p /tmp/mnist-convnet-raw
curl --fail --location https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz --output /tmp/mnist-convnet-raw/train-images-idx3-ubyte.gz
curl --fail --location https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz --output /tmp/mnist-convnet-raw/train-labels-idx1-ubyte.gz
.venv-convnet/bin/python "$S/prepare_draws.py" --raw /tmp/mnist-convnet-raw --output /tmp/mnist-convnet-reproduction
uvx --with numpy==2.2.6 modal==1.5.5 run "$S/modal_accuracy.py" --study /tmp/mnist-convnet-reproduction
.venv-convnet/bin/python "$S/evaluate_frozen.py" freeze --study /tmp/mnist-convnet-reproduction
.venv-convnet/bin/python "$S/evaluate_frozen.py" evaluate --study /tmp/mnist-convnet-reproduction --raw /tmp/mnist-convnet-raw --output /tmp/mnist-convnet-reproduction/accuracy.json
```

Use a fresh output directory. Preparation refuses to overwrite an existing
protocol; the runner refuses to overwrite predictions or resume a partial run.
It verifies original IDX checksums and source hashes. Each remote call receives
only one draw's `train_images`, `train_labels`, and `test_images`, with no raw
IDX files, archives from other draws, or reference checkpoints.

The runner freezes all 11 ensemble predictions and diagnostic member logits
before the separate evaluator opens the raw labels and derives test-label
vectors. The evaluator verifies the complete evidence freeze before counting
any result. Accuracy is the unrounded aggregate fraction; sample SD uses the
11 dataset-level percentages and denominator 10. It is neither standard error
nor variability across the three member seeds.

Saved predictions can be audited without retraining, preserving the original
freeze and result by using fresh output paths:

```sh
.venv-convnet/bin/python "$S/evaluate_frozen.py" freeze --study "$S" --freeze-manifest /tmp/mnist-convnet-audit-freeze.json
.venv-convnet/bin/python "$S/evaluate_frozen.py" evaluate --study "$S" --raw /tmp/mnist-convnet-raw --freeze-manifest /tmp/mnist-convnet-audit-freeze.json --output /tmp/mnist-convnet-audit-accuracy.json
```

## Performance scope and model scoring

The A100 experiment begins only after the 11-dataset accuracy gate. It repeats
the entire frozen three-member learner on draw 00, including fresh training,
prediction, and FP64 ensemble construction. The report documents timing and
NVML boundaries, idle subtraction, raw trials, and output-identity checks.

After the accuracy result has passed, reproduce the measurement with:

```sh
uvx --with numpy==2.2.6 modal==1.5.5 run "$S/gpu_benchmark.py" --study /tmp/mnist-convnet-reproduction --output /tmp/mnist-convnet-reproduction/benchmark
```

This performs one complete warmup and three measured tasks on one A100.
The runner compares each fresh result to draw 00's retained logits,
predictions, and state-tensor hashes. The results include paired idle NVML
samples, gross and adjusted GPU-board energy, wall time, and CUDA-event time.

An exact Dally v4 translation is a separate requirement. The frozen learner
uses nonlinear functions, FP64 ensemble arithmetic, and GPU reduction semantics
that are not represented by the existing affine scorer. Missing model fields
are left unavailable; tensor FLOP counts are not substituted for v4 scores.
See `scoring-feasibility.md` for the specific lowering work still required.

Human-readable cost units are **ms**, **mJ**, **mm²**, and **s** for time to
score, with two significant figures. Accuracy mean and SD use one decimal
place; the retained JSON keeps exact counts and full precision. In these units,
`energy_mJ = power_W × time_ms`.

## Rebuild the report

```sh
.venv-convnet/bin/python -m pip install -r "$S/requirements-report.txt"
.venv-convnet/bin/python "$S/write_report.py"
.venv-convnet/bin/python "$S/build_pages.py"
```

The report is generated from retained evidence, without retraining. The session
export includes visible user and assistant messages only, excluding internal
reasoning, instructions, metadata, and raw tool logs. Its timestamp marks the
export cutoff.

- [Standalone report](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-convnet-11draw-20260911/)
- [Separate ambiguities and problems](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-convnet-11draw-20260911/ambiguities.html)
- [Human-readable session](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-convnet-11draw-20260911/session.html)
