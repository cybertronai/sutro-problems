# MNIST-medium: fixed MLP submission attempt

**5,755/6,000 correct (96%)**: below the requested 98% goal by 125 correct
predictions. The frozen model has 512 hidden units, 200 training epochs,
learning rate 0.1, batch size 30, and seed 101.

This submission trains an 81 → H → 10 ReLU network from the supplied 6,000
training examples, then predicts the 6,000 canonical 9 × 9 test images.
`config.json` fixes the width, training epochs, learning rate, seed, and batch
size. Every measured task starts again from seed-only initial parameter
constants and performs the complete training and prediction computation.

The requested goal is **98% accuracy**, or **5,880/6,000 correct**. The
repository's current medium threshold is **98.14%**, or **5,889/6,000 correct**.
These are separate checks. A lower-accuracy attempt remains useful to report,
but does not qualify at a threshold it misses. Exact integer counts determine
the status even when displayed percentages round to the same value.

Contributors: Yaroslav Bulatov (task and benchmark requirements), Codex
(implementation, measurements, verification, and report). No W&B run was
created; measurement and verification evidence is retained beside the source.

## Selection and arithmetic

The predeclared search has 12 width/rate trajectories: widths 128, 256, and 512;
learning rates 0.01, 0.03, 0.1, and 0.2; and checkpoints at 25, 50, 100, 200, and
300 epochs. Its single random, non-stratified training/validation split contains
4,800/1,200 rows and uses PCG64 seed 20260913. Search initialization uses seed
11. Selection first seeks 1,176/1,200 validation hits and minimizes epochs ×
width, then width, then learning rate; if no checkpoint passes, it maximizes
validation hits with those same cost tie-breaks. This proxy is not the exact
physical-model score. The selected configuration is frozen with final seed
101 before test evaluation; no further search follows the test result.

The learner uses fixed contiguous minibatches of 30, input transform
`float32(x * 4) - float32(0.5)`, squared-error gradients, and ascending FP32
reductions with separate multiply and add operations. Initial weights are
uniform within ±1/√81 and ±1/√H, respectively; biases start at zero. These
seed-only constants contain no learned parameters. The historical CNN result
that motivated the repository's 98.14% target used a different 10,000/10,000
dataset; its weights are not used here.

## Reproduce from a clone

The submission branch is `codex/mnist-medium-submission`. After it is merged,
the same commands work on `main`. Run from the repository root:

```sh
git clone --branch codex/mnist-medium-submission https://github.com/cybertronai/sutro-problems.git
cd sutro-problems
python3.11 -m venv .venv-medium
.venv-medium/bin/python -m pip install -r mnist/submissions/medium-affine-20260911/requirements.txt
.venv-medium/bin/python -m mnist.code.data --output mnist/data --seed 20260910
S=mnist/submissions/medium-affine-20260911
.venv-medium/bin/python "$S/learner.py" --data mnist/data/medium.npz --config "$S/config.json" --output /tmp/mnist-medium-reproduction
.venv-medium/bin/python -m mnist.code.evaluate --tier medium --predictions /tmp/mnist-medium-reproduction/predictions.npy --data-dir mnist/data --output /tmp/mnist-medium-reproduction/accuracy.json
.venv-medium/bin/python "$S/score.py" --config "$S/config.json" --output /tmp/mnist-medium-reproduction
.venv-medium/bin/python "$S/verify.py" --data mnist/data/medium.npz --config "$S/config.json" --artifacts /tmp/mnist-medium-reproduction --output /tmp/mnist-medium-reproduction/verification.json
```

The evaluator reports the repository threshold; the verifier reports both the
requested and repository thresholds. Valid predictions below a threshold still
produce a successful evaluator command with `meets_accuracy_target=false`.
This flag checks classification accuracy, not complete benchmark acceptance.

The learner imports numerical routines from the checked-in
`mnist/experiments/accuracy-il-20260911/accuracy_study.py`. The model builder uses
the unchanged `il.py` in that directory. Neither dependency loads saved weights
or predictions. Canonical input array hashes are checked against
`mnist/doc/dataset_manifest.json`. The learner reads only `train_images`,
`train_labels`, and `test_images`; separate evaluation and verification code
may read test labels after predictions are fixed.

Verification compares a fresh complete fit with the retained parameter, score,
and prediction bits, including a learner CLI run whose NPZ contains only the
three allowed members. It independently checks ordered reductions on multiple
shapes, one complete training epoch, and final inference. It does not replay
every epoch through the independent ordered reference or numerically expand the
full affine program. Small expanded-program tests and an exact regression to
the previous small program are recorded in `model-ir-validation.json`.

The optional search reproduction below is substantially more work than
reproducing the selected learner. It writes to a separate directory and refuses
to overwrite existing search evidence. It reads training data only and does
not replace the frozen submission configuration:

```sh
.venv-medium/bin/python "$S/search.py" --data mnist/data/medium.npz --output /tmp/mnist-medium-search
```

## Reproduce the A100 measurements

With configured Modal credentials and `uv` installed, first generate the CPU
artifacts above, then run:

```sh
uvx --with numpy==2.2.6 modal==1.5.5 run "$S/gpu_benchmark.py" --data mnist/data/medium.npz --config "$S/config.json" --reference /tmp/mnist-medium-reproduction --output /tmp/mnist-medium-gpu
.venv-medium/bin/python -m mnist.code.evaluate --tier medium --predictions /tmp/mnist-medium-gpu/gpu_predictions.npy --data-dir mnist/data --output /tmp/mnist-medium-gpu/accuracy.json
```

The reference directory must contain `parameters.npz` with `w1`, `b1`, `w2`,
and `b2`, plus `output-scores.npy` and `predictions.npy`. These are host-only
verification oracles: no learned oracle weights, scores, or predictions are
copied to the GPU or supplied to its learner kernels. Full parameter, score,
and prediction bits are checked against the CPU artifacts before and after
measurement. The remote container pins NumPy 2.2.6; the local CPU and scoring
environment pins NumPy 2.4.6.

Each complete task replays three captured graph types: initialization once,
the one-epoch graph E times, and inference once, for E + 2 graph replays.
Initialization transforms training and test pixels, creates targets, and resets
parameters. Every task trains afresh. Three trials measure repeated,
GPU-resident complete tasks. Transfers, allocations, JIT compilation, capture,
verification, and host energy are excluded. The manually implemented GPU
kernels follow the learner equations; they are not generated by an automatic
IL-to-PTX compiler.

## Model scoring and report generation

The proposed `sutro-affine-v4/0.1` representation encodes bounded loops and
affine addresses, then counts every expanded v4 primitive and memory access.
Training data remain resident while test queries are received, normalized,
classified, and emitted one at a time. This keeps the searched models below
the unchanged scorer's one-million-word memory guard. The GPU uses its own
parallel buffer layout; the model's occupied scratch area is not GPU memory
usage or die area.

The scorer validates the restricted program, proves initialized reads under
its supported rules, constructs placement and per-address histograms, and
sums exact costs. It does not execute the billions of primitive instructions
numerically. Its timing excludes program construction, JSON loading, numerical
training, accuracy checks, and file output.

Human-readable results use two significant figures: execution time in ms,
energy in mJ, occupied scratch-cell area in mm², and time to score in s.
Exact measurements and the model's native ps/fJ/µm² quantities remain in JSON.
In these display units, `E_mJ = P_W × t_ms`.

To rebuild the report and Pages files from retained evidence:

```sh
.venv-medium/bin/python -m pip install -r "$S/requirements-report.txt"
.venv-medium/bin/python "$S/write_report.py"
.venv-medium/bin/python "$S/build_pages.py"
```

Report generation does not rerun training or GPU trials. The readable session
is a timestamped snapshot of visible user and assistant messages; hidden
reasoning, system instructions, and tool payloads are omitted.

- [Standalone report](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-affine-20260911/)
- [Separate ambiguities and problems](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-affine-20260911/ambiguities.html)
- [Human-readable session](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-affine-20260911/session.html)
