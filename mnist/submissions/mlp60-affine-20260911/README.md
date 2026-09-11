# MNIST-small: 60% target submission

Fixed 9 → 32 → 10 ReLU network, 300 epochs of ordered FP32 minibatch SGD,
learning rate 0.2, batch size 30, seed 101. **374/600 correct (62%)**, exceeding
the required 360/600. Each invocation learns from the supplied training arrays.

Contributors: Yaroslav Bulatov (task and benchmark requirements), Codex
(implementation, measurements, verification, and report). No W&B run was
created for this submission; measured evidence is checked in beside the source.

This is a submission attempt using a proposed compact affine v4 representation.
The benchmark's acceptance of the representation and arithmetic/tape conventions
remains subject to review. Selection of this submission from the published
feasibility study occurred after that study's test results were known.

## Reproduce from a clone

The submission branch is `codex/mnist-small-60-submission`. After it is merged,
the same commands work on `main`. Run from the repository root:

```sh
git clone --branch codex/mnist-small-60-submission https://github.com/cybertronai/sutro-problems.git
cd sutro-problems
python3.11 -m venv .venv-mlp60
.venv-mlp60/bin/python -m pip install -r mnist/submissions/mlp60-affine-20260911/requirements.txt
.venv-mlp60/bin/python -m mnist.code.data --output mnist/data --seed 20260910
.venv-mlp60/bin/python mnist/submissions/mlp60-affine-20260911/learner.py --output /tmp/mlp60-reproduction
.venv-mlp60/bin/python -m mnist.code.evaluate --tier small --predictions /tmp/mlp60-reproduction/predictions.npy --data-dir mnist/data --output /tmp/mlp60-reproduction/accuracy.json
.venv-mlp60/bin/python mnist/submissions/mlp60-affine-20260911/score.py --output /tmp/mlp60-reproduction
.venv-mlp60/bin/python mnist/submissions/mlp60-affine-20260911/verify.py --artifacts /tmp/mlp60-reproduction --output /tmp/mlp60-reproduction/verification.json
```

The learner and generator import the checked-in feasibility study implementation
under `mnist/experiments/accuracy-il-20260911/`. They do not import saved weights
or predictions. Input array content hashes are checked against the canonical
manifest. The evaluator alone opens test labels after predictions are fixed.
Verification also runs the learner with an archive containing only the three
allowed input members.

With configured Modal credentials and `uv` installed, reproduce the complete
task on an A100-40GB:

```sh
uvx --with numpy==2.2.6 modal==1.5.5 run mnist/submissions/mlp60-affine-20260911/gpu_benchmark.py --data mnist/data/small.npz --output /tmp/mlp60-gpu
.venv-mlp60/bin/python -m mnist.code.evaluate --tier small --predictions /tmp/mlp60-gpu/gpu_predictions.npy --data-dir mnist/data --output /tmp/mlp60-gpu/accuracy.json
```

The GPU command runs three trials after numerical verification. Each graph
replay resets parameters, prepares inputs and targets, trains all 300 epochs,
and predicts all 600 test labels. Host/device transfers, JIT, graph capture,
allocation, and validation are excluded from steady-state GPU measurements.
The remote container pins NumPy 2.2.6; the local CPU and scoring environment
used NumPy 2.4.6. Parameter, score, and prediction bits match across them.

To regenerate the report from the saved evidence:

```sh
.venv-mlp60/bin/python -m pip install -r mnist/submissions/mlp60-affine-20260911/requirements-report.txt
.venv-mlp60/bin/python mnist/submissions/mlp60-affine-20260911/write_report.py
.venv-mlp60/bin/python mnist/submissions/mlp60-affine-20260911/build_pages.py
```

The report build uses checked-in measurements, never reruns GPU trials. The
readable session is a timestamped snapshot of visible user and assistant
messages; hidden reasoning, system instructions, and tool payloads are omitted.

- [Standalone report](https://cybertronai.github.io/sutro-problems/docs/submissions/mlp60-affine-20260911/)
- [Separate ambiguities and problems](https://cybertronai.github.io/sutro-problems/docs/submissions/mlp60-affine-20260911/ambiguities.html)
- [Human-readable session](https://cybertronai.github.io/sutro-problems/docs/submissions/mlp60-affine-20260911/session.html)
