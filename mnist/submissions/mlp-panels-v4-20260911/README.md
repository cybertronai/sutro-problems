# MNIST-small: Energy-Aware Panel MLP

Historical-specification study: 600 training / 600 test examples and the former
single-core-with-tape model. This is not a current qualifying submission.

Upstream now requires 1,000/1,000 examples, 67% mean accuracy and spatial-computer
grid scoring. The measurements here must remain in the historical results table;
current-specification accuracy and grid costs have not been measured.

The primary `energy` variant optimizes all seven matrix products of the fixed
9-32-10 ReLU MLP. Its model energy is 14.14% lower than the original; its measured
A100 time is 20.34% lower and idle-adjusted board energy is 17.95% lower in the
primary paired experiment. See the report for exact scope, variability, and caveats.

Fresh submission evaluation: **4292/6600 correct, 65.0% +/- 2.1 pp**, over eleven
new predeclared draws, with learner seed 101 on every draw. The architecture,
learning rate, stopping rule and arithmetic were frozen before this evaluation.

## Requirements

- Run commands from the repository root with Python 3.11 or newer.
- CPU reproduction needs the pinned NumPy dependency below.
- GPU reproduction additionally needs `uv`, a configured Modal account and paid
  A100 access. Do not run it accidentally while reviewing this package.
- The package uses the existing tracked affine-v4 scorer, original MLP helpers,
  and the tracked canonical input tape. It does not require the untracked research
  directories or saved learned weights.

```sh
python3 -m venv .venv-panels
.venv-panels/bin/python -m pip install -r mnist/submissions/mlp-panels-v4-20260911/requirements.txt
```

The measurement environment was Python 3.14.7 / NumPy 2.4.1 on macOS ARM64 for
CPU scoring/evaluation, and the pinned PyTorch/Triton/CUDA image described in
the report for A100. Host scoring runtime depends on the machine.

## Canonical Reproduction

Recover the three allowlisted canonical arrays from the already tracked input
tape. This checks their hashes against the canonical manifest and avoids silently
accepting platform-dependent resize differences. No test labels are recovered.

```sh
.venv-panels/bin/python mnist/submissions/mlp-panels-v4-20260911/prepare.py
.venv-panels/bin/python mnist/submissions/mlp-panels-v4-20260911/learner.py --data mnist/submissions/mlp-panels-v4-20260911/generated/canonical-inputs.npz --manifest mnist/doc/dataset_manifest.json --output mnist/submissions/mlp-panels-v4-20260911/generated/canonical-replay
.venv-panels/bin/python mnist/submissions/mlp-panels-v4-20260911/score.py --output mnist/submissions/mlp-panels-v4-20260911/generated/model-replay
.venv-panels/bin/python mnist/submissions/mlp-panels-v4-20260911/verify_submission.py --full
```

`learner.py` accepts any valid MNIST-small draw, not only the canonical hashes.
Pass that draw's own manifest. It writes `predictions.npy` as a one-dimensional
int64 vector, plus scores, parameters and provenance in the chosen output folder.
The generated cache is ignored by Git. `--variant baseline` and
`--variant no_slowdown` are supporting comparison options.

Full verification downloads the small official training-label source if its
verified local cache is absent; it never launches GPU work.

`score.py` verifies exact model totals against the measured artifact. It records
five new host timing samples after a warmup; it never launches a GPU.

## Reproduce the Eleven-Draw Evaluation

Use a fresh output directory. The fixed seeds are 2026091301 through 2026091311.
The script uses the repository's sampling and resize functions, builds only the
small tier, and retains a per-draw manifest. Source MNIST files are MD5-verified
and downloaded only if the existing raw cache is absent.

```sh
.venv-panels/bin/python mnist/submissions/mlp-panels-v4-20260911/evaluate_draws.py predeclare --output mnist/submissions/mlp-panels-v4-20260911/generated/reproduce11
.venv-panels/bin/python mnist/submissions/mlp-panels-v4-20260911/evaluate_draws.py predict --output mnist/submissions/mlp-panels-v4-20260911/generated/reproduce11
.venv-panels/bin/python mnist/submissions/mlp-panels-v4-20260911/evaluate_draws.py score --output mnist/submissions/mlp-panels-v4-20260911/generated/reproduce11
```

The scoring phase starts only after all prediction files and hashes exist.
Do not tune the learner after inspecting its output. Different BLAS/CPU resize
rounding can change exact input bytes; manifests identify the actual arrays.
The archived submitted results correspond to the recorded evaluation environment.

For an ordinary repository-generated `small.npz`, the repository's one-draw
evaluator can also score the resulting predictions. Its single-draw pass flag is
not the eleven-draw qualification:

```sh
.venv-panels/bin/python -m mnist.code.evaluate --tier small --data-dir /path/to/draw --predictions /path/to/learner-output/predictions.npy --output /path/to/one-draw-score.json
```

## A100 Reproduction: Paid

The portable runner changes only its local reference-file path. Its remote
benchmark function is AST-identical to the primary measured snapshot; kernel
source and the unchanged original baseline are verified by hash/AST checks.

```sh
MODAL_PROFILE=vargapowercouple uvx --with numpy==2.2.6 modal==1.5.5 run --profile vargapowercouple mnist/submissions/mlp-panels-v4-20260911/gpu_benchmark.py --output mnist/submissions/mlp-panels-v4-20260911/generated/new-gpu-run
```

The explicit profile guard prevents accidental billing to another account.
Other users must deliberately configure that profile or adapt only the local
billing guard. Do not change the remote algorithm/protocol when reproducing.
Use a fresh output path to preserve completed measurements.

The command runs all three variants in one A100-40GB container, validates full
learning and mutation cases, checks 24,004 nodes in each graph, and measures three
cyclic rounds with CUDA events and paired-idle NVML counters. Every timed replay
trains from scratch. Transfers, JIT, capture, allocation, validation, startup and
host CPU energy are outside the steady-state task metrics.

## Build and Review Documentation

```sh
.venv-panels/bin/python mnist/submissions/mlp-panels-v4-20260911/build_report.py
.venv-panels/bin/python mnist/submissions/mlp-panels-v4-20260911/verify_submission.py --full
```

The report builder uses completed local evidence and produces a static HTML
preview; it neither reruns measurements nor publishes Pages. On a clone without
the original generated cache, the archived accuracy evidence is sufficient for
documentation rebuilding.

## Documentation

- [Standalone report and required metrics](report.md)
- [Local HTML preview, not published](../../../docs/submissions/mlp-panels-v4-20260911/index.html)
- [Requirements checklist and review caveats](REVIEW_CHECKLIST.md)
- [Historical-study PR description](PR_DRAFT.md)
- [Full-precision metrics](metrics.json)
- [Artifact checksums](artifact_manifest.json)
