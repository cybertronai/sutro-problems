# MNIST-medium ConvNet accuracy study

> **Accuracy scope:** This is a one-dataset study. The current small/medium requirement is mean ± sample SD over 11 independently resampled datasets; those aggregate statistics have not been measured here.

**The fixed ensemble reached 98.2% (5,894 / 6,000). The validation-selected single model reached 97.8% (5,868 / 6,000).** The frozen selection remains unchanged; promoting the diagnostic ensemble after test evaluation would be test-informed.

This experiment tests ten ConvNet configurations on the canonical 6,000 / 6,000, 9 × 9 problem, with six validation replications and three frozen full-data fits. It is the accuracy stage requested before translation and measurement. Exact outcomes are in `test_results.json`; the report explains selection, counts, and limitations. No ConvNet theoretical score or controlled performance/energy measurement is claimed.

Use the source branch `codex/mnist-medium-convnet` until the accompanying accuracy-study pull request is merged. The primary result uses the learner named in `selection.json`, not whichever saved test seed is best.

## Reproduce the frozen learner

From the repository root, with Python 3.11+, NumPy, uv/uvx installed, and configured Modal credentials:

```sh
python3 -m venv .venv-convnet
.venv-convnet/bin/python -m pip install numpy==2.2.6
.venv-convnet/bin/python -m mnist.code.data --output mnist/data --seed 20260910
study_source=mnist/experiments/medium-convnet-20260911
reproduction_dir=/tmp/mnist-convnet-reproduction
mkdir -p "$reproduction_dir"
.venv-convnet/bin/python "$study_source/prepare_inputs.py" --phase refit \
  --selection "$study_source/selection.json" \
  --output "$reproduction_dir/refit-input.npz" --record "$reproduction_dir/refit-input.json"
uvx --with numpy==2.2.6 modal==1.5.5 run "$study_source/refit.py" \
  --data "$reproduction_dir/refit-input.npz" \
  --selection "$study_source/selection.json" --output "$reproduction_dir/final"
.venv-convnet/bin/python "$study_source/evaluate_frozen.py" freeze \
  --selection "$study_source/selection.json" --selected-name seed101 \
  --predictions "seed101=$reproduction_dir/final/predictions-seed101.npy" \
  --predictions "seed102=$reproduction_dir/final/predictions-seed102.npy" \
  --predictions "seed103=$reproduction_dir/final/predictions-seed103.npy" \
  --predictions "ensemble=$reproduction_dir/final/predictions-ensemble.npy" \
  --manifest "$reproduction_dir/prediction_manifest.json"
.venv-convnet/bin/python "$study_source/evaluate_frozen.py" evaluate \
  --selection "$study_source/selection.json" \
  --manifest "$reproduction_dir/prediction_manifest.json" \
  --output "$reproduction_dir/test_results.json"
```

The runner refuses a full dataset archive: its input must contain exactly `train_images`, `train_labels`, and `test_images`. It accepts no checkpoints, learned weights, or prediction oracles. Every seed initializes fresh parameters, fits all 6,000 allowed training rows for the frozen 71 epochs, then predicts all test images. The three-model ensemble is retained as a predeclared diagnostic, but validation selected the single seed-101 model. All four prediction arrays are frozen before evaluation. Do not change the selected predictor using the test results.

The runner pins its container image and dependencies, disables TF32/autocast, and enables deterministic PyTorch/cuDNN behavior. Its source, input, protocol, and validation-selection hashes must match the frozen record. Reproducing on different software, hardware, or arithmetic may change results. Metadata timestamps and serialized checkpoint hashes need not repeat even when tensors do.

## Repeat the training-only search

This is optional and more work than reproducing the fixed learner. Use an empty destination; preserve the checked-in experiment evidence.

```sh
study_source=mnist/experiments/medium-convnet-20260911
search_dir=/tmp/mnist-convnet-search
mkdir -p "$search_dir"
.venv-convnet/bin/python "$study_source/prepare_inputs.py" --phase search \
  --output "$search_dir/train-only.npz" --record "$search_dir/search_input.json"
uvx --with numpy==2.2.6 modal==1.5.5 run "$study_source/modal_train.py" \
  --phase prepare --data "$search_dir/train-only.npz" --output "$search_dir"
uvx --with numpy==2.2.6 modal==1.5.5 run "$study_source/modal_train.py" \
  --phase search --data "$search_dir/train-only.npz" --output "$search_dir"
.venv-convnet/bin/python "$study_source/audit_search.py" \
  --study "$search_dir" --data "$search_dir/train-only.npz" --output "$search_dir/search_audit.json"
uvx --with numpy==2.2.6 modal==1.5.5 run "$study_source/modal_train.py" \
  --phase replicate --data "$search_dir/train-only.npz" --output "$search_dir"
.venv-convnet/bin/python "$study_source/audit_search.py" --replications-complete \
  --study "$search_dir" --data "$search_dir/train-only.npz" --output "$search_dir/validation_audit.json"
.venv-convnet/bin/python "$study_source/audit_selection.py" \
  --study "$search_dir" --data "$search_dir/train-only.npz" --output "$search_dir/selection_audit.json"
.venv-convnet/bin/python "$study_source/freeze_selection.py" \
  --study "$search_dir" --output "$search_dir/selection.json"
```

Check the new selection before any test evaluation. Use its file in the refit commands above, and its `selected_name` in the prediction-freeze command. These may differ if a reproduction changes numerical results. All selection must remain training-only.

The initial ten models use seed 11. The top three repeat seeds 22 and 33. Selection first ranks mean best-checkpoint validation correct count, then mean loss, parameter count, and configuration ID. The refit epoch is the median selected epoch. The chosen architecture becomes an ensemble only when the validation logits average beats its fixed seed-11 checkpoint; ties favor a single model. This policy, final seeds 101/102/103, split, optimizer, augmentation, and schedule are frozen in `protocol.json` before search.

`audit_search.py` checks every retained result, source/config/provenance, checkpoint and logit hash, validation prediction/loss, history-based checkpoint choice, and parameter count. Run it before reusing saved results for replication or selection. Search checkpoints are ignored by Git and regenerated by search; final checkpoints and all validation logits/curves are published. The search and refit runners never open test labels.

## Build the reports

```sh
.venv-convnet/bin/python -m pip install -r "$study_source/requirements-report.txt"
.venv-convnet/bin/python "$study_source/plot_search.py"
.venv-convnet/bin/python "$study_source/write_report.py"
.venv-convnet/bin/python "$study_source/build_pages.py"
```

`export_session.py` makes a timestamped export from a local session log. It includes visible user and assistant messages only, from the ConvNet request onward, and excludes hidden reasoning, system/developer instructions, raw tool payloads, and environment metadata. Raw session logs are not published. The report, separate issue report, and export are static GitHub Pages files served from `main`.

Execution times in later benchmark tables will use **ms**, energies **mJ**, occupied-cell areas **mm²**, and time to score **s**, with two significant figures. Experimental elapsed durations in raw JSON are not benchmark measurements.

- [Standalone accuracy report](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-convnet-20260911/)
- [Separate ambiguities and problems](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-convnet-20260911/ambiguities.html)
- [Human-readable session](https://cybertronai.github.io/sutro-problems/docs/submissions/medium-convnet-20260911/session.html)
