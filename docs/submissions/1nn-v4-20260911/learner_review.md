# Independent review of MNIST-small 1NN reference learner

Reviewed `learner.py`, SHA-256 `1815ae94fab2ab4d8f938d9aff2617bed3f6fed4399af7b7001861459f4dfaac`, against baseline `e70f9c9e1db65b62d9256f7b1f9b668cf4c48909`.

**No blocking integrity or correctness defect found for the canonical MNIST-small inputs.** This review covers the CPU reference learner and saved CPU predictions, not the separate tape-program scorer or A100 kernel.

## Integrity and scope

- Lines 43–45 explicitly decode only `train_images`, `train_labels`, and `test_images`. Lines 47–52 check each allowed array's shape, dtype and content hash against the manifest before computing predictions. No source indices or test-label values enter the learning or prediction computation.
- Line 66 reads the complete compressed archive to compute an NPZ identity hash *after predictions are computed*. Those raw bytes include all archive members when the original NPZ is used. This is checksum bookkeeping, not decoding or consulting the `test_labels` array. Phrase the guarantee as “the learner never decodes or uses test labels” rather than “no byte from the test-label member is ever read.” A successful CLI run using an NPZ that contains only the three allowed arrays directly establishes that labels and source indices are unnecessary for prediction.
- The submitted algorithm has no configurable model hyperparameters. Lines 54–58 run a training-only check using the disclosed seed, and lines 59–60 fit/memorize all training examples and predict. No conditional candidate selection follows validation. The record of when this algorithm was selected must come from the session chronology rather than code inspection alone.
- Explicit float32 subtract, multiply, and ordered add at lines 27–31 implement the stated numerical convention. NumPy `argmin` supplies first-row tie breaking. The finite-input checks cover canonical pixels; with nine coordinates in [0, 1], squared-distance arithmetic cannot overflow.
- The CPU timing includes copying the training image/label arrays and prediction, but excludes prior validation, input loading/hash verification and output writing. The saved `cpu_reference_scope` describes it correctly. The dense CPU distance matrix is not the memory allocation of the separately scored tape program.

## Executed checks

All **12** checks passed:

1. Equal-distance alternatives select the first training row.
2. Duplicate images with different labels retain the first row's label.
3. A single training example predicts its label for every supplied query.
4. Nonfinite train/query pixels are rejected.
5. Prediction leaves all three input arrays unchanged.
6. Canonical prediction output exactly matches the saved 600 predictions.
7. An independent scalar float32 oracle agrees on 12 spread-out queries, checking all 600 training rows for each.
8. Replacing training labels with `(label + 1) mod 10` changes output labels correspondingly, while nearest-row indices remain unchanged.
9. Saved source and prediction hashes match the files reviewed.
10. The recorded training/validation rows partition all 600 training examples without overlap; the validation result independently reproduces **63/120**.
11. The actual CLI succeeds and reproduces all predictions with no `test_labels`, `train_indices`, or `test_indices` members present in its NPZ input.
12. A one-ULP modification to one training pixel fails the canonical hash check before a prediction file is created.

An independent integrity-only audit also checked all six arrays in the clean checkout's canonical dataset: every shape, dtype and content SHA-256 matches the canonical manifest, and the train/test source-index intersection is empty. That audit does not pass test labels to the learner.

## Limits and optional wording improvements

The helper `predict` is intended for this benchmark and is protected by the canonical CLI input guard. Its API does not independently enforce integer label dtype/range, and malformed empty arrays can fail during reshape before reaching its custom shape error. These are not defects affecting canonical runs; extending generic validation is unnecessary for this submission.

The measured test result is **308/600 (51%)**, eight correct predictions above the original 50% threshold. That historical pass is below the current 60% requirement (360/600). It does not measure uncertainty across alternative dataset samples. This review did not search alternate algorithms, hyperparameters, seeds or numerical conventions after the score was known.

Reproduce the independent review with:

```bash
python mnist/submissions/1nn-v4-20260911/verify_learner.py \
  --data mnist/data/small.npz \
  --manifest mnist/doc/dataset_manifest.json \
  --output mnist/submissions/1nn-v4-20260911
```

Run from the repository root after installing the submission requirements. Omitting `--output` writes results to a new temporary directory. The data and manifest flags default to the paths shown.

[Verification script](verify_learner.py) · [Check results](learner_review.json) · [All-six dataset audit](dataset_verification.json) · [Rules and ambiguities](ambiguities.md)
