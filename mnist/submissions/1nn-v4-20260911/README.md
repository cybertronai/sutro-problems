# MNIST-small: fixed 1-nearest-neighbor, v4 tape attempt

**308/600 correct (51.33%)** on canonical `competition-v2`, seed 20260910. The current 50% accuracy requirement is met. This is a submission attempt for review, with explicit FP32 arithmetic, tape serialization, and occupied-cell area conventions.

The implementation memorizes the 600 supplied training examples and predicts all 600 test labels. It uses ordered FP32 squared Euclidean distance and first-training-row tie breaking. No extra labeled data, pretrained weights, test-label selection or hyperparameter search is used.

The standalone report explains all six requested metrics, versions, boundaries, raw measurements and reproduction commands. Model scores use the v4 single-core-with-tape specification at `26abcca402de647381d31286d42dfbb7a001763d`; benchmark rules/data are pinned to `e70f9c9e1db65b62d9256f7b1f9b668cf4c48909`. The custom scorer's results require acceptance of the stated conventions.

## Reproduce

From the repository root, using Python 3.11 or newer:

```bash
python3 -m venv mnist/.venv
S=mnist/submissions/1nn-v4-20260911
mnist/.venv/bin/python -m pip install -r "$S/requirements.txt"
mnist/.venv/bin/python -m mnist.code.data --output mnist/data --seed 20260910
mnist/.venv/bin/python "$S/learner.py" --data mnist/data/small.npz --output /tmp/mnist-1nn-cpu
mnist/.venv/bin/python -m mnist.code.evaluate --tier small --data-dir mnist/data \
  --predictions /tmp/mnist-1nn-cpu/predictions.npy --output /tmp/mnist-1nn-cpu/accuracy.json
mnist/.venv/bin/python "$S/verify_learner.py" --data mnist/data/small.npz
mnist/.venv/bin/python "$S/score_v4.py" --self-test --data mnist/data/small.npz \
  --output /tmp/mnist-1nn-model --emit-ir /tmp/mnist-1nn-model/program.v4.gz
mnist/.venv/bin/python "$S/score_v4.py" --data mnist/data/small.npz \
  --replay-ir /tmp/mnist-1nn-model/program.v4.gz --output /tmp/mnist-1nn-replay
uvx --with numpy==2.2.6 modal==1.5.5 run "$S/gpu_benchmark.py" \
  --data mnist/data/small.npz --output /tmp/mnist-1nn-gpu
```

The last command needs a configured Modal account and runs one A100-40GB. It measures repeated GPU-resident complete tasks with training copies, excluding transfers, allocation, JIT, graph capture and startup. Model Time to score covers one full instruction-generation/interpreter/accounting run; optional IR emission and replay have separately reported times.

Reproduction outputs go to new temporary directories so the submitted measurement evidence remains intact. The expanded IR is 220,146,130 bytes; gzip generation is about 11.1 MB. Only the compact generator, excerpt and expanded-text checksum are checked in. All six canonical dataset hashes should be verified; `verify_learner.py` does this separately from label-blind learning and runs twelve integrity/correctness checks.

## Reports and evidence

- [Standalone results and reproduction report](https://cybertronai.github.io/sutro-problems/docs/submissions/1nn-v4-20260911/)
- [Human-readable session export](https://cybertronai.github.io/sutro-problems/docs/submissions/1nn-v4-20260911/session.html)
- [Separate ambiguities and problems report](https://cybertronai.github.io/sutro-problems/docs/submissions/1nn-v4-20260911/ambiguities.html)
- [Report source](report.md), [scoring conventions](scoring-notes.md), [accuracy](accuracy.json), [model score](model-score.json), [expanded-IR replay](replay-score.json), [A100 measurements](gpu_results.json)
- [CPU learner](learner.py), [v4 generator/interpreter](score_v4.py), [Triton benchmark](gpu_benchmark.py), [dataset verification](dataset_verification.json), [independent learner review](learner_review.md)
- [Visible session Markdown](session.md), [session JSON](session.json), [exporter](export_session.py), [HTML renderer](build_pages.py)
