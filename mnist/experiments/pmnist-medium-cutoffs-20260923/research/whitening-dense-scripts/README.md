# Scripts behind `research/whitening-dense-results.json` / `.md`

CPU only, `/tmp/pmnist-env/bin/python` (3.11, numpy 1.26.4, scipy 1.11.4,
scikit-learn 1.4.2, torch 2.2.2) on an Intel Mac. No Modal, no GPU, no network.
They import the study's `study.py`, `classical.py`, `learners.py`, `topology.py`
and the new `whiten.py` read-only; nothing in the live study was modified.

| script | what it produced |
| --- | --- |
| `eps_cv.py` | the train-only 5-fold CV sweep that chose `epsilon` (arc-cosine depth 3, N=1000) |
| `run_dense.py <classical\|mlp> <variant> [kernel] <N>` | every dense query-error number |
| `attack_topology.py <variant> <N> [--cnn]` | `recover_layout` / QAP-only attack + the cnn-09 arm |
| `collate.py` | assembles `research/whitening-dense-results.{json,md}` |

Variants live in `run_dense.VARIANTS`. Scratch JSON went to
`/tmp/whiten/results/`; rerunning the scripts regenerates it.

Dataset seed 2026092301 (a dev seed) throughout. No final seed
(2026092001..2026092011) was touched. Query labels were read only through
`raw/pool_labels.npy` indexed by `study.query_indices(2026092301)`, for scoring
these CPU pilots.

The attack side of the same question (topographic ICA and friends) was worked in
parallel by a second agent and lands in `research/whitening-attack-results.*`
with its own scripts under `research/whitening-attack-scripts/`.
