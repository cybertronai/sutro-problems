# Scripts behind `research/whitening-attack-results.json`

Every number in the results file and in
`research/whitening-attack-results.md` comes from one of these, run with
`/tmp/pmnist-env/bin/python` (3.11, numpy 1.26.4, scipy 1.11.4, scikit-learn
1.4.2, torch 2.2.2 CPU) on an Intel Mac.  CPU only: no Modal, no GPU, no network.
They import `research/attack_ica_topographic.py`, which imports the study's
`study.py` / `topology.py` / `whiten.py` / `learners.py` / `spatial_learner.py`
read-only.

Scratch arrays (ICA sources, correlation matrices, layouts) were written under
`/tmp/whiten-attack/` and are not checked in; rerunning `stage1.py` regenerates
them.

| script | what it produced |
| --- | --- |
| `sweep1.py N` | first ICA localisation sweep (contrast function x whitening mode x n_components) |
| `affinity_probe.py N` | neighbour-vs-far separation of the energy affinities |
| `qap1.py N` | first lattice-QAP layouts at N=1000 (the layout the N=1000 CNN arms use) |
| `stage1.py N out.json` | the reproducible pipeline: `run_attack` for three ICA settings x three affinities |
| `batchA.py` / `batchB.py` | cnn-09 downstream arms (CPU, 1 member) |
| `seedvar.py` | member-seed spread of the two key CNN arms |
| `dense.py N epochs` | dense standardised-MLP baselines, whitened vs permuted pixels |
| `secondorder.py` | the existing second-order attack applied to the whitened variant |
| `ceiling.py`, `ceiling2.py` | how close any *white* basis can get to the pixel basis |
| `objective.py` | the ICA contrast at the FastICA solution vs at the pixel-aligned basis |

Seeds: dataset seed 2026092301 (a dev seed) throughout.  No final seed
(2026092001..11) was touched.  Query labels were read only through
`raw/pool_labels.npy` indexed by `study.query_indices(2026092301)`, for scoring
CPU pilots.

Added after the ICA result came back negative, to test a *different* prior
(non-negativity / zero-inflation instead of independence):

| script | what it produced |
| --- | --- |
| `nonneg.py N` | non-negative-ICA style rotation search (weak; reported as a failed attempt) |
| `atom.py` | single-facet probe: is ONE pixel direction findable? |
| `facet.py N eigfloor restarts` | deflation over many restarts; how many pixels come back |
| `facet2.py N epochs` | the same attack end to end: pixels -> `recover_layout` -> cnn-09 |
