# Red-team submissions

Attacks against this harness live here: replay and caching, timer tampering,
side-stream work, MNIST label lookup, trivial-work shortcuts. Each file is an
ordinary submission that `run_modal.py --submission redteam/<file>` can run, and
each should carry a docstring saying what it attacks and what the harness is
expected to do about it.

## replay / memoization (lens: replay-cache, 2026-09-22)

| File | Attacks | CPU dry-run verdict |
| --- | --- | --- |
| `replay-cache.py` | pool memoization: every draw re-splits the *same* 60k pool, so earlier training halves answer later test halves | **escaped** — 12% band passes at 94.90% with a 79% payload (`results/redteam-replay-cache-12pct.json`) |
| `replay-cache_persist.py` | same table, written outside the scratch directory so it survives to the next submission | **escaped** — 2% and 5% bands pass at 98.10% (`results/redteam-replay-cache-persist-{2,5}pct.json`) |
| `replay-cache_weights.py` | fit once, recover each draw's label permutation from a 1,000-image probe, reuse the weights; retrain only for the hold-out | **escaped** — 9.4x lower ranked time at equal accuracy (`results/redteam-replay-weights-5pct{,-control}.json`) |
| `replay-cache_naive.py` | control: replay call 1's answer by call count | caught by the accuracy gate (10.22%) |

Reproduce with `--local --pool-cache <raw dir>`; see each file's docstring.

## Status of these verdicts

Every "escaped" above was against harness **1.0.0**. They were all re-run
against **1.1.0** after the fixes; the before/after table is in
[`../DESIGN.md`](../DESIGN.md) under "Harness changes from red-teaming", and the
GPU-only ones that still have to be checked on an A100 are listed in
[`../tests/GPU_CHECKS.md`](../tests/GPU_CHECKS.md). Four of these files are now
rejected before they are even imported, because they are larger than the
submission size cap; that is the intended outcome, not a harness error.

`timer-patch_cellsteal.py` was added during the fix pass rather than by the red
team: it defeats the in-process hardening the way the red-team report predicted
(walk `gc` for the cell holding the evaluator's captured `perf_counter` and
overwrite it) and is rejected by the parent-side clock bound. It is the
regression test for that bound on the CPU path.

## the linear release (lens: release, 2026-09-23)

Harness 1.2.0 stops releasing 9x9 pixels: every draw is served as
`z = Q W (x - mu)`, exact PCA whitening onto the top 60 principal directions of
that draw's own training rows followed by a secret per-draw Haar rotation
(`eval.release_map`). [`release/`](release/) is the red-team evidence for what
that leaves open. Unlike the files above these are not submissions -- they are
attack scripts that call `eval.make_draw(..., release_dims=60)` directly, so
they measure the real release without a GPU.

| Script | Attacks | CPU verdict |
| --- | --- | --- |
| `release/blind_attack.py` | blind unmixing + lattice recovery: an entrant with no source data tries to find the pixel grid and convolve | **no gain** — 4.0x chance on lattice edges, but the CNN on the recovered lattice (95.69%) ties the CNN on a *random* layout (95.68%) and the honest MLP (95.79%) |
| `release/informed_attack.py` | public MNIST + labelled moment matching: recover the map, then fire an offline CNN at the reconstructed pixels | **escapes the release** — pixels back at 0.077 rel-RMS, 98.52% vs the honest MLP's 95.79% (+2.73 points). The secret label permutation is recovered 10/10 in 0.2 s and is not an obstacle |
| `release/artifact_size.py` | can that be packed into `submission.py`? | **caught by the size cap** — 985,188 bytes of payload, 48.1x the 20,480-byte cap; exact whitening blocks the obvious shrink (k=30 statistics fit but the attack collapses to 46.49%), and the largest artifact that does fit loses 5.04 points to the honest MLP |
| `release/holdout_check.py` | the same informed pipeline fired at a Fashion-MNIST hold-out draw | **caught by the hold-out** — 57.12% against the 70% floor (honest MLP 85.95%) — but the smuggler can *detect* the hold-out from its own moment residual (65 vs 753) and fall back, so the cap is the load-bearing defence, not this |
| `release/exact_source.py` | norm re-identification of released rows against the public pool | **caught twice** — 15.89% labels because `(mu, W)` is fitted per draw on secret rows (with the organiser's fit set it is 99.96%), and the cheapest norm→label table is 36,156 bytes, 1.8x the cap, with an entropy floor of 33,200 bytes that no coder beats |

Full tables, the residual risks and reproduction commands are in
[`release/README.md`](release/README.md); merged numbers in
`release/results.json`.
