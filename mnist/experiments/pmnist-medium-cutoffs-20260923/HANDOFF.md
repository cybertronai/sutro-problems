# Handoff: permutation-invariant MNIST-medium cutoffs

Study id `pmnist-medium-cutoffs-20260923`, dated 2026-09-23/24. The requested experiment is
**complete, frozen and scored**. This document is enough to continue without the original
conversation; the numbers and their derivations are in [README.md](README.md).

## What the user asked

> "For the permutation-invariant version of MNIST-medium (9x9 MNIST, 10k train, 10k test, both
> drawn disjointly from the official training split), find the largest achievable result if
> allowed one A100 for 20 minutes per fit, then create a five-level ladder whose top is that
> result at 10,000 examples and whose bottom is the best state-of-the-art dense network at 1,000
> examples."

Compute authorisation (frozen in [authorization.json](authorization.json)): "Feel free to use
modal for experiments, don't exceed 20 dollars" and, later, "feel free to use up to 8 A100s in
parallel".

A follow-up question was asked while the study ran: **can whitening / decorrelating the pixels
before permuting remove the topology-recovery loophole?** That was answered with CPU pilots
only, on a development seed, and is summarised in the README's whitening section.

## What is done

- Both ladders are measured on all eleven final dataset seeds (2026092001..2026092011) at all
  five levels: 1,000 / 1,778 / 3,162 / 5,623 / 10,000. 121 final jobs, 0 failed, 0 truncated.
- **Ladder A (dense permutation-invariant)**: 5.4627% error at 1,000 (`ladder-retuned-tq`) ->
  4.9464% / 3.8173% / 3.0755% / **2.4682%** at 10,000 (`kr-arccos1-d3`).
- **Ladder B (unrestricted, topology recovery + CNN)**: 3.9991% at 1,000 -> 3.1445% / 2.4127% /
  1.8964% / **1.5455%** at 10,000 (`topo-cnn09-x3`).
- The 20-minute question is answered: the winning dense recipe uses **no GPU at all** (kernel
  ridge, 124.3 s mean CPU wall per fit at N=10,000), and every high-compute GPU arm scored
  worse on the development seed. Labels and the 9x9 bottleneck bind, not compute.
- The topology loophole is demonstrated and quantified: recovery costs 6.24-9.10 s and the
  resulting CNN matches the true-grid spatial reference (+0.04 pp at 10,000, CI includes zero).
- The whitening follow-up has a two-agent CPU answer: **rotation, not whitening, is what closes
  the loophole**; whitening is a pure accuracy cost; ICA attacks fail structurally; a
  non-negativity/facet attack recovers a dozen pixels exactly and is the open risk.
- The combined two-ladder figure `analysis/figures/ladders.png` / `.svg` exists, with its
  plotted values in `analysis/figures/ladders_data.json`; every value there matches
  `analysis/ladders.json` and the reference `learning_curves.csv` exactly.
- README.md and this handoff are written. No paper, no page, no leaderboard row has been
  produced from this study.

## Exact state

**Frozen and immutable** (do not regenerate, do not edit):

| File | State |
| --- | --- |
| `protocol.json` | `status: frozen`, `frozen_at_utc` 2026-09-24T01:07:18Z; carries the frozen source hashes for all thirteen executed modules |
| `selection.json` | `status: frozen` 2026-09-24T00:19:35Z; SHA-256 `ef5753c0f3b411c3c4ebd87736ee92faa219c822a4cf1da4425ace9bbd667daa` |
| `predictions/final_freeze.json` | frozen 2026-09-24T01:07:18Z, `scored: true` 2026-09-24T01:07:29Z; current SHA-256 `e6940da36cb428c3d381395692cf40627a87413dc447e6ab041f2f45b80b607c` (the pre-scoring value `6804e9d6...` is what `protocol.json` records) |
| `plans/final_all.json` | SHA-256 `0607055651d4ae30fde299f219589003cd729890c2660fa84e55a983a028318a`, 121 jobs |
| `results/` | 121 final job records + 72 dev records + `scores_final.json` / `scores_dev.json` |
| `predictions/` | one `.npz` per job (local only, 77 MB, git-ignored), hashes recorded in `results/*.json` and the freeze |
| `raw/data_manifest.json` | SHA-256 `a557e524f939b043c81513d7fd6dba84ddb3dd1156dd1dcc4e7442399f10791b` |

`score.py` has already marked the freeze scored and will refuse to re-freeze without
`--refreeze`; re-running the scorer on this directory is not a safe way to "check" anything.

**Ledger.** [budget-ledger.json](budget-ledger.json) totals **$14.0241** charged upper envelope
across 8 apps against the user's $20 cap: $0.17 smoke, $10.46 development (rounds 1 and 2),
$3.39 final GPU. This is an upper envelope (every reserved worker billed for the whole app
lifetime at the published A100-40GB rate $0.00065316/worker-second), **not a provider invoice
and not an energy measurement**. The policy was amended mid-study from 2 to 8 maximum containers
after the user authorised 8 parallel A100s; the amendment and both policies are recorded in the
file. $5.976 of the $20 cap is unbilled, of which **$3.976 is available allowance** under
`budget.py` (the remaining $2 is the frozen contingency, which `budget.py` nets out exactly as
the README's disclosures 6 and 7 do), but **no further paid compute is authorised** without a
new user instruction.

**No compute is running for this study.** `uvx modal app list` on 2026-09-23 shows the four
apps of this study that are still in the listing (`ap-0hnxop3LzUZTvjE5JJdMkF`,
`ap-1ZusaytP70O7Pbki94t2ex`, `ap-nu1UHXTwhxcwXNRrJDQYmg`, `ap-05G2istJ7nxnSiZ0hY61iG`) in state
`stopped` with 0 tasks. The study created **eight** apps in all, and all eight carry
`verified_stopped: true` in `budget-ledger.json` with matching `logs/*-stopped.json` evidence.
The list also contains apps belonging to **other, unrelated** studies — `rotation-ob…`
(two ephemeral, with running tasks), `aminist21-o…`, `kimi-k3-con…` and the deployed
`sutro-mnist…` submission site. **Do not stop them.**

**Not finished:**

- `research/whitening-deterrent.md` (503 lines) is the consolidated whitening/rotation write-up, with scripts under `research/whitening-*-scripts/`; the README's whitening section summarises it. Its two headline findings: a public-pool re-identification attack inverts ANY fixed invertible transform (also the current permutation) in seconds, so topology secrecy is a rule, not a property of the data; and a minimum-volume cone fit recovers most pixel axes from rotated/whitened data without external data.
- `analysis/final-ladder-retuned-tq/` is an empty directory: `analysis.py` builds a five-level
  ladder and that candidate has only N=1,000, so no per-candidate summary exists for it. Its
  numbers live in `analysis/ladders.json` under `candidates["ladder-retuned-tq@1000"]`.
- `analysis/dev-preview/dev_table.md` predates the GPU rounds and covers the CPU classical
  candidates only. Regenerate a complete one into a **new** directory with
  `/tmp/pmnist-env/bin/python analysis.py --dev --scores results/scores_dev.json --out analysis/dev`
  if you need it; the README's GPU dev tables were computed directly from
  `results/scores_dev.json`.

## What another agent should do first

1. **Read [README.md](README.md), then `protocol.json` and `selection.json`.** The selection rule
   was frozen before any GPU development result existed; every claim in the report traces to
   `results/scores_final.json`, `results/scores_dev.json`, `analysis/*`, `budget-ledger.json` or
   `protocol.json`. Do not re-derive a number by re-running a fit.
2. **Do not launch paid compute.** There is no outstanding job. The study's question is answered;
   any new fit is a new study and needs a new user authorisation and a new ledger reservation.
3. **Read the ladders figure before quoting it.** On log-log axes Ladder B and the spatial
   reference overlap from 1,778 upward; that is the honest visual form of the +0.04 pp paired
   result, and prose must not claim the recovered-topology ladder beats the true grid.
4. **Reproduce cheaply if you need confidence**: `study.py --prepare` verifies the pool arrays
   against the data manifest, `python -m unittest test_study test_learners test_classical
   test_topology test_analysis test_whiten` runs the test suite, and `ladders.py
   results/scores_final.json` rebuilds `analysis/ladders.{md,json}` from the scored counts — all
   on CPU, all free. Note that `ladders.py` **overwrites** those two files; they are derived
   artefacts, so that is safe, but diff the result rather than assuming.
5. **Never read `raw/pool_labels.npy` or any label array directly.** Every number in the report
   is available from the scored JSON. `score.py` enforces the stage/seed binding; a throwaway
   script bypasses it and contaminates the study (this has already happened three times —
   disclosures 1 and 2 and the whitening pilots in `research/`; only the first touched a final
   seed).

## Optional follow-ups

None of these is running, scheduled or promised. Freeze the scope before producing new scores.

1. **VAT with standardized inputs.** The four round-1 VAT arms collapsed because of the
   `4x - 0.5` noise-unit error, and the corrected version was dropped for budget. It is the
   largest untested cell in the dense sweep and the most plausible challenger to the kernel at
   N=10,000, since it is the one arm that would use the 10,000 unlabelled query images the way
   the transductive Ladder does. Budget: ~8 GPU fits for two dev seeds at two levels.
2. **A second development seed for the GPU candidates.** The frozen rule allowed it and the
   budget did not. Differences under ~0.2 pp at N=10,000 among the GPU arms are currently
   unresolved, including the 2.54% / 2.55% ordering of the two Ladder variants.
3. **The Ladder network at the middle levels (1,778 / 3,162 / 5,623).** Ladder A's first step is
   only 1.104x because the 1,000 rung uses a different recipe from the rest. Measuring
   `ladder-retuned-tq` at 1,778 and 3,162 (~320-900 s per fit, 22 fits) would tell us whether
   the mixed ladder should extend further up, and would make the A ladder's spacing honest.
4. **Adopt a rotation in the protocol, if the permutation is meant to bind.** The research says
   release `Q (x - mean)` with a published Haar seed and no whitening: free for the RBF kernel,
   +0.14 to +0.23 pp for the arc-cosine kernel, and it puts the second-order topology attack at
   chance. Adopting it changes the task and would need a fresh dense sweep plus a restated
   ladder; it should also ship with a stated bounty on recovering k pixels, because the
   non-negativity/facet attack is unresolved.
5. **Improve the facet attack** — proper deflation and a joint refit. Section 6 of
   `research/whitening-attack-results.md` already measures the current version end to end at
   49.66% (N=1,000) and 46.07% (N=10,000) against 10.58% and 4.44% for no attack, i.e. far worse
   than doing nothing: the leak is real at the level of individual pixel functionals (a dozen
   pixels at |r| > 0.99) but has no path yet to accuracy. Also try topographic ICA
   (Hyvarinen-Hoyer-Inki) before claiming any rotation-based deterrent is safe.
6. **Energy measurement.** Nothing here is measured in joules, so no MNIST-medium leaderboard row
   can be filled from this study. A PCA-QDA-style NVML measurement of the winning kernel and of
   the topo-CNN would be needed, and the kernel currently runs on CPU.
7. **Publish.** If this becomes a repo page or a leaderboard row, keep the query-set caveat
   attached: these 10,000 queries are held-out rows of the official *training* split, not the
   official test split.

## Hard rules

- **Never edit** `results/`, `predictions/`, `protocol.json`, `selection.json`,
  `authorization.json`, `budget-ledger.json` or anything under `plans/`. They define the
  executed, frozen experiment. Editing a learner module invalidates the results by design:
  `runner.py` compares recorded SHA-256 of `learners.py`, `ladder_model.py`,
  `spatial_learner.py` and `topology.py` before reusing any result and re-runs the job if they
  differ.
- **Never read query labels outside `score.py`.** No throwaway scripts against
  `raw/pool_labels.npy`.
- **Never launch Modal, or any paid compute, without a fresh user authorisation** recorded in a
  ledger reservation. The existing $3.98 of available allowance under the cap is not an
  authorisation.
- **New work goes under `runs/` or a new dated directory**, never on top of this one. Copy what
  you need; leave the record intact.
- Do not stop, modify or inspect other sessions' Modal apps.
