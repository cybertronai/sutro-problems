# Submission Requirements and Pre-PR Review

This checklist maps the package to the historical rules in `mnist/instructions.md`.
The updated `mnist/README.md` explicitly supersedes those rules: current small
uses 1,000/1,000 examples, a 67% target and the spatial-computer grid model.
This is a historical study, not current qualification or maintainer acceptance
of the affine representation. The new row belongs only in the historical table,
with spatial-grid metrics left unmeasured.

## Required Contents

| Requirement | Package evidence |
|---|---|
| Solution under `mnist/submissions/<name>/` | This directory |
| Source or generator | `learner.py`, `panels.py`, `reference.py`, selected config and executable IL |
| Training, prediction and scoring reproduction | `README.md`, `prepare.py`, `score.py` |
| MNIST-small only, 600/600, 3x3 | Input checks, manifests, fixed learner |
| Eleven predeclared dataset draws | `evidence/accuracy/plan.json` |
| Fresh training on each draw; fixed learner randomness | Seed 101, reset parameters, no transferred state |
| Predictions frozen before test-label scoring | `predictions_frozen.json`, timestamps and hashes |
| Mean accuracy and sample SD across draws | 65.0% +/- 2.1 pp; exact 4292/6600 |
| All per-draw counts, seeds, manifests and predictions | `evidence/accuracy/` |
| Time, Energy, Area, Time to score | `costs.json`, report headline |
| Actual A100 time and idle-adjusted NVML energy | `evidence/gpu/results.json` |
| Hardware/software versions and measurement scope | Report and raw GPU metadata |
| Two significant figures for costs; one decimal for accuracy/SD | Report and proposed MNIST table row |
| Standalone report link | `report.md` and local HTML preview |
| Contributors and prior-work credit | SecurityQQ, OpenCode and credited prior work |
| W&B runs | None created; stated explicitly |
| Matching MNIST results-table row | Local change to `mnist/README.md` |

## Important Caveats

- The affine-v4 scorer is the repository's research prototype. Its arithmetic,
  tape and scratch-area conventions remain subject to maintainer review.
- Full-program costs are static exact access counts. Actual IL interpretation
  covers bounded complete training/inference cases, not all 641 million steps.
- The GPU implementation adapts panel reuse through SIMD/registers. It is not a
  literal physical mapping of Dally distances or serial memory lifetimes.
- The primary GPU run uses original-style separate aligned tensors. Preliminary
  misaligned-buffer results and both compilation failures are retained separately.
- Fresh submission accuracy is 65.0% +/- 2.1 pp on seeds 2026091301..2026091311.
  Earlier exploratory accuracy on different seeds is not the headline result.
- Dataset resizing uses float32 matrix operations; byte-exact reproduction on
  another BLAS/CPU may differ. Do not silently bypass manifest validation.
- Model-energy optimization does not guarantee GPU improvement. The supporting
  `no_slowdown` model profile loses on A100; it is not promoted as the headline.
- No test-label array is used by the learner. The separate dataset preparer and
  scorer necessarily have access to the public benchmark labels.

## Before Opening a PR

- Contributor handle verified through the authenticated GitHub account: SecurityQQ.
- Review the proposed table row and standalone report.
- Run `verify_submission.py --full` and inspect the saved verification output.
- Include only this package, its static report page, and the intended MNIST row.
- Exclude ignored generated inputs/checkpoints, virtual environments, unrelated
  untracked research directories, worker transcripts and local configuration.
- Inspect `git status`, `git diff`, and the eventual base-branch diff.
- Decide whether to include `PR_DRAFT.md` as a repository document or use it only
  as the future PR description.
- Commit/push/open a PR only after explicit authorization; the user has now authorized a PR.
