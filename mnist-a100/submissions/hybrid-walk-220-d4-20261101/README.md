# Fisher-PCG hybrid with a without-replacement row walk (220 updates): difficulty 4

## Independent reviewer reproduction — November 1, 2026

The exact submitted source passed its first fixed group of **3/3 fresh canonical sandboxed Modal A100-80GB runs**, with valid energy in **3/3**. Median ranked time: **22.069657 ms/call**; median GPU-board energy above idle: **2,259.293532 mJ/call**. All three complete runs separately passed the MNIST accuracy, foreign-data and timing gates; pooled MNIST error was **2.176061%** (descriptive). Optional energy acceptance is assessed separately from score qualification.

| Reviewer run | MNIST error | Margin, labels | Ranked ms | Above-idle mJ |
| - | -: | -: | -: | -: |
| 1 | 2.260000% | +44 | 22.069657 | 2287.351889 |
| 2 | 2.074545% | +248 | 22.018177 | 2259.293532 |
| 3 | 2.193636% | +117 | 22.280011 | 2228.175903 |

Source SHA-256: `02755ca6031242b0ef29ded5ba2f4f78ee7e592e120694edb3a4d2ed65f9d010`. The scorer and Modal runner are unchanged from main `09979ba2ce5bba4ef80432ec5004b27686541bdf`; CPU, memory, sandbox, environment and draw-generation defaults are unchanged. Source bytes and harness hashes were frozen before launch. Random fresh draws are generated inside the scorer; the records do not contain replayable draw seeds. All initial outcomes, logs and energy windows are retained; no failed-run replacement or recipe tuning was performed. An external 1,200-second group deadline and owned-app cleanup guarded resource use without changing scoring.

Energy excludes compilation and host/system energy. Host type and draw variation limit comparisons with earlier groups. The author chose the update count using the supplied 24 split diagnostics; those results describe paired development tests, not an untouched confirmation set. The reported ±0.005 percentage points is a standard error, rather than a 95% confidence interval. Their JSONL files do not pin the executed source hash, so their attribution and ablation claims remain author evidence. [Source review](evidence/reviewer-20261101/source-review.json), [author-record reconstruction](evidence/reviewer-20261101/author-audit.json), [fresh independent audit](evidence/reviewer-20261101/fresh-independent-audit.json), and [raw records plus exact reconstruction](evidence/reviewer-20261101/) document the scope.

Saved-record check from this directory: `python evidence/reviewer-20261101/verify_records.py evidence/reviewer-20261101 submission.py --difficulty 4 --function hybrid_d4 --output /tmp/pr142-verification.json`.

## Original author report (as submitted)

The dated reviewer section above records independently reproduced results; the original figures below remain author evidence.


Entrant: [@spartanhaden](https://github.com/spartanhaden).

This is @sjbaebae's merged [codesign D4](../codesign-d4-d5-20261024/README.md) (#140) with work taken out of it. The Fisher-PCG teacher, the three width-128 tangent students, 220 updates and the HKNN readout are unchanged in structure. Nearly every change below removes something.

Three sandboxed Modal A100-80GB runs of the unchanged scorer (`mnist-a100/1.2.0`, stock `run_modal.py --difficulty 4 --runs 3`) all passed. **Median ranked time: 21.636 ms; median energy: 2,249 mJ per call.**

| Run | GPU | MNIST error | Margin (labels) | Hold-out | Ranked ms | mJ per call |
| ---: | --- | ---: | ---: | --- | ---: | ---: |
| [1](evidence/official/run-1.json) | A100-SXM4-80GB | 2.205% | +104 | fashion 88.58% | 21.443 | 2,248.7 |
| [2](evidence/official/run-2.json) | A100-SXM4-80GB | 2.110% | +209 | fashion 88.47% | 22.072 | 2,223.9 |
| [3](evidence/official/run-3.json) | A100-SXM4-80GB | 2.085% | +236 | KMNIST 96.88% | 21.636 | 2,316.6 |

The current record, #140's reviewer median, is 25.3 ms and 3,347 mJ. This entry is 1.17× faster and uses 33% less energy, though the host mix differs. A paired in-process test on one GPU type isolates the change:

| 24 fresh pool splits × 11 calls, A100-SXM4-80GB | Mean error | Split sd | Worst split | Splits over 2.30% | Median ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| #140 D4, unchanged | 2.196% | 0.075 | 2.368% | 2 | 23.5 |
| this file | 2.162% | 0.067 | 2.303% | 1 | 18.8 |

The paired difference is −0.034 ± 0.005 points (23 of 24 splits better) at −20% time. Band minus mean is 2.1 split sds, up from 1.4. Split *s* is a `mnist.Pool` from `default_rng(1031000 + s)` with its 11 draws ([records](evidence/fresh-splits/confirm-seed1031000.jsonl), [runner](evidence/fresh-splits/runner_modal.py)).

`submission.py` is 20,429 bytes, SHA-256 `02755ca6031242b0ef29ded5ba2f4f78ee7e592e120694edb3a4d2ed65f9d010`, matching `source_sha256` in all three records ([manifest](evidence/official/manifest.json)). It passes `check_source`.

## Method

Time is paired on one host at each step of the stack. Error comes from the replica described below, on the development splits.

| Change from #140 | Time | Error (points) |
| --- | ---: | ---: |
| Neighbour search within each class (10 × 2,400²) instead of all pairs | −1.6 ms | level |
| Rows from a walk `(a·j + b) mod n`, without replacement, with fresh `a` and `b` per student and chunk | level | −0.017 ± 0.005 |
| No isotropic input noise, so each row draws only its 4 tangent coefficients | −0.7 ms | −0.000 ± 0.005 |
| No cross-teaching | −0.5 ms | −0.007 ± 0.006 |
| 256 unlabelled rows per batch instead of 1,024 | −1.3 ms | +0.001 ± 0.006 |
| bf16 storage for the input and tangent tables | −0.5 ms | together below |
| Two tangent power iterations instead of four | not timed alone | together below |
| 1,280 labelled rows, learning rate 0.024, Adam β₂ 0.88 | about +0.4 ms | −0.036 ± 0.004 for these three rows |

Five other cross-teaching schedules, and none at all, moved error by at most 0.011 ± 0.006 points. Tangent jitter is what matters: removing it costs 0.19 points.

## Development

- Error screens ran on a plain-PyTorch replica of the kernel's arithmetic on an RTX 4090, on 12 splits of seed 1030000. Replica minus kernel was +0.002 ± 0.007.
- Timing came from the Triton file on Modal, each batch sharing one host.
- The update count was chosen on the confirmation splits above. On the replica, 180, 200 and 220 updates gave 2.187%, 2.173% and 2.158%. Only 220 reaches band − mean ≥ 2 sd.
- No gain: weight decay, prediction averaging, label smoothing, orthogonal initialization, an extra radial input, warm-up, one-hidden-layer students and larger batches (slower). Two students buy error at about the same rate per millisecond as more updates.

## Credits

- Everything not listed above is #140 by @sjbaebae, which derives from our [#138](../hybrid-fisher-triton-220-d4-20261020/README.md). See their READMEs for the lineage; the source's credit lines are kept.

Co-developed with Claude.
