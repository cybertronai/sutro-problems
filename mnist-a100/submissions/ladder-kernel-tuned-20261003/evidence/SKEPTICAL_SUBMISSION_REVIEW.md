# Skeptical submission review

Reviewed locally on 2026-10-04. Target: `ladder_d3.py:classify`, difficulty 3.

## Verdict

**Source legitimacy: PASS on the inspected source and canonical rules.** I found no test-label access, embedded trained prior, retained learned state, timer/energy manipulation, dataset-dependent shortcut, or omitted training/prediction work that would justify rejection.

**Readiness for submission: GO, with a material accuracy caveat.** The frozen source, three saved canonical results, reproducible local verification command, attribution and report form a coherent submission package. **Concrete blockers found: none.** This is readiness to submit for maintainer review, not acceptance, an independent GPU reproduction, or a guarantee that a fresh run passes. The third run passes its mean error gate by only 52 correct labels.

I reached this assessment without reading the existing independent review. I deliberately tried to invalidate the source, qualification records and reported numbers before arriving at the verdict.

## Findings and residual caveats

1. **Medium — accuracy qualification is fragile.** [README.md:66](../README.md#L66), [summary.json:51](official/summary.json#L51), [mnist.py:1199](../../../mnist.py#L1199). Run 3 has 107,082/110,000 correct; the 2.70% band requires 107,030. Its margin is 0.047273 percentage points, or 52 labels. The three mean errors span 2.421818% to 2.652727%, a 0.230909-point range, nearly five times this final margin. All three pass individually, but pooling them into 2.546061% does not strengthen a later run's own gate. Draws share each run's dataset split, and repeated development fits use the same draws; treating all image decisions or fits as independent trials would exaggerate confidence. Three runs and one chosen learner seed cannot establish a reliable future failure probability. This is disclosed correctly and is a residual risk, not a failure of the saved qualification.

2. **Low — the packaged verifier is narrower than its opening description suggests.** [verify_records.py:5](../verify_records.py#L5), [verify_records.py:25](../verify_records.py#L25), [verify_records.py:66](../verify_records.py#L66), [verify_records.py:84](../verify_records.py#L84). It reconstructs judge/energy values, checks source hash and counts, and adds UUID distinctness. It does not run `check_source`, enforce the entry point or scorer version, compare scorer/runner hashes, or validate `summary.json` aggregates. The entry point is recorded as a fact without a corresponding check. With `--source`, it compares the source to the manifest hash but does not check the manifest's embedded source string. These omissions do not invalidate this package: I performed those checks separately and they pass. Its success message should be read as saved-record reconstruction within those limits, not a complete legitimacy certificate.

3. **Low — source has little size headroom.** [ladder_d3.py:1](../ladder_d3.py#L1), [README.md:64](../README.md#L64), [mnist.py:763](../../../mnist.py#L763). The exact file is 20,194 bytes, only 286 below the 20,480-byte canonical cap. It passes today; additions to the candidate, including comments, can invalidate this result or change its hash. The scorer's size review flag is advisory and does not reject the current file.

4. **Informational — board variation limits the comparison's interpretation.** [README.md:27](../README.md#L27), [README.md:11](../README.md#L11), [summary.json:17](official/summary.json#L17). The three official energy devices are distinct A100-SXM4-80GB boards, with recorded power limits of 500/400/400 W. All pass canonical energy telemetry checks; the rules do not require identical limits or distinct UUIDs. The 19.83% time and 24.97% energy reductions compare separately measured, rounded published baseline values. They are valid leaderboard comparisons, not a paired isolation of kernel tuning. The README discloses both the separate-run comparison and the 900-to-800 step reduction.

5. **Informational — cost, cleanup and remote provenance are not independently established here.** [README.md:88](../README.md#L88), [README.md:92](../README.md#L92), [summary.json:87](official/summary.json#L87), [runs.log:147](official/runs.log#L147). Saved logs identify the qualification app and show its completion/stop. They do not independently prove the claimed 386.081-second controller lifetime, settled billing, that every related app was stopped, or the final live container inventory. The stated cost formula gives $1.173243, consistent with the rounded $1.1732 estimate. Those administrative claims are ancillary to canonical eligibility. I did not query Modal, launch work, or authenticate the remote history.

## Independent source audit

- Inputs route from `classify` into copied training labels and the combined **feature-only** training/test reconstruction pool ([ladder_d3.py:383](../ladder_d3.py#L383)). Supervised loss uses `y_labelled`; reconstruction uses supplied unlabelled features ([ladder_d3.py:247](../ladder_d3.py#L247)). There is no file/dataset fetch, image identity lookup, label permutation lookup, or test-label path. Transduction matches the accepted ancestor and is not prohibited by the current rules.
- The only module state is constants, allowed torch flags and `CACHE`. `os.environ.setdefault` selects a writable Triton compile-cache directory; it does not read scoring data. No timer, event, NVML, subprocess, network, frame inspection or scoring-function hook appears in the candidate.
- `Trainer` construction allocates buffers and warms/captures a full update on dummy buffers ([ladder_d3.py:272](../ladder_d3.py#L272)). Every real call overwrites all input/label buffers and calls `initialise` before replay. That resets every encoder/decoder/combinator/beta/gamma parameter, Adam moments and step, scores, counter and unlabelled indices, and seeds the CUDA RNG ([ladder_d3.py:333](../ladder_d3.py#L333)). The learning-rate tensor is overwritten from the immutable schedule in each step. Calibration statistics are recomputed locally for prediction. The fused-kernel RNG seed depends on the reset counter and layer, not labels or earlier learned state.
- Captured gradient buffers are reused through the standard capture pattern: gradients are set to `None` before capturing backward. The source has no explicit retained-gradient accumulation path. Graph/RNG behavior still depends on the supplied PyTorch CUDA implementation; this audit did not execute it or inspect GPU state after foreign warmup.
- For the canonical 10,000 training rows, `steps_per_epoch=10`, `epochs=80`, and `total=800`. All calls execute all 800 graph replays and clean prediction. Kernel width/batch specialization changes launch geometry, with row/column masks preserved. Fixed 60-dimensional inputs, ten outputs and batch 1,000 are compatible with the published API. General support for arbitrary training sizes is not promised or required; the ragged kernel checks do not prove arbitrary-size end-to-end training.
- A full diff against the current accepted sampled-Ladder source confirms the disclosed changes: 900→800 steps, forward tiling, decoder gradient-store order, backward launch geometry, attribution and entry-point rename. No undisclosed objective, prediction or optimizer change was found.

## Rule freshness and identity

`git ls-remote origin refs/heads/main` independently returned `aa6c51e6ad661089cbf71a39c95c577c216b627f`, matching local `origin/main`. I read the README from that revision rather than relying on the working-tree README. Its difficulty-3 best remains 1,443.3 ms and 201,792 mJ. Local scorer and runner bytes exactly match their upstream versions.

| File | Recomputed SHA-256 |
| --- | --- |
| Candidate | `5052f86e9cdf6086783bd64c90a31fcb86b3d19d1d0f27ef8f21457923c713e1` |
| `mnist.py` | `1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064` |
| `run_modal.py` | `1180b843cf6c32414774da90c835a8b7298d2db8719989237a793ca1c62a10ab` |
| Accepted sampled-Ladder ancestor | `0c11521f8ef34e4356ac335160fc1537f16e6805dbbeba9870646614a869452c` |

The candidate matches the manifest's embedded source **byte for byte**, all record hashes, and both final development snapshots. The canonical source/import gate passes. Its only review flag is size proximity.

## Saved canonical results recomputed

I separately checked record identity/version/entry point, successful return codes, sandbox flags, 11 MNIST plus four correct holdout calls per run, all scalar/count consistency, and the absence of recorded problems. I invoked current `judge` and `energy_summary`, compared the entire reconstructed energy object after normalizing JSON tuples/lists, and independently recomputed dataset mean time floors and idle/control-subtracted energy arithmetic. All match exactly. The packaged verifier also passes unchanged.

| Run | Ranked ms | Energy mJ | MNIST correct / total | Mean error | Worst error | Holdout accuracy | Mean-gate surplus |
| ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |
| 1 | 1,157.137595 | 151,759.932481 | 107,336 / 110,000 | 2.421818% | 2.62% | KMNIST 96.1650% | 306 labels |
| 2 | 1,156.458740 | 151,403.970125 | 107,180 / 110,000 | 2.563636% | 2.77% | KMNIST 96.0325% | 150 labels |
| 3 | 1,160.505341 | 149,399.706870 | 107,082 / 110,000 | 2.652727% | 2.81% | Fashion 87.5425% | 52 labels |

The record score is the **slower dataset's mean**, including the parent-clock floor, not a median of the 15 calls. The submission score is the **median of the three run scores**, 1,157.137595 ms; its separately aggregated energy median is 151,403.970125 mJ. Total MNIST correctness is 321,598/330,000, yielding 2.546061% error. All relevant `summary.json` values and README rounding agree.

The difficulty-3 mean gate is 2.70%, but its per-draw gate is **4.20%**, or at least 9,580 correct. Thus the saved 2.77% and 2.81% individual draws are valid. Holdout accuracy, timing dispersion, per-call and grouped parent-clock gates pass. All saved calls are below the 60-second limit. Canonical worker output validation happens during execution; saved records do not preserve original prediction tensors for a new output audit.

All three energy windows contain 18 complete calls over at least 20 seconds. Minimum energy-window correctness is 9,721/9,716/9,703. The exact reference products pass; reference readings are 7.586/7.843/7.963 J/TFLOP at 18.349/18.312/18.100 TFLOP/s, inside the A100 bands. Idle utilization/context checks, positive windows, energy-call timing consistency and positive net energy pass. **Energy failure would leave the energy column empty; it is not a canonical time-score disqualification** ([mnist.py:65](../../../mnist.py#L65)). The three distinct UUIDs are useful evidence and an extra verifier condition, not a canonical rejection rule. Canonical remote containers are single use, but distinct physical boards are not thereby guaranteed.

## Development evidence and portability

Both paired profile files' source hashes recompute correctly; each final snapshot is the exact packaged candidate. Their configuration order is reversed between SXM4 and PCIe. Eight unique split/draw pairs, three fits each, give 24 fits per variant. Their medians are 1,188.596008 versus 1,155.406128 ms; mean errors are 2.482917% versus 2.445833%, matching the report. All recorded comparison checks pass; maximum kernel relative L2 is `1.440240140482274e-7`, and maximum whole-loss/gradient discrepancy is `0.0001618572132429108`. These are recorded comparisons, not comparisons I executed; they were not canonical premeasurement rejection gates.

The warp-sweep snapshot also hashes correctly and is a distinct intermediate source. Final launch choices and results are instead anchored by the exact final snapshots and official hash. The final profile's GEMM/split-K sum is 3,085.120 µs of 7,291.719 µs, or 42.309913%; custom-family durations, scaled columns and stage times reconcile with the README. Development records mark `official=false` and are not substituted into ranked results.

The candidate is self-contained apart from torch/Triton supplied by the official image. The saved-record verifier needs the repository's `mnist.py` and standard Python; it ran successfully here on Python 3.10.15 without importing the candidate or requiring a GPU. All relative README artifact links resolve. Its documented canonical reproduction command is valid against current `run_modal.py`; it intentionally launches fresh paid work, which I did not run. The additional qualification controller's concurrency/timeout overrides are disclosed, but its full orchestration script and billing/cleanup evidence are not packaged. They do not alter the scorer's gates or prevent canonical reproduction.

The package explicitly says it is local and unaccepted ([README.md:5](../README.md#L5)); summary flags also say `submitted=false` and `maintainer_accepted=false`. Nothing examined falsely claims publication or acceptance.

## Audit boundary

This review used only local reads, standard-library calculations, canonical saved-record reconstruction and a read-only upstream ref query. I changed only this new review file. No GPU/Modal experiment, candidate change, raw-record change, publication, push, PR or external message was performed. Hash consistency binds the included artifacts together; editable local JSON and logs are not an independent attestation of their remote origin. Maintainer reproduction and source judgment remain necessary for acceptance.
