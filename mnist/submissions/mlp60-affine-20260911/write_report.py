"""Render human units from exact saved evidence; never rerun measurements."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/name).read_text())


def sig(value):
    return format(Decimal(format(float(value), '.1e')), 'f')


def main():
    model, cpu, gpu = read('model-score.json'), read('cpu_results.json'), read('gpu_results.json')
    verification, accuracy = read('verification.json'), read('accuracy.json')
    assert verification['all_passed'] and accuracy['meets_accuracy_target']
    assert accuracy['correct'] == 374 and accuracy['total'] == 600
    assert gpu['source_sha256'] == hashlib.sha256((HERE/'gpu_benchmark.py').read_bytes()).hexdigest()
    assert gpu['prediction_sha256_int64_le'] == cpu['prediction_sha256_int64_le']
    assert gpu['protocol']['target_active_seconds_per_trial'] == 10
    assert gpu['validation']['after_all_timed_graph_replays']['score_bitwise_matches'] == 6000
    assert gpu['validation']['canonical']['scores_sha256_float32_le'] == cpu['output_scores_sha256']
    trials = gpu['trials']
    for trial in trials:
        assert 0.8 < trial['cuda_event_duration_s']/trial['wall_duration_s'] < 1.2
    summary, versions, hardware = gpu['summary'], gpu['versions'], gpu['hardware']
    gpu_ms = summary['cuda_event_us_per_invocation']['mean']/1000
    gpu_mj = summary['idle_adjusted_j_per_invocation']['mean']*1000
    model_ms, model_mj, area = model['time_ms'], model['energy_mj'], model['area_mm2_occupied_cells']
    trial_rows = '\n'.join(
        f"| {t['trial']} | 62% (374/600) | {t['invocations']} | {sig(t['cuda_event_us_per_invocation']/1000)} | "
        f"{sig(t['idle_before']['average_power_w'])} / {sig(t['idle_after']['average_power_w'])} | "
        f"{sig(t['unadjusted_j_per_invocation']*1000)} | {sig(t['idle_adjusted_j_per_invocation']*1000)} | "
        f"{sig(t['before_only_adjusted_j_per_invocation']*1000)} / {sig(t['after_only_adjusted_j_per_invocation']*1000)} |"
        for t in trials)
    input_hashes = '\n'.join(f'- `{name}`: `{digest}`.' for name,digest in cpu['input_sha256'].items())
    source_hashes = '\n'.join(f'- `{name}`: `{digest}`.' for name,digest in model['source_sha256'].items())
    readme = (HERE/'README.md').read_text().split('## Reproduce from a clone\n',1)[1]
    text = f'''# MNIST-small: a submission for the 60% target

> **Accuracy scope:** These are historical results on one fixed dataset. The current small/medium rule requires **mean ± sample SD over 11 independently resampled datasets**; that aggregate has not been measured here. Threshold checks below describe the fixed dataset. Training-seed repeats do not supply across-dataset SD. [Current accuracy protocol](https://github.com/cybertronai/sutro-problems/blob/main/mnist/instructions.md#accuracy-over-11-random-datasets).

**374/600 correct (62%)** on the canonical 600-training / 600-test, 3 × 3
MNIST-small dataset. The fixed 32-unit network exceeds the per-draw diagnostic threshold
of **360/600 (60%)** by 14 correct predictions. Both the model scores and the
A100 measurements cover fresh training and all 600 predictions.

Contributors: Yaroslav Bulatov (task and benchmark requirements) and Codex
(implementation, experiments, verification, and reporting). No W&B run was
created for this submission; raw evidence is supplied below.

This is a submission attempt. The compact intermediate language and the
declared model conventions still require benchmark review. This particular
configuration was chosen for submission after the feasibility study's test
results were known; the configuration and seed themselves were fixed before
that study's test evaluation. See the separate ambiguity report for the
selection boundary.

[TOC]

## Comparable measurements

Human displays use two significant figures. **Execution time is in ms, energy
in mJ, area in mm², and time to score in s.** Exact measurements and primitive
counts remain in the linked JSON. The same fixed model and predictions appear
in both rows; area applies to the theoretical scratch-cell model.

| Execution | Accuracy | Time (ms) | Energy (mJ) | Area (mm²) | Time to score (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dally v4, complete compact program | 62% (374/600) | {sig(model_ms)} | {sig(model_mj)} | {sig(area)} | {sig(model['time_to_score_seconds'])} |
| A100, complete task; mean of three trials | 62% (374/600) | {sig(gpu_ms)} | {sig(gpu_mj)} | — | — |

The A100 time is {sig(gpu_ms/model_ms)} times the modeled serial time. Its
idle-adjusted board energy is {sig(gpu_mj/model_mj)} times the model energy.
These ratios compare the same learning task under different hardware and
accounting assumptions; they do not calibrate the theoretical model to silicon.
GPU energy includes the entire board's incremental energy, whereas the model
charges scratch accesses under its declared tape and arithmetic conventions.

## Fixed learning algorithm

The network is `9 → 32 ReLU → 10`, with 650 scalar FP32 parameters. Inputs use
`FP32(FP32(x * 4) - 0.5)`. Training uses squared error against one-hot labels,
300 epochs, contiguous minibatches of 30, no shuffle, and learning rate 0.2.
There are exactly 20 minibatches per epoch and 6,000 updates per task.

Seed 101 initializes PCG64. Input weights use a uniform distribution on
`[-1/3, 1/3]`; output weights use `[-1/sqrt(32), 1/sqrt(32)]`; biases start at
zero. The 650 initialization words are data-independent program constants.
Generating those constants is outside the task; writing/resetting every
parameter is included in both model and GPU execution.

For a batch, compute the hidden activations and linear outputs, then
`delta2 = output - target` and
`delta1 = (delta2 @ old_W2.T) * (hidden > 0)`. Sum gradients in increasing batch
index and subtract `FP32(0.2 / 30) * gradient` from each parameter. Backpropagation
uses the old output weights. Every multiplication and addition rounds separately
to FP32, with ascending-index reductions and no fused multiply-add. The final
prediction is the first maximum among the ten output scores.

The [feasibility study](../accuracy-il-20260911/) froze this configuration as
the best training-validation choice at a 300-epoch budget. Its three predeclared
seeds scored 374/600, 371/600, and 377/600. This submission uses the first seed,
101, without further tuning or selection among seeds. Choosing this budget for
the new target happened after those study results were public.

## Complete-task model and compact scoring

The program uses the single-core-with-tape model and v4 specification at commit
`{model['model_spec_commit']}`. It receives 11,400 tape words (all training
pixels and labels, then all test pixels), performs input and target preparation,
sets initial parameters and work buffers, executes every training update, and
sends 600 predicted labels. Tape reads and writes have the pinned model's zero
access charge; ordinary scratch operands retain all their charges.

The compact program has {model['static_nodes']:,} syntax nodes, including
{model['static_leaf_nodes']:,} primitive leaves. It represents
**{model['total_instructions']:,} primitive instructions** and
**{model['charged_accesses']:,} charged operand accesses**. Each fixed loop
contributes its exact multiplicity. Affine address histograms retain every
source read and destination write, including repeated or aliased operands.
There is no fitted training cost or free matrix operation.

Scratch storage occupies {model['peak_initialized_scratch_words']:,} words in
13 non-overlapping regions. At one square micrometre per word, the reported
area is **{sig(area)} mm²**. This is occupied cell area, not the placement's
enclosing rectangle, die area, or GPU allocation. Dividing µm² by 10⁶ changes
only the presentation units.

The five complete static scoring calls have median
**{sig(model['time_to_score_seconds'])} s**. This includes schema, bounds and
definite-initialization checks, placement, exact address histograms, integer
cost sums, and canonical hashing. It excludes JSON loading, numerical execution,
accuracy checking, and output files. CPU: Python {cpu['software']['python'].split()[0]},
NumPy {cpu['software']['numpy']}, `{cpu['software']['platform']}`.

The exact model uses integer 0.2-ps ticks and integer fJ internally. Conversion
to the display is `ticks / 5e9` ms, `fJ / 1e12` mJ, and `µm² / 1e6` mm².
The [language specification](../accuracy-il-20260911/il.html) explains the cost
equations, fixed half-diamond placement, initialization proof, and restrictions.

## Independent numerical verification

All 300 epochs were replayed using explicit, ordered FP32 reductions. All four
parameter arrays, every one of the 6,000 output scores, and all 600 predictions
match bit for bit. Parameter hashes and predictions also match the frozen study;
that study did not save its full output-score matrix.

Running the actual learner CLI with only `train_images`, `train_labels`, and
`test_images` in the input archive produces identical outputs. Changing every
training label to `(label + 1) % 10` changes learned parameters and 589 predictions.
The separate evaluator confirms the exact per-draw threshold, independent of rounded
display percentages. Test labels are never learner inputs.

| Separate host work | Accuracy checked | Time (ms) | Scope |
| --- | ---: | ---: | --- |
| CPU learner | 62% (374/600) | {sig(cpu['cpu_reference_seconds']*1000)} | Fresh normalization, initialization, training, prediction |
| Ordered 300-epoch replay | 62% (374/600) | {sig(verification['stage_wall_seconds']['ordered_300_epoch_replay_seconds']*1000)} | Independent reduction implementation |
| Complete CPU verification suite | 62% (374/600) | {sig(verification['total_verification_wall_seconds']*1000)} | Replay, hashes, allowed-input CLI, mutation, static rescore, accuracy |

The full 618-million-instruction program was not numerically interpreted after
expansion. The earlier IL tests execute small expanded MLPs and check lowering
semantics and address counts; the full ordered replay checks this learner's
numerical result. The distinction is recorded in [verification.json](verification.json).

## A100 implementation and measurement

The measured device was **{hardware['gpu_name']}**, with
{hardware['multiprocessors']} streaming multiprocessors, compute capability
{hardware['compute_capability']}, and board power limit {sig(hardware['power_limit_w'])} W.
It ran through Modal using a pinned container. Software: Python
{versions['python']}, PyTorch {versions['torch']}, Triton {versions['triton']},
NumPy {versions['numpy']}, CUDA runtime {versions['cuda_runtime']}, NVIDIA driver
{versions['nvidia_driver']}, NVML {versions['nvml']}, and
`nvidia-ml-py` {versions['nvidia_ml_py']}.

Container: `{versions['container_image']}`.

Eight Triton kernels implement input normalization, parameter/target
initialization, hidden forward, output/error, hidden error, parameter updates,
and the two inference stages. Each minibatch runs four kernels; every graph
replay includes all 24,000 minibatch kernel launches plus preparation and
inference. The graph is replayed for a complete fresh learning task each time.
GPU memory is reused, but learned parameters are reset and recomputed.

The manual GPU implementation is checked against the ordered CPU calculation
for all 650 parameter words, all 6,000 output-score words, and all 600 predictions
before and after timing. Changed queries and changed training labels also pass
independent reference checks. The exported PTX contains no FP32 fused multiply-add
or flush-to-zero instruction. This establishes agreement for the tested data;
it is not a proof for every possible input.

Each trial targets ten seconds of active graph execution. Idle power is measured
for three seconds before and after the active window, with a three-second
settling gap before each idle sample. CUDA events and host wall time both cover
the repeated graph replays. A consistency assertion catches unit mismatches. Calibration selected 122
complete tasks per trial; actual active intervals were about 8,700 ms, shorter
than the calibration target after execution warmed up.

NVML cumulative board energy is sampled at active-window boundaries. If the
paired idle powers are `P_before` and `P_after`, the per-task adjusted energy is
`1000 * (E_active_J - (P_before + P_after)/2 * active_seconds) / tasks`, in mJ.
The subtraction uses the NVML counter's active interval, not the CUDA duration.
No negative energy is clipped. Means below are unweighted across three trials.

| Trial | Accuracy | Complete tasks | Time (ms/task) | Idle before / after (W) | Raw energy (mJ/task) | Paired-idle energy (mJ/task) | Before-only / after-only energy (mJ/task) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{trial_rows}

Mean CUDA time is **{sig(gpu_ms)} ms**, with sample standard deviation
**{sig(summary['cuda_event_us_per_invocation']['sample_stddev']/1000)} ms**.
Mean host wall time is **{sig(summary['wall_us_per_invocation']['mean']/1000)} ms**.
Mean idle-adjusted energy is **{sig(gpu_mj)} mJ**, with sample standard deviation
**{sig(summary['idle_adjusted_j_per_invocation']['sample_stddev']*1000)} mJ**.
Mean raw board energy is **{sig(summary['unadjusted_j_per_invocation']['mean']*1000)} mJ**.
The separate baseline alternatives show sensitivity to the idle-power choice.
The third trial has a 2.3 W difference between its paired idle means; the
one-sided baseline energy envelope across trials is about 1,900–2,200 mJ/task.

Host/device transfers, allocations, JIT compilation, graph capture, validation,
cold start, and host CPU energy are excluded. These are GPU-resident steady-state
complete-task measurements. Repeated execution may benefit from cache residency.
The first run had a timing conversion bug and is excluded from every metric
above; the corrected second run supplies all results. The measurement history
retains that failure and its correction.

## Data and evidence fingerprints

The dataset is `competition-v2`, canonical seed 20260910. The 600 training and
600 test examples are disjoint subsets of the original MNIST training pool,
downsampled to 3 × 3. The manifest file SHA-256 is
`{verification['canonical_manifest_file_sha256']}`.
Allowed array SHA-256 values (C-order, little-endian content):

{input_hashes}

Prediction content SHA-256: `{cpu['prediction_sha256_int64_le']}`.
Output-score content SHA-256: `{cpu['output_scores_sha256']}`.
Canonical IL hash: `{model['program_sha256']}`.
GPU source SHA-256: `{gpu['source_sha256']}`.

Scoring source fingerprints:

{source_hashes}

Additional source, parameter, and PTX fingerprints are recorded in the raw
evidence. The program-file byte hash differs from the canonical IL hash because
the canonical hash normalizes JSON formatting.

## Reproduction

{readme}

## Download the evidence

- [CPU learner](learner.py), [CPU results](cpu_results.json), [predictions](predictions.npy), [scores](output-scores.npy), [parameters](parameters.npz), [accuracy](accuracy.json).
- [Compact program](program.il.json), [score driver](score.py), [exact model score](model-score.json).
- [Independent verifier](verify.py), [verification evidence](verification.json).
- [GPU benchmark](gpu_benchmark.py), [GPU results and trial telemetry](gpu_results.json), [GPU verification](gpu_validation.json), [GPU predictions](gpu_predictions.npy), [measurement history](gpu_run_history.txt).
- [Separate ambiguities and problems](ambiguities.html), [human-readable session export](session.html).
'''
    (HERE/'report.md').write_text(text)
    print(HERE/'report.md')


if __name__ == '__main__':
    main()
