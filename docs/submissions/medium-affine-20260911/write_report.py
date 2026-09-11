"""Make the standalone report from saved measurements and exact counts."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/name).read_text())


def sig(value):
    return format(Decimal(format(float(value),'.1e')),',f')


def main():
    config=read('config.json')
    cpu,model,gpu=read('cpu_results.json'),read('model-score.json'),read('gpu_results.json')
    accuracy,goal=read('accuracy.json'),read('goal-status.json')
    search=read('validation_results.json')
    verification=read('verification.json')
    host=read('host.json')
    assert gpu['source_sha256']==hashlib.sha256((HERE/'gpu_benchmark.py').read_bytes()).hexdigest()
    assert gpu['prediction_sha256_int64_le']==cpu['prediction_sha256_int64_le']
    assert gpu['model_config']==config==cpu['configuration']
    assert gpu['config_sha256']==cpu['config_file_sha256']==hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest()
    assert gpu['input_sha256']==cpu['input_sha256']
    for name in ('canonical','captured_task','after_all_timed_graph_replays'):
        validation=gpu['validation'][name]
        assert validation['prediction_matches']==6000
        assert validation['score_bitwise_matches']==60000
        assert validation['parameter_bitwise_matches']==(81+11)*config['width']+10
        assert validation['scores_sha256_float32_le']==cpu['output_scores_sha256']
    assert accuracy['correct']==goal['correct'] and accuracy['total']==6000
    assert verification['all_passed']
    for key in ('width','epochs','learning_rate','seed','features','n_train','n_test'):
        assert model['configuration'][key]==config[key]
    assert model['configuration']['batch']==config['batch_size']
    for record in (cpu,model,verification):
        for name,digest in record['source_sha256'].items():
            assert hashlib.sha256((HERE.parents[2]/name).read_bytes()).hexdigest()==digest,name
    correct=accuracy['correct']
    pct=sig(100*correct/6000)
    achieved=f'{pct}% ({correct:,}/6,000)'
    user_status='meets' if goal['user_target_met'] else 'does not meet'
    repo_status='meets' if accuracy['meets_accuracy_target'] else 'does not meet'
    summary=gpu['summary']
    gpu_ms=summary['cuda_event_us_per_invocation']['mean']/1000
    gpu_mj=summary['idle_adjusted_j_per_invocation']['mean']*1000
    rows=search['rows']
    groups=sorted({(row['width'],row['learning_rate']) for row in rows})
    valrows=[]
    for width,rate in groups:
        finite=[row for row in rows if row['width']==width and row['learning_rate']==rate and row['status']=='finite']
        if not finite:
            valrows.append(f'| {width} | {rate} | — | Nonfinite training | — |')
            continue
        best=min(finite,key=lambda row:(-row['validation_correct'],row['epochs']))
        valrows.append(f"| {width} | {rate} | {best['epochs']} | {sig(100*best['validation_correct']/1200)}% ({best['validation_correct']}/1200) | {sig(100*best['training_correct']/4800)}% ({best['training_correct']}/4800) |")
    trialrows=[]
    for trial in gpu['trials']:
        trialrows.append(f"| {trial['trial']} | {achieved} | {trial['invocations']} | {sig(trial['cuda_event_us_per_invocation']/1000)} | {sig(trial['unadjusted_j_per_invocation']*1000)} | {sig(trial['idle_adjusted_j_per_invocation']*1000)} | {sig(trial['before_only_adjusted_j_per_invocation']*1000)} / {sig(trial['after_only_adjusted_j_per_invocation']*1000)} |")
    trials='\n'.join(trialrows)
    valtable='\n'.join(valrows)
    versions=gpu['versions']
    configtext=f"81 → {config['width']} ReLU → 10; {config['epochs']} epochs; learning rate {config['learning_rate']}; batch size 30; seed 101"
    gap_user=goal['user_required_correct']-correct
    gap_repo=accuracy['required_correct']-correct
    status_details=(f'The attempt is {gap_user} correct predictions short of your 98% goal and {gap_repo} short of the repository threshold.'
                    if gap_user>0 else f'Threshold status is evaluated using exact counts; your goal requires 5,880 correct and the repository threshold requires 5,889.')
    hashes='\n'.join(f'- `{key}`: `{value}`.' for key,value in cpu['input_sha256'].items())
    reproduction=(HERE/'README.md').read_text().split('## Reproduce from a clone\n',1)[1]
    text=f'''# MNIST-medium: an attempt at 98% accuracy

The fixed network achieves **{correct:,}/6,000 correct ({pct}%)** on the canonical
MNIST-medium split. It **{user_status} your 98% goal** and **{repo_status} the
repository's 98.14% requirement**. {status_details}
This report records the attempted solution and its measured performance.

**Frozen algorithm:** {configtext}. It learns from all 6,000 supplied training
examples and predicts all 6,000 test examples. Configuration selection uses
only a training-data validation split; the final seed and configuration were
frozen before the test labels were opened.

Contributors: Yaroslav Bulatov (task and requirements), Codex (implementation,
experiments, measurements, and reporting). No W&B run was created; evidence is
preserved in the files linked below. Publication is a submission attempt, not
benchmark acceptance.

[TOC]

## Results in the same units

Execution times use **ms**, energies **mJ**, modeled area **mm²**, and scoring
time **s**. Human values have two significant figures; exact prediction counts,
thresholds, and raw measurements remain unrounded in the evidence. In these
units, energy in mJ equals power in W multiplied by time in ms.

| Execution | Accuracy | Time (ms) | Energy (mJ) | Area (mm²) | Time to score (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dally v4, complete learning task | {achieved} | {sig(model['time_ms'])} | {sig(model['energy_mj'])} | {sig(model['area_mm2_occupied_cells'])} | {sig(model['time_to_score_seconds'])} |
| A100, complete learning task; three-trial mean | {achieved} | {sig(gpu_ms)} | {sig(gpu_mj)} | — | — |

Every measured A100 task resets parameters, prepares inputs and targets, trains
the full frozen epoch budget, and predicts all test labels. The comparison
covers the same learning algorithm. The physical energy accounting differs:
the theoretical model prices scratch accesses, while NVML measures GPU-board
energy after subtracting idle power. Modeled area counts occupied scratch cells. The ratio of A100 to modeled time is
{sig(gpu_ms/model['time_ms'])}; the energy ratio is {sig(gpu_mj/model['energy_mj'])}.

## Train-only selection and the target

The predeclared grid contains 12 training runs: widths 128, 256, and 512; learning
rates 0.01, 0.03, 0.1, and 0.2. Each finite run is examined at epochs 25, 50, 100,
200, and 300. A fixed PCG64 permutation with seed 20260913 partitions only the
training arrays into 4,800 fitting examples and 1,200 validation examples. The
split is not stratified. All search runs initialize with seed 11.

The selection rule first seeks 1,176/1,200 validation predictions (98%), choosing
the least `epochs × width` work among candidates that meet it. If none meets it,
the rule maximizes validation correct, breaking ties by work, width, then rate.
One configuration and final seed 101 are frozen; it is refitted from scratch on
all 6,000 training examples. Predictions are saved before separate evaluation.
There is no further configuration or seed search after test evaluation.

Best observed checkpoint within each predeclared width/rate run:

| Width | Learning rate | Epochs | Validation accuracy | Fitting accuracy |
| ---: | ---: | ---: | ---: | ---: |
{valtable}

The selected validation result is **{search['selected']['validation_correct']}/1,200**.
The complete checkpoint records, nonfinite stops if any, split indices, source
hashes, selection rule, and freeze timestamps are retained in the raw evidence.
One validation example changes accuracy by about 0.083 percentage points; tiny
ranking differences are not strong generalization evidence.

This is a bounded attempt using an algorithm that can be exactly represented
by the current compact IL. It does not establish that 98% is impossible for
MNIST-medium. The historical 98.14% result used a deeper CNN, a different
10,000/10,000 dataset, AdamW, cross-entropy, and other modeling choices. Its
weights are not reused: some examples in that historical training set occur in
the current test split. Only algorithm ideas and published summary results
inform the choice of model family.

## Exact learning algorithm

Input pixels are transformed as `FP32(FP32(x × 4) − 0.5)`. The first layer uses
ReLU and the second is linear. The loss is half squared error against one-hot
targets, averaged over each batch. Batches contain 30 contiguous examples in
supplied order, without shuffling, giving 200 updates per epoch.

PCG64 seed 101 generates input weights uniformly within `[-1/9, 1/9]` and output
weights within `[-1/sqrt(width), 1/sqrt(width)]`; biases start at zero. The initial
parameter words are data-independent program constants. Generating these
constants is outside model/GPU execution; writing/resetting them is included.

For a batch, compute `hidden = ReLU(x @ W1 + b1)`, `output = hidden @ W2 + b2`,
`delta2 = output − target`, and `delta1 = (delta2 @ old_W2.T) * (hidden > 0)`.
Sum gradients in ascending batch-row order and subtract
`FP32(learning_rate / 30) × gradient` from every parameter. All backpropagation
uses pre-update weights. Products and additions round separately to FP32 with
no fusion. Reduction indices increase monotonically. Inference chooses the
first class on an exact maximum-score tie.

## Compact program and modeled area

The model is single-core-with-tape, v4, pinned at
`{model['model_spec_commit']}`. The builder emits fixed loops, affine addresses,
and priced primitive instructions. It uses the unchanged compact scorer from
the prior study; no free matrix operation or learner-specific cost certificate
is introduced.

The program receives **{model['input_tape_words']:,} input words**, executes
**{model['total_instructions']:,} primitive instructions** with
**{model['charged_accesses']:,} charged operand accesses**, and sends 6,000
predictions. Fixed half-diamond placement determines each scratch access cost.
Tape operations retain their zero-access-cost convention at the pinned revision.

The medium builder streams test queries after training: receive one image,
normalize it, run inference, send a prediction, then reuse the query buffer.
It allocates **{model['peak_initialized_scratch_words']:,} scratch words**,
reported as **{sig(model['area_mm2_occupied_cells'])} mm²** at one square
micrometre per word. This is occupied cell area, excluding tape and instruction
storage, not a bounding rectangle or A100 allocation. Streaming keeps the
program within the existing scorer's allocation and initialization-proof guards.

The canonical program hash is `{model['program_sha256']}`. The serialized IL
has {model['static_nodes']:,} syntax nodes and {model['static_leaf_nodes']:,}
primitive leaves. Exact affine address histograms aggregate all dynamic reads
and writes. Source aliases and unchosen select operands retain their charges.

The reported static scoring time is the median of
{len(model['time_to_score_samples_seconds'])} complete calls, including schema,
bounds and definite-initialization validation, placement, histograms, exact
integer sums, and canonical hashing. Program construction, JSON loading, numeric training, accuracy
evaluation, and file output are excluded. Scoring runs on {host['cpu_model']}
with {host['logical_processors']} logical processors and
{sig(host['memory_bytes']/2**30)} GiB RAM; Python
{model['software']['python'].split()[0]} and NumPy {model['software']['numpy']}.
Native integer units convert as
`time_ticks_0_2_ps / 5e9` ms, `energy_fj / 1e12` mJ, and
`area_um2_occupied_cells / 1e6` mm².

## Numerical and scoring verification

The CPU learner's matrix operations are checked against explicit ordered FP32
reductions for the medium shapes, including transposed gradients and row sums.
A complete training epoch is compared in both implementations. The entire
final GPU run is checked against all CPU parameter words, every output-score
word, and every prediction, before and after timing.

The generalized builder also passes four tests: unchanged small-program
regression, independently expanded toy semantics including 81-feature inputs,
per-address cost agreement, and allocation/guard behavior. These establish
different properties. The full medium program is not expanded and interpreted
instruction by instruction, and a second slow ordered CPU replay of every
training epoch is not performed.

The allowed-input-only CLI check removes test labels and source indices, then
refits the model and reproduces the saved outputs. Independent verification
records the exact scope, source fingerprints, matrix checks, and timings.
The primary CPU refit took **{sig(cpu['cpu_reference_seconds']*1000)} ms**,
separate from the static scoring time and A100 measurements.

## A100 implementation and measurements

Hardware: **{gpu['hardware']['gpu_name']}**. Software: Python {versions['python']},
PyTorch {versions['torch']}, Triton {versions['triton']}, NumPy {versions['numpy']},
CUDA {versions['cuda_runtime']}, NVIDIA driver {versions['nvidia_driver']},
NVML {versions['nvml']}, and `nvidia-ml-py` {versions['nvidia_ml_py']}.
Container: `{versions['container_image']}`.

The manual Triton implementation preserves ascending FP32 reductions and
disables fusion. Large reduction loops remain runtime loops to keep compilation
bounded. It is a port of the fixed algorithm, not an automatic IL-to-PTX compiler.

Three CUDA graphs hold initialization, one complete training epoch, and
inference. Each measured task runs initialization once, the epoch graph
{config['epochs']} times, and inference once. A graph replay is therefore not
an entire task; the trial counters record complete tasks separately. All
parameter reset and training are repeated. The GPU materializes all output
scores for verification, and that extra work is included in its measurements.

Independent CPU parameters, scores, and predictions are host-only verification
oracles. They are not copied to the GPU or consulted by learner kernels. The
GPU learner receives only the three allowed data arrays and seed-only initial
literals. Changed-query and bounded changed-training-label checks exercise
input dependence; their exact scope is retained in the GPU evidence.

Each of the three trials executes three complete tasks, about 14,000 ms of
active work. Each 3,000 ms idle sample follows a 3,000 ms settling gap, before
and after the active window. CUDA events and wall time measure complete repeated
tasks. NVML cumulative
board energy is differenced over each active interval, then adjusted by the
mean of bracketing idle power measurements. Per-task mJ is
`1000 × (active_J − mean_idle_W × active_seconds) / complete_tasks`.
The subtraction uses the NVML interval; no negative value is clipped.

| Trial | Accuracy | Complete tasks | Time (ms/task) | Raw energy (mJ/task) | Paired-idle energy (mJ/task) | Before-only / after-only energy (mJ/task) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{trials}

Mean task time is **{sig(gpu_ms)} ms** (sample SD
{sig(summary['cuda_event_us_per_invocation']['sample_stddev']/1000)} ms).
Mean idle-adjusted board energy is **{sig(gpu_mj)} mJ** (sample SD
{sig(summary['idle_adjusted_j_per_invocation']['sample_stddev']*1000)} mJ).
The raw counter readings, idle measurements, graph/task counts, trial durations,
hardware telemetry, and both one-sided baseline alternatives are retained.

Transfers, allocations, compilation, graph capture, verification, cold start,
and host CPU energy are excluded. Repeated GPU-resident tasks may benefit from
cache residency. These measurements describe this implementation and boundary;
they are not a claim of optimal A100 performance or calibrated per-kernel energy.

## Provenance and reproduction

Dataset: `competition-v2`, canonical seed 20260910, 6,000/6,000 disjoint examples
from the original MNIST training pool, area-downsampled to 9 × 9. Allowed array
content SHA-256 values, C-order little-endian:

{hashes}

Predictions: `{cpu['prediction_sha256_int64_le']}`.
Output scores: `{cpu['output_scores_sha256']}`.
GPU source: `{gpu['source_sha256']}`.
Further manifest, parameter, source, and PTX hashes are preserved in the evidence.

{reproduction}

## Download the evidence

- [Frozen configuration](config.json), [predeclared search plan](predeclared_plan.json), [validation results](validation_results.json), [split indices](validation_split.json).
- [Learner](learner.py), [CPU results](cpu_results.json), [predictions](predictions.npy), [parameters](parameters.npz), [output scores](output-scores.npy).
- [Official accuracy evaluation](accuracy.json), [requested 98% goal status](goal-status.json).
- [Compact program](program.il.json), [builder](model_ir.py), [score wrapper](score.py), [model scores](model-score.json), [independent count audit](model-score-audit.json), [builder validation](model-ir-validation.json).
- [Independent verifier](verify.py), [verification evidence](verification.json).
- [GPU source](gpu_benchmark.py), [GPU results](gpu_results.json), [measurement notes](gpu-notes.md).
- [Separate ambiguities and problems](ambiguities.html), [human-readable session](session.html).
'''
    (HERE/'report.md').write_text(text)
    print(HERE/'report.md')


if __name__=='__main__':
    main()
