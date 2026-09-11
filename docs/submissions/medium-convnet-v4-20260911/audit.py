"""Verify retained submission evidence without reading raw query labels or training.

Only the requested audit JSON is written. No model scoring, GPU invocation, or
raw IDX access occurs. Use --output to preserve the published audit record.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def close(actual, expected, message):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), message)


def audit(check_inputs=True):
    import evaluation as ev

    # Make the label boundary executable, including accidental future changes.
    def forbid_raw(*args, **kwargs):
        raise ValueError('Evidence audit attempted raw IDX access')
    ev.generator.read_idx = forbid_raw
    protocol, master, draws = ev.verify_protocol()
    rows, predictions = ev.validate_outputs(protocol, draws)
    frozen = read(HERE / 'prediction_manifest.json')
    accuracy = read(HERE / 'accuracy.json')
    config = read(HERE / 'config.json')
    require(config == protocol['config'], 'Config differs from frozen protocol')
    require(frozen['predictions'] == rows, 'Global prediction freeze differs from current files')
    require(frozen['protocol_sha256'] == sha(HERE / 'protocol.json'), 'Frozen protocol hash')
    require(frozen['draw_manifest_sha256'] == sha(HERE / 'draw_manifest.json'), 'Frozen draw manifest hash')
    require(frozen['total_predictions'] == 110000 and not frozen['test_labels_opened'], 'Global freeze scope')
    require(accuracy['prediction_manifest_sha256'] == sha(HERE / 'prediction_manifest.json'), 'Accuracy freeze identity')
    require(accuracy['protocol_sha256'] == sha(HERE / 'protocol.json'), 'Accuracy protocol identity')
    require(accuracy['evaluator_sha256'] == sha(HERE / 'evaluation.py'), 'Accuracy evaluator identity')
    require(accuracy['predictions_globally_frozen_before_query_labels'], 'Missing evaluation phase assertion')
    instant = datetime.fromisoformat
    require(instant(protocol['frozen_at_utc']) < instant(frozen['frozen_at_utc']) < instant(accuracy['evaluated_at_utc']), 'Phase timestamp order')
    for draw in draws:
        train = np.asarray(draw['train_indices'])
        query = np.asarray(draw['test_indices'])
        require(len(train) == len(query) == 10000, 'Dataset dimensions')
        require(np.unique(np.concatenate((train, query))).size == 20000, 'Within-draw overlap')
        require(train.min() >= 0 and query.min() >= 0 and train.max() < 60000 and query.max() < 60000, 'Source row bounds')
        if check_inputs:
            path = ev.safe_path(draw['archive'])
            require(sha(path) == draw['archive_sha256'], 'Input archive changed')
            with np.load(path, allow_pickle=False) as arrays:
                require(set(arrays.files) == set(ev.INPUT_NAMES), 'Unexpected learner input')
                for name in arrays.files:
                    spec = draw['arrays'][name]
                    require(list(arrays[name].shape) == spec['shape'] and str(arrays[name].dtype) == spec['dtype'], 'Input shape/dtype')
                    require(array_sha(arrays[name]) == spec['sha256'], 'Input array hash')

    # Independently derive aggregate statistics from retained per-draw counts.
    require([d['dataset_seed'] for d in accuracy['draws']] == list(range(20261101, 20261112)), 'Accuracy draw order')
    require([d['draw_index'] for d in accuracy['draws']] == list(range(11)), 'Accuracy draw indices')
    counts = [d['correct'] for d in accuracy['draws']]
    require(all(type(c) is int and 0 <= c <= 10000 for c in counts), 'Invalid correct counts')
    require(all(d['total'] == 10000 and d['accuracy_percent'] == d['correct'] / 100 for d in accuracy['draws']), 'Per-draw accuracy arithmetic')
    mean = Fraction(sum(counts), 1100)
    variance = sum((Fraction(c, 100) - mean) ** 2 for c in counts) / 10
    require(accuracy['total_correct'] == sum(counts) and accuracy['total_predictions'] == 110000, 'Aggregate counts')
    require(accuracy['mean_accuracy_percent'] == float(mean), 'Mean accuracy')
    require(accuracy['mean_error_rate_percent'] == float(100 - mean), 'Mean error')
    require(accuracy['sample_variance_exact'] == dict(numerator=variance.numerator, denominator=variance.denominator), 'Exact sample variance')
    require(accuracy['ddof'] == 1, 'SD denominator')
    close(accuracy['sample_standard_deviation_pp'], math.sqrt(float(variance)), 'Sample SD')
    for error in (10, 8, 6, 4, 2):
        required = 1100 * (100 - error)
        require(accuracy['error_targets'][str(error)] == dict(required_correct=required, meets_target=sum(counts) >= required), 'Error target arithmetic')
    require(accuracy['required_correct'] == 105600 and accuracy['target_percent'] == 96, 'Attempt threshold')
    require(accuracy['meets_target'] == (sum(counts) >= 105600), 'Attempt decision')
    require(accuracy['margin_correct'] == sum(counts) - 105600, 'Threshold margin')
    require(accuracy['strictest_target_percent'] == 98 and accuracy['meets_strictest_target'] == (sum(counts) >= 107800), 'Strictest target')

    constants = read(HERE / 'constants/constants.json')
    require(constants['config'] == config and constants['config_sha256'] == sha(HERE / 'config.json'), 'Compiler config')
    require(constants['n_train'] == 10000 and constants['seeds'] == config['member_seeds'] == config['seeds'], 'Compiler seed order')
    for name, digest in constants['source_sha256'].items():
        require(sha(HERE / 'ordered_backend' / name) == digest, 'Constant generator source changed')
    require([r['seed'] for r in constants['records']] == config['seeds'], 'Constant record order')
    for record in constants['records']:
        initial = record['initial']
        path = HERE / 'constants' / initial['path']
        require(sha(path) == initial['file_sha256'], 'Initial archive hash')
        with np.load(path, allow_pickle=False) as arrays:
            require(set(arrays.files) == set(initial['arrays']), 'Initial array names')
            for name in arrays.files:
                spec = initial['arrays'][name]
                require(str(arrays[name].dtype) == spec['dtype'] == 'float32' and list(arrays[name].shape) == spec['shape'], 'Initial shape/dtype')
                require(array_sha(arrays[name]) == spec['sha256'], 'Initial literal bits')
    results = [read(HERE / f'results/draw-{i:02d}.json') for i in range(11)]
    for result in results:
        require(instant(result['completed_at_utc']) < instant(frozen['frozen_at_utc']), 'Output completed after global freeze')
        for member, record in zip(result['members'], constants['records']):
            require(member['seed'] == record['seed'], 'Member seed mismatch')
            require(member['schedule_manifests'] == record['epochs'] and len(record['epochs']) == 8, 'Compiler/learner schedule mismatch')
            require(member['initial_parameter_sha256'] == {n: s['sha256'] for n, s in record['initial']['arrays'].items()}, 'Compiler/learner initialization mismatch')
            require(member['minibatches_per_epoch'] == 79 and member['epochs'] == 8, 'Complete training scope')

    score = read(HERE / 'model-score.json')
    program = read(HERE / 'program.il.json')
    require(score['config'] == program['metadata']['config'] == config, 'Scored algorithm config')
    require(score['constants_manifest_sha256'] == sha(HERE / 'constants/constants.json'), 'Scored constants identity')
    digest = hashlib.sha256(json.dumps(program, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(digest == score['program_sha256'], 'Scored IL identity')
    for name, digest in score['source_sha256'].items():
        require(sha(ROOT / name) == digest, 'Scorer source changed: ' + name)
    require(score['scorer_source_sha256'] == sha(HERE / 'ir_core.py'), 'Core scorer identity')
    require(score['time_ms'] == score['time_ticks_0_2_ps'] / 5 / 1e9, 'Model time conversion')
    require(score['energy_mj'] == score['energy_fj'] / 1e12, 'Model energy conversion')
    require(score['area_mm2'] == score['area_um2_occupied_cells'] / 1e6, 'Model area conversion')
    require(score['time_to_score_seconds'] == statistics.median(score['time_to_score_samples_seconds']), 'Scoring-time median')
    require(sum(r['words'] for r in program['regions']) == score['area_um2_occupied_cells'] == score['peak_initialized_scratch_words'], 'Physical scratch allocation')
    counts_ir = Counter()
    def visit(body, multiple=1):
        for node in body:
            if 'loop' in node:
                visit(node['body'], multiple * node['count'])
            else:
                counts_ir[node['op']] += multiple
    visit(program['body'])
    require(dict(counts_ir) == score['instructions'] and sum(counts_ir.values()) == score['total_instructions'], 'Static instruction multiplicities')
    arities = dict(set=0, recv=0, send=0, copy=1, add=2, sub=2, mul=2, div=2, cmp=2, select=3)
    require(sum(arities[k] * v for k, v in counts_ir.items()) == score['charged_reads'], 'Charged source multiplicity')
    require(sum(v for k, v in counts_ir.items() if k not in ('recv', 'send')) == score['charged_writes'], 'Charged destination multiplicity')
    require(counts_ir['recv'] == score['input_tape_words'] == 1630000 and counts_ir['send'] == score['output_tape_words'] == 10000, 'Complete tape streams')

    for filename in ('ir-core-validation.json', 'ir-conv-validation.json', 'ir-model-validation.json'):
        record = read(HERE / filename)
        require(record['all_passed'], 'IR validation did not pass')
        for name, digest in record['source_sha256'].items():
            require(sha(ROOT / name) == digest, 'IR validation source changed: ' + name)
    gpu_validation = read(HERE / 'ordered_backend/submission-validation/results.json')
    require(gpu_validation['status'] == 'passed', 'Current-source GPU validation did not pass')
    for name, digest in gpu_validation['source_sha256'].items():
        require(sha(HERE / 'ordered_backend' / name) == digest, 'GPU validation source changed')
    checks = [c for case in gpu_validation['cases'] for c in case['checks']]
    require(all(c['bitwise_match'] for c in checks), 'GPU/CPU array comparison failed')
    require(gpu_validation['ptx_checks']['no_fp32_fma'] and gpu_validation['ptx_checks']['no_ftz'] and gpu_validation['ptx_checks']['no_tensorcore_instructions'], 'GPU validation PTX restriction')

    benchmark = read(HERE / 'benchmark/results.json')
    measured_result = read(HERE / 'benchmark/results/draw-00.json')
    require(benchmark['learner_result_sha256'] == sha(HERE / 'benchmark/results/draw-00.json'), 'Measurement learner identity')
    require(measured_result['source_sha256'] == protocol['source_sha256'] and measured_result['protocol_sha256'] == sha(HERE / 'protocol.json'), 'Measured frozen source identity')
    require(measured_result['config'] == config and not measured_result['test_labels_opened'], 'Measured learner scope')
    require(measured_result['input_sha256'] == results[0]['input_sha256'] and measured_result['dataset_seed'] == results[0]['dataset_seed'] and measured_result['draw_index'] == 0, 'Measured dataset identity')
    hardware = benchmark['hardware']
    require(hardware['cuda_uuid'] == hardware['nvml_uuid'] and hardware['uuid_match'], 'CUDA/NVML UUID mismatch')
    require(hardware['cuda_uuid'].removeprefix('GPU-') == measured_result['hardware']['uuid'].removeprefix('GPU-'), 'Learner/instrument GPU mismatch')
    require('A100' in hardware['cuda_name'] and hardware['cuda_name'] == hardware['nvml_name'] and hardware['mig_mode'] == [0, 0], 'A100 hardware identity')
    fingerprint = benchmark['state_fingerprint']
    require([m['seed'] for m in measured_result['members']] == [m['seed'] for m in fingerprint['members']] == config['seeds'], 'Measured member order/count')
    for expected, measured, actual in zip(results[0]['members'], measured_result['members'], fingerprint['members']):
        for field in ('seed', 'config', 'epochs', 'minibatches_per_epoch', 'final_parameter_sha256', 'final_buffer_sha256', 'final_velocity_sha256', 'initial_parameter_sha256', 'logits_sha256', 'schedule_manifests'):
            require(expected[field] == measured[field], 'Benchmark differs from frozen accuracy member: ' + field)
        for key, field in (('parameters', 'final_parameter_sha256'), ('buffers', 'final_buffer_sha256'), ('velocities', 'final_velocity_sha256'), ('scores', 'logits_sha256')):
            require(actual[key] == expected[field], 'Timed fingerprint differs from accuracy: ' + key)
        require(actual['seed'] == expected['seed'] and actual['schedule_position'] == array_sha(np.array(80000, dtype=np.int32)), 'Timed schedule position')
    with np.load(HERE / 'predictions/draw-00.npz', allow_pickle=False) as original:
        total = np.zeros((10000, 10), dtype=np.float32)
        for seed in config['seeds']:
            total = np.add(total, original[f'logits_seed{seed}'], dtype=np.float32)
        require(fingerprint['ensemble_scores'] == array_sha(total) and fingerprint['predictions'] == array_sha(original['predictions']), 'Timed ensemble differs from frozen accuracy')
        measured_path = HERE / 'benchmark/predictions/draw-00.npz'
        require(sha(measured_path) == measured_result['prediction_archive_sha256'], 'Measured prediction archive hash')
        with np.load(measured_path, allow_pickle=False) as measured:
            require(set(measured.files) == set(original.files), 'Measured prediction archive schema')
            require(all(array_sha(measured[n]) == array_sha(original[n]) for n in original.files), 'Measured output bits differ')
    require(len(benchmark['trials']) == benchmark['protocol']['trials'] == 3, 'Measurement trial count')
    for trial in benchmark['trials']:
        repeats = trial['invocations']
        require(repeats == benchmark['invocations_per_trial'] and repeats >= 1, 'Measurement repetition count')
        for key in ('active', 'idle_before', 'idle_after'):
            interval = trial[key]
            require(all(math.isfinite(interval[end]['read_duration_seconds']) and interval[end]['read_duration_seconds'] >= 0 for end in ('start', 'end')), 'NVML read duration')
            duration = interval['end']['time_seconds'] - interval['start']['time_seconds']
            energy = interval['end']['counter_mj'] - interval['start']['counter_mj']
            require(duration > 0 and energy >= 0, 'NVML counter interval')
            close(interval['duration_seconds'], duration, 'NVML duration arithmetic')
            require(interval['energy_mj'] == energy, 'NVML counter subtraction')
            close(interval['average_power_w'], energy / duration / 1000, 'NVML power units')
        active = trial['active']
        before = trial['idle_before']['average_power_w']
        after = trial['idle_after']['average_power_w']
        paired = (before + after) / 2
        close(trial['paired_idle_power_w'], paired, 'Paired idle average')
        close(trial['gross_energy_mj_per_task'], active['energy_mj'] / repeats, 'Gross energy per task')
        for field, baseline in (('idle_adjusted_energy_mj_per_task', paired), ('before_only_adjusted_mj_per_task', before), ('after_only_adjusted_mj_per_task', after)):
            close(trial[field], (active['energy_mj'] - baseline * active['duration_seconds'] * 1000) / repeats, 'Idle energy correction')
        close(trial['wall_ms_per_task'], (trial['wall_end_seconds'] - trial['wall_start_seconds']) * 1000 / repeats, 'Wall time per task')
        require(active['start']['time_seconds'] <= trial['wall_start_seconds'] < trial['wall_end_seconds'] <= active['end']['time_seconds'], 'Energy interval does not enclose timed work')
        require(active['start']['time_seconds'] + active['start']['read_duration_seconds'] / 2 <= trial['wall_start_seconds'] + 1e-9, 'Opening NVML API call overlaps timed work')
        require(trial['wall_end_seconds'] <= active['end']['time_seconds'] - active['end']['read_duration_seconds'] / 2 + 1e-9, 'Closing NVML API call overlaps timed work')
        require(trial['idle_before']['end']['time_seconds'] < active['start']['time_seconds'] and active['end']['time_seconds'] < trial['idle_after']['start']['time_seconds'], 'Idle/active ordering')
        require(trial['cuda_ms_per_task'] * repeats >= 8000 and .8 < trial['cuda_ms_per_task'] / trial['wall_ms_per_task'] < 1.2, 'Timing sanity bounds')
        require(trial['repeated_full_state_fingerprint_matches'], 'Timed result mismatch')
    for key, summary in benchmark['summary'].items():
        values = [trial[key] for trial in benchmark['trials']]
        for name, value in dict(median=statistics.median(values), mean=statistics.mean(values), min=min(values), max=max(values), sample_standard_deviation=statistics.stdev(values)).items():
            close(summary[name], value, 'Benchmark summary: ' + key + '/' + name)

    ptx_checked = 0
    ptx_sets = [(HERE / 'ptx', result['ptx_sha256']) for result in results]
    ptx_sets += [(HERE / 'benchmark/ptx', measured_result['ptx_sha256']),
                 (HERE / 'ordered_backend/submission-validation', gpu_validation['ptx_checks']['sha256'])]
    for folder, manifests in ptx_sets:
        for name, digest in manifests.items():
            path = folder / (name + '.ptx')
            require(sha(path) == digest, 'Declared PTX hash mismatch')
            text = path.read_text()
            require(not re.search(r'\bfma(?:\.[a-z0-9]+)*\.f32\b', text), 'FP32 fused multiply-add in retained PTX')
            require('.ftz.' not in text and not re.search(r'\b(?:mma|wmma|wgmma)\.', text), 'FTZ/tensor core instruction in retained PTX')
            ptx_checked += 1

    supplemental = read(HERE / 'supplemental-argmax/results.json')
    require(supplemental['all_passed'] and not any(supplemental['instruction_audit'].values()), 'Supplemental argmax checks')
    require(supplemental['protocol_sha256'] == sha(HERE / 'protocol.json') and supplemental['original_benchmark_result_sha256'] == benchmark['learner_result_sha256'], 'Supplemental provenance')
    require(supplemental['software'] == measured_result['software'], 'Supplemental compiler environment differs')
    require(not any(supplemental[key] for key in ('dataset_loaded', 'learned_parameters_loaded', 'original_predictions_loaded')), 'Supplemental fixture isolation')
    for name, digest in supplemental['source_sha256'].items():
        require(sha(HERE / name) == digest, 'Supplemental source changed')
    compiler = supplemental['compiler']
    require(compiler['specialization'] == dict(B=10000, BLOCK=128) and compiler['launch_grid'] == [79], 'Supplemental argmax shape')
    require(compiler['launch_options'] == dict(num_warps=4, enable_fp_fusion=False), 'Supplemental compiler flags')
    require(compiler['ptx_target'] == 'sm_80' and supplemental['hardware']['compute_capability'] == [8, 0] and 'A100' in supplemental['hardware']['name'], 'Supplemental GPU architecture')
    path = HERE / 'supplemental-argmax' / supplemental['ptx_file']
    require(sha(path) == supplemental['ptx_sha256'], 'Supplemental PTX hash')
    require(supplemental['ptx_sha256'] not in measured_result['ptx_sha256'].values(), 'Expected original-run coverage gap changed')
    text = path.read_text()
    require(not re.search(r'\b(?:fma|mad)(?:\.[a-z0-9]+)*\.f32\b', text) and '.ftz.' not in text and not re.search(r'\b(?:mma|wmma|wgmma)\.', text), 'Supplemental PTX instruction restriction')
    # Regenerate only the tiny synthetic argument fixtures, not any learner.
    rng = np.random.default_rng(20260911)
    random_values = rng.standard_normal((10000, 10), dtype=np.float32)
    ties = rng.integers(-3, 4, (10000, 10)).astype(np.float32)
    ties[0] = 0; ties[1] = -4
    ties[2] = -5; ties[2, [2, 7]] = 6
    ties[3] = np.array([-0., 0.] * 5, dtype=np.float32)
    ties[4] = 0; ties[4, [3, 9]] = np.nextafter(np.float32(0), np.float32(1))
    ties[5] = -np.finfo(np.float32).max; ties[5, [1, 8]] = np.finfo(np.float32).max
    ties[6] = -7; ties[6, 9] = 8
    ties[-1] = -9; ties[-1, [4, 9]] = 10
    require([row['name'] for row in supplemental['checks']] == ['finite_random', 'ties_and_boundary_rows'], 'Supplemental case coverage')
    for values, row in zip((random_values, ties), supplemental['checks']):
        expected = values.argmax(1).astype(np.int64)
        require(row['all_predictions_match'] and row['rows'] == 10000 and row['input_shape'] == [10000, 10] and row['output_shape'] == [10000], 'Supplemental case shape')
        require(row['input_sha256'] == array_sha(values) and row['output_sha256'] == array_sha(expected), 'Supplemental independent input/output hashes')
        require(row['first_eight_predictions'] == expected[:8].tolist() and row['last_prediction'] == int(expected[-1]), 'Supplemental tie/boundary outputs')

    retained = ['protocol.json', 'config.json', 'draw_manifest.json', 'prediction_manifest.json', 'accuracy.json',
                'constants/constants.json', 'program.il.json', 'model-score.json', 'benchmark/results.json',
                'benchmark/results/draw-00.json', 'ir-model-validation.json',
                'ordered_backend/submission-validation/results.json', 'supplemental-argmax/results.json']
    return dict(status='passed', audited_at_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(Path(__file__)), evidence_sha256={name: sha(HERE / name) for name in retained},
        raw_query_labels_read=False, training_or_scoring_rerun=False,
        input_archive_hashes_checked=check_inputs, draws=11, trained_members=33,
        total_correct=sum(counts), mean_accuracy_percent=float(mean), sample_standard_deviation_pp=math.sqrt(float(variance)),
        target_required_correct=105600, target_met=sum(counts) >= 105600,
        compiler_initial_and_schedule_hashes_match_all_members=True,
        current_source_gpu_cpu_array_checks=len(checks), declared_ptx_files_checked_with_repetitions=ptx_checked,
        separately_regenerated_argmax_source_ptx_and_fixture_hashes_verified=True,
        benchmark_fingerprint_matches_frozen_accuracy=True, benchmark_raw_arithmetic_and_summaries_verified=True,
        model_units_and_instruction_multiplicities_verified=True,
        median_gpu_ms=benchmark['summary']['cuda_ms_per_task']['median'],
        median_idle_adjusted_energy_mj=benchmark['summary']['idle_adjusted_energy_mj_per_task']['median'],
        scope_limits=['Accuracy statistics recomputed from retained counts; raw ground-truth labels are deliberately not reopened.',
            'Geometry-weighted costs are covered by retained expanded-program tests; this audit does not rerun the scorer.',
            'GPU/IL equivalence evidence covers complete synthetic small networks and component source review, not full numerical interpretation of the 2.8-trillion-instruction production program.',
            'PTX checks cover all hashes declared in retained manifests. The original final full-query argmax capture gap is supplemented by independently verified same-source regeneration, explicitly not an original-run capture. PyTorch memory-operation kernels are not retained.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'audit.json')
    parser.add_argument('--skip-input-files', action='store_true', help='Audit retained manifests without requiring regenerated learner input archives; this limitation is recorded.')
    args = parser.parse_args()
    result = audit(check_inputs=not args.skip_input_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'draws', 'trained_members', 'mean_accuracy_percent', 'sample_standard_deviation_pp', 'median_gpu_ms', 'median_idle_adjusted_energy_mj')}))


if __name__ == '__main__':
    main()
