"""Verify the frozen medium learner with bounded independent numerical checks.

Checks all medium reduction shapes, one complete ordered training epoch, final
inference, and a fresh full learner CLI run using only the three allowed arrays.
It does not replay every training epoch with the slow ordered CPU reductions.
Test labels are opened only in the final, separate evaluation stage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import learner
from mnist.code.evaluate import accuracy_target_status, load_accuracy_target, score_predictions

NAMES = ('w1', 'b1', 'w2', 'b2')
ALLOWED = ('train_images', 'train_labels', 'test_images')


def require(value, message):
    if not value:
        raise AssertionError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    little = np.ascontiguousarray(value.astype(value.dtype.newbyteorder('<'), copy=False))
    return hashlib.sha256(little.tobytes()).hexdigest()


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(
        np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))


def explicit_mm(a, b):
    """Separate float32 products and ordered accumulation, without einsum/BLAS."""
    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    for k in range(a.shape[1]):
        product = np.multiply(a[:, k:k + 1], b[k:k + 1, :], dtype=np.float32)
        result = np.add(result, product, dtype=np.float32)
    return result


def explicit_rows(a):
    result = np.zeros(a.shape[1], dtype=np.float32)
    for row in range(a.shape[0]):
        result = np.add(result, a[row], dtype=np.float32)
    return result


def parameter_hashes(values):
    return dict(zip(NAMES, (array_hash(value) for value in values)))


def reduction_checks(widths, batch=30):
    rng = np.random.Generator(np.random.PCG64(20260914))
    records = []
    for width in widths:
        def random(shape):
            return rng.normal(0.0, 0.25, size=shape).astype(np.float32)
        x, w1, h = random((batch, 81)), random((81, width)), random((batch, width))
        w2, d2, d1 = random((width, 10)), random((batch, 10)), random((batch, width))
        queries, qhidden = random((6000, 81)), random((6000, width))
        operations = [
            ('hidden_forward', x, w1), ('output_forward', h, w2),
            ('hidden_backward_transposed_w2', d2, w2.T),
            ('w1_gradient_transposed_x', x.T, d1),
            ('w2_gradient_transposed_hidden', h.T, d2),
            ('all_query_hidden', queries, w1), ('all_query_output', qhidden, w2),
        ]
        for name, a, b in operations:
            expected = explicit_mm(a, b)
            fast = learner.fast_mm(a, b)
            reference_ordered = learner.ordered_mm(a, b)
            require(same_bits(expected, fast), f'fast_mm disagrees for H{width} {name}')
            require(same_bits(expected, reference_ordered), f'ordered_mm disagrees for H{width} {name}')
            records.append({'width': width, 'operation': name,
                            'left_shape': list(a.shape), 'right_shape': list(b.shape),
                            'left_c_contiguous': bool(a.flags.c_contiguous),
                            'right_c_contiguous': bool(b.flags.c_contiguous),
                            'fast_and_ordered_and_explicit_bits_equal': True})
        for name, values in [('hidden_bias_gradient', d1), ('output_bias_gradient', d2)]:
            expected = explicit_rows(values)
            require(same_bits(expected, learner.fast_rows(values)), f'fast_rows disagrees for {name}')
            require(same_bits(expected, learner.ordered_rows(values)), f'ordered_rows disagrees for {name}')
            records.append({'width': width, 'operation': name, 'shape': list(values.shape),
                            'fast_and_ordered_and_explicit_bits_equal': True})
    return records


def frozen_protocol(config, directory, input_hashes):
    plan_path = directory / 'predeclared_plan.json'
    split_path = directory / 'validation_split.json'
    results_path = directory / 'validation_results.json'
    plan = json.loads(plan_path.read_text())
    split = json.loads(split_path.read_text())
    results = json.loads(results_path.read_text())
    require(plan['widths'] == [128, 256, 512], 'Unexpected predeclared widths')
    require(plan['learning_rates'] == [0.01, 0.03, 0.1, 0.2], 'Unexpected rate grid')
    require(plan['checkpoint_epochs'] == [25, 50, 100, 200, 300], 'Unexpected epoch checkpoints')
    require(plan['validation_split_seed'] == 20260913, 'Unexpected split seed')
    require(plan['validation_fit_rows'] == 4800 and plan['validation_holdout_rows'] == 1200,
            'Unexpected split sizes')
    expected_order = np.random.Generator(np.random.PCG64(plan['validation_split_seed'])).permutation(6000)
    require(split['training_rows'] == expected_order[:4800].tolist(), 'Training split changed')
    require(split['validation_rows'] == expected_order[4800:].tolist(), 'Validation split changed')
    require(set(split['training_rows']).isdisjoint(split['validation_rows']), 'Validation overlap')
    require(plan['training_input_sha256'] == {k: input_hashes[k] for k in ('train_images', 'train_labels')},
            'Search inputs differ from canonical learner inputs')
    for name, expected in plan['source_sha256'].items():
        require(file_hash(HERE / name) == expected, f'Search source differs from predeclaration: {name}')
    rows = results['rows']
    for width in plan['widths']:
        for rate in plan['learning_rates']:
            group = [row for row in rows if row['width'] == width and row['learning_rate'] == rate]
            require(bool(group), 'A predeclared configuration is missing')
            stops = [row for row in group if row['status'] == 'nonfinite_training']
            finite = [row for row in group if row['status'] == 'finite']
            require(len(stops) <= 1 and len(stops) + len(finite) == len(group), 'Invalid candidate status')
            stop = stops[0]['epochs'] if stops else max(plan['checkpoint_epochs']) + 1
            require(sorted(row['epochs'] for row in finite) == [e for e in plan['checkpoint_epochs'] if e < stop],
                    'Candidate checkpoint sequence differs from predeclared budget')
            require(all(row['validation_total'] == 1200 and row['training_total'] == 4800 for row in finite),
                    'Candidate score used the wrong split size')
    require(all(row['width'] in plan['widths'] and row['learning_rate'] in plan['learning_rates'] for row in rows),
            'An undeclared candidate was included')
    finite = [row for row in rows if row['status'] == 'finite']
    eligible = [row for row in finite if row['validation_correct'] >= 1176]
    cost = lambda row: (row['epochs'] * row['width'], row['width'], row['learning_rate'])
    chosen = min(eligible, key=cost) if eligible else min(finite, key=lambda row: (-row['validation_correct'], *cost(row)))
    require(chosen == results['selected'], 'Selected validation record disagrees with predeclared ranking')
    expected_config = {k: chosen[k] for k in ('width', 'epochs', 'learning_rate')}
    expected_config.update({'seed': plan['final_seed'], 'batch_size': 30, 'features': 81,
                            'n_train': 6000, 'n_test': 6000})
    require(config == results['config'] == expected_config, 'Final configuration was not the validation-selected one')
    require(config['seed'] == 101, 'Unexpected final seed')
    created = datetime.fromisoformat(plan['created_at_utc'])
    frozen = datetime.fromisoformat(results['frozen_at_utc'])
    require(created < frozen, 'Invalid recorded protocol chronology')
    return {'predeclared_plan_file_sha256': file_hash(plan_path),
            'validation_split_file_sha256': file_hash(split_path),
            'validation_results_file_sha256': file_hash(results_path),
            'predeclared_at_utc': plan['created_at_utc'], 'configuration_frozen_at_utc': results['frozen_at_utc'],
            'candidate_training_runs': len(plan['widths']) * len(plan['learning_rates']),
            'inspected_finite_checkpoints': len(finite),
            'selected_validation_correct': chosen['validation_correct'], 'validation_total': 1200,
            'split_kind': 'fixed unstratified PCG64 permutation; 4800 fit / 1200 validation',
            'selection_recomputed_from_frozen_training_only_evidence': True}, plan


def read_parameters(directory, width):
    with np.load(directory / 'parameters.npz', allow_pickle=False) as saved:
        require(set(saved.files) == set(NAMES), 'Unexpected parameter archive members')
        values = [saved[name] for name in NAMES]
    shapes = [(81, width), (width,), (width, 10), (10,)]
    require(all(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all()
                for value, shape in zip(values, shapes)), 'Invalid parameter shape, dtype or values')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'mnist/data/medium.npz')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'mnist/doc/dataset_manifest.json')
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--artifacts', type=Path, default=HERE)
    parser.add_argument('--protocol', type=Path, default=HERE)
    parser.add_argument('--output', type=Path, default=HERE / 'verification.json')
    args = parser.parse_args()
    started = time.perf_counter()
    stages = {}
    sources = [Path(__file__), HERE / 'learner.py', HERE / 'search.py', learner.REFERENCE,
               ROOT / 'mnist/code/evaluate.py', ROOT / 'mnist/doc/accuracy_targets.json']
    source_hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in sources}
    config_hash = file_hash(args.config)
    config = json.loads(args.config.read_text())
    data = learner.inputs(args.data, args.manifest)
    require(tuple(data) == ALLOWED, 'Unexpected learner inputs')
    input_hashes = {key: array_hash(value) for key, value in data.items()}
    protocol, plan = frozen_protocol(config, args.protocol, input_hashes)
    artifact_names = ('cpu_results.json', 'parameters.npz', 'output-scores.npy', 'predictions.npy')
    artifact_hashes = {name: file_hash(args.artifacts / name) for name in artifact_names}
    cpu = json.loads((args.artifacts / 'cpu_results.json').read_text())
    params = read_parameters(args.artifacts, config['width'])
    saved_scores = np.load(args.artifacts / 'output-scores.npy', allow_pickle=False)
    predictions = np.load(args.artifacts / 'predictions.npy', allow_pickle=False)
    require(saved_scores.shape == (6000, 10) and saved_scores.dtype == np.float32 and np.isfinite(saved_scores).all(),
            'Invalid saved scores')
    require(predictions.shape == (6000,) and predictions.dtype == np.int64 and np.all((predictions >= 0) & (predictions < 10)),
            'Invalid saved predictions')
    require(cpu['configuration'] == config and cpu['config_file_sha256'] == config_hash, 'CPU configuration mismatch')
    require(cpu['decoded_input_members'] == list(ALLOWED) and cpu['input_sha256'] == input_hashes, 'CPU input evidence mismatch')
    require(cpu['parameter_sha256'] == parameter_hashes(params), 'Saved parameter hashes mismatch')
    require(cpu['output_scores_sha256'] == array_hash(saved_scores), 'Saved score hash mismatch')
    require(cpu['prediction_sha256_int64_le'] == array_hash(predictions), 'Saved prediction hash mismatch')
    require(all(file_hash(ROOT / name) == expected for name, expected in cpu['source_sha256'].items()),
            'CPU results were not generated by current learner sources')

    begin = time.perf_counter()
    arithmetic = reduction_checks(plan['widths'], config['batch_size'])
    stages['all_medium_operation_shape_checks_seconds'] = time.perf_counter() - begin
    print(f'Passed {len(arithmetic)} ordered reduction checks across all three widths.', flush=True)

    begin = time.perf_counter()
    one_epoch_config = {**config, 'epochs': 1}
    fast_epoch = learner.fit(data['train_images'], data['train_labels'], one_epoch_config)
    ordered_epoch = learner.fit(data['train_images'], data['train_labels'], one_epoch_config, explicit_mm, explicit_rows)
    require(all(same_bits(a, b) for a, b in zip(fast_epoch, ordered_epoch)), 'Complete ordered first epoch differs')
    stages['one_complete_6000_example_ordered_epoch_and_fast_comparison_seconds'] = time.perf_counter() - begin
    print('One complete 6000-example epoch: all four parameter arrays match bitwise.', flush=True)

    begin = time.perf_counter()
    queries = learner.transform(data['test_images'])
    recomputed = learner.forward(queries, params)[2]
    ordered_scores = learner.forward(queries, params, explicit_mm)[2]
    require(same_bits(saved_scores, recomputed) and same_bits(saved_scores, ordered_scores), 'Final inference score bits differ')
    require(same_bits(predictions, ordered_scores.argmax(axis=1).astype(np.int64)), 'Final prediction bits differ')
    stages['final_inference_from_saved_parameters_seconds'] = time.perf_counter() - begin

    begin = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='mnist-medium-allowed-only-') as temp:
        temp = Path(temp)
        input_path = temp / 'allowed-only.npz'
        output_path = temp / 'fresh'
        np.savez(input_path, **data)
        with np.load(input_path, allow_pickle=False) as archive:
            require(set(archive.files) == set(ALLOWED), 'Forbidden members present in CLI test input')
        subprocess.run([sys.executable, str(HERE / 'learner.py'), '--data', str(input_path),
                        '--manifest', str(args.manifest.resolve()), '--config', str(args.config.resolve()),
                        '--output', str(output_path)], check=True, capture_output=True, text=True)
        fresh = json.loads((output_path / 'cpu_results.json').read_text())
        fresh_params = read_parameters(output_path, config['width'])
        require(all(same_bits(a, b) for a, b in zip(params, fresh_params)), 'Full allowed-only refit parameter mismatch')
        require(same_bits(saved_scores, np.load(output_path / 'output-scores.npy', allow_pickle=False)), 'Full refit score mismatch')
        require(same_bits(predictions, np.load(output_path / 'predictions.npy', allow_pickle=False)), 'Full refit prediction mismatch')
        for key in ('configuration', 'input_sha256', 'decoded_input_members', 'prediction_sha256_int64_le',
                    'parameter_sha256', 'output_scores_sha256', 'source_sha256', 'config_file_sha256'):
            require(fresh[key] == cpu[key], f'Full refit metadata mismatch: {key}')
    stages['fresh_full_training_cli_allowed_only_seconds'] = time.perf_counter() - begin
    print('Full fresh learner CLI with allowed-only input reproduced every parameter, score and prediction bit.', flush=True)

    # No preceding computation has decoded test labels. This final stage is an
    # evaluator and dataset-identity audit, separate from the learner.
    evaluation_started = datetime.now(timezone.utc).isoformat()
    require(datetime.fromisoformat(protocol['configuration_frozen_at_utc']) < datetime.fromisoformat(evaluation_started),
            'Verification evaluation predates configuration freeze')
    manifest = json.loads(args.manifest.read_text())
    dataset_identity = {}
    with np.load(args.data, allow_pickle=False) as archive:
        for name, expected in manifest['tiers']['medium']['arrays'].items():
            value = archive[name]
            actual = {'shape': list(value.shape), 'dtype': str(value.dtype),
                      'sha256_c_order_little_endian': array_hash(value)}
            require(actual == expected, f'Noncanonical array in evaluation-stage integrity audit: {name}')
            dataset_identity[name] = actual
        labels = archive['test_labels']
        overlap = len(set(archive['train_indices']) & set(archive['test_indices']))
    require(overlap == 0, 'Train/test source indices overlap')
    evaluation = score_predictions(predictions, labels)
    require(evaluation['total'] == 6000, 'Wrong evaluation denominator')
    repository = accuracy_target_status(evaluation['correct'], evaluation['total'], load_accuracy_target('medium'))
    user_target = accuracy_target_status(evaluation['correct'], evaluation['total'], '98')
    require(repository['required_correct'] == 5889 and user_target['required_correct'] == 5880, 'Target policy changed')
    score_path = args.artifacts / 'accuracy.json'
    if score_path.exists():
        packaged = json.loads(score_path.read_text())
        require(all(packaged[key] == evaluation[key] for key in ('correct', 'total', 'accuracy')),
                'Packaged evaluation differs from independent evaluation')
        require(all(packaged[key] == value for key, value in repository.items()), 'Packaged repository status differs')
    goal_path = args.artifacts / 'goal-status.json'
    if goal_path.exists():
        packaged_goal = json.loads(goal_path.read_text())
        require(packaged_goal['correct'] == evaluation['correct'] and packaged_goal['total'] == evaluation['total'],
                'Packaged user-goal counts differ')
        require(packaged_goal['user_target_percent'] == 98 and packaged_goal['user_required_correct'] == 5880,
                'Packaged user-goal threshold differs')
        require(packaged_goal['user_target_met'] == user_target['meets_accuracy_target'],
                'Packaged user-goal status differs')
    require(file_hash(args.config) == config_hash, 'Configuration changed during verification')
    require(all(file_hash(args.artifacts / name) == expected for name, expected in artifact_hashes.items()),
            'CPU artifacts changed during verification')
    require(all(file_hash(ROOT / name) == expected for name, expected in source_hashes.items()),
            'Verification or learner sources changed during verification')

    report = {
        'all_passed': True, 'configuration': config, 'config_file_sha256': config_hash,
        'frozen_selection_protocol': protocol,
        'canonical_allowed_input_hashes': input_hashes, 'all_six_canonical_arrays': dataset_identity,
        'train_test_source_index_overlap_count': overlap,
        'arithmetic_shape_checks': arithmetic,
        'one_complete_ordered_training_epoch': {'epochs': 1, 'examples': 6000, 'minibatches': 200,
                                                'all_four_parameter_arrays_bitwise_equal': True,
                                                'parameter_sha256': parameter_hashes(ordered_epoch)},
        'fresh_full_training_from_allowed_only_archive': {
            'epochs': config['epochs'], 'training_examples': 6000,
            'removed_members': ['test_labels', 'train_indices', 'test_indices'],
            'all_final_parameter_score_prediction_bits_equal': True,
            'arithmetic_engine': 'declared fast einsum reductions, not slow ordered CPU replay'},
        'all_final_scores_from_saved_parameters_match_ordered_recomputation': True,
        'parameter_sha256': parameter_hashes(params), 'output_scores_sha256': array_hash(saved_scores),
        'predictions_sha256_int64_le': array_hash(predictions),
        'classification': {**{key: evaluation[key] for key in ('correct', 'total', 'accuracy')},
                           'user_98_percent_goal': user_target, 'repository_98_14_percent_requirement': repository},
        'separate_evaluation_started_at_utc': evaluation_started,
        'stage_wall_seconds': stages, 'total_verification_wall_seconds': time.perf_counter() - started,
        'source_sha256': source_hashes, 'reviewed_cpu_artifact_file_sha256': artifact_hashes,
        'canonical_manifest_file_sha256': file_hash(args.manifest),
        'software': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()},
        'scope': [
            'All medium training/inference reduction shapes and transposed gradient layouts were checked across widths 128, 256 and 512.',
            'Exactly one complete 6000-example training epoch was independently replayed with explicit ordered float32 reductions.',
            'The full selected training run was reproduced through the actual learner CLI with no test-label or source-index input members.',
            'Every final score was independently recomputed from saved parameters with explicit ordered float32 reductions.',
            'The complete selected training run was not replayed with slow ordered CPU reductions or expanded v4 interpretation.',
            'Full GPU-versus-NumPy training equality is a separate check reported by the GPU benchmark; this file makes no GPU-validation claim.',
            'Stored protocol timestamps, source identities and selection rules were checked; these are not an independent proof of all past test-access history.',
            'Below-target accuracy does not make these reproducibility checks fail; both exact target statuses are reported separately.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'all_passed': True, 'configuration': config, 'classification': report['classification'],
                      'arithmetic_shape_checks': len(arithmetic), 'stage_wall_seconds': stages}, indent=2))


if __name__ == '__main__':
    main()
