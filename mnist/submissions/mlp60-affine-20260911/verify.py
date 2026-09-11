"""Reproduce the fixed 60% candidate and verify its CPU/IL evidence.

Numerical replay uses explicit ordered FP32 reductions for all 300 epochs.
The learner receives only three allowed input arrays. Ground-truth test labels
are opened later by a separate evaluation stage, after predictions are fixed.
GPU verification and measurements belong to gpu_benchmark.py, not this script.
"""
from __future__ import annotations

import argparse
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
EXPERIMENT = HERE.parents[1] / 'experiments' / 'accuracy-il-20260911'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENT))
import learner
import accuracy_study as reference
from il import score as static_score
from mnist.code.evaluate import accuracy_target_status, load_accuracy_target, score_predictions

PARAMETER_NAMES = ('w1', 'b1', 'w2', 'b2')
ALLOWED_INPUTS = ('train_images', 'train_labels', 'test_images')
CONFIG_ID = 'h32-e300-lr0.2'
SEED = 101


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(
        np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8)
    )


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def parameter_hashes(params):
    return {name: reference.digest(value) for name, value in zip(PARAMETER_NAMES, params)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'mnist/data/small.npz')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'mnist/doc/dataset_manifest.json')
    parser.add_argument('--artifacts', type=Path, default=HERE,
                        help='Directory containing outputs from learner.py and score.py')
    parser.add_argument('--output', type=Path, default=HERE / 'verification.json')
    args = parser.parse_args()
    started = time.perf_counter()
    sources = [Path(__file__), HERE / 'learner.py', HERE / 'score.py',
               EXPERIMENT / 'accuracy_study.py', EXPERIMENT / 'il.py',
               EXPERIMENT / 'mlp_il.py', ROOT / 'mnist/code/evaluate.py',
               ROOT / 'mnist/doc/accuracy_targets.json']
    source_hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in sources}
    frozen_path = EXPERIMENT / 'frozen_predictions.json'
    frozen = json.loads(frozen_path.read_text())
    run = next(row for row in frozen['runs'] if row['config_id'] == CONFIG_ID and row['seed'] == SEED)
    require(frozen['source_sha256'] == file_hash(EXPERIMENT / 'accuracy_study.py'),
            'Reference source differs from the frozen study')

    # reference.inputs checks each allowed array against the canonical manifest;
    # it opens no test labels or source indices.
    data = reference.inputs(args.data, args.manifest)
    require(tuple(data) == ALLOWED_INPUTS, 'Unexpected decoded learner input member')
    input_hashes = {key: reference.digest(value) for key, value in data.items()}
    stages = {}
    begin = time.perf_counter()
    predictions, params, scores = learner.learn(data)
    stages['fresh_learner_seconds'] = time.perf_counter() - begin
    require(predictions.shape == (600,) and scores.shape == (600, 10), 'Wrong output shape')
    require(all(np.isfinite(value).all() for value in [*params, scores]), 'Nonfinite numerical output')
    require([reference.digest(value) for value in params] == run['final_parameter_sha256'],
            'Fresh learner parameter bits differ from the frozen study hashes')
    require(reference.digest(predictions) == run['predictions_sha256_int64_le'],
            'Fresh learner predictions differ from the frozen study hash')
    require(np.array_equal(predictions, np.asarray(run['predictions'], dtype=np.int64)),
            'Fresh learner predictions differ from the frozen prediction array')
    require([reference.digest(value) for value in reference.parameters(32, SEED)] == run['initial_parameter_sha256'],
            'Seed-only initialization differs from the frozen study')

    # Independently exercise every training reduction as separate FP32 multiply
    # and add operations; this is an arithmetic replay, not an expansion of the
    # entire 619-million-instruction IL program.
    begin = time.perf_counter()
    x = reference.transform(data['train_images'])
    q = reference.transform(data['test_images'])
    targets = (data['train_labels'][:, None] == np.arange(10)).astype(np.float32)
    ordered_params = reference.parameters(32, SEED)
    for _ in range(300):
        ordered_params = reference.epoch(x, targets, ordered_params, 0.2,
                                         reference.ordered_mm, reference.ordered_rows)
    ordered_scores = reference.forward(q, ordered_params, reference.ordered_mm)[2]
    ordered_predictions = ordered_scores.argmax(axis=1).astype(np.int64)
    stages['ordered_300_epoch_replay_seconds'] = time.perf_counter() - begin
    require(all(same_bits(a, b) for a, b in zip(params, ordered_params)),
            'Ordered replay parameter bits differ')
    require(same_bits(scores, ordered_scores), 'Ordered replay output score bits differ')
    require(same_bits(predictions, ordered_predictions), 'Ordered replay predictions differ')

    # Check the supplied reproducible package, not merely an in-memory run.
    require(same_bits(predictions, np.load(args.artifacts / 'predictions.npy', allow_pickle=False)),
            'Packaged predictions differ from the fresh learner')
    require(same_bits(scores, np.load(args.artifacts / 'output-scores.npy', allow_pickle=False)),
            'Packaged score vectors differ from the fresh learner')
    with np.load(args.artifacts / 'parameters.npz', allow_pickle=False) as saved:
        require(all(same_bits(value, saved[name]) for name, value in zip(PARAMETER_NAMES, params)),
                'Packaged parameter bits differ from the fresh learner')
    cpu_record = json.loads((args.artifacts / 'cpu_results.json').read_text())
    require(cpu_record['input_sha256'] == input_hashes, 'Packaged canonical input hashes differ')
    require(cpu_record['parameter_sha256'] == parameter_hashes(params), 'Packaged parameter hashes differ')
    require(cpu_record['output_scores_sha256'] == reference.digest(scores), 'Packaged score-vector hash differs')

    # The real learner CLI must run after all forbidden NPZ members have been
    # removed. Temporary outputs never overwrite the published evidence.
    begin = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='mnist-mlp60-input-check-') as directory:
        directory = Path(directory)
        stripped = directory / 'allowed-only.npz'
        np.savez(stripped, **data)
        with np.load(stripped, allow_pickle=False) as archive:
            require(set(archive.files) == set(ALLOWED_INPUTS), 'Stripped archive has unexpected members')
        result = subprocess.run(
            [sys.executable, str(HERE / 'learner.py'), '--data', str(stripped),
             '--manifest', str(args.manifest.resolve()), '--output', str(directory / 'result')],
            check=True, capture_output=True, text=True,
        )
        record = json.loads(result.stdout)
        require(record['decoded_input_members'] == list(ALLOWED_INPUTS), 'CLI reported unexpected input members')
        require(record['input_sha256'] == input_hashes, 'Stripped-input canonical hashes differ')
        require(record['parameter_sha256'] == parameter_hashes(params), 'Stripped-input parameter hashes differ')
        require(record['prediction_sha256_int64_le'] == reference.digest(predictions),
                'Stripped-input predictions differ')
        require(record['output_scores_sha256'] == reference.digest(scores), 'Stripped-input score vectors differ')
        require(same_bits(scores, np.load(directory / 'result/output-scores.npy', allow_pickle=False)),
                'Stripped-input actual score array differs')
    stages['allowed_only_npz_cli_seconds'] = time.perf_counter() - begin

    begin = time.perf_counter()
    changed = {**data, 'train_labels': (data['train_labels'] + 1) % 10}
    changed_predictions, changed_params, _ = learner.learn(changed)
    changed_count = int(np.count_nonzero(changed_predictions != predictions))
    require(changed_count > 0, 'Changing training labels did not affect predictions')
    require(any(not same_bits(a, b) for a, b in zip(params, changed_params)),
            'Changing training labels did not affect learned parameters')
    stages['changed_training_labels_seconds'] = time.perf_counter() - begin

    program_path = args.artifacts / 'program.il.json'
    program = json.loads(program_path.read_text())
    canonical_hash = hashlib.sha256(json.dumps(program, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    saved_score = json.loads((args.artifacts / 'model-score.json').read_text())
    prior_score = json.loads((EXPERIMENT / f'{CONFIG_ID}.score.json').read_text())
    begin = time.perf_counter()
    recalculated = static_score(program)
    stages['fresh_static_score_seconds'] = time.perf_counter() - begin
    exact_fields = ('program_sha256', 'instructions', 'total_instructions', 'charged_reads',
                    'charged_writes', 'charged_accesses', 'time_ticks_0_2_ps', 'energy_fj',
                    'area_um2_occupied_cells', 'peak_initialized_scratch_words',
                    'input_tape_words', 'output_tape_words')
    require(all(recalculated[key] == prior_score[key] == saved_score[key] for key in exact_fields),
            'Static costs differ from the frozen study or supplied package')
    require(canonical_hash == prior_score['program_sha256'], 'Canonical program hash differs')
    require(sum(item['words'] for item in program['regions']) == 20290,
            'Unexpected scratch-region allocation')
    require(recalculated['total_instructions'] == 618842313 and recalculated['charged_accesses'] == 1833513513,
            'Unexpected instruction or access total')
    require(recalculated['time_ticks_0_2_ps'] == 464041510156 and recalculated['energy_fj'] == 113965836674,
            'Unexpected exact time or energy total')

    # Separate evaluator stage: only now open test ground truth. It never enters
    # learner.learn, the stripped-input CLI, or the program's scoring inputs.
    with np.load(args.data, allow_pickle=False) as archive:
        test_labels = archive['test_labels']
    manifest = json.loads(args.manifest.read_text())
    require(reference.digest(test_labels) == manifest['tiers']['small']['arrays']['test_labels']['sha256_c_order_little_endian'],
            'Noncanonical test labels in evaluation stage')
    evaluation = score_predictions(predictions, test_labels)
    evaluation.update(accuracy_target_status(evaluation['correct'], evaluation['total'], load_accuracy_target('small')))
    require(evaluation['correct'] == 374 and evaluation['total'] == 600, 'Unexpected canonical accuracy')
    require(evaluation['accuracy_target_percent'] == 60 and evaluation['required_correct'] == 360
            and evaluation['meets_accuracy_target'], 'Current small classification target not met')
    require(all(file_hash(ROOT / name) == digest for name, digest in source_hashes.items()),
            'Source changed while verification was running')

    report = {
        'all_passed': True,
        'configuration': {'hidden_width': 32, 'epochs': 300, 'learning_rate': 0.2, 'batch_size': 30, 'seed': SEED},
        'canonical_allowed_input_hashes': input_hashes,
        'frozen_study_parameter_hashes_match': True,
        'frozen_study_prediction_array_and_hash_match': True,
        'all_300_epochs_replayed_with_explicit_ordered_fp32_reductions': True,
        'all_four_parameter_arrays_bitwise_equal': True,
        'all_600_by_10_output_scores_bitwise_equal': True,
        'all_600_predictions_bitwise_equal': True,
        'packaged_learner_artifacts_match_fresh_execution': True,
        'allowed_only_npz_cli_reproduces_identical_parameters_scores_predictions': True,
        'removed_npz_members': ['test_labels', 'train_indices', 'test_indices'],
        'changed_training_labels': {'transformation': '(label + 1) modulo 10',
                                    'training_labels_changed': int(np.count_nonzero(changed['train_labels'] != data['train_labels'])),
                                    'predictions_changed': changed_count,
                                    'learned_parameters_changed': True},
        'parameter_sha256': parameter_hashes(params),
        'output_scores_sha256': reference.digest(scores),
        'prediction_sha256_int64_le': reference.digest(predictions),
        'program_sha256': canonical_hash,
        'exact_score_fields_match_frozen_study': list(exact_fields),
        'total_instructions': recalculated['total_instructions'],
        'charged_accesses': recalculated['charged_accesses'],
        'time_ticks_0_2_ps': recalculated['time_ticks_0_2_ps'],
        'energy_fj': recalculated['energy_fj'],
        'occupied_scratch_words': 20290,
        'area_mm2_occupied_cells': 20290 / 1e6,
        'classification': {key: evaluation[key] for key in ('correct', 'total', 'accuracy',
                           'accuracy_target_percent', 'required_correct', 'meets_accuracy_target')},
        'stage_wall_seconds': stages,
        'total_verification_wall_seconds': time.perf_counter() - started,
        'software': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()},
        'source_sha256': source_hashes,
        'frozen_predictions_file_sha256': file_hash(frozen_path),
        'canonical_manifest_file_sha256': file_hash(args.manifest),
        'supplied_program_file_sha256': file_hash(program_path),
        'scope': [
            'Frozen study preserves final parameter hashes and predictions; it does not preserve the full output-score array.',
            'Full score-vector equality is established against the independent ordered 300-epoch replay and the supplied submission array.',
            'The ordered replay shares the declared high-level update equations with the learner but uses explicit FP32 reductions instead of optimized einsum.',
            'Static cost verification traverses the compact IL; the full 618842313-instruction program is not numerically interpreted here.',
            'Test labels are opened only by the separate evaluation stage after predictions are fixed; the learner receives three allowed arrays.',
            'This verifies reproducibility and classification/cost evidence, not a fresh blind selection protocol or GPU performance.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
