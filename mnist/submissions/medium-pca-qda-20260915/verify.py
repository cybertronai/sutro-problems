"""Verify all 11 frozen draws, ordered-FP32 PCA-QDA parameter and score hashes.

The default checks imported evidence without modifying it. --evidence-dir checks
an independent prepare/freeze/score reproduction. --output writes a fresh report.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run
import reference
from run import ds, require, read_json


def source_hashes():
    files = {name: HERE / name for name in
             ('run.py', 'verify.py', 'reference.py', 'protocol.json', 'requirements.txt')}
    files['mnist/code/data.py'] = Path(ds.__file__)
    return {name: ds.file_hash(path) for name, path in files.items()}


def verify(evidence):
    start = time.perf_counter()
    sources = source_hashes()
    raw_sources = run.verify_raw()
    protocol = read_json(HERE / 'protocol.json')
    require(protocol['dataset_seeds'] == run.SEEDS and protocol['planned_draws'] == 11,
            'Protocol draw count or seeds differ')
    require(protocol['target_total_correct'] == 104500 and protocol['target_accuracy'] == .95,
            'Protocol accuracy target differs')
    draw_manifest = read_json(evidence / 'draw_manifest.json')
    run.validate_records(draw_manifest, 'draw manifest')
    manifest, predictions = run.frozen_predictions(evidence)
    accuracy = read_json(evidence / 'accuracy.json')
    require(len(accuracy['draws']) == 11, 'Accuracy must contain all 11 draws')
    evaluation = read_json(evidence / 'evaluation_freeze.json')
    require(evaluation['prediction_manifest_sha256'] == ds.file_hash(evidence / 'prediction_manifest.json'),
            'Evaluation freeze does not reference the frozen prediction manifest')
    frozen_at = datetime.fromisoformat(manifest['frozen_at_utc'].replace('Z', '+00:00'))
    evaluated_at = datetime.fromisoformat(evaluation['evaluated_at_utc'].replace('Z', '+00:00'))
    require(frozen_at <= evaluated_at, 'Evaluation timestamp precedes prediction freeze')

    print('Canonical MNIST sources, complete manifests, and evaluation freeze hash verified', flush=True)

    counts, verified_draws = [], []
    for index, (draw, record, pred, expected_accuracy) in enumerate(zip(
            draw_manifest['draws'], manifest['draws'], predictions, accuracy['draws'], strict=True)):
        draw_start = time.perf_counter()
        seed = draw['dataset_seed']
        train, test, x, q, ytrain, pixels, labels = run._load(seed)
        run.check_inputs(draw, train, test, x, q)
        params, scores, again = reference.train_predict(x, ytrain, q)
        for name in ('m', 'W', 'mu', 'packed', 'kappa'):
            require(ds.array_hash(params[name]) == record[name + '_sha256'],
                    f'Draw {index}: reference parameter hash differs: {name}')
        require(ds.array_hash(scores) == record['scores_sha256'], f'Draw {index}: reference score hash differs')
        require(np.array_equal(again, pred), f'Draw {index}: reference predictions differ')
        native_x = ds.area_resize(pixels[train].astype(np.float32) / np.float32(255), 9).reshape(10000, 81)
        native_q = ds.area_resize(pixels[test].astype(np.float32) / np.float32(255), 9).reshape(10000, 81)
        count = {'draw': index, 'dataset_seed': seed, 'correct': int((pred == labels[test]).sum()), 'total': 10000}
        require(count == expected_accuracy, f'Draw {index}: recorded accuracy differs')
        counts.append(count)
        verified_draws.append({
            **count, 'indices_and_normalized_input_hashes_match': True,
            'training_labels_sha256': ds.array_hash(ytrain), 'test_labels_sha256': ds.array_hash(labels[test]),
            'reference_parameter_hashes_match': True, 'reference_scores_sha256': ds.array_hash(scores),
            'prediction_file_sha256': record['prediction_sha256'], 'prediction_array_sha256': ds.array_hash(pred),
            'reference_predictions_match': True,
            'native_blas_train_input_sha256': ds.array_hash(native_x),
            'native_blas_test_input_sha256': ds.array_hash(native_q),
            'native_blas_inputs_match_recorded': np.array_equal(native_x, x) and np.array_equal(native_q, q),
            'recorded_train_input_sha256': ds.array_hash(x),
            'recorded_test_input_sha256': ds.array_hash(q),
            'parameter_sha256': {key: ds.array_hash(params[key]) for key in ('m', 'W', 'mu', 'packed', 'kappa')},
            'verification_seconds': time.perf_counter() - draw_start,
        })
        print(f'Draw {index:02d}, seed {seed}: {count["correct"]}/10000; '
              'input, parameter, score, and prediction hashes match', flush=True)

    recomputed_accuracy = run.accuracy_result(counts)
    require(recomputed_accuracy == accuracy, 'Recorded aggregate accuracy or sample standard deviation differs')
    require(source_hashes() == sources, 'Verification source files changed during this run')
    evidence_hashes = {name: ds.file_hash(evidence / name) for name in
                       ('draw_manifest.json', 'prediction_manifest.json', 'accuracy.json', 'evaluation_freeze.json')}
    protocol_at = datetime.fromisoformat(protocol['created_at_utc'].replace('Z', '+00:00'))
    return {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
        'scope': 'All 11 recorded draws: canonical raw MNIST, exact indices and input hashes, frozen '
                 'prediction files, reference parameters/scores, evaluation freeze hash, accuracy/sample SD. '
                 'Spatial scoring and reduced numerical execution are recorded separately.',
        'software': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                     'machine': platform.machine()},
        'source_sha256': sources, 'raw_mnist': raw_sources, 'evidence_sha256': evidence_hashes,
        'accuracy': recomputed_accuracy, 'draws': verified_draws,
        'preprocessing': {
            'implementation': 'run.resize_recorded: repository box-area weights, sequential accumulation '
                              'with FP64 intermediates and an FP32 cast after each multiply-add',
            'all_recorded_input_hashes_match': True,
            'native_blas_all_recorded_inputs_match': all(d['native_blas_inputs_match_recorded'] for d in verified_draws),
            'reason': 'Native BLAS matrix-product rounding depends on CPU dispatch. Explicit accumulation '
                      'reproduces all 22 archived input hashes on this CPU.',
        },
        'spatial_validation': {'performed_by_this_verifier': False, 'report': 'results/grid_verification.json'},
        'provenance_limits': {
            'protocol_timestamp_precedes_prediction_freeze': protocol_at <= frozen_at,
            'historical_preselection_independently_proven': False,
            'note': 'Hashes and reproducibility do not establish historical learner selection or label-access order. '
                    'The imported protocol timestamp is later than the imported prediction freeze.',
        },
        'verification_seconds': time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, default=run.EVIDENCE)
    parser.add_argument('--output', type=Path, help='Write a fresh JSON verification report')
    args = parser.parse_args()
    if args.output is not None:
        require(not args.output.resolve().is_relative_to((HERE / 'evidence').resolve()),
                'Write verification reports outside the imported evidence directory')
    result = verify(args.evidence_dir)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print('Wrote', args.output)
    accuracy = result['accuracy']
    print(f'Verified {accuracy["correct"]}/{accuracy["total"]} = {accuracy["mean_accuracy"] * 100:.8f}%; '
          f'sample SD {accuracy["sample_sd_pp"]:.8f} percentage points')


if __name__ == '__main__':
    main()
