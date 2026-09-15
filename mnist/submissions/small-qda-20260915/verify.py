"""Verify all 11 frozen draws, ordered-FP32 reference hashes, and spatial execution.

The default checks imported evidence without modifying it. --evidence-dir checks
an independent prepare/freeze/score reproduction. --output writes a fresh report.
"""
import argparse
from datetime import datetime, timezone
import hashlib
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
import spatial_program as sp
from score import score as score_grid
from run import ds, require, read_json


def sha256_json(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_hashes():
    files = {name: HERE / name for name in
             ('run.py', 'verify.py', 'reference.py', 'spatial_program.py', 'protocol.json', 'protocol_fresh.json', 'protocol_beacon.json', 'requirements.txt')}
    files['mnist/code/data.py'] = Path(ds.__file__)
    shared = HERE.parent / 'grid-mlp-scoring-20260912'
    files.update({f'../grid-mlp-scoring-20260912/{name}': shared / name for name in ('affine.py', 'score.py')})
    return {name: ds.file_hash(path) for name, path in files.items()}


def verify(evidence):
    start = time.perf_counter()
    sources = source_hashes()
    raw_sources = run.verify_raw()
    protocol = read_json(run.PROTOCOL)
    if 'dataset_seeds' in protocol:
        require(protocol['dataset_seeds'] == run.SEEDS and protocol['planned_draws'] == 11, 'Protocol draw count or seeds differ')
    else:   # beacon-seeded: re-derive the seeds from the stored pulse and check its declared time
        pulse = read_json(HERE / protocol['seed_source']['pulse_file'])
        require(run.beacon_seeds(protocol, pulse) == run.SEEDS and protocol['planned_draws'] == 11, 'Beacon-derived seeds differ')
        require(read_json(evidence / 'draw_manifest.json')['seeds'] == run.SEEDS, 'Draw manifest seeds differ from the beacon derivation')
    require(protocol['target_total_correct'] == 7370 and protocol['target_accuracy'] == .67,
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

    document = sp.build_submitted(1000, 1000)
    grid = read_json(HERE / 'grid/grid-score.json')
    program_hash = sha256_json(document)
    require(program_hash == grid['program_sha256'], 'Generated spatial program differs from scored program')
    # Check the exact serialized artifact hash even when the regenerable JSON is absent.
    generated_bytes = (json.dumps(document, indent=2) + '\n').encode()
    require(hashlib.sha256(generated_bytes).hexdigest() == grid['program_file_sha256'],
            'Generated spatial JSON bytes differ from the recorded artifact hash')
    program_path = HERE / 'grid/program.spatial.json'
    if program_path.exists():
        require(program_path.read_bytes() == generated_bytes, 'On-disk program differs from generated program')
    shared = HERE.parent / 'grid-mlp-scoring-20260912'
    for name, path in (('affine.py', shared / 'affine.py'), ('score.py', shared / 'score.py'),
                       ('spatial_program.py', HERE / 'spatial_program.py')):
        require(ds.file_hash(path) == grid['source_sha256'][name], f'Scored source hash differs: {name}')
    fresh_grid = score_grid(document)
    for key, value in fresh_grid.items():
        if key != 'time_to_score_seconds':
            require(grid[key] == value, f'Spatial scorer result differs: {key}')
    print(f'Canonical MNIST, manifests, and all deterministic spatial score fields verified; '
          f'{grid["energy_fj"]} fJ, {grid["cycles"]} cycles', flush=True)

    counts, verified_draws = [], []
    for index, (draw, record, pred, expected_accuracy) in enumerate(zip(
            draw_manifest['draws'], manifest['draws'], predictions, accuracy['draws'], strict=True)):
        draw_start = time.perf_counter()
        seed = draw['dataset_seed']
        train, test, x, q, ytrain, pixels, labels = run._load(seed)
        run.check_inputs(draw, train, test, x, q)
        params, scores, again = reference.train_predict(x, ytrain, q)
        for name in ('mu', 'packed', 'kappa'):
            require(ds.array_hash(params[name]) == record[name + '_sha256'],
                    f'Draw {index}: reference parameter hash differs: {name}')
        require(ds.array_hash(scores) == record['scores_sha256'], f'Draw {index}: reference score hash differs')
        require(np.array_equal(again, pred), f'Draw {index}: reference predictions differ')
        xr = run.resize_recorded(pixels[train].astype(np.float32) / np.float32(255)).reshape(1000, 9)
        qr = run.resize_recorded(pixels[test].astype(np.float32) / np.float32(255)).reshape(1000, 9)
        tape = sp.tape_words(xr, ytrain, qr)
        require(len(tape) == grid['input_tape_words'], f'Draw {index}: spatial input tape length differs')
        out = np.asarray(sp.execute(document, tape), dtype=np.int64)
        require(out.shape == pred.shape and np.array_equal(out, pred),
                f'Draw {index}: executed spatial program predictions differ')
        require(ds.array_hash(out) == record['array_sha256'], f'Draw {index}: spatial output hash differs')
        count = {'draw': index, 'dataset_seed': seed, 'correct': int((pred == labels[test]).sum()), 'total': 1000}
        require(count == expected_accuracy, f'Draw {index}: recorded accuracy differs')
        counts.append(count)
        verified_draws.append({
            **count, 'indices_and_normalized_input_hashes_match': True,
            'training_labels_sha256': ds.array_hash(ytrain), 'test_labels_sha256': ds.array_hash(labels[test]),
            'reference_parameter_hashes_match': True, 'reference_scores_sha256': ds.array_hash(scores),
            'prediction_file_sha256': record['prediction_sha256'], 'prediction_array_sha256': ds.array_hash(pred),
            'reference_predictions_match': True, 'spatial_predictions_match': True,
            'spatial_output_sha256': ds.array_hash(out),
            'spatial_input_tape_sha256': ds.array_hash(np.asarray(tape, dtype=np.uint32)),
            'verification_seconds': time.perf_counter() - draw_start,
        })
        print(f'Draw {index:02d}, seed {seed}: {count["correct"]}/1000; '
              'input, parameter, score, prediction hashes and spatial labels match', flush=True)

    recomputed_accuracy = run.accuracy_result(counts)
    require(recomputed_accuracy == accuracy, 'Recorded aggregate accuracy or sample standard deviation differs')
    require(source_hashes() == sources, 'Verification source files changed during this run')
    evidence_hashes = {name: ds.file_hash(evidence / name) for name in
                       ('draw_manifest.json', 'prediction_manifest.json', 'accuracy.json', 'evaluation_freeze.json')}
    protocol_at = datetime.fromisoformat(protocol['created_at_utc'].replace('Z', '+00:00'))
    return {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
        'scope': 'All 11 recorded draws: canonical raw MNIST, exact indices and input hashes, frozen '
                 'prediction files, reference parameters/scores, executed spatial output labels, '
                 'evaluation freeze hash, accuracy/sample SD, and all deterministic spatial scorer fields.',
        'software': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                     'machine': platform.machine()},
        'source_sha256': sources, 'raw_mnist': raw_sources, 'evidence_sha256': evidence_hashes,
        'accuracy': recomputed_accuracy, 'draws': verified_draws,
        'grid': {**{key: fresh_grid[key] for key in
                   ('program_sha256', 'energy_fj', 'energy_mj', 'cycles', 'time_ms',
                    'peak_allocated_scratch_bytes', 'total_executed_instructions', 'placement_sha256')},
                 'program_file_sha256': grid['program_file_sha256'],
                 'grid_score_file_sha256': ds.file_hash(HERE / 'grid/grid-score.json'),
                 'all_deterministic_score_fields_match': True,
                 'fresh_scoring_seconds': fresh_grid['time_to_score_seconds']},
        'protocol': run.PROTOCOL.name,
        'provenance_limits': {
            'protocol_timestamp_precedes_prediction_freeze': protocol_at <= frozen_at,
            'historical_preselection_independently_proven': run.PROTOCOL.name in ('protocol_fresh.json', 'protocol_beacon.json'),
            'note': ('The seeds are derived from a NIST randomness-beacon pulse whose time the protocol declared before the pulse '
                     'existed (protocol_beacon.json, published and hashed on the pull request before that time); anyone can '
                     're-derive them from the public pulse.' if run.PROTOCOL.name == 'protocol_beacon.json' else
                     'The fresh protocol (protocol_fresh.json) was committed before its draws were prepared; the freeze '
                     'and evaluation commits follow it in the repository history.' if run.PROTOCOL.name == 'protocol_fresh.json' else
                     'Hashes and reproducibility do not establish historical learner selection or label-access order. '
                     'The imported protocol timestamp is later than the imported prediction freeze.'),
        },
        'verification_seconds': time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, default=run.EVIDENCE)
    parser.add_argument('--output', type=Path, help='Write a fresh JSON verification report')
    args = parser.parse_args()
    if args.evidence_dir.resolve().is_relative_to(run.FRESH.resolve()):
        require(run.PROTOCOL.name == 'protocol_fresh.json', 'Verify evidence/fresh with SUTRO_PROTOCOL=protocol_fresh.json')
    if args.evidence_dir.resolve().is_relative_to(run.BEACON.resolve()):
        require(run.PROTOCOL.name == 'protocol_beacon.json', 'Verify evidence/beacon with SUTRO_PROTOCOL=protocol_beacon.json')
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
