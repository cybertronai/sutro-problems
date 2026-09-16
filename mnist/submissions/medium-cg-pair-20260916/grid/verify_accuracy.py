"""Independently score frozen spatial-grid predictions against canonical labels."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mnist.code import data

SEEDS = range(2026091600, 2026091611)
PREDICTIONS_PER_DRAW = 10000
TOTAL_PREDICTIONS = len(SEEDS) * PREDICTIONS_PER_DRAW


def artifact_bytes(path):
    """Read original artifact bytes from a raw file, gzip file or ZIP archive."""
    path = Path(path)
    try:
        if path.exists():
            return path.read_bytes()
        compressed = Path(str(path) + '.gz')
        if compressed.exists():
            return gzip.decompress(compressed.read_bytes())
        with zipfile.ZipFile(path.parent / 'outputs.npz') as archive:
            return archive.read(path.name)
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError(f'{path}: cannot read artifact: {error}') from error


def artifact_hash(path):
    return hashlib.sha256(artifact_bytes(path)).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f'{path}: cannot read JSON: {error}') from error


def verify_freeze(results_dir):
    """Bind the completed prediction freeze to the recorded run identity."""
    freeze = read_json(results_dir / 'prediction_freeze.json')
    manifest = read_json(results_dir / 'run_manifest.json')
    try:
        identity_bytes = json.dumps(
            manifest['identity'], sort_keys=True, separators=(',', ':')
        ).encode()
        fingerprint = hashlib.sha256(identity_bytes).hexdigest()
        if manifest['fingerprint'] != fingerprint:
            raise ValueError('run_manifest.json: run fingerprint mismatch')
        if freeze['run_fingerprint'] != fingerprint:
            raise ValueError('prediction_freeze.json: run fingerprint mismatch')
        if (not freeze['complete'] or freeze['draws'] != len(SEEDS)
                or freeze['predictions'] != TOTAL_PREDICTIONS
                or freeze['test_labels_read'] is not False):
            raise ValueError('prediction_freeze.json: incomplete prediction freeze')
    except (KeyError, TypeError) as error:
        raise ValueError(f'{results_dir}: invalid run manifest or prediction freeze: {error}') from error
    return freeze, fingerprint


def load_canonical_labels(raw_dir):
    filename, expected_md5 = data.SOURCES['train_labels']
    path = raw_dir / filename
    try:
        if data.file_hash(path, 'md5') != expected_md5:
            raise ValueError('MNIST label hash mismatch')
        return data.read_idx(path, 60000, False)
    except (OSError, ValueError) as error:
        raise ValueError(f'{path}: {error}') from error


def load_verified_array(results_dir, item):
    """Check both serialized file bytes and the decoded NumPy array."""
    path = results_dir / item['filename']
    try:
        payload = artifact_bytes(path)
        if hashlib.sha256(payload).hexdigest() != item['file_sha256']:
            raise ValueError('artifact file hash mismatch')
        array = np.load(io.BytesIO(payload), allow_pickle=False)
        if array_hash(array) != item['array_sha256']:
            raise ValueError('artifact array hash mismatch')
        if list(array.shape) != item['shape'] or str(array.dtype) != item['dtype']:
            raise ValueError('artifact shape or dtype differs from recorded metadata')
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'{path.name}: {error}') from error
    return array


def verify_outputs(predictions, scores, record):
    """Check classifier outputs and the recorded intermediate finiteness checks."""
    if predictions.shape != (PREDICTIONS_PER_DRAW,) or predictions.dtype != np.int64:
        raise ValueError('predictions: expected shape (10000,) and dtype int64')
    if scores.shape != (PREDICTIONS_PER_DRAW, 10) or scores.dtype != np.float32:
        raise ValueError('scores: expected shape (10000, 10) and dtype float32')
    if np.any((predictions < 0) | (predictions > 9)):
        raise ValueError('predictions: class outside 0..9')
    if not np.isfinite(scores).all():
        raise ValueError('scores: nonfinite values')
    if not np.array_equal(scores.argmax(1), predictions):
        raise ValueError('predictions: do not match score argmax')
    if record['all_scores_finite'] is not True:
        raise ValueError('draw record: all_scores_finite is not true')
    for name, snapshot in record['snapshots'].items():
        if not snapshot['all_finite']:
            raise ValueError(f'snapshot {name}: nonfinite intermediate')


def verify_dataset(seed, record, truth):
    """Rebuild the canonical disjoint train/query split and return query labels."""
    order = np.random.Generator(np.random.PCG64(seed)).permutation(60000)
    train = order[:10000]
    query = order[10000:20000]
    if np.intersect1d(train, query).size:
        raise ValueError('dataset: train/query overlap')
    expected = record['input_arrays']
    arrays = {'train_indices': train, 'query_indices': query, 'test_labels': truth[query]}
    for name, array in arrays.items():
        if array_hash(array) != expected[name]:
            raise ValueError(f'dataset: frozen {name} hash mismatch')
    return arrays['test_labels']


def verify_draw(results_dir, draw, seed, freeze, fingerprint, truth):
    path = results_dir / f'draw_{draw:02d}.json'
    try:
        if artifact_hash(path) != freeze['draw_record_sha256'][path.name]:
            raise ValueError('draw record changed after freeze')
        record = read_json(path)
        if (record['draw'] != draw or record['seed'] != seed
                or record['run_fingerprint'] != fingerprint):
            raise ValueError('draw identity mismatch')
        predictions = load_verified_array(results_dir, record['predictions'])
        scores = load_verified_array(results_dir, record['scores'])
        verify_outputs(predictions, scores, record)
        query_labels = verify_dataset(seed, record, truth)
        correct = int(np.count_nonzero(predictions == query_labels))
        return {
            'draw': draw,
            'seed': seed,
            'correct': correct,
            'total': PREDICTIONS_PER_DRAW,
            'errors': PREDICTIONS_PER_DRAW - correct,
            'host_execution_seconds': record['host_execution_seconds'],
        }
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError(f'draw {draw:02d} ({path.name}): {error}') from error


def verify_accuracy(raw_dir, results_dir):
    """Verify every frozen draw and return the reproducible accuracy report."""
    raw_dir = Path(raw_dir)
    results_dir = Path(results_dir)
    freeze, fingerprint = verify_freeze(results_dir)
    truth = load_canonical_labels(raw_dir)
    rows = [
        verify_draw(results_dir, draw, seed, freeze, fingerprint, truth)
        for draw, seed in enumerate(SEEDS)
    ]
    total_correct = sum(row['correct'] for row in rows)
    total_errors = TOTAL_PREDICTIONS - total_correct
    return {
        'run_fingerprint': fingerprint,
        'prediction_freeze_sha256': artifact_hash(results_dir / 'prediction_freeze.json'),
        'independently_verified': True,
        'all_scores_finite': True,
        'all_argmax_predictions_verified': True,
        'completed_draws': len(SEEDS),
        'total_correct': total_correct,
        'total_predictions': TOTAL_PREDICTIONS,
        'total_errors': total_errors,
        'mean_accuracy_percent': total_correct / 1100,
        'mean_error_percent': total_errors / 1100,
        'sample_sd_percentage_points': statistics.stdev(row['correct'] / 100 for row in rows),
        'minimum_correct_for_2_percent': 107800,
        'meets_2_percent_target': total_correct >= 107800,
        'mean_host_execution_seconds': statistics.mean(row['host_execution_seconds'] for row in rows),
        'draws': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, required=True)
    args = parser.parse_args()
    report = verify_accuracy(args.raw_dir, args.results_dir)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
