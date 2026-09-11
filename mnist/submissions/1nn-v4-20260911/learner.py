"""Fixed 1-nearest-neighbor baseline; the learner never opens test_labels."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np


def array_hash(a):
    a = np.ascontiguousarray(a.astype(a.dtype.newbyteorder('<'), copy=False))
    return hashlib.sha256(a.tobytes()).hexdigest()


def predict(train_images, train_labels, test_images):
    """FP32 subtract, multiply, then ordered add; ties use first training row."""
    x = np.asarray(train_images, dtype=np.float32).reshape(len(train_images), -1).copy()
    y = np.asarray(train_labels, dtype=np.int64).copy()
    q = np.asarray(test_images, dtype=np.float32).reshape(len(test_images), -1)
    if len(x) == 0 or len(y) != len(x) or x.shape[1] != q.shape[1]:
        raise ValueError('Invalid shapes')
    if not np.isfinite(x).all() or not np.isfinite(q).all():
        raise ValueError('Nonfinite pixels')
    distances = np.zeros((len(q), len(x)), dtype=np.float32)
    for feature in range(x.shape[1]):
        difference = q[:, feature, None] - x[None, :, feature]
        squared = difference * difference
        distances = distances + squared
    nearest = np.argmin(distances, axis=1)  # NumPy chooses the first equal minimum.
    return y[nearest], nearest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--manifest', type=Path, default=Path('mnist/doc/dataset_manifest.json'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # This explicit allowlist is the only NPZ access in the learner.
    with np.load(args.data, allow_pickle=False) as z:
        allowed = {key: z[key] for key in ('train_images', 'train_labels', 'test_images')}
    manifest = json.loads(args.manifest.read_text())
    canonical = manifest['tiers']['small']['arrays']
    for key, value in allowed.items():
        if list(value.shape) != canonical[key]['shape'] or str(value.dtype) != canonical[key]['dtype']:
            raise ValueError(f'Wrong shape/dtype: {key}')
        if array_hash(value) != canonical[key]['sha256_c_order_little_endian']:
            raise ValueError(f'Noncanonical data: {key}')
    # One fixed candidate, train-only sanity check. No hyperparameter search.
    order = np.random.Generator(np.random.PCG64(20260911)).permutation(600)
    train, validation = order[:480], order[480:]
    vpred, _ = predict(allowed['train_images'][train], allowed['train_labels'][train],
                       allowed['train_images'][validation])
    vcorrect = int((vpred == allowed['train_labels'][validation]).sum())
    t0 = time.perf_counter()
    predictions, nearest = predict(**allowed)
    runtime = time.perf_counter() - t0
    np.save(args.output / 'predictions.npy', predictions)
    metadata = {
        'algorithm': 'fixed 1NN, squared Euclidean distance, FP32, ordered sum, first-row ties',
        'dataset_profile': manifest['profile'], 'dataset_seed': manifest['seed'],
        'dataset_sha256_npz': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'allowed_input_hashes': {k: array_hash(v) for k, v in allowed.items()},
        'predictions_sha256_int64_le': array_hash(predictions),
        'predictions': predictions.tolist(), 'nearest_train_rows': nearest.tolist(),
        'training_only_validation': {'seed': 20260911, 'train_count': 480,
           'validation_count': 120, 'correct': vcorrect, 'accuracy': vcorrect / 120,
           'training_rows': train.tolist(), 'validation_rows': validation.tolist(),
           'candidate_count': 1, 'hyperparameter_selection': 'none'},
        'cpu_reference_wall_seconds': runtime,
        'cpu_reference_scope': 'copy/memorize training + vectorized prediction; not Dally time to score',
        'software': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / 'cpu_results.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: metadata[k] for k in ('algorithm', 'dataset_sha256_npz',
                       'predictions_sha256_int64_le', 'cpu_reference_wall_seconds')}, indent=2))
    print(f'Train-only validation: {vcorrect}/120; saved 600 predictions without test-label access.')


if __name__ == '__main__':
    main()
